"""
The graph builder: tasks × conditions × trials → a DAG the engine can run.

`bench/engine.py` runs a graph of async functions correctly but knows nothing
about shops, seeds, or graders. This module is the translation layer: it turns
"run 5 tasks under 7 site conditions, 3 trials each" into the concrete `Node`s
the engine schedules, and turns the engine's `RunReport` back into honest
metrics a human can read.

The shape of the graph:

    seed ──┬── run::task::cond::0 ── grade::task::cond::0 ──┐
           ├── run::task::cond::1 ── grade::task::cond::1 ──┤
           └── ...                                          ├── aggregate
                                                            │
           (one run+grade pair per task × condition × trial)┘

Three invariants from the roadmap are enforced by *where* work happens, not by
convention:

1. **Retry infra errors, never agent mistakes.** The run node raises only when
   the injected `RunFn` raises (an infra failure); an agent that finishes — right
   or wrong — makes `RunFn` return, so the run node returns and is never retried.
   Retrying a wrong answer would hide the flakiness the bench exists to measure.

2. **Every retry starts from a fresh DB.** `prepare_case` lives *inside* the run
   node, so each of the engine's retry attempts re-copies the seed. A previous
   attempt that dirtied the database can't leak into the next one.

3. **Grade nodes are pure.** A grade node reads one local SQLite file and nothing
   else — no network, no live app. It depends on its run node, so if the run was
   infra-skipped the grade skips too (the engine's default cascade), and every
   case that actually ran gets graded.
"""

from __future__ import annotations

import inspect
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional

from bench.engine import Node, NodeResult, RunReport, run_dag
from bench.env import build_seed, prepare_case
from bench.grader import grade
from bench.model import Case, Grade, Task

# --------------------------------------------------------------------- conditions


@dataclass(frozen=True)
class Condition:
    """
    A named variant of the site under test.

    `mutations` is an opaque payload handed to the `RunFn`, which is the only
    thing that knows how to serve a mutated app. `clean` (empty mutations) is
    the untouched baseline. v1's `app/mutations.py` adds six more, giving the
    seven conditions the roadmap's headline matrix runs against.
    """

    name: str
    mutations: tuple = ()
    faults: tuple = ()        # v3: infra faults, a SEPARATE axis from UI mutations


CLEAN = Condition("clean")
DEFAULT_CONDITIONS: tuple[Condition, ...] = (CLEAN,)


def seven_conditions() -> tuple[Condition, ...]:
    """
    The roadmap's headline matrix: `clean` plus one condition per UI mutation.

    Built from `app.mutations.SEVEN_CONDITIONS` (plain `(name, mutation-names)`
    data) so this module never has to know what any mutation *does* to the markup
    — the app layer owns that. The import is local to keep the graph builder above
    free of the app, exactly like the CLI's transport.
    """
    from app.mutations import SEVEN_CONDITIONS

    return tuple(Condition(name, muts) for name, muts in SEVEN_CONDITIONS)


def fault_conditions() -> tuple[Condition, ...]:
    """
    The v3 axis: a clean baseline plus one condition per infra fault. Faults go on
    a clean UI (not combined with mutations) so a failure is unambiguously the
    fault's doing. The point of the baseline is contrast — the same agent that
    succeeds on `clean` is the one we watch misbehave under a flaky checkout.
    """
    from app.faults import FAULT_NAMES

    return (CLEAN,) + tuple(
        Condition(f"fault_{name}", faults=(name,)) for name in FAULT_NAMES
    )


# A RunFn drives the agent against a prepared case under one condition and trial.
# CONTRACT: it must raise on infrastructure failure (so the engine retries) and
# return normally the moment the agent finishes — whether it succeeded or not.
# It must leave the case's final state in `case.db_path` for the grader to read.
RunFn = Callable[[Case, Condition, int], object]


# ------------------------------------------------------------------ node outputs


@dataclass(frozen=True)
class RunOutcome:
    """What a run node returns once the agent has finished (right or wrong)."""

    case: Case
    condition: str
    trial: int


@dataclass(frozen=True)
class CellGrade:
    """One trial's verdict, tagged with where in the matrix it sits."""

    task_id: str
    condition: str
    trial: int
    status: str                      # "passed" | "failed" | "skipped"
    grade: Optional[Grade] = None
    error: Optional[str] = None

    @property
    def attempted(self) -> bool:
        # The agent actually ran. Infra-skipped trials never got a fair shot.
        return self.status != "skipped"


# ------------------------------------------------------------------- node factories
# Factories (not inline closures) so each node captures its own task/cond/trial;
# a bare loop closure would bind late and every node would see the last values.


def _run_node_fn(task: Task, cond: Condition, trial: int, root: Path, run_fn: RunFn):
    async def fn(deps: dict[str, NodeResult]):
        seed = deps["seed"].value
        # Fresh DB per attempt: prepare_case is here, inside the retried unit.
        workdir = root / task.id / cond.name / str(trial)
        case = prepare_case(task, workdir, seed)
        # RunFn may be sync (the oracle) or async (the real agent loop); await
        # the latter so a transport error still raises => infra => engine retries.
        result = run_fn(case, cond, trial)   # raises => infra => retried on a new DB
        if inspect.isawaitable(result):
            await result
        return RunOutcome(case=case, condition=cond.name, trial=trial)

    return fn


def _grade_node_fn(task: Task, cond: Condition, trial: int, run_id: str):
    def fn(deps: dict[str, NodeResult]):
        outcome: RunOutcome = deps[run_id].value
        return grade(outcome.case)   # pure: reads outcome.case.db_path, nothing else

    return fn


def _aggregate_fn(coords: list[tuple[str, str, int, str]]):
    """coords: (task_id, condition, trial, grade_node_id) for every planned trial."""

    def fn(deps: dict[str, NodeResult]) -> "RunSummary":
        cell_grades: list[CellGrade] = []
        for task_id, cond, trial, gid in coords:
            res = deps.get(gid)
            if res is None or res.status == "skipped":
                cell_grades.append(CellGrade(task_id, cond, trial, "skipped"))
            elif res.status == "failed":
                # A grade node should not raise (grade() swallows check crashes),
                # but if it does the trial ran — count it attempted, not passed.
                cell_grades.append(
                    CellGrade(task_id, cond, trial, "failed", error=repr(res.error))
                )
            else:
                g: Grade = res.value
                status = "passed" if g.passed else "failed"
                cell_grades.append(CellGrade(task_id, cond, trial, status, grade=g))
        return RunSummary.from_cell_grades(cell_grades)

    return fn


# ------------------------------------------------------------------- graph builder


def build_graph(
    tasks: tuple[Task, ...],
    conditions: tuple[Condition, ...],
    trials: int,
    *,
    root: str | Path,
    run_fn: RunFn,
    seed_path: str | Path | None = None,
    run_retries: int = 2,
    run_timeout_s: Optional[float] = None,
    solari_group: str = "solari",
) -> list[Node]:
    """
    Build the full DAG: one seed root, a run+grade pair per (task, condition,
    trial), and one aggregate sink.

    - Run nodes share `solari_group` so the engine can cap them at the plan's
      concurrency limit, and carry `run_retries`/`run_timeout_s` because the
      agent run is the only place infra can fail. The per-node timeout is the
      real deadline (a Solari `timeout_ms` is only a rolling idle window).
    - Grade nodes are ungrouped, un-retried, and depend solely on their run node.
    - The aggregate tolerates dependency failure so it still produces a summary
      when some trials were infra-skipped.
    """
    if trials < 1:
        raise ValueError(f"trials must be >= 1, got {trials}")

    root = Path(root)
    seed_file = Path(seed_path) if seed_path is not None else root / "seed.db"

    nodes: list[Node] = [Node("seed", lambda _deps: build_seed(seed_file))]
    coords: list[tuple[str, str, int, str]] = []

    for task in tasks:
        for cond in conditions:
            for trial in range(trials):
                run_id = f"run::{task.id}::{cond.name}::{trial}"
                grade_id = f"grade::{task.id}::{cond.name}::{trial}"
                nodes.append(
                    Node(
                        run_id,
                        _run_node_fn(task, cond, trial, root, run_fn),
                        deps=("seed",),
                        group=solari_group,
                        retries=run_retries,
                        timeout_s=run_timeout_s,
                    )
                )
                nodes.append(
                    Node(
                        grade_id,
                        _grade_node_fn(task, cond, trial, run_id),
                        deps=(run_id,),
                    )
                )
                coords.append((task.id, cond.name, trial, grade_id))

    nodes.append(
        Node(
            "aggregate",
            _aggregate_fn(coords),
            deps=tuple(c[3] for c in coords),
            tolerate_dep_failure=True,
        )
    )
    return nodes


# ----------------------------------------------------------------------- metrics


@dataclass(frozen=True)
class Cell:
    """One (task, condition) square of the matrix, across its trials."""

    task_id: str
    condition: str
    grades: tuple[CellGrade, ...]

    @property
    def planned(self) -> int:
        return len(self.grades)

    @property
    def attempted(self) -> int:
        return sum(1 for g in self.grades if g.attempted)

    @property
    def passed(self) -> int:
        return sum(1 for g in self.grades if g.status == "passed")

    @property
    def skipped(self) -> int:
        return self.planned - self.attempted

    @property
    def pass_rate(self) -> Optional[float]:
        """passed / attempted — None when nothing was attempted (all skipped)."""
        return self.passed / self.attempted if self.attempted else None

    @property
    def pass_hat_k(self) -> bool:
        """
        Reliability: every planned trial was attempted AND passed. A single
        infra skip or one wrong run breaks it — that's the point.
        """
        return self.planned > 0 and self.passed == self.planned


@dataclass(frozen=True)
class RunSummary:
    cells: tuple[Cell, ...]

    @classmethod
    def from_cell_grades(cls, cell_grades: list[CellGrade]) -> "RunSummary":
        buckets: dict[tuple[str, str], list[CellGrade]] = {}
        order: list[tuple[str, str]] = []
        for cg in cell_grades:
            key = (cg.task_id, cg.condition)
            if key not in buckets:
                buckets[key] = []
                order.append(key)
            buckets[key].append(cg)
        cells = tuple(
            Cell(t, c, tuple(sorted(buckets[(t, c)], key=lambda g: g.trial)))
            for (t, c) in order
        )
        return cls(cells=cells)

    # Honest totals: attempted and planned are reported separately, and skipped
    # (infra never let the agent try) is never folded into "failed".
    @property
    def planned(self) -> int:
        return sum(c.planned for c in self.cells)

    @property
    def attempted(self) -> int:
        return sum(c.attempted for c in self.cells)

    @property
    def passed(self) -> int:
        return sum(c.passed for c in self.cells)

    @property
    def skipped(self) -> int:
        return self.planned - self.attempted

    @property
    def passed_over_attempted(self) -> Optional[float]:
        return self.passed / self.attempted if self.attempted else None

    @property
    def passed_over_planned(self) -> Optional[float]:
        return self.passed / self.planned if self.planned else None

    @property
    def reliable_cells(self) -> list[Cell]:
        """Cells that passed every trial (pass^k)."""
        return [c for c in self.cells if c.pass_hat_k]

    def format(self) -> str:
        def pct(x: Optional[float]) -> str:
            return f"{x:.0%}" if x is not None else "  –"

        lines = [f"{'task':<20} {'condition':<12} {'pass':>7} {'skip':>5} {'pass^k':>7}"]
        lines.append("-" * 56)
        for c in self.cells:
            passes = f"{c.passed}/{c.attempted}" if c.attempted else "  –"
            lines.append(
                f"{c.task_id:<20} {c.condition:<12} {passes:>7} "
                f"{c.skipped:>5} {('yes' if c.pass_hat_k else 'no'):>7}"
            )
        lines.append("-" * 56)
        lines.append(
            f"planned {self.planned}  attempted {self.attempted}  "
            f"passed {self.passed}  skipped {self.skipped}"
        )
        lines.append(
            f"passed/attempted {pct(self.passed_over_attempted)}   "
            f"passed/planned {pct(self.passed_over_planned)}   "
            f"reliable cells {len(self.reliable_cells)}/{len(self.cells)}"
        )
        return "\n".join(lines)


# --------------------------------------------------------------------- top-level


async def run_matrix(
    tasks: tuple[Task, ...],
    conditions: tuple[Condition, ...],
    trials: int,
    *,
    root: str | Path,
    run_fn: RunFn,
    seed_path: str | Path | None = None,
    run_retries: int = 2,
    run_timeout_s: Optional[float] = None,
    max_concurrency: int = 8,
    solari_concurrency: int = 4,
    solari_group: str = "solari",
) -> tuple[RunSummary, RunReport]:
    """
    Build the graph, run it, and return (summary, raw report).

    `solari_concurrency` caps how many agent runs happen at once — the plan
    limit becomes the engine's per-group cap. The summary is the aggregate
    node's value; the report is every node's result for debugging.
    """
    nodes = build_graph(
        tasks,
        conditions,
        trials,
        root=root,
        run_fn=run_fn,
        seed_path=seed_path,
        run_retries=run_retries,
        run_timeout_s=run_timeout_s,
        solari_group=solari_group,
    )
    report = await run_dag(
        nodes,
        max_concurrency=max_concurrency,
        group_limits={solari_group: solari_concurrency},
    )
    summary: RunSummary = report["aggregate"].value
    return summary, report


# ----------------------------------------------------------------------- CLI
# A runnable end-to-end demonstration: the oracle for every task, driven through
# the real engine instead of selfcheck's straight-line loop. Transport lives
# here, local to main, so the graph builder above stays free of the app/browser.


def _compose_app(db_path, cond: Condition):
    """Build the app a cell actually serves: clean -> mutated -> faulty, in that order."""
    from app.mutations import create_mutated_app

    app = create_mutated_app(db_path, mutations=cond.mutations)
    if cond.faults:
        from app.faults import inject_faults

        app = inject_faults(app, cond.faults)
    return app


def naive_recovery_run_fn(retries: int = 1) -> RunFn:
    """
    The simplest 'recovery' agent there is: run the task's oracle, and on ANY error
    re-run the whole thing from scratch (fresh login, re-add, re-checkout).

    This is the v3 instrument. The fault is deterministic, not flaky, so recovery
    is the *agent's* job — this RunFn NEVER raises (it would only trigger an engine
    retry against the same fault). Under `drop_checkout_response` the first attempt
    commits the order then errors, so the retry commits a SECOND one: the classic
    recovery double-order, which `exactly_one_new_order` then catches in SQL.
    """
    from fastapi.testclient import TestClient

    def run_fn(case: Case, cond: Condition, trial: int) -> None:
        for _ in range(retries + 1):
            try:
                with TestClient(_compose_app(case.db_path, cond)) as client:
                    case.task.oracle(client)
                return                      # clean run — done
            except Exception:
                continue                    # naive: the agent just tries again
        return                              # gave up; a finished (wrong) agent, not infra

    return run_fn


def recovery_outcome(case: Case) -> str:
    """Classify a finished case by how many orders the agent left behind."""
    from app.db import connect
    from bench.tasks import new_orders

    conn = connect(case.db_path)
    try:
        n = len(new_orders(conn, case.baseline, case.task.email))
    finally:
        conn.close()
    return "double" if n >= 2 else "ok" if n == 1 else "none"


def _testclient_oracle_run_fn() -> RunFn:
    """
    A RunFn that runs each task's oracle in-process via Starlette's TestClient,
    against the app *mutated* for this cell's condition. The oracle drives
    endpoints, so it sails through every surface mutation — which is the point:
    it proves each cell is achievable before any real agent is scored there.
    """
    from fastapi.testclient import TestClient

    from app.mutations import create_mutated_app

    def run_fn(case: Case, cond: Condition, trial: int) -> None:
        app = create_mutated_app(case.db_path, mutations=cond.mutations)
        with TestClient(app) as client:
            case.task.oracle(client)

    return run_fn


def main() -> int:
    import asyncio
    import sys
    import tempfile

    from bench.tasks import TASKS

    with tempfile.TemporaryDirectory() as tmp:
        summary, _report = asyncio.run(
            run_matrix(
                TASKS,
                seven_conditions(),
                trials=3,
                root=tmp,
                run_fn=_testclient_oracle_run_fn(),
            )
        )
    print(summary.format())
    # The oracle must pass every planned trial with nothing skipped, or the
    # graph builder — not the tasks — is broken.
    ok = summary.passed == summary.planned and summary.skipped == 0
    print("\nrun graph OK: oracle passed the full matrix" if ok
          else "\nrun graph BROKEN: oracle did not sweep the matrix")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
