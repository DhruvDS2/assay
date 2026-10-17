"""
A2, MEASURED: the hardening wins on the FAULT axis, and here is the number.

A2 shipped the *disciplines* (bench/hardened.py): verify-before-stop and safe
recovery, wrapped around a scripted base brain. This module closes two gaps that
stood between "built" and "measured":

  1. **The disciplines wrap the real brain, not just a script.** `HardenedLLMPolicy`
     puts `HardenedPolicy` around an `LLMPolicy` — the same wrapper, the same
     `decide` contract, so a live model's loop gains verify-before-stop and safe
     recovery with no change to the model, the browser, or the grader. Construction
     only: building one makes no network call (the `LLMPolicy` is handed an injected
     client, or left to lazily need a key only if `decide` is ever called). The
     pass^k gain of a *live* model waits on A1's key; the seam that delivers it does not.

  2. **The win has a measurement.** A1's 73/105 is the mutation-only, no-memory
     reference — that matrix ran `seven_conditions()` (UI perturbations) and never
     touched a fault, so it cannot show a recovery win. A2's win lives on the
     *other* axis: `fault_conditions()` (clean + error_on_checkout +
     drop_checkout_response + expire_session). `measure_recovery` runs naive vs
     hardened across exactly those cells and reports the **double-order rate** per
     cell — the one number that separates a blind retry from a state-checked one.

Keyless by construction. The measurement reuses `naive_recovery_run_fn` and
`hardened_recovery_run_fn` from bench/run.py / bench/hardened.py, both of which
drive a *scripted* base brain through the real shop + fault middleware. No
`anthropic` import is reachable from `measure_recovery`; the live model is only a
possible base brain for the wrapper, exercised in A1.

Invariants honored (CLAUDE.md):
  * **Retry infra errors, never agent mistakes.** This module never raises on a
    wrong agent — it reuses the run_fns that return a finished (possibly wrong) run
    so the engine grades it instead of retrying. A fault is deterministic, so the
    double-order is the agent's to avoid, not infra's to retry away.
  * **Grade nodes are pure.** Classification reads the local DB (via
    `recovery_outcome`) and the grader reads the local SQLite file; no network.
  * **Integer cents.** Untouched — this layer only counts orders, never money.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Sequence

from bench.agent import Action, LLMPolicy, Observation, Policy, Step
from bench.hardened import (
    HardenedPolicy,
    TaskModel,
    hardened_policy_for,
    hardened_recovery_run_fn,
)
from bench.run import (
    Condition,
    RunFn,
    fault_conditions,
    naive_recovery_run_fn,
    recovery_outcome,
)
from bench.tasks import BUY_BLUE_SHIRT, BUY_SOCKS_X3, BUY_TOTE_AND_MUG, Task


# The three purchase tasks — the only ones a checkout fault can bite (cancel/empty
# don't place an order, so there's no double-order to make). The A2 measurement is
# about them.
PURCHASE_TASKS: tuple[Task, ...] = (BUY_BLUE_SHIRT, BUY_SOCKS_X3, BUY_TOTE_AND_MUG)


# ------------------------------------------------------- hardening around the real brain


class HardenedLLMPolicy(HardenedPolicy):
    """
    `HardenedPolicy` with an `LLMPolicy` as its base brain: the loop disciplines
    (verify-before-stop, safe recovery) wrap the real model unchanged.

    Construction makes no live call. The `LLMPolicy` is either handed an injected
    `client` (tests / A1 harness) or left to acquire one lazily — and even then
    only when `decide` is first invoked, never at construction. So you can build a
    `HardenedLLMPolicy` keyless to prove the seam; running it against a live model
    is A1's job.

    This is the only new behavior: the wrapper's state machine, the browser it
    drives, and the SQL grader are all inherited / reused without a line changed.
    """

    def __init__(
        self,
        model_spec: TaskModel,
        *,
        max_rounds: int = 1,
        llm_model: str = "claude-opus-4-8",
        client: object = None,
        max_history: int = 8,
        max_tokens: int = 1024,
    ) -> None:
        base: Policy = LLMPolicy(
            model=llm_model,
            client=client,
            max_history=max_history,
            max_tokens=max_tokens,
        )
        super().__init__(base, model_spec, max_rounds=max_rounds)


def hardened_llm_policy_for(
    task: Task,
    *,
    max_rounds: int = 1,
    llm_model: str = "claude-opus-4-8",
    client: object = None,
) -> Policy:
    """
    Build the hardened real-brain agent for a task.

    Mirrors `hardened_policy_for`, but the base brain is an `LLMPolicy` instead of
    a scripted stand-in. A task with no hardening model (cancel/empty) gets a bare
    `LLMPolicy` — nothing to verify-before-stop or recover for a non-purchase task
    that a checkout fault can't touch.
    """
    from bench.hardened import HARDENED_MODELS

    model_spec = HARDENED_MODELS.get(task.id)
    if model_spec is None:
        return LLMPolicy(model=llm_model, client=client)
    return HardenedLLMPolicy(
        model_spec, max_rounds=max_rounds, llm_model=llm_model, client=client
    )


# ------------------------------------------------------------------ the A/B measurement


@dataclass(frozen=True)
class CellResult:
    """One (strategy, task, condition) square: the outcome and whether it passed."""

    strategy: str
    task_id: str
    condition: str
    outcome: str          # "double" | "ok" | "none"  (from recovery_outcome)
    passed: bool

    @property
    def double(self) -> bool:
        return self.outcome == "double"


@dataclass(frozen=True)
class StrategyTally:
    """A strategy's totals across every fault cell — the headline numbers."""

    strategy: str
    cells: tuple[CellResult, ...]

    @property
    def n(self) -> int:
        return len(self.cells)

    @property
    def double_orders(self) -> int:
        return sum(1 for c in self.cells if c.double)

    @property
    def passed(self) -> int:
        return sum(1 for c in self.cells if c.passed)

    @property
    def double_rate(self) -> Optional[float]:
        return self.double_orders / self.n if self.n else None

    def double_orders_on(self, condition: str) -> int:
        return sum(1 for c in self.cells if c.condition == condition and c.double)


@dataclass(frozen=True)
class Measurement:
    """naive vs hardened across the fault axis — the A2 win, as data."""

    naive: StrategyTally
    hardened: StrategyTally

    def cell(self, strategy: str, task_id: str, condition: str) -> CellResult:
        tally = self.naive if strategy == "naive" else self.hardened
        for c in tally.cells:
            if c.task_id == task_id and c.condition == condition:
                return c
        raise KeyError((strategy, task_id, condition))

    def format(self) -> str:
        def pct(x: Optional[float]) -> str:
            return f"{x:.0%}" if x is not None else "  -"

        lines = [
            f"{'strategy':<10} {'task':<18} {'condition':<28} {'orders':>7} {'pass':>5}"
        ]
        lines.append("-" * 72)
        for tally in (self.naive, self.hardened):
            for c in tally.cells:
                lines.append(
                    f"{c.strategy:<10} {c.task_id:<18} {c.condition:<28} "
                    f"{c.outcome:>7} {('yes' if c.passed else 'no'):>5}"
                )
            lines.append("-" * 72)
        for tally in (self.naive, self.hardened):
            lines.append(
                f"{tally.strategy:<10} double-order {tally.double_orders}/{tally.n} "
                f"({pct(tally.double_rate)})   passed {tally.passed}/{tally.n}"
            )
        return "\n".join(lines)


def _drive(strategy: str, case, cond: Condition, *, max_rounds: int) -> None:
    """
    Run one strategy's scripted agent against a prepared case under `cond`.

    Both run_fns are the real v3 instruments: they drive a scripted base brain
    through the real shop + fault middleware and NEVER raise (a deterministic fault
    is the agent's to recover from, not infra's to retry). Keyless — no model here.
    """
    if strategy == "naive":
        naive_recovery_run_fn()(case, cond, 0)
    else:
        asyncio.run(hardened_recovery_run_fn(max_rounds=max_rounds)(case, cond, 0))


def measure_recovery(
    tasks: Sequence[Task] = PURCHASE_TASKS,
    conditions: Optional[Sequence[Condition]] = None,
    *,
    root: str | Path,
    seed_path: str | Path,
    max_rounds: int = 1,
) -> Measurement:
    """
    The A2 A/B: run naive and hardened recovery across the fault axis and tally the
    double-order rate per cell.

    `conditions` defaults to `fault_conditions()` — clean + the three faults — the
    axis A1's mutation matrix never exercised. Each cell gets a fresh DB copied
    from `seed_path` (isolation), the agent runs, then the case is classified by
    `recovery_outcome` (how many orders it left) and graded by SQL. Reuses the
    exact run_fns the v3 harness uses, so this is the same instrument, read for a
    different number.
    """
    from bench.env import prepare_case

    conds = tuple(conditions) if conditions is not None else fault_conditions()
    root = Path(root)

    def tally(strategy: str) -> StrategyTally:
        cells: list[CellResult] = []
        for task in tasks:
            for cond in conds:
                workdir = root / strategy / task.id / cond.name
                case = prepare_case(task, workdir, seed_path)
                _drive(strategy, case, cond, max_rounds=max_rounds)
                outcome = recovery_outcome(case)
                passed = grade_passed(case)
                cells.append(CellResult(strategy, task.id, cond.name, outcome, passed))
        return StrategyTally(strategy, tuple(cells))

    return Measurement(naive=tally("naive"), hardened=tally("hardened"))


def grade_passed(case) -> bool:
    """Pure SQL verdict for a finished case — reads only the local DB file."""
    from bench.grader import grade

    return grade(case).passed


# ----------------------------------------------------------------------- CLI


def main() -> int:
    import tempfile

    from bench.env import build_seed

    with tempfile.TemporaryDirectory() as tmp:
        seed = build_seed(f"{tmp}/seed.db")
        m = measure_recovery(root=tmp, seed_path=seed)

    print(m.format())
    print()

    drop = "fault_drop_checkout_response"
    naive_drop = m.naive.double_orders_on(drop)
    hardened_drop = m.hardened.double_orders_on(drop)
    print(
        f"A2 headline (axis: faults, not A1's mutations): under {drop}, "
        f"naive double-orders {naive_drop}/{len(PURCHASE_TASKS)} cells, "
        f"hardened {hardened_drop}/{len(PURCHASE_TASKS)}."
    )
    # The win is real iff the naive agent double-orders somewhere the hardened one
    # never does. (It won't exit non-zero on a tie: the measurement is the artifact,
    # not a gate.)
    won = m.hardened.double_orders < m.naive.double_orders
    print(
        "A2 MEASURED: hardening eliminates recovery double-orders the naive loop makes."
        if won
        else "A2: no double-order gap observed on this run (check the fault axis)."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
