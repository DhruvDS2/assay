"""
v2 deliverable: the grader-comparison pass — the evidence for SQL-as-truth.

`bench/graders.py` gives us the pieces (SQL / DOM / LLM-screenshot graders,
`grade_all`, `false_success_rate`). This module is the *pass* that drives them:
it runs a population of agent behaviours — a correct run and the single most
plausible wrong run per task — across the site conditions, grades every one with
all three graders, and prints the disagreement table whose headline is each cheap
grader's **false-success rate**: how often it certified "done" when SQL knows the
agent got it wrong. That number is the whole argument for grading by database.

Why a *population* and not just the oracle: if every run succeeds, every grader
agrees and the table is empty. The disagreement — and the lie — only shows up on
the wrong runs, which is exactly where real agents land under mutation and fault.

Keyless by default. SQL + DOM run with no network, so the printed numbers are
always real. The LLM screenshot judge is added only when `ANTHROPIC_API_KEY` is
set — never faked into the headline. The keyless finding stands on its own: even
the DOM grader false-passes `cancel_latest_order` and `empty_cart`, because the
screen looks right and only the database knows an order was wrongly left standing.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional

from bench.confidence import false_success_band
from bench.env import build_seed, prepare_case
from bench.graders import (
    DomGrader,
    LlmScreenshotGrader,
    SqlGrader,
    Verdict,
    false_success_rate,
    grade_all,
)
from bench.receipt_grader import ReceiptGrader
from bench.model import Task
from bench.tasks import (
    ALICE,
    BUY_BLUE_SHIRT,
    BUY_SOCKS_X3,
    BUY_TOTE_AND_MUG,
    CANCEL_LATEST_ORDER,
    EMPTY_CART,
    add,
    checkout,
    login,
)

# --------------------------------------------------------------- the population
# For each task: the oracle (correct → SQL passes) and the single most plausible
# wrong run (→ SQL fails). The wrong runs mirror the adversarial tests in
# tests/test_ground_truth.py — the mistakes a real agent actually makes. The
# comment on each names the grader it's expected to fool.


def _wrong_blue_shirt_item(c):
    # Bought the look-alike White shirt. DOM reads the item name and catches it.
    login(c, *ALICE)
    add(c, "SHIRT-WHT")
    checkout(c, "12 Elm St, Austin, TX 78701")


def _wrong_socks_qty(c):
    # Bought one pair instead of three. DOM looks for a "3" on the order page.
    login(c, *ALICE)
    add(c, "SOCK-WOOL", 1)
    checkout(c, "400 Main St, Dallas, TX 75201")


def _wrong_tote_keeps_cap(c):
    # Ignored the pre-existing Black Cap in the cart, so it ships too.
    login(c, *ALICE)
    add(c, "TOTE-CNV")
    add(c, "MUG-ENML")
    checkout(c, "9 Oak Ave, Houston, TX 77002")


def _wrong_cancel_older(c):
    # Cancelled the OLDER order (id 1, the belt) instead of the latest. The orders
    # list still shows one "cancelled" and one "placed" — so the DOM grader is
    # fooled. Only SQL knows the wrong order was cancelled.
    login(c, *ALICE)
    c.post("/orders/1/cancel")


def _wrong_empty_by_buying(c):
    # "Emptied" the cart by checking out. The cart page truthfully says it's empty
    # — DOM passes — but an order was placed, which only SQL's side-effect check sees.
    login(c, *ALICE)
    checkout(c, "1 Anywhere St, Austin, TX 78701")


@dataclass(frozen=True)
class Run:
    """One agent behaviour on one task: a correct or a plausibly-wrong run."""

    task: Task
    label: str                       # "correct" | "wrong"
    agent: Callable[[object], None]


DEFAULT_RUNS: tuple[Run, ...] = (
    Run(BUY_BLUE_SHIRT, "correct", BUY_BLUE_SHIRT.oracle),
    Run(BUY_BLUE_SHIRT, "wrong", _wrong_blue_shirt_item),
    Run(BUY_SOCKS_X3, "correct", BUY_SOCKS_X3.oracle),
    Run(BUY_SOCKS_X3, "wrong", _wrong_socks_qty),
    Run(BUY_TOTE_AND_MUG, "correct", BUY_TOTE_AND_MUG.oracle),
    Run(BUY_TOTE_AND_MUG, "wrong", _wrong_tote_keeps_cap),
    Run(CANCEL_LATEST_ORDER, "correct", CANCEL_LATEST_ORDER.oracle),
    Run(CANCEL_LATEST_ORDER, "wrong", _wrong_cancel_older),
    Run(EMPTY_CART, "correct", EMPTY_CART.oracle),
    Run(EMPTY_CART, "wrong", _wrong_empty_by_buying),
)


# --------------------------------------------------------------- run the comparison


@dataclass(frozen=True)
class CellVerdicts:
    """Every grader's verdict on one (task, condition, run) — one row to score."""

    task_id: str
    condition: str
    label: str
    verdicts: tuple[Verdict, ...]

    def by_grader(self) -> dict[str, Verdict]:
        return {v.grader: v for v in self.verdicts}


def default_graders() -> tuple[list, bool]:
    """
    SQL + DOM always; the LLM screenshot judge only when a key is present, so the
    headline never contains a faked number. Returns (graders, screenshot_live).
    """
    graders = [SqlGrader(), DomGrader(), ReceiptGrader()]
    live = bool(os.environ.get("ANTHROPIC_API_KEY"))
    if live:
        graders.append(LlmScreenshotGrader())
    return graders, live


def run_comparison(
    runs: tuple[Run, ...],
    conditions,
    graders,
    *,
    root: str | Path,
    seed_path: str | Path | None = None,
) -> list[CellVerdicts]:
    """
    For every (condition, run): fresh case → drive the agent against the app
    *mutated* for that condition → grade with all graders against the same
    mutated surface. One `CellVerdicts` per cell.

    The agent runs against the mutated app so it drives exactly the surface the
    graders then read — form actions/names are byte-identical under mutation
    (the mutation invariant), so the wrong run stays wrong in the same way.
    """
    from fastapi.testclient import TestClient

    from app.mutations import create_mutated_app

    root = Path(root)
    seed = build_seed(Path(seed_path) if seed_path is not None else root / "seed.db")

    cells: list[CellVerdicts] = []
    for cond in conditions:
        for run in runs:
            workdir = root / run.task.id / cond.name / run.label
            case = prepare_case(run.task, workdir, seed)
            with TestClient(create_mutated_app(case.db_path, mutations=cond.mutations)) as c:
                run.agent(c)
            verdicts = grade_all(case, graders, mutations=cond.mutations)
            cells.append(CellVerdicts(run.task.id, cond.name, run.label, tuple(verdicts)))
    return cells


# ----------------------------------------------------------------------- report


def _disagreements(cells: list[CellVerdicts]) -> list[CellVerdicts]:
    """Cells where some non-SQL grader's verdict differs from SQL truth."""
    out = []
    for cell in cells:
        by = cell.by_grader()
        truth = by.get("sql")
        if truth is None:
            continue
        if any(v.passed != truth.passed for v in cell.verdicts
               if v.grader != "sql" and not v.abstained):
            out.append(cell)
    return out


def format_report(cells: list[CellVerdicts], *, screenshot_live: bool) -> str:
    rows = [list(c.verdicts) for c in cells]
    scores = false_success_rate(rows)

    lines: list[str] = []
    lines.append(f"grader comparison — {len(cells)} cells "
                 f"({len({c.task_id for c in cells})} tasks × "
                 f"{len({c.condition for c in cells})} conditions × correct/wrong)")
    lines.append("")

    # Headline: each cheap grader scored against SQL truth, with a 95% Wilson band
    # on the false-success rate so a 0/70 reads as "0, give or take", not "proven 0".
    lines.append(f"{'grader':<12} {'agree':>7} {'false-success':>15} "
                 f"{'95% CI':>15} {'false-failure':>15}")
    lines.append("-" * 68)
    for name, s in scores.items():
        b = false_success_band(s)
        lines.append(
            f"{name:<12} {f'{s.agree}/{s.total}':>7} "
            f"{f'{s.false_success}/{s.total} ({s.false_success_rate:.0%})':>15} "
            f"{f'[{b.lower:.0%}-{b.upper:.0%}]':>15} "
            f"{f'{s.false_failure}/{s.total}':>15}"
        )
    lines.append("-" * 68)

    # The receipts: every cell where a cheap grader lied, and which way.
    dis = _disagreements(cells)
    lines.append("")
    lines.append(f"disagreements with SQL ({len(dis)}):")
    if not dis:
        lines.append("  (none — every grader agreed with the database)")
    for cell in dis:
        by = cell.by_grader()
        truth = by["sql"]
        for v in cell.verdicts:
            if v.grader == "sql" or v.abstained or v.passed == truth.passed:
                continue
            kind = "FALSE-SUCCESS" if v.passed else "false-failure"
            lines.append(
                f"  {kind:<13} {cell.task_id}/{cell.label} [{cell.condition}] "
                f"— {v.grader} said {'PASS' if v.passed else 'FAIL'}, "
                f"SQL said {'PASS' if truth.passed else 'FAIL'}"
            )

    lines.append("")
    if screenshot_live:
        lines.append("screenshot judge: LIVE (ANTHROPIC_API_KEY set)")
    else:
        lines.append("screenshot judge: OMITTED — set ANTHROPIC_API_KEY to score it "
                     "live. SQL/DOM numbers above are real.")
    return "\n".join(lines)


# ----------------------------------------------------------------------- CLI


def main() -> int:
    import tempfile

    from bench.run import seven_conditions

    graders, live = default_graders()
    with tempfile.TemporaryDirectory() as tmp:
        cells = run_comparison(DEFAULT_RUNS, seven_conditions(), graders, root=tmp)
    print(format_report(cells, screenshot_live=live))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
