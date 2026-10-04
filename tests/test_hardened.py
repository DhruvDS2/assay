"""
A2: the hardened agent holds up where the naive loop breaks — proven against the
real shop + faults, keyless.

Two disciplines, two headline tests:

  * safe recovery — under `drop_checkout_response` the order commits but the client
    sees a 500. The naive recovery re-places it (double-order, FAIL). The hardened
    agent checks /orders, sees the order already there, and stops: exactly one
    order, the cell PASSES.
  * verify-before-stop — a base brain that quits before checking out would leave
    the task unfinished. The hardened agent re-reads state, sees no order, and
    finishes the checkout: the cell PASSES.
"""

from __future__ import annotations

import asyncio

from bench.agent import Action, TestClientBrowser, ScriptedPolicy, run
from bench.env import prepare_case
from bench.grader import grade
from bench.hardened import (
    HardenedPolicy,
    HARDENED_MODELS,
    _login_steps,
    hardened_policy_for,
    hardened_recovery_run_fn,
)
from bench.run import (
    CLEAN,
    Condition,
    _compose_app,
    fault_conditions,
    naive_recovery_run_fn,
    recovery_outcome,
    run_matrix,
)
from bench.tasks import BUY_BLUE_SHIRT, BUY_SOCKS_X3, BUY_TOTE_AND_MUG

PURCHASE = (BUY_BLUE_SHIRT, BUY_SOCKS_X3, BUY_TOTE_AND_MUG)
DROP = Condition("drop", faults=("drop_checkout_response",))


def _run_hardened(case, cond, **kw):
    asyncio.run(hardened_recovery_run_fn(**kw)(case, cond, 0))


# ---------------------------------------------------------------- safe recovery


def test_hardened_recovers_drop_without_double_order(seed, tmp_path):
    """The A2 headline: drop_checkout_response → one order, cell passes."""
    case = prepare_case(BUY_BLUE_SHIRT, tmp_path / "hardened", seed)
    _run_hardened(case, DROP)
    assert recovery_outcome(case) == "ok"        # exactly one order — NOT double
    assert grade(case).passed                    # the order was correct; the cell passes


def test_naive_double_orders_where_hardened_does_not(seed, tmp_path):
    """Direct A/B against the v3 instrument, same fault, same task."""
    naive = prepare_case(BUY_BLUE_SHIRT, tmp_path / "naive", seed)
    naive_recovery_run_fn()(naive, DROP, 0)
    assert recovery_outcome(naive) == "double"   # the blind retry places a second order
    assert not grade(naive).passed

    hardened = prepare_case(BUY_BLUE_SHIRT, tmp_path / "hard", seed)
    _run_hardened(hardened, DROP)
    assert recovery_outcome(hardened) == "ok"    # the state-checked retry does not


def test_hardened_never_double_orders_on_any_purchase_task_or_fault(seed, tmp_path):
    for task in PURCHASE:
        for cond in fault_conditions():
            case = prepare_case(task, tmp_path / task.id / cond.name, seed)
            _run_hardened(case, cond)
            assert recovery_outcome(case) != "double", (task.id, cond.name)


# ---------------------------------------------------------------- verify-before-stop


def _deficient_base(task):
    """A base brain that logs in and gives up — it never places the order."""
    return ScriptedPolicy(_login_steps() + [Action.stop("done (wrongly)")])


def test_verify_before_stop_finishes_a_premature_agent(seed, tmp_path):
    """
    The base brain stops without ordering. On its own that's a failed run; wrapped,
    the hardened loop sees no order on file and completes the checkout.
    """
    case = prepare_case(BUY_BLUE_SHIRT, tmp_path / "premature", seed)
    policy = HardenedPolicy(_deficient_base(BUY_BLUE_SHIRT), HARDENED_MODELS["buy_blue_shirt"])
    browser = TestClientBrowser(_compose_app(case.db_path, CLEAN))
    asyncio.run(run(BUY_BLUE_SHIRT.instruction("http://shop"), browser, policy, max_steps=40))
    assert recovery_outcome(case) == "ok"
    assert grade(case).passed


# ---------------------------------------------------------------- through the matrix


def test_hardened_passes_drop_cells_through_the_engine(seed, tmp_path):
    summary, _report = asyncio.run(
        run_matrix(
            PURCHASE,
            fault_conditions(),
            trials=1,
            root=tmp_path / "matrix",
            run_fn=hardened_recovery_run_fn(),
        )
    )
    by = {(c.task_id, c.condition): c for c in summary.cells}
    for task in PURCHASE:
        assert by[(task.id, "clean")].passed == 1                          # baseline still works
        assert by[(task.id, "fault_drop_checkout_response")].passed == 1   # recovered, not doubled
    assert summary.skipped == 0                                            # nothing infra-skipped
