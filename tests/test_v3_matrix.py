"""
v3 wired into the matrix: faults are a condition axis, and a naive-recovery agent's
double-order is caught in SQL — both directly and through the real engine.

The headline: under `drop_checkout_response` the agent's retry places a SECOND
order. `recovery_outcome` classifies it as "double" and `exactly_one_new_order`
fails. The clean baseline, same agent, is a single clean order.
"""

from __future__ import annotations

import asyncio

import pytest

from bench.env import prepare_case
from bench.grader import grade
from bench.run import (
    CLEAN,
    Condition,
    fault_conditions,
    naive_recovery_run_fn,
    recovery_outcome,
    run_matrix,
)
from bench.tasks import BUY_BLUE_SHIRT


def _run(case, cond):
    naive_recovery_run_fn()(case, cond, 0)   # the run_fn is sync; drive it directly


def test_drop_checkout_response_makes_recovery_double_order(seed, tmp_path):
    case = prepare_case(BUY_BLUE_SHIRT, tmp_path / "drop", seed)
    _run(case, Condition("drop", faults=("drop_checkout_response",)))
    assert recovery_outcome(case) == "double"
    failed = {r.name for r in grade(case).results if not r.passed}
    assert "exactly_one_new_order" in failed


def test_clean_recovery_is_one_clean_order(seed, tmp_path):
    case = prepare_case(BUY_BLUE_SHIRT, tmp_path / "clean", seed)
    _run(case, CLEAN)
    assert recovery_outcome(case) == "ok"
    assert grade(case).passed


def test_error_on_checkout_loses_the_order(seed, tmp_path):
    case = prepare_case(BUY_BLUE_SHIRT, tmp_path / "err", seed)
    _run(case, Condition("err", faults=("error_on_checkout",)))
    assert recovery_outcome(case) == "none"
    assert not grade(case).passed


def test_fault_axis_runs_through_the_matrix(seed, tmp_path):
    summary, _report = asyncio.run(
        run_matrix(
            (BUY_BLUE_SHIRT,),
            fault_conditions(),
            trials=1,
            root=tmp_path / "matrix",
            run_fn=naive_recovery_run_fn(),
        )
    )
    by_cond = {c.condition: c for c in summary.cells}
    assert by_cond["clean"].passed == 1                          # baseline succeeds
    assert by_cond["fault_drop_checkout_response"].passed == 0   # double-order caught
    assert by_cond["fault_error_on_checkout"].passed == 0        # order lost
    assert by_cond["fault_expire_session"].passed == 0           # bounced to login
    # Nothing was infra-skipped: the agent "finished" every cell (right or wrong).
    assert summary.skipped == 0
