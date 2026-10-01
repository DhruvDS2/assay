"""
The graph builder tells the truth.

selfcheck proves the *grader* separates success from failure on a straight
loop. These tests prove the *run graph* preserves that truth once it runs
through the real engine — and that the three invariants hold where it counts:

    oracle sweeps the matrix        every planned trial passes, nothing skipped
    noop is caught                  agent finished wrong => attempted, not passed
    infra failure is skipped        a raising RunFn => skipped, never "failed"
    retries start from a fresh DB   a dirtied attempt can't leak into the next
"""

from __future__ import annotations

import asyncio
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from app.db import connect
from app.main import create_app
from bench.env import build_seed
from bench.run import (
    CLEAN,
    DEFAULT_CONDITIONS,
    Condition,
    build_graph,
    run_matrix,
)
from bench.tasks import BUY_BLUE_SHIRT, TASKS


def oracle_run_fn(case, cond, trial) -> None:
    with TestClient(create_app(case.db_path)) as client:
        case.task.oracle(client)


def noop_run_fn(case, cond, trial) -> None:
    """The agent 'finished' having done nothing. Not an infra failure."""


def always_infra_fail(case, cond, trial) -> None:
    raise RuntimeError("browser would not launch")


@pytest.fixture
def seed(tmp_path: Path) -> Path:
    return build_seed(tmp_path / "seed.db")


# --------------------------------------------------------------------- structure


def test_graph_has_a_run_and_grade_per_trial_plus_seed_and_aggregate(tmp_path):
    tasks = TASKS[:2]
    conditions = (CLEAN, Condition("mutant"))
    trials = 3
    nodes = build_graph(
        tasks, conditions, trials, root=tmp_path, run_fn=noop_run_fn
    )
    ids = {n.id for n in nodes}

    cases = len(tasks) * len(conditions) * trials
    assert len(nodes) == 1 + cases * 2 + 1  # seed + (run+grade)/case + aggregate
    assert "seed" in ids and "aggregate" in ids
    assert sum(i.startswith("run::") for i in ids) == cases
    assert sum(i.startswith("grade::") for i in ids) == cases


def test_run_nodes_carry_retries_and_the_solari_group(tmp_path):
    nodes = build_graph(
        (BUY_BLUE_SHIRT,), DEFAULT_CONDITIONS, 1,
        root=tmp_path, run_fn=noop_run_fn, run_retries=2, solari_group="solari",
    )
    run = next(n for n in nodes if n.id.startswith("run::"))
    grade = next(n for n in nodes if n.id.startswith("grade::"))
    agg = next(n for n in nodes if n.id == "aggregate")

    assert run.group == "solari" and run.retries == 2
    assert grade.group is None and grade.retries == 0     # grade nodes are pure
    assert agg.tolerate_dep_failure                       # summarizes the wreckage


def test_trials_must_be_positive(tmp_path):
    with pytest.raises(ValueError):
        build_graph((BUY_BLUE_SHIRT,), DEFAULT_CONDITIONS, 0,
                    root=tmp_path, run_fn=noop_run_fn)


# ------------------------------------------------------------------- calibration


def test_oracle_sweeps_the_full_matrix(tmp_path, seed):
    summary, _ = asyncio.run(
        run_matrix(TASKS, DEFAULT_CONDITIONS, 2,
                   root=tmp_path, run_fn=oracle_run_fn, seed_path=seed)
    )
    assert summary.planned == len(TASKS) * 2
    assert summary.passed == summary.planned
    assert summary.skipped == 0
    assert summary.passed_over_planned == 1.0
    assert len(summary.reliable_cells) == len(summary.cells) == len(TASKS)


def test_noop_is_attempted_but_never_passes(tmp_path, seed):
    summary, _ = asyncio.run(
        run_matrix(TASKS, DEFAULT_CONDITIONS, 1,
                   root=tmp_path, run_fn=noop_run_fn, seed_path=seed)
    )
    # Doing nothing is a finished agent, not an infra failure: fully attempted,
    # zero passes, nothing skipped. This is the run-graph analogue of selfcheck.
    assert summary.attempted == summary.planned
    assert summary.skipped == 0
    assert summary.passed == 0
    assert summary.reliable_cells == []


# ---------------------------------------------------------------------- invariants


def test_infra_failure_is_skipped_not_failed(tmp_path, seed):
    summary, report = asyncio.run(
        run_matrix((BUY_BLUE_SHIRT,), DEFAULT_CONDITIONS, 2,
                   root=tmp_path, run_fn=always_infra_fail, seed_path=seed,
                   run_retries=1)
    )
    # A raising RunFn exhausts retries -> run node "failed" -> grade cascades to
    # "skipped". The summary must count those as skipped, never folded into fails.
    assert summary.attempted == 0
    assert summary.passed == 0
    assert summary.skipped == summary.planned == 2
    assert all(not c.pass_hat_k for c in summary.cells)
    # And the run node really did retry (1 extra attempt = 2 invocations).
    run_res = report["run::buy_blue_shirt::clean::0"]
    assert run_res.status == "failed" and run_res.attempts == 2


def test_every_retry_starts_from_a_fresh_db(tmp_path, seed):
    """A dirtied, failed attempt must not leak into the next attempt's DB."""
    calls = {"n": 0}

    def flaky_run_fn(case, cond, trial):
        # Invariant under test: at the START of every attempt the DB is clean,
        # i.e. prepare_case re-copied the seed and re-ran setup.
        conn = connect(case.db_path)
        new_orders = conn.execute(
            "SELECT COUNT(*) FROM orders WHERE id > ?",
            (case.baseline.max_order_id,),
        ).fetchone()[0]
        conn.close()
        assert new_orders == 0, "DB was not reset between retries"

        n = calls["n"]
        calls["n"] += 1
        if n < 2:  # first two attempts: dirty the DB, then fail like infra
            conn = connect(case.db_path)
            with conn:
                conn.execute(
                    "INSERT INTO orders (user_id, status, total_cents, ship_address) "
                    "VALUES (1, 'placed', 1, 'junk from a doomed attempt')"
                )
            conn.close()
            raise RuntimeError("infra boom mid-run")
        # third attempt sees a clean DB (asserted above) and does the task right
        oracle_run_fn(case, cond, trial)

    summary, report = asyncio.run(
        run_matrix((BUY_BLUE_SHIRT,), DEFAULT_CONDITIONS, 1,
                   root=tmp_path, run_fn=flaky_run_fn, seed_path=seed,
                   run_retries=2)
    )
    assert calls["n"] == 3                       # two failures, then success
    assert summary.passed == summary.planned == 1
    assert summary.skipped == 0
    assert report["run::buy_blue_shirt::clean::0"].attempts == 3
