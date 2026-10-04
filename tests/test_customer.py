"""
Is the customer config layer (B2) telling the truth on a schema that isn't our shop?

The worked example in `bench.customer` is a billing DB — accounts + subscriptions, no
products or orders. These tests apply exactly the calibration discipline CLAUDE.md demands
of every task, but against that foreign schema:

  calibration  — the oracle passes every check, doing nothing fails at least one
  adversarial  — the most plausible wrong agent still fails, on the SPECIFIC named check
                 that describes its mistake

If this holds, the harness really is grading the customer's tables, not ours. Everything is
hermetic: each test runs on its own seed + case under pytest's tmp_path — `data/app.db` is
never touched.
"""

import pytest

from bench.customer import (
    ACME,
    BILLING_DB,
    CUSTOMER_TASKS,
    PLAN_PRO_CENTS,
    UPGRADE_TO_PRO,
    build_seed,
    noop,
    run_case,
    _account_id,
)


@pytest.fixture
def seed(tmp_path):
    """A fresh billing seed database, built once per test under tmp_path."""
    return build_seed(BILLING_DB, tmp_path / "seed.db")


@pytest.fixture
def run(seed, tmp_path):
    """run(task, agent) -> Grade, on a fresh isolated copy of the billing seed."""
    def _run(task, agent):
        return run_case(BILLING_DB, task, agent, tmp_path / task.id, seed)
    return _run


# ---------------------------------------------------------------- calibration


@pytest.mark.parametrize("task", CUSTOMER_TASKS, ids=lambda t: t.id)
def test_oracle_passes(run, task):
    g = run(task, task.oracle)
    assert g.passed, [r for r in g.results if not r.passed]


@pytest.mark.parametrize("task", CUSTOMER_TASKS, ids=lambda t: t.id)
def test_doing_nothing_fails(run, task):
    """An idle agent leaves the subscription on Basic — it must fail a check."""
    g = run(task, noop)
    assert not g.passed
    assert "subscription_is_pro" in g.failed_checks


# ---------------------------------------------------------------- adversarial
# The plausible wrong agent "upgrades" by creating a NEW Pro subscription and cancelling the
# old Basic one, so the end-state looks perfect — one active Pro sub at $25. The state-only
# checks are fooled; only the baseline-backed side-effect check sees the spurious row.


def test_recreate_instead_of_upgrade_is_caught(run):
    def agent(conn):
        aid = _account_id(conn, ACME)
        conn.execute(
            "UPDATE subscriptions SET status = 'cancelled' WHERE account_id = ? AND status = 'active'",
            (aid,),
        )
        conn.execute(
            "INSERT INTO subscriptions (account_id, plan, status, monthly_cents) "
            "VALUES (?, 'pro', 'active', ?)",
            (aid, PLAN_PRO_CENTS),
        )

    g = run(UPGRADE_TO_PRO, agent)
    # Covering your tracks doesn't erase the side effect: exactly one named check fails.
    assert g.failed_checks == ["no_new_subscription_rows"]


def test_duplicate_subscription_is_caught(run):
    """The sloppier mistake: add a Pro sub but forget to cancel the Basic one. Now two active
    subscriptions — a second, independent named check also catches it."""
    def agent(conn):
        conn.execute(
            "INSERT INTO subscriptions (account_id, plan, status, monthly_cents) "
            "VALUES (?, 'pro', 'active', ?)",
            (_account_id(conn, ACME), PLAN_PRO_CENTS),
        )

    g = run(UPGRADE_TO_PRO, agent)
    assert set(g.failed_checks) == {"exactly_one_active_subscription", "no_new_subscription_rows"}


# ---------------------------------------------------------------- through the engine
# The customer layer isn't just a standalone CLI — it flows through the same async DAG
# engine the shop's matrix uses, with the same fresh-DB-per-attempt / pure-grade contract.


def test_oracle_sweeps_the_engine(tmp_path):
    import asyncio

    from bench.customer import run_customer_matrix

    cells = asyncio.run(
        run_customer_matrix(BILLING_DB, CUSTOMER_TASKS, lambda t: t.oracle,
                            root=tmp_path / "ok", trials=2)
    )
    assert cells and all(c.passed for c in cells)          # oracle passes every trial
    assert len(cells) == len(CUSTOMER_TASKS) * 2


def test_noop_fails_through_the_engine(tmp_path):
    import asyncio

    from bench.customer import run_customer_matrix

    cells = asyncio.run(
        run_customer_matrix(BILLING_DB, CUSTOMER_TASKS, lambda t: noop, root=tmp_path / "noop")
    )
    assert cells and all(not c.passed for c in cells)      # doing nothing fails a named check
    assert all("subscription_is_pro" in c.failed_checks for c in cells)
