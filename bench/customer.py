"""
The customer config layer (roadmap B2): prove the harness is *not* welded to our shop.

The whole bench rests on one bet — "did the agent succeed?" is a SQL query against a
database, not an LLM's guess about a screenshot. That bet only sells if a *customer* can
make it against *their own* store: their schema, their seed, their tasks, their SQL checks.
This module is the seam that lets them, without editing a single line of our shop (`app/`)
or the core grader.

A customer brings four things, and nothing else:

    CustomerDB    how to build their clean world — DDL (`init`) + starting rows (`seed`),
                  plus a `baseline` snapshot taken before the agent runs so side-effect
                  checks can tell "new" from "already there" (our shop keys this off
                  `max_order_id`; a customer keys it off whatever their schema needs).
    CustomerTask  a goal in plain English + a `setup` that puts the world in its start
                  state + named SQL `Check`s that read THEIR tables + an `oracle` (a
                  scripted correct run that must pass every check).

We deliberately reuse the core vocabulary — `Check`, `CheckResult`, `Grade` from
`bench.model` are domain-agnostic already (a `Check` is just a name + a function of
`(connection, baseline) -> (passed, detail)`), and the grade loop mirrors
`bench.grader.grade` exactly. What generalizes is the *plumbing* (build → copy → setup →
baseline → grade); what the customer supplies is the *content*.

How an agent "acts" here: in our shop the oracle drives real web forms, because the point
there is to exercise the UI. In a customer deployment the oracle/agent drives the
customer's own site, whose actions land in their DB. For this config-layer demo (and for a
hermetic test) agents are callables that mutate the connection directly — the same rows
their app would write — because B2 is about generalizing the *grading*, not the web driver.

Everything money-shaped is integer cents. No floats, ever (see the worked example below).

Run the worked example's calibration with:

    python -m bench.customer        # oracle must PASS, noop must FAIL — else "NOT calibrated"
"""

from __future__ import annotations

import shutil
import sqlite3
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

# Read-only reuse of the core vocabulary. We do not touch these modules; a Check is already
# domain-agnostic, so a customer's checks are ordinary Checks over their own tables.
from bench.model import Check, CheckResult, Grade

# An "agent" (oracle, noop, or a real one) performs the work by mutating the customer's DB,
# exactly as the customer's live site would when an agent drives it.
Agent = Callable[[sqlite3.Connection], None]


def connect(path: str | Path) -> sqlite3.Connection:
    """A plain SQLite connection with row access by name and FK enforcement.

    Local on purpose: the customer layer depends only on `bench.model`, not on `app/`, so
    the generality claim ("not welded to our shop") holds at import time, not just in spirit.
    """
    conn = sqlite3.connect(str(path))
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    return conn


@dataclass(frozen=True)
class CustomerDB:
    """How a customer builds, seeds, and baselines THEIR database.

    init      run once on a fresh file — create the customer's tables (DDL).
    seed      insert the starting rows that define the clean world.
    baseline  snapshot taken after setup, before the agent runs. Returns whatever the
              customer's side-effect checks need to distinguish agent-created rows from
              pre-existing ones (our shop returns max order id; the example below returns
              the max subscription id). The returned value is handed to every Check.
    """

    init: Callable[[sqlite3.Connection], None]
    seed: Callable[[sqlite3.Connection], None]
    baseline: Callable[[sqlite3.Connection], Any]


@dataclass(frozen=True)
class CustomerTask:
    """One task against the customer's schema: a goal, a start state, checks, an oracle.

    Mirrors `bench.model.Task` but drops the shop-specific fields (login email/password,
    web-form oracle signature). A task passes only if ALL its checks pass; each check is
    named so a failure is explainable, not just counted.
    """

    id: str
    goal: str                                      # plain-English, what the agent is told
    setup: Callable[[sqlite3.Connection], None]    # puts the world in its start state
    checks: tuple[Check, ...]                       # ALL must pass
    oracle: Agent                                   # scripted perfect run; proves it is doable


@dataclass(frozen=True)
class CustomerCase:
    """One task on one fresh database file, with its pre-run baseline captured."""

    task: CustomerTask
    db_path: Path
    baseline: Any


# ---------------------------------------------------------------- the plumbing
# Identical in shape to bench/env.py + bench/grader.py, just parameterised by CustomerDB.


def build_seed(db: CustomerDB, path: str | Path) -> Path:
    """Build the pristine seed database once: schema + starting rows. Copy it per case."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        path.unlink()
    conn = connect(path)
    try:
        with conn:
            db.init(conn)
            db.seed(conn)
    finally:
        conn.close()
    return path


def prepare_case(db: CustomerDB, task: CustomerTask, workdir: str | Path, seed: str | Path) -> CustomerCase:
    """Fresh copy of the seed -> task setup -> baseline snapshot.

    After this returns, the file is exactly the world the agent should start in, and the
    baseline records what existed before it touched anything.
    """
    workdir = Path(workdir)
    workdir.mkdir(parents=True, exist_ok=True)
    db_path = workdir / f"{task.id}.db"
    shutil.copyfile(seed, db_path)

    conn = connect(db_path)
    try:
        with conn:
            task.setup(conn)
        baseline = db.baseline(conn)
    finally:
        conn.close()
    return CustomerCase(task=task, db_path=db_path, baseline=baseline)


def grade(case: CustomerCase) -> Grade:
    """Open the case's database, run every check, report each one.

    A straight copy of `bench.grader.grade`'s contract: a crashing check is a failed check,
    never a silent pass; the grader looks only at what actually landed in the DB.
    """
    conn = connect(case.db_path)
    results = []
    try:
        for check in case.task.checks:
            try:
                passed, detail = check.fn(conn, case.baseline)
            except Exception as e:  # a crashing check is a failed check, never a pass
                passed, detail = False, f"check crashed: {type(e).__name__}: {e}"
            results.append(CheckResult(check.name, bool(passed), detail))
    finally:
        conn.close()
    return Grade(task_id=case.task.id, results=tuple(results))


def run_case(db: CustomerDB, task: CustomerTask, agent: Agent, workdir: str | Path, seed: str | Path) -> Grade:
    """prepare -> let the agent mutate the DB -> grade. One isolated database per call."""
    case = prepare_case(db, task, workdir, seed)
    conn = connect(case.db_path)
    try:
        with conn:  # commit the agent's work (or roll back if it raises)
            agent(conn)
    finally:
        conn.close()
    return grade(case)


def noop(conn: sqlite3.Connection) -> None:
    """The laziest possible agent. It must fail at least one check of a well-built task."""


def calibrate(db: CustomerDB, tasks: tuple[CustomerTask, ...]) -> bool:
    """The same discipline as bench/selfcheck: oracle must PASS, noop must FAIL, per task.

    Returns True iff the grader separates a perfect run from doing nothing on every task.
    """
    healthy = True
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        seed = build_seed(db, tmp / "seed.db")
        print(f"{'task':<28} {'oracle':<8} noop")
        print("-" * 60)
        for task in tasks:
            o = run_case(db, task, task.oracle, tmp / "oracle", seed)
            n = run_case(db, task, noop, tmp / "noop", seed)
            o_txt = "PASS" if o.passed else "BROKEN"
            n_txt = f"fail ({', '.join(n.failed_checks)})" if not n.passed else "BROKEN (passed!)"
            print(f"{task.id:<28} {o_txt:<8} {n_txt}")
            if not o.passed:
                healthy = False
                for r in o.results:
                    if not r.passed:
                        print(f"    oracle failed {r.name}: {r.detail}")
            if n.passed:
                healthy = False
                print("    doing nothing passed this task — its checks are too weak")
    print("-" * 60)
    print("calibrated: grader separates success from failure" if healthy
          else "NOT calibrated — fix the tasks above")
    return healthy


# ---------------------------------------------------------------- through the engine
# The customer layer is not just a standalone CLI: it plugs into the SAME async DAG
# engine (bench/engine.py) the shop's matrix uses. We build seed -> run+grade pair per
# (task, trial) -> aggregate, honoring the shop's invariants: prepare_case lives inside
# the run node (fresh DB per attempt), and the grade node only reads the case's file.


@dataclass(frozen=True)
class CustomerCellGrade:
    """One (task, trial) result from an engine run of the customer matrix."""

    task_id: str
    trial: int
    passed: bool
    failed_checks: tuple[str, ...]


def _cust_run_node(db: CustomerDB, task: CustomerTask, agent: Agent, trial: int, root: Path):
    def fn(deps: dict) -> CustomerCase:
        seed = deps["seed"].value
        case = prepare_case(db, task, root / task.id / str(trial), seed)  # fresh DB per attempt
        conn = connect(case.db_path)
        try:
            with conn:
                agent(conn)   # raises => infra => engine retries; a wrong-but-finished agent returns
        finally:
            conn.close()
        return case

    return fn


def _cust_grade_node(run_id: str):
    def fn(deps: dict) -> Grade:
        return grade(deps[run_id].value)   # pure: reads the case's db file, nothing else

    return fn


async def run_customer_matrix(
    db: CustomerDB,
    tasks: tuple[CustomerTask, ...],
    agent_for: Callable[[CustomerTask], Agent],
    *,
    root: str | Path,
    trials: int = 1,
    max_concurrency: int = 4,
) -> list[CustomerCellGrade]:
    """Run the customer's tasks through the real async engine and return per-cell grades.

    `agent_for(task)` yields the agent to drive each task (e.g. `lambda t: t.oracle`).
    Proves the generality claim end-to-end: the customer's schema flows through the same
    scheduler, retries, and pure-grade contract as our own shop's matrix.
    """
    from bench.engine import Node, run_dag

    root = Path(root)
    nodes = [Node("seed", lambda _deps: build_seed(db, root / "seed.db"))]
    coords: list[tuple[str, int, str]] = []
    for task in tasks:
        for trial in range(trials):
            run_id, grade_id = f"run::{task.id}::{trial}", f"grade::{task.id}::{trial}"
            nodes.append(Node(run_id, _cust_run_node(db, task, agent_for(task), trial, root), deps=("seed",)))
            nodes.append(Node(grade_id, _cust_grade_node(run_id), deps=(run_id,)))
            coords.append((task.id, trial, grade_id))

    def _aggregate(deps: dict) -> list[CustomerCellGrade]:
        out = []
        for task_id, trial, gid in coords:
            res = deps.get(gid)
            if res is None or not res.ok:
                out.append(CustomerCellGrade(task_id, trial, False, ("<infra-skipped>",)))
            else:
                g: Grade = res.value
                out.append(CustomerCellGrade(task_id, trial, g.passed, tuple(g.failed_checks)))
        return out

    nodes.append(Node("aggregate", _aggregate, deps=tuple(c[2] for c in coords), tolerate_dep_failure=True))
    report = await run_dag(nodes, max_concurrency=max_concurrency)
    return report["aggregate"].value


# ================================================================================
# WORKED EXAMPLE — a different domain than our shop, to prove generality.
#
# "Billing": a tiny SaaS subscription database. Two tables, no products/cart/orders in
# sight. If the harness can grade a correct plan-upgrade here — and catch the plausible
# wrong one — then it is genuinely the customer's schema doing the work, not ours.
# ================================================================================

# Plan catalogue, prices in integer cents (no floats, ever).
PLAN_BASIC_CENTS = 1000   # $10.00 / month
PLAN_PRO_CENTS = 2500     # $25.00 / month

_SCHEMA = """
CREATE TABLE accounts (
    id    INTEGER PRIMARY KEY,
    email TEXT NOT NULL UNIQUE
);
CREATE TABLE subscriptions (
    id            INTEGER PRIMARY KEY,
    account_id    INTEGER NOT NULL REFERENCES accounts(id),
    plan          TEXT NOT NULL,                 -- 'basic' | 'pro'
    status        TEXT NOT NULL DEFAULT 'active', -- 'active' | 'cancelled'
    monthly_cents INTEGER NOT NULL
);
"""

ACME = "billing@acme.com"


def _init(conn: sqlite3.Connection) -> None:
    conn.executescript(_SCHEMA)


def _seed(conn: sqlite3.Connection) -> None:
    """The clean world: one customer, no subscription yet. Tasks set up their own state."""
    conn.execute("INSERT INTO accounts (email) VALUES (?)", (ACME,))


def _baseline(conn: sqlite3.Connection):
    """Highest subscription id before the agent runs — anything above it, the agent made."""
    return conn.execute("SELECT COALESCE(MAX(id), 0) FROM subscriptions").fetchone()[0]


BILLING_DB = CustomerDB(init=_init, seed=_seed, baseline=_baseline)


def _account_id(conn: sqlite3.Connection, email: str) -> int:
    return conn.execute("SELECT id FROM accounts WHERE email = ?", (email,)).fetchone()[0]


# ---- the task: upgrade an existing subscription in place --------------------------------


def _setup_basic_subscription(conn: sqlite3.Connection) -> None:
    """ACME starts with one active Basic subscription — the thing to be upgraded."""
    conn.execute(
        "INSERT INTO subscriptions (account_id, plan, status, monthly_cents) "
        "VALUES (?, 'basic', 'active', ?)",
        (_account_id(conn, ACME), PLAN_BASIC_CENTS),
    )


def _active_subs(conn: sqlite3.Connection, email: str):
    return conn.execute(
        "SELECT s.* FROM subscriptions s JOIN accounts a ON a.id = s.account_id "
        "WHERE a.email = ? AND s.status = 'active' ORDER BY s.id",
        (email,),
    ).fetchall()


def _check_subscription_is_pro(conn, baseline) -> tuple[bool, str]:
    """The account's active subscription is the Pro plan at exactly $25/mo. Doing nothing
    leaves it on Basic, so this is the check an idle agent fails."""
    subs = _active_subs(conn, ACME)
    if not subs:
        return False, "no active subscription to inspect"
    s = subs[-1]
    ok = s["plan"] == "pro" and s["monthly_cents"] == PLAN_PRO_CENTS
    return ok, f"active plan={s['plan']!r} monthly_cents={s['monthly_cents']}"


def _check_exactly_one_active_subscription(conn, baseline) -> tuple[bool, str]:
    """State check: the account must end with exactly one active subscription — not two."""
    n = len(_active_subs(conn, ACME))
    return n == 1, f"{n} active subscriptions (want exactly 1)"


def _check_no_new_subscription_rows(conn, baseline) -> tuple[bool, str]:
    """Side-effect check (keyed off the baseline): a correct upgrade edits the existing row
    in place, so NO subscription row above the baseline id should exist. An agent that
    "upgrades" by inserting a fresh Pro row — even if it cancels the old one to cover its
    tracks so the end-state looks right — created a spurious row, and this catches it.
    This is the billing-domain analogue of our shop's `exactly_one_new_order`."""
    n = conn.execute("SELECT COUNT(*) FROM subscriptions WHERE id > ?", (baseline,)).fetchone()[0]
    return n == 0, f"{n} subscription rows created during the run (want 0)"


def _oracle_upgrade_to_pro(conn: sqlite3.Connection) -> None:
    """The correct run: modify the existing subscription in place — no new rows."""
    conn.execute(
        "UPDATE subscriptions SET plan = 'pro', monthly_cents = ? "
        "WHERE account_id = ? AND status = 'active'",
        (PLAN_PRO_CENTS, _account_id(conn, ACME)),
    )


UPGRADE_TO_PRO = CustomerTask(
    id="upgrade_to_pro",
    goal=(
        f"Upgrade the subscription for account {ACME} from the Basic plan to the Pro plan "
        f"(${PLAN_PRO_CENTS // 100}/month). Change the existing subscription — do not create "
        "a second one."
    ),
    setup=_setup_basic_subscription,
    checks=(
        Check("subscription_is_pro", _check_subscription_is_pro),
        Check("exactly_one_active_subscription", _check_exactly_one_active_subscription),
        Check("no_new_subscription_rows", _check_no_new_subscription_rows),
    ),
    oracle=_oracle_upgrade_to_pro,
)


CUSTOMER_TASKS: tuple[CustomerTask, ...] = (UPGRADE_TO_PRO,)


def main() -> int:
    return 0 if calibrate(BILLING_DB, CUSTOMER_TASKS) else 1


if __name__ == "__main__":
    sys.exit(main())
