"""
v3 faults: break the infrastructure on purpose and show what the grader sees.

The trio:
- error_on_checkout       — 500 before the write; no order exists afterward.
- drop_checkout_response  — the order commits, the client sees 500; a naive retry
                            places it AGAIN. SQL catches the double-order.
- expire_session          — the agent is bounced to /login mid-task.

These aren't agent mistakes; they're the world misbehaving. What matters is the
*recovery*, and whether recovery leaves a mess the screen can't show but SQL can.
"""

from __future__ import annotations

import sqlite3

import pytest
from fastapi.testclient import TestClient

from app.faults import create_faulty_app, inject_faults
from app.main import create_app
from bench.env import prepare_case
from bench.grader import grade
from bench.tasks import BUY_BLUE_SHIRT, add, login


def _n_orders(db_path) -> int:
    c = sqlite3.connect(db_path)
    try:
        return c.execute("SELECT COUNT(*) FROM orders").fetchone()[0]
    finally:
        c.close()


@pytest.fixture
def case(seed, tmp_path):
    return prepare_case(BUY_BLUE_SHIRT, tmp_path / "blue", seed)


def test_no_faults_is_a_normal_checkout(case):
    with TestClient(create_faulty_app(case.db_path, faults=())) as c:
        login(c, "alice@example.com", "password123")
        add(c, "SHIRT-BLU")
        r = c.post("/checkout", data={"ship_address": "12 Elm St"})
        assert r.status_code == 200
    assert _n_orders(case.db_path) == 1


def test_error_on_checkout_creates_no_order(case):
    with TestClient(create_faulty_app(case.db_path, faults=("error_on_checkout",))) as c:
        login(c, "alice@example.com", "password123")
        add(c, "SHIRT-BLU")
        r = c.post("/checkout", data={"ship_address": "12 Elm St"}, follow_redirects=False)
        assert r.status_code == 500
    assert _n_orders(case.db_path) == 0        # nothing committed


def test_drop_checkout_response_commits_but_looks_failed(case):
    """The order IS created; the client just sees a 500. This is the trap."""
    with TestClient(create_faulty_app(case.db_path, faults=("drop_checkout_response",))) as c:
        login(c, "alice@example.com", "password123")
        add(c, "SHIRT-BLU")
        r = c.post("/checkout", data={"ship_address": "12 Elm St"}, follow_redirects=False)
        assert r.status_code == 500            # the agent sees failure...
    assert _n_orders(case.db_path) == 1        # ...but the order exists


def test_naive_recovery_double_orders_and_sql_catches_it(case):
    """A retry that re-adds items under drop_checkout_response places a SECOND order."""
    with TestClient(create_faulty_app(case.db_path, faults=("drop_checkout_response",))) as c:
        login(c, "alice@example.com", "password123")
        add(c, "SHIRT-BLU")
        c.post("/checkout", data={"ship_address": "12 Elm St"}, follow_redirects=False)  # order 1
        # Agent saw a 500, thinks it failed; cart now looks empty, so it re-adds and retries.
        add(c, "SHIRT-BLU")
        c.post("/checkout", data={"ship_address": "12 Elm St"}, follow_redirects=False)  # order 2

    assert _n_orders(case.db_path) == 2
    g = grade(case)
    failed = {r.name for r in g.results if not r.passed}
    assert "exactly_one_new_order" in failed    # the double-order is caught in SQL
    assert not g.passed


def test_expire_session_bounces_to_login(case):
    with TestClient(create_faulty_app(case.db_path, faults=("expire_session",))) as c:
        c.post("/login", data={"email": "alice@example.com", "password": "password123"})
        r = c.get("/")                          # should be redirected to the login page
        assert r.url.path == "/login"
        assert "Log in" in r.text


def test_unknown_fault_is_rejected(case):
    with pytest.raises(ValueError):
        inject_faults(create_app(case.db_path), ("no_such_fault",))
