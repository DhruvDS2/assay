"""
Keyless tests for C3 (bench/tenancy.py): the multi-tenant control plane.

Two things to prove: (1) the store authenticates and keeps each tenant's runs in their
OWN SQLite file so nothing leaks across tenants, and (2) the dashboard app enforces that
— session login for humans, API key for the harness, and a tenant can never read another
tenant's run even by guessing its id.
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from bench.tenancy import TenantStore, create_control_app


def _store(tmp_path):
    return TenantStore(tmp_path / "control")


# --------------------------------------------------------------- store + auth

def test_create_tenant_returns_a_working_api_key(tmp_path):
    store = _store(tmp_path)
    tenant, key = store.create_tenant("acme", "Acme Inc", "ops@acme.com", "s3cret")
    assert key.startswith("ak_")
    assert store.tenant_by_api_key(key) == tenant
    assert store.tenant_by_api_key("ak_wrong") is None
    assert store.tenant_by_api_key("") is None


def test_login_checks_the_password(tmp_path):
    store = _store(tmp_path)
    tenant, _ = store.create_tenant("acme", "Acme", "ops@acme.com", "s3cret")
    assert store.tenant_by_login("ops@acme.com", "s3cret") == tenant
    assert store.tenant_by_login("ops@acme.com", "nope") is None
    assert store.tenant_by_login("ghost@acme.com", "s3cret") is None


def test_runs_are_isolated_per_tenant(tmp_path):
    store = _store(tmp_path)
    a, _ = store.create_tenant("a", "A", "a@x.com", "pw")
    b, _ = store.create_tenant("b", "B", "b@x.com", "pw")
    ra = store.record_run(a, agent="llm", conditions="seven", trials=3, planned=105, passed=73)
    rb1 = store.record_run(b, agent="oracle", conditions="clean", trials=1, planned=5, passed=5)
    rb2 = store.record_run(b, agent="oracle", conditions="seven", trials=1, planned=5, passed=4)

    assert [r.id for r in store.list_runs(a)] == [ra]
    assert [r.id for r in store.list_runs(b)] == [rb2, rb1]

    # Separate files: the same id in each tenant's DB is a DIFFERENT run (both start at 1).
    assert ra == rb1 == 1
    assert store.get_run(a, 1)[0].agent == "llm"
    assert store.get_run(b, 1)[0].agent == "oracle"
    # An id that only B has (rb2 = 2) simply doesn't exist in A's DB.
    assert store.get_run(a, rb2) is None


def test_list_runs_is_newest_first(tmp_path):
    store = _store(tmp_path)
    t, _ = store.create_tenant("a", "A", "a@x.com", "pw")
    first = store.record_run(t, agent="llm", conditions="clean", trials=1, planned=5, passed=4)
    second = store.record_run(t, agent="llm", conditions="seven", trials=3, planned=105, passed=90)
    assert [r.id for r in store.list_runs(t)] == [second, first]


# --------------------------------------------------------------- the dashboard

def _client(store):
    return TestClient(create_control_app(store, secret_key="test-key"))


def test_dashboard_requires_login(tmp_path):
    client = _client(_store(tmp_path))
    r = client.get("/", follow_redirects=False)
    assert r.status_code == 303 and r.headers["location"] == "/login"


def test_login_then_dashboard_lists_runs(tmp_path):
    store = _store(tmp_path)
    tenant, _ = store.create_tenant("acme", "Acme Inc", "ops@acme.com", "s3cret")
    store.record_run(tenant, agent="llm", conditions="seven", trials=3, planned=105, passed=73,
                     report_html="<html><body>REPORT-BODY</body></html>")
    client = _client(store)

    bad = client.post("/login", data={"email": "ops@acme.com", "password": "nope"},
                      follow_redirects=False)
    assert bad.status_code == 401 and "Wrong email or password" in bad.text

    ok = client.post("/login", data={"email": "ops@acme.com", "password": "s3cret"},
                     follow_redirects=False)
    assert ok.status_code == 303 and ok.headers["location"] == "/"

    page = client.get("/")   # TestClient keeps the session cookie
    assert page.status_code == 200
    assert "Acme Inc" in page.text and "73/105" in page.text


def test_run_report_served_only_to_its_owner(tmp_path):
    store = _store(tmp_path)
    a, _ = store.create_tenant("a", "A", "a@x.com", "pw")
    b, _ = store.create_tenant("b", "B", "b@x.com", "pw")
    rb = store.record_run(b, agent="llm", conditions="clean", trials=1, planned=5, passed=5,
                          report_html="<html><body>B-ONLY-REPORT</body></html>")
    client = _client(store)
    client.post("/login", data={"email": "a@x.com", "password": "pw"})

    # A is logged in; B's run id must 404 for A (not leak B's report).
    miss = client.get(f"/runs/{rb}")
    assert miss.status_code == 404 and "B-ONLY-REPORT" not in miss.text


def test_run_report_body_is_served_to_owner(tmp_path):
    store = _store(tmp_path)
    t, _ = store.create_tenant("acme", "Acme", "ops@acme.com", "pw")
    rid = store.record_run(t, agent="llm", conditions="seven", trials=3, planned=105, passed=73,
                           report_html="<html><body>FULL-REPORT-HERE</body></html>")
    client = _client(store)
    client.post("/login", data={"email": "ops@acme.com", "password": "pw"})
    r = client.get(f"/runs/{rid}")
    assert r.status_code == 200 and "FULL-REPORT-HERE" in r.text


def test_api_records_a_run_with_a_valid_key(tmp_path):
    store = _store(tmp_path)
    tenant, key = store.create_tenant("acme", "Acme", "ops@acme.com", "pw")
    client = _client(store)

    bad = client.post("/api/runs", headers={"X-API-Key": "ak_nope"},
                      json={"agent": "llm", "conditions": "seven", "planned": 105, "passed": 73})
    assert bad.status_code == 401

    missing = client.post("/api/runs",
                          json={"agent": "llm", "conditions": "seven", "planned": 105, "passed": 73})
    assert missing.status_code == 401

    ok = client.post("/api/runs", headers={"X-API-Key": key},
                     json={"agent": "llm", "conditions": "seven", "trials": 3,
                           "planned": 105, "passed": 73})
    assert ok.status_code == 201
    run_id = ok.json()["run_id"]
    # it landed in THIS tenant's isolated db
    assert [r.id for r in store.list_runs(tenant)] == [run_id]


def test_api_rejects_a_bad_payload(tmp_path):
    store = _store(tmp_path)
    _, key = store.create_tenant("acme", "Acme", "ops@acme.com", "pw")
    client = _client(store)
    r = client.post("/api/runs", headers={"X-API-Key": key},
                    json={"agent": "llm"})   # missing planned/passed
    assert r.status_code == 400


def test_eval_run_payload_lands_on_the_dashboard(tmp_path):
    """The C1->C3 loop, end to end (keyless): a real oracle eval -> build_run_payload ->
    POST /api/runs -> the run shows on the dashboard and its B1 report is served."""
    import asyncio
    import tempfile

    from bench.adapter import resolve_agent
    from bench.eval import build_run_payload
    from bench.run import CLEAN, run_matrix
    from bench.tasks import TASKS

    task = next(t for t in TASKS if t.id == "empty_cart")
    with tempfile.TemporaryDirectory() as root:
        summary, _ = asyncio.run(
            run_matrix((task,), (CLEAN,), 1, root=root, run_fn=resolve_agent("oracle"))
        )
    payload = build_run_payload(summary, agent="oracle", conditions="clean", trials=1)
    assert payload["planned"] == 1 and payload["passed"] == 1 and "<" in payload["report_html"]

    store = _store(tmp_path)
    _, key = store.create_tenant("acme", "Acme", "ops@acme.com", "pw")
    client = _client(store)

    posted = client.post("/api/runs", headers={"X-API-Key": key}, json=payload)
    assert posted.status_code == 201
    run_id = posted.json()["run_id"]

    client.post("/login", data={"email": "ops@acme.com", "password": "pw"})
    assert "1/1" in client.get("/").text                       # headline on the dashboard
    report = client.get(f"/runs/{run_id}")
    assert report.status_code == 200 and "empty_cart" in report.text   # the B1 report


def test_logout_clears_the_session(tmp_path):
    store = _store(tmp_path)
    store.create_tenant("acme", "Acme", "ops@acme.com", "pw")
    client = _client(store)
    client.post("/login", data={"email": "ops@acme.com", "password": "pw"})
    assert client.get("/", follow_redirects=False).status_code == 200
    client.post("/logout")
    assert client.get("/", follow_redirects=False).status_code == 303
