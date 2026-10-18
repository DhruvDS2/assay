"""
C3 — multi-tenancy, auth, and the dashboard: the control plane that turns the eval
harness into a hosted service many customers can log into.

This is the first thin end-to-end slice: a tenant authenticates, owns their eval runs
(strongly isolated), and views them on a served dashboard that reuses the B1 report.

Design decisions (chosen with the user, 2026-10-17):
  * **One SQLite DB per tenant** — `<root>/tenants/<slug>/runs.db`, matching the repo's
    one-db-per-case ethos. A tenant's queries only ever touch THEIR file, so a run can't
    leak across tenants by a forgotten WHERE clause — isolation is structural. A small
    control DB (`<root>/control.db`) maps tenants -> credentials.
  * **Two auth paths** — a per-tenant API key (header `X-API-Key`) for the harness/CI to
    push runs (fits C1's eval CLI / Action), and a session-cookie login for humans on the
    dashboard (reuses app/'s SessionMiddleware). API keys are high-entropy, so we store a
    plain SHA-256 for O(1) lookup; passwords use app.db's salted PBKDF2.
  * **Server-rendered, no JavaScript** — same contract as the shop and the B1 report: the
    page is exactly what the server sent. HTML is built with f-strings + html.escape.

Reuses app.db (connect/hash_password/verify_password) and, at the edges, bench.report.
"""

from __future__ import annotations

import hashlib
import html
import secrets
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, Form, Request
from fastapi.responses import HTMLResponse, JSONResponse, RedirectResponse
from starlette.middleware.sessions import SessionMiddleware

from app.db import connect, hash_password, verify_password

CONTROL_SCHEMA = """
CREATE TABLE IF NOT EXISTS tenants (
  id            INTEGER PRIMARY KEY,
  slug          TEXT UNIQUE NOT NULL,
  name          TEXT NOT NULL,
  email         TEXT UNIQUE NOT NULL,
  password_hash TEXT NOT NULL,
  api_key_hash  TEXT UNIQUE NOT NULL,
  created_at    TEXT NOT NULL DEFAULT (datetime('now'))
);
"""

RUNS_SCHEMA = """
CREATE TABLE IF NOT EXISTS runs (
  id          INTEGER PRIMARY KEY,
  agent       TEXT NOT NULL,
  conditions  TEXT NOT NULL,
  trials      INTEGER NOT NULL,
  planned     INTEGER NOT NULL,
  passed      INTEGER NOT NULL,
  report_html TEXT NOT NULL DEFAULT '',
  created_at  TEXT NOT NULL DEFAULT (datetime('now'))
);
"""


@dataclass(frozen=True)
class Tenant:
    id: int
    slug: str
    name: str
    email: str


@dataclass(frozen=True)
class Run:
    id: int
    agent: str
    conditions: str
    trials: int
    planned: int
    passed: int
    created_at: str

    @property
    def pass_rate(self) -> float:
        return self.passed / self.planned if self.planned else 0.0


def _api_key_hash(key: str) -> str:
    # API keys are high-entropy random tokens, so a fast deterministic hash is safe and
    # lets us look a tenant up by key in one indexed query (unlike the salted password).
    return hashlib.sha256(key.encode()).hexdigest()


def _row_to_tenant(row) -> Tenant:
    return Tenant(id=row["id"], slug=row["slug"], name=row["name"], email=row["email"])


class TenantStore:
    """All tenant state: the control DB + one runs DB per tenant. The only object that
    touches a tenant's file is a method given that tenant, so isolation is structural."""

    def __init__(self, root: str | Path) -> None:
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self._control_path = self.root / "control.db"
        conn = connect(self._control_path)
        try:
            conn.executescript(CONTROL_SCHEMA)
            conn.commit()
        finally:
            conn.close()

    # -- control DB ----------------------------------------------------------

    def _control(self):
        return connect(self._control_path)

    def create_tenant(self, slug: str, name: str, email: str, password: str) -> tuple[Tenant, str]:
        """Create a tenant and return it plus its API key. The key is shown ONCE — only
        its hash is stored, so it can't be recovered later (rotate by re-creating)."""
        api_key = "ak_" + secrets.token_urlsafe(24)
        conn = self._control()
        try:
            cur = conn.execute(
                "INSERT INTO tenants (slug, name, email, password_hash, api_key_hash) "
                "VALUES (?, ?, ?, ?, ?)",
                (slug, name, email, hash_password(password), _api_key_hash(api_key)),
            )
            conn.commit()
            tid = cur.lastrowid
        finally:
            conn.close()
        self._runs(slug).close()   # materialize the tenant's isolated runs DB
        return Tenant(id=tid, slug=slug, name=name, email=email), api_key

    def tenant_by_id(self, tenant_id: int) -> Optional[Tenant]:
        conn = self._control()
        try:
            row = conn.execute("SELECT * FROM tenants WHERE id = ?", (tenant_id,)).fetchone()
        finally:
            conn.close()
        return _row_to_tenant(row) if row else None

    def tenant_by_api_key(self, key: str) -> Optional[Tenant]:
        if not key:
            return None
        conn = self._control()
        try:
            row = conn.execute(
                "SELECT * FROM tenants WHERE api_key_hash = ?", (_api_key_hash(key),)
            ).fetchone()
        finally:
            conn.close()
        return _row_to_tenant(row) if row else None

    def tenant_by_login(self, email: str, password: str) -> Optional[Tenant]:
        conn = self._control()
        try:
            row = conn.execute("SELECT * FROM tenants WHERE email = ?", (email,)).fetchone()
        finally:
            conn.close()
        if row and verify_password(password, row["password_hash"]):
            return _row_to_tenant(row)
        return None

    # -- per-tenant runs DB (isolated file) ----------------------------------

    def _runs_path(self, slug: str) -> Path:
        return self.root / "tenants" / slug / "runs.db"

    def _runs(self, slug: str):
        path = self._runs_path(slug)
        path.parent.mkdir(parents=True, exist_ok=True)
        conn = connect(path)
        conn.executescript(RUNS_SCHEMA)
        conn.commit()
        return conn

    def record_run(
        self,
        tenant: Tenant,
        *,
        agent: str,
        conditions: str,
        trials: int,
        planned: int,
        passed: int,
        report_html: str = "",
    ) -> int:
        conn = self._runs(tenant.slug)
        try:
            cur = conn.execute(
                "INSERT INTO runs (agent, conditions, trials, planned, passed, report_html) "
                "VALUES (?, ?, ?, ?, ?, ?)",
                (agent, conditions, int(trials), int(planned), int(passed), report_html),
            )
            conn.commit()
            return cur.lastrowid
        finally:
            conn.close()

    def list_runs(self, tenant: Tenant) -> list[Run]:
        conn = self._runs(tenant.slug)
        try:
            rows = conn.execute(
                "SELECT id, agent, conditions, trials, planned, passed, created_at "
                "FROM runs ORDER BY id DESC"      # newest-first; id, not clock (B1 lesson)
            ).fetchall()
        finally:
            conn.close()
        return [
            Run(r["id"], r["agent"], r["conditions"], r["trials"], r["planned"],
                r["passed"], r["created_at"])
            for r in rows
        ]

    def get_run(self, tenant: Tenant, run_id: int) -> Optional[tuple[Run, str]]:
        """A (Run, report_html) pair, or None — looked up in THIS tenant's DB only, so a
        tenant can never fetch another's run even by guessing the id."""
        conn = self._runs(tenant.slug)
        try:
            r = conn.execute("SELECT * FROM runs WHERE id = ?", (run_id,)).fetchone()
        finally:
            conn.close()
        if not r:
            return None
        run = Run(r["id"], r["agent"], r["conditions"], r["trials"], r["planned"],
                  r["passed"], r["created_at"])
        return run, r["report_html"]


# ----------------------------------------------------------------- HTML (no JS)


def _page(title: str, body: str) -> str:
    return (
        "<!DOCTYPE html><html lang='en'><head><meta charset='utf-8'>"
        f"<title>{html.escape(title)}</title>"
        "<style>body{font:15px/1.5 system-ui,sans-serif;max-width:860px;margin:2rem auto;"
        "padding:0 1rem;color:#1a1a1a}h1{font-size:1.4rem}table{border-collapse:collapse;"
        "width:100%}th,td{text-align:left;padding:.5rem .6rem;border-bottom:1px solid #eee}"
        "a{color:#0b5fff;text-decoration:none}a:hover{text-decoration:underline}"
        ".err{color:#b00020}.muted{color:#666}.pill{font-weight:600}</style></head>"
        f"<body>{body}</body></html>"
    )


def _login_html(error: str = "") -> str:
    err = f"<p class='err'>{html.escape(error)}</p>" if error else ""
    return _page("Assay — sign in", (
        "<h1>Assay</h1><p class='muted'>Sign in to view your eval runs.</p>" + err +
        "<form method='post' action='/login'>"
        "<p><label>Email<br><input name='email' type='email' required></label></p>"
        "<p><label>Password<br><input name='password' type='password' required></label></p>"
        "<p><button type='submit'>Sign in</button></p></form>"
    ))


def _dashboard_html(tenant: Tenant, runs: list[Run]) -> str:
    if runs:
        rows = "".join(
            f"<tr><td><a href='/runs/{r.id}'>#{r.id}</a></td>"
            f"<td>{html.escape(r.agent)}</td><td>{html.escape(r.conditions)}</td>"
            f"<td class='pill'>{r.passed}/{r.planned}</td>"
            f"<td class='muted'>{html.escape(r.created_at)}</td></tr>"
            for r in runs
        )
        table = ("<table><thead><tr><th>Run</th><th>Agent</th><th>Conditions</th>"
                 "<th>Passed</th><th>When</th></tr></thead><tbody>" + rows + "</tbody></table>")
    else:
        table = "<p class='muted'>No runs yet. Push one with your API key via <code>POST /api/runs</code>.</p>"
    return _page(f"Assay — {tenant.name}", (
        f"<h1>{html.escape(tenant.name)}</h1>"
        "<form method='post' action='/logout' style='float:right'>"
        "<button type='submit'>Sign out</button></form>"
        "<p class='muted'>Your eval runs. Each is graded by SQL truth; click one for the full report.</p>"
        + table
    ))


# -------------------------------------------------------------------- the app


def create_control_app(store: TenantStore, secret_key: str = "dev-only-secret"):
    app = FastAPI()
    app.add_middleware(SessionMiddleware, secret_key=secret_key)

    def current_tenant(request: Request) -> Optional[Tenant]:
        tid = request.session.get("tenant_id")
        return store.tenant_by_id(tid) if tid else None

    @app.get("/health")
    def health():
        return {"ok": True}

    @app.get("/login", response_class=HTMLResponse)
    def login_form():
        return HTMLResponse(_login_html())

    @app.post("/login")
    def login(request: Request, email: str = Form(...), password: str = Form(...)):
        tenant = store.tenant_by_login(email, password)
        if not tenant:
            return HTMLResponse(_login_html("Wrong email or password."), status_code=401)
        request.session.clear()
        request.session["tenant_id"] = tenant.id
        return RedirectResponse("/", status_code=303)

    @app.post("/logout")
    def logout(request: Request):
        request.session.clear()
        return RedirectResponse("/login", status_code=303)

    @app.get("/", response_class=HTMLResponse)
    def dashboard(request: Request):
        tenant = current_tenant(request)
        if not tenant:
            return RedirectResponse("/login", status_code=303)
        return HTMLResponse(_dashboard_html(tenant, store.list_runs(tenant)))

    @app.get("/runs/{run_id}", response_class=HTMLResponse)
    def run_report(request: Request, run_id: int):
        tenant = current_tenant(request)
        if not tenant:
            return RedirectResponse("/login", status_code=303)
        got = store.get_run(tenant, run_id)   # scoped to this tenant's DB only
        if not got:
            return HTMLResponse(_page("Not found", "<h1>404</h1><p>No such run.</p>"),
                                status_code=404)
        run, report = got
        return HTMLResponse(report or _page(
            f"Run #{run.id}",
            f"<h1>Run #{run.id}</h1><p>{run.passed}/{run.planned} passed "
            f"(agent {html.escape(run.agent)}, {html.escape(run.conditions)}).</p>"
            "<p class='muted'>No stored report for this run.</p><p><a href='/'>&larr; back</a></p>",
        ))

    @app.post("/api/runs")
    async def api_record_run(request: Request):
        tenant = store.tenant_by_api_key(request.headers.get("x-api-key", ""))
        if not tenant:
            return JSONResponse({"error": "invalid or missing X-API-Key"}, status_code=401)
        try:
            body = await request.json()
        except Exception:
            return JSONResponse({"error": "body must be JSON"}, status_code=400)
        try:
            run_id = store.record_run(
                tenant,
                agent=str(body["agent"]),
                conditions=str(body["conditions"]),
                trials=int(body.get("trials", 0)),
                planned=int(body["planned"]),
                passed=int(body["passed"]),
                report_html=str(body.get("report_html", "")),
            )
        except (KeyError, TypeError, ValueError) as e:
            return JSONResponse({"error": f"bad run payload: {e}"}, status_code=400)
        return JSONResponse({"run_id": run_id}, status_code=201)

    return app


def default_control_app():
    """Factory for `uvicorn bench.tenancy:default_control_app --factory`."""
    import os

    root = os.environ.get("GT_CONTROL_ROOT", "data/control")
    return create_control_app(TenantStore(root), os.environ.get("GT_SECRET_KEY", "dev-only-secret"))


# -------------------------------------------------------------------- CLI


def main(argv: Optional[list[str]] = None) -> int:
    import os
    import sys

    argv = list(sys.argv[1:] if argv is None else argv)
    root = os.environ.get("GT_CONTROL_ROOT", "data/control")
    if not argv or argv[0] in ("-h", "--help"):
        print("usage: python -m bench.tenancy <create|serve> ...")
        print("  create <slug> <name> <email> <password>   create a tenant, print its API key")
        print("  serve  [--port N]                          run the dashboard (uvicorn)")
        return 0 if argv else 2

    cmd, rest = argv[0], argv[1:]
    if cmd == "create":
        if len(rest) != 4:
            print("usage: create <slug> <name> <email> <password>")
            return 2
        store = TenantStore(root)
        tenant, api_key = store.create_tenant(*rest)
        print(f"created tenant {tenant.slug!r} (id={tenant.id})")
        print(f"API key (shown once): {api_key}")
        return 0
    if cmd == "serve":
        import uvicorn

        port = int(rest[rest.index("--port") + 1]) if "--port" in rest else 8080
        uvicorn.run(default_control_app(), host="127.0.0.1", port=port)
        return 0
    print(f"unknown command: {cmd!r}")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
