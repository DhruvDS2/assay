"""
C2 — Solari sandbox/hosting: run the shop on real cloud infra, keep the grader honest.

The bet is unchanged: "did the agent succeed?" is a SQL query against a SQLite file.
C2 only moves *where the shop runs* and *where the browser lives* — never how we grade.
Per case:

    seed -> fresh case.db  (local, by the engine's run node)
      -> a Provisioner boots the shop in a Solari microVM serving THAT db, returns a
         public preview URL
      -> a Browser (a Solari cloud page) drives the agent through the real forms
      -> the Provisioner pulls the mutated db BACK into case.db_path
      -> the grade node reads case.db_path locally and purely — the SAME grader as v0

Two seams plus glue:
  * `SolariBrowser` — the `bench.agent.Browser` protocol over a Playwright/patchright
    page (what a Solari cloud browser hands you). The agent loop is unchanged.
  * `Provisioner` / `BrowserFactory` — ASYNC context managers (the Solari sandbox
    handle's methods are coroutines). `serve()` yields the shop's base URL and, on exit,
    pulls the db back then tears the VM down. `open()` yields a Browser and closes the
    cloud session.
  * `solari_run_fn` — a `bench.run.RunFn` wiring the two behind the existing seam.

Everything except the LIVE classes is transport-agnostic and keyless; `tests/test_solari.py`
drives it with fakes. The live classes touch the network and need SOLARI_API_KEY.

CLAUDE.md gotchas honored: `kill()` ends the VM and `close()` releases the browser — both
in `finally`; preview URLs are public and signed, so every run gets a FRESH GT_SECRET_KEY;
`run_code` payloads are argv lists, never shell strings; concurrency is the engine's
`group="solari"` cap, upstream.
"""

from __future__ import annotations

import os
import secrets
from contextlib import asynccontextmanager
from pathlib import Path
from typing import AsyncIterator, Callable, Optional, Protocol
from urllib.parse import urljoin, urlsplit, urlunsplit

from bench.agent import AgentActionError, Browser, Policy, run as _agent_run
from bench.model import Case

SOLARI_BASE_URL = os.environ.get("SOLARI_BASE_URL", "https://api.getsolari.com")
SHOP_PORT = int(os.environ.get("GT_SHOP_PORT", "8000"))


# --------------------------------------------------------------------------- browser
# The agent speaks one small selector vocabulary (bench/agent.py `_LLM_SYSTEM`):
#   goto  -> a path like "/cart"
#   fill  -> "#<id>" or "name=<name>"
#   click -> "text=<label>" (a link or a submit button), or a bare "/path"
# A real Playwright page understands this subset directly; we only translate the two
# bench-isms ("name=x" -> "[name='x']", a bare "/path" click -> a navigation).


def _to_css(selector: str) -> str:
    s = selector.strip()
    if s.startswith("name="):
        return f"[name={s[len('name='):].strip()!r}]"
    return s  # "#id" and raw CSS pass straight through


def _with_path(base_url: str, path: str) -> str:
    """Join a path onto a preview URL while PRESERVING its ?slr_token query.

    Solari preview URLs are signed with a one-hour `slr_token`. urljoin() would drop
    the query, so we splice the path in and keep the token on every request — the
    proxy authorizes each one (and sets a cookie the cloud browser then reuses)."""
    parts = urlsplit(base_url)
    new_path = path if path.startswith("/") else "/" + path
    return urlunsplit((parts.scheme, parts.netloc, new_path, parts.query, ""))


class SolariBrowser:
    """A `bench.agent.Browser` over a Playwright/patchright async page. Drop-in for
    `TestClientBrowser`; the agent loop, policy, and grader are all unchanged."""

    def __init__(self, page, base_url: str) -> None:
        self._page = page
        self._base = base_url   # full preview URL, incl. ?slr_token

    def _abs(self, path: str) -> str:
        return _with_path(self._base, path)

    @property
    def url(self) -> str:
        return self._page.url

    async def content(self) -> str:
        return await self._page.content()

    async def goto(self, url: str) -> None:
        target = url if url.startswith("http") else self._abs(url)
        await self._page.goto(target, wait_until="domcontentloaded")

    async def fill(self, selector: str, value: str) -> None:
        try:
            await self._page.fill(_to_css(selector), value)
        except Exception as e:   # a bad selector is an AGENT mistake, not infra
            raise AgentActionError(f"fill {selector!r} failed: {e}") from e

    async def click(self, selector: str) -> None:
        s = selector.strip()
        if s.startswith("/"):            # a bare path is a navigation
            await self.goto(s)
            return
        try:
            if s.startswith("text="):
                label = s[len("text="):].strip().strip("\"'")
                # Target an INTERACTIVE control — a link, then a submit button — never a
                # heading that merely contains the text (get_by_text matched the <h1>Log in</h1>
                # before the <button>Log in</button> and clicks went nowhere). Mirrors
                # TestClientBrowser's link-first-then-button resolution.
                link = self._page.get_by_role("link", name=label, exact=False)
                if await link.count() > 0:
                    await link.first.click()
                else:
                    await self._page.get_by_role("button", name=label, exact=False).first.click()
            else:
                await self._page.click(s)
        except AgentActionError:
            raise
        except Exception as e:
            raise AgentActionError(f"click {selector!r} failed: {e}") from e


# ----------------------------------------------------------------------- protocols
# Both are ASYNC context managers because the Solari handle's methods are coroutines.


class Provisioner(Protocol):
    def serve(
        self, db_path: Path, *, secret_key: str, mutations: tuple = (), faults: tuple = ()
    ):  # -> async context manager yielding the shop's public base URL
        ...


class BrowserFactory(Protocol):
    def open(self, base_url: str):  # -> async context manager yielding a Browser
        ...


# ------------------------------------------------------------------------- the RunFn


def fresh_secret_key() -> str:
    """A new signing key per run — preview URLs are PUBLIC (CLAUDE.md)."""
    return secrets.token_urlsafe(32)


def solari_run_fn(
    policy_for: Callable[[Case], Policy],
    *,
    provisioner: Provisioner,
    browser_factory: BrowserFactory,
    max_steps: int = 25,
):
    """Build a `bench.run.RunFn` driving the agent on Solari, leaving the final state
    in `case.db_path` for the local grader.

    CONTRACT (mirrors run.py): raise only on infrastructure failure (engine retries on a
    fresh db); return the moment the agent finishes, right or wrong. The provisioner's
    context manager owns pull-the-db-back + tear-down in its `finally`, so a crash mid-run
    still leaves a gradeable db and never leaks a VM.
    """

    async def run_fn(case: Case, condition, trial: int):
        mutations = tuple(getattr(condition, "mutations", ()) or ())
        faults = tuple(getattr(condition, "faults", ()) or ())
        async with provisioner.serve(
            Path(case.db_path), secret_key=fresh_secret_key(),
            mutations=mutations, faults=faults,
        ) as base_url:
            instruction = case.task.instruction(base_url)
            async with browser_factory.open(base_url) as browser:
                return await _agent_run(instruction, browser, policy_for(case), max_steps=max_steps)

    return run_fn


# =========================================================================== LIVE
# Talks to Solari; needs SOLARI_API_KEY. SDKs imported lazily so the keyless build and
# tests never require them. The flow below is VERIFIED against a real VM: create ->
# connect -> run_code works, PyPI egress works (pip install the public repo), a Popen'd
# uvicorn survives, preview_url(port) is reachable. File transfer is via run_code base64
# (the Python Sandbox has no files.write), which keeps the whole path on one RPC channel.


def _require_key() -> str:
    key = os.environ.get("SOLARI_API_KEY", "").strip()
    if not key or key == "slr_live_replace_me":
        raise RuntimeError("SOLARI_API_KEY missing or placeholder — put a real key in .env.")
    return key


_REMOTE_DB = "/root/case.db"
_REPO = os.environ.get("GT_SOLARI_REPO", "git+https://github.com/DhruvDS2/assay.git")


def _check(result, what: str) -> str:
    """run_code returns a RunCodeResult; raise on error, else join its text output."""
    if getattr(result, "error", None):
        raise RuntimeError(f"{what} failed in sandbox: {result.error}")
    return "".join(i.text for i in result.results if getattr(i, "text", None)).strip()


class SandboxProvisioner:
    """Live `Provisioner`: a Solari microVM runs the shop; the db is pushed in and pulled
    back. With GT_SOLARI_SNAPSHOT set, the shop+deps are baked in and we skip the install
    (the CLAUDE.md design); otherwise we `pip install` the public repo at boot."""

    def __init__(
        self,
        *,
        snapshot: Optional[str] = None,
        base_url: str = SOLARI_BASE_URL,
        repo: str = _REPO,
        ready_timeout_s: float = 150.0,
        create_timeout_ms: int = 180_000,
    ) -> None:
        from solari_sandbox import SandboxClient  # lazy: live-only dep

        self._client = SandboxClient(api_key=_require_key(), base_url=base_url)
        self._snapshot = snapshot or os.environ.get("GT_SOLARI_SNAPSHOT")
        self._repo = repo
        self._ready_timeout_s = ready_timeout_s
        self._create_timeout_ms = create_timeout_ms

    @asynccontextmanager
    async def serve(self, db_path, *, secret_key, mutations=(), faults=()) -> AsyncIterator[str]:
        db_path = Path(db_path)
        kwargs = {"from_snapshot": self._snapshot} if self._snapshot else {}
        sb = await self._client.create(timeout_ms=self._create_timeout_ms, **kwargs)
        try:
            await sb.connect()
            await self._provision(sb)
            await self._push_db(sb, db_path)
            await self._launch(sb, secret_key=secret_key, mutations=mutations, faults=faults)
            base_url = await self._preview(sb)
            await self._await_ready(base_url)
            yield base_url
        finally:
            # Pull the (mutated) db back FIRST (even a mid-run crash stays gradeable),
            # THEN tear down. close() releases, kill() ends the VM — both in finally.
            try:
                await self._pull_db(sb, db_path)
            finally:
                try:
                    await sb.close()
                finally:
                    await sb.kill()

    async def _provision(self, sb) -> None:
        if self._snapshot:
            return
        code = (
            "import subprocess\n"
            f"p = subprocess.run(['pip','install','--quiet','{self._repo}'],"
            " capture_output=True, text=True, timeout=600)\n"
            "print('rc', p.returncode)\n"
            "assert p.returncode == 0, (p.stderr or p.stdout)[-800:]\n"
        )
        _check(await sb.run_code(code), "pip install shop")

    async def _push_db(self, sb, db_path: Path) -> None:
        import base64

        b64 = base64.b64encode(db_path.read_bytes()).decode()
        code = (
            "import base64, pathlib\n"
            f"pathlib.Path({_REMOTE_DB!r}).write_bytes(base64.b64decode({b64!r}))\n"
            "print('wrote db')\n"
        )
        _check(await sb.run_code(code), "push db")

    async def _launch(self, sb, *, secret_key, mutations, faults) -> None:
        env = {
            "GT_DB_PATH": _REMOTE_DB,
            "GT_SECRET_KEY": secret_key,
            "GT_MUTATIONS": ",".join(mutations),
            "GT_FAULTS": ",".join(faults),
        }
        code = (
            "import subprocess, sys, os\n"
            f"env = dict(os.environ); env.update({env!r})\n"
            "subprocess.Popen([sys.executable, '-m', 'uvicorn', 'app.main:default_app',"
            f" '--factory', '--host', '0.0.0.0', '--port', '{SHOP_PORT}'], env=env)\n"
            "print('launched')\n"
        )
        _check(await sb.run_code(code), "launch uvicorn")

    async def _preview(self, sb) -> str:
        info = await sb.preview_url(SHOP_PORT)
        url = info.get("url") if isinstance(info, dict) else str(info)
        if not url:
            raise RuntimeError(f"no preview url for port {SHOP_PORT}: {info!r}")
        return url

    async def _await_ready(self, base_url: str) -> None:
        import asyncio

        import httpx

        loop = asyncio.get_event_loop()
        deadline = loop.time() + self._ready_timeout_s
        last = None
        async with httpx.AsyncClient(timeout=5.0) as client:
            while loop.time() < deadline:
                try:
                    r = await client.get(_with_path(base_url, "/login"))
                    if r.status_code < 500:
                        return
                    last = r.status_code
                except Exception as e:   # not listening yet; keep polling
                    last = e
                await asyncio.sleep(2.0)
        raise RuntimeError(f"shop never became ready at {base_url} (last={last!r})")

    async def _pull_db(self, sb, db_path: Path) -> None:
        # Binary comes back over a presigned download URL, NOT run_code stdout — stdout
        # truncates a multi-KB blob. download_url() -> {"url": ".../files/download?token="}.
        import httpx

        info = await sb.download_url(_REMOTE_DB)
        url = info["url"] if isinstance(info, dict) else info
        async with httpx.AsyncClient(timeout=60.0) as client:
            resp = await client.get(url)
            resp.raise_for_status()
        Path(db_path).write_bytes(resp.content)


class SolariBrowserFactory:
    """Live `BrowserFactory`: a Solari cloud browser session, one page per run."""

    def __init__(self, *, recording: bool = False, region: str = "us-west") -> None:
        from solari_browser import Solari  # lazy: live-only dep

        self._solari = Solari(_require_key(), region=region)
        self._recording = recording

    @staticmethod
    async def _maybe_await(value):
        """Solari's browser methods may be sync or async across SDK builds — await either."""
        if hasattr(value, "__await__"):
            return await value
        return value

    @asynccontextmanager
    async def open(self, base_url: str) -> AsyncIterator[Browser]:
        session = await self._maybe_await(self._solari.launch(recording=self._recording))
        try:
            page = await self._maybe_await(session.new_page())
            yield SolariBrowser(page, base_url)
        finally:
            try:
                await self._maybe_await(session.close())   # release the cloud session
            finally:
                await self._maybe_await(self._solari.close())


# ------------------------------------------------------------------------- smoke CLI


def smoke(task_id: str = "empty_cart") -> int:
    """One live case end-to-end: boot a VM, drive the real LLM agent, pull the db, grade.

    Needs SOLARI_API_KEY and ANTHROPIC_API_KEY. Set GT_SOLARI_SNAPSHOT to skip the pip
    install if you have a baked snapshot; otherwise the shop is pip-installed from the repo.
    """
    import asyncio

    from bench.agent import LLMPolicy
    from bench.env import build_seed, prepare_case
    from bench.grader import grade
    from bench.run import CLEAN
    from bench.tasks import TASKS

    task = next(t for t in TASKS if t.id == task_id)
    root = Path("runs/solari_smoke")
    root.mkdir(parents=True, exist_ok=True)
    seed = build_seed(root / "seed.db")
    case = prepare_case(task, root / task_id, seed)

    run_fn = solari_run_fn(
        lambda c: LLMPolicy(),
        provisioner=SandboxProvisioner(),
        browser_factory=SolariBrowserFactory(recording=True),
    )

    print(f"[smoke] booting a Solari VM for task {task_id!r} ...")
    transcript = asyncio.run(run_fn(case, CLEAN, 0))
    if transcript is not None:
        print(f"[smoke] agent took {transcript.n_steps} steps "
              f"(stopped: {transcript.stopped_reason!r}):")
        for s in transcript.steps:
            err = f"  -> ERROR: {s.error}" if s.error else ""
            print(f"    {s.index}: {s.action.kind} {s.action.target!r} {s.action.value!r}{err}")
    g = grade(case)
    print(f"[smoke] graded {task_id}: {'PASS' if g.passed else 'FAIL'}")
    for cr in g.results:
        print(f"    {'ok ' if cr.passed else 'FAIL'} {cr.name}: {cr.detail}")
    return 0 if g.passed else 1


def main(argv: Optional[list[str]] = None) -> int:
    import sys

    argv = sys.argv[1:] if argv is None else argv
    return smoke(argv[0] if argv else "empty_cart")


if __name__ == "__main__":
    raise SystemExit(main())
