"""
C2 — Solari sandbox/hosting: run the shop on real cloud infra, keep the grader honest.

The bet is unchanged: "did the agent succeed?" is a SQL query against a SQLite file.
C2 only moves *where the shop runs* and *where the browser lives* — never how we grade.
Per case:

    seed -> fresh case.db  (local, by the engine's run node)
      -> a Provisioner boots the shop in a Solari microVM serving THAT db, returns a
         public preview URL
      -> a Browser (a Solari cloud page) drives the agent through the real forms
      -> the Provisioner pulls the mutated db back into case.db_path
      -> the grade node reads case.db_path locally and purely — the SAME grader as v0

So this file is two seams plus the glue:

  * `SolariBrowser` — the `bench.agent.Browser` protocol over a Playwright/patchright
    page (what a Solari cloud browser hands you). The agent loop is unchanged.
  * `Provisioner` — boot a shop for one db, yield its base URL, and on exit pull the db
    BACK so the local grader sees the agent's real effects. Live impl = `SandboxProvisioner`.
  * `solari_run_fn` — a `bench.run.RunFn` wiring the two behind the existing transport seam.

Everything above the live classes is transport-agnostic and keyless; `tests/test_solari.py`
drives it with fakes. The live classes touch the network and need SOLARI_API_KEY.

CLAUDE.md gotchas honored here: `kill()` ends the VM and `close()` releases the browser —
both go in `finally`; preview URLs are public, so every run gets a FRESH GT_SECRET_KEY;
commands are argv, not shell; concurrency is the engine's `group="solari"` cap, upstream.
"""

from __future__ import annotations

import os
import secrets
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Awaitable, Callable, Iterator, Optional, Protocol
from urllib.parse import urljoin

from bench.agent import AgentActionError, Browser, Policy, run as _agent_run
from bench.model import Case

SOLARI_BASE_URL = os.environ.get("SOLARI_BASE_URL", "https://api.getsolari.com")
SHOP_PORT = int(os.environ.get("GT_SHOP_PORT", "8000"))


# --------------------------------------------------------------------------- browser
# The agent speaks one small selector vocabulary (see bench/agent.py `_LLM_SYSTEM`):
#   goto  -> a path like "/cart"
#   fill  -> "#<id>" or "name=<name>"
#   click -> "text=<label>" (a link or a submit button), or a bare "/path"
# A real Playwright page understands this subset directly; we only translate the two
# bench-isms ("name=x" -> "[name='x']", a bare "/path" click -> a navigation).


def _to_css(selector: str) -> str:
    """Translate a fill selector into something a Playwright page accepts."""
    s = selector.strip()
    if s.startswith("name="):
        return f"[name={s[len('name='):].strip()!r}]"
    return s  # "#id" and raw CSS pass straight through


class SolariBrowser:
    """A `bench.agent.Browser` backed by a Playwright/patchright async page.

    Swap this in for `TestClientBrowser` and the agent loop, the policy, and the
    grader are all unchanged — only the transport is the cloud.
    """

    def __init__(self, page, base_url: str) -> None:
        self._page = page
        self._base = base_url.rstrip("/") + "/"

    def _abs(self, path: str) -> str:
        return urljoin(self._base, path.lstrip("/"))

    @property
    def url(self) -> str:
        return self._page.url

    async def content(self) -> str:
        return await self._page.content()

    async def goto(self, url: str) -> None:
        await self._page.goto(self._abs(url), wait_until="domcontentloaded")

    async def fill(self, selector: str, value: str) -> None:
        try:
            await self._page.fill(_to_css(selector), value)
        except Exception as e:   # a bad selector is an AGENT mistake, not infra
            raise AgentActionError(f"fill {selector!r} failed: {e}") from e

    async def click(self, selector: str) -> None:
        s = selector.strip()
        # A bare path (or explicit href=) is a navigation, same as TestClientBrowser.
        if s.startswith("/"):
            await self.goto(s)
            return
        try:
            if s.startswith("text="):
                label = s[len("text="):].strip().strip("\"'")
                # Playwright's text engine is case-insensitive substring; .first keeps
                # us out of strict-mode ambiguity. Matches a link or a submit button —
                # clicking a link navigates, clicking a submit posts the current form.
                await self._page.get_by_text(label).first.click()
            else:
                await self._page.click(s)
        except AgentActionError:
            raise
        except Exception as e:
            raise AgentActionError(f"click {selector!r} failed: {e}") from e


# ----------------------------------------------------------------------- provisioner


class Provisioner(Protocol):
    """Boots a shop serving one local db and, on context exit, pulls the db back."""

    def serve(
        self,
        db_path: Path,
        *,
        secret_key: str,
        mutations: tuple = (),
        faults: tuple = (),
    ) -> "ServeHandle": ...


class ServeHandle(Protocol):
    """A context manager yielding the shop's public base URL. On exit it MUST copy the
    VM's (possibly-mutated) db back into `db_path` and then tear the VM down."""

    def __enter__(self) -> str: ...
    def __exit__(self, *exc) -> Optional[bool]: ...


class BrowserFactory(Protocol):
    """Opens a Browser against a base URL; the context closes the cloud session."""

    def open(self, base_url: str) -> "BrowserHandle": ...


class BrowserHandle(Protocol):
    def __enter__(self) -> Browser: ...
    def __exit__(self, *exc) -> Optional[bool]: ...


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
    """Build a `bench.run.RunFn` that drives the agent on Solari and leaves the final
    state in `case.db_path` for the local grader.

    CONTRACT (mirrors run.py): raise only on infrastructure failure (the engine then
    retries on a fresh db); return the moment the agent finishes, right or wrong. The
    provisioner's context manager owns the pull-the-db-back + tear-down in its `finally`,
    so a crash mid-run still leaves a gradeable db and never leaks a VM.
    """

    async def run_fn(case: Case, condition, trial: int):
        mutations = tuple(getattr(condition, "mutations", ()) or ())
        faults = tuple(getattr(condition, "faults", ()) or ())
        instruction_base = None
        with provisioner.serve(
            Path(case.db_path),
            secret_key=fresh_secret_key(),
            mutations=mutations,
            faults=faults,
        ) as base_url:
            instruction = case.task.instruction(base_url)
            with browser_factory.open(base_url) as browser:
                await _agent_run(instruction, browser, policy_for(case), max_steps=max_steps)
        return instruction_base  # unused by the engine; db_path carries the real result

    return run_fn


# =========================================================================== LIVE
# Everything below talks to Solari and needs SOLARI_API_KEY. Kept out of the import
# path for the keyless build: the SDKs are imported lazily inside the constructors.
#
# ‼️ LIVE STATUS (2026-10-16) — the seam above is built + keyless-tested (223 passed);
# the live provisioner below is an UNVERIFIED scaffold. A connectivity probe against a
# real VM surfaced three facts the SDK type signatures hid, which this scaffold does
# NOT yet handle correctly — finish these before trusting the live path:
#   1. `SyncSandboxClient.create()` returns a handle whose methods (preview_url,
#      upload_url, download_url, kill, close) are ASYNC COROUTINES — they must be
#      awaited. Prefer the async `SandboxClient` throughout (this run_fn is already async).
#   2. A `Sandbox` has NO `exec` (only `Desktop` does). Serve the shop either via
#      `create_desktop()` (has `exec`) or via `Sandbox.run_code(...)` launching uvicorn
#      with subprocess — `_provision`/`_launch_shop` below assume `exec` and will fail.
#   3. The intended design is a PREBUILT SNAPSHOT with the shop + deps baked in
#      (CLAUDE.md) — set GT_SOLARI_SNAPSHOT and the live bootstrap becomes just "push db,
#      start server", sidestepping (2). Build it once via TemplateClient/Image.
# Also verify the upload_url/download_url/preview_url wire shapes (presigned PUT vs POST).
# Resolve against docs.getsolari.com, then this scaffold can be made real + smoke-tested.


def _require_key() -> str:
    key = os.environ.get("SOLARI_API_KEY", "").strip()
    if not key or key == "slr_live_replace_me":
        raise RuntimeError(
            "SOLARI_API_KEY is missing or still the placeholder — put a real key in .env."
        )
    return key


# The server command run INSIDE the VM. The shop is the same FastAPI app; default_app()
# reads GT_DB_PATH / GT_SECRET_KEY / GT_MUTATIONS / GT_FAULTS from the environment, so a
# perturbed condition is just env vars — no code difference between here and localhost.
_REMOTE_DB = "/root/case.db"
_REMOTE_APP_DIR = "/root/shop"


class SandboxProvisioner:
    """Live `Provisioner`: a Solari microVM runs the shop; the db is pushed in and pulled back.

    Two provisioning modes:
      * `snapshot` (recommended, the CLAUDE.md design) — a prebuilt snapshot with the shop +
        deps already baked in; we only push the db and start the server. Fast.
      * otherwise — a best-effort live bootstrap: upload app/, pip install, launch uvicorn.
        Slower and template-dependent; good enough for a smoke.
    """

    def __init__(
        self,
        *,
        snapshot: Optional[str] = None,
        template: Optional[str] = None,
        base_url: str = SOLARI_BASE_URL,
        app_dir: Path = Path(__file__).resolve().parent.parent / "app",
        ready_timeout_s: float = 90.0,
    ) -> None:
        from solari_sandbox import SyncSandboxClient  # lazy: live-only dep

        self._client = SyncSandboxClient(api_key=_require_key(), base_url=base_url)
        self._snapshot = snapshot or os.environ.get("GT_SOLARI_SNAPSHOT")
        self._template = template or os.environ.get("GT_SOLARI_TEMPLATE")
        self._app_dir = Path(app_dir)
        self._ready_timeout_s = ready_timeout_s

    @contextmanager
    def serve(self, db_path, *, secret_key, mutations=(), faults=()) -> Iterator[str]:
        db_path = Path(db_path)
        kwargs = {}
        if self._snapshot:
            kwargs["from_snapshot"] = self._snapshot
        elif self._template:
            kwargs["template"] = self._template
        sandbox = self._client.create(**kwargs)
        try:
            self._provision(sandbox)
            self._upload(sandbox, db_path, _REMOTE_DB)
            self._launch_shop(sandbox, secret_key=secret_key, mutations=mutations, faults=faults)
            base_url = self._preview(sandbox)
            self._await_ready(sandbox, base_url)
            yield base_url
        finally:
            # Pull the (mutated) db back FIRST so even a mid-run crash is gradeable,
            # THEN tear down. kill() ends the VM; both in finally (CLAUDE.md).
            try:
                self._download(sandbox, _REMOTE_DB, db_path)
            finally:
                try:
                    sandbox.close()
                finally:
                    sandbox.kill()

    # -- provisioning steps (argv, never shell-interpreted) --------------------

    def _provision(self, sandbox) -> None:
        if self._snapshot:
            return  # shop + deps already baked into the snapshot
        # Best-effort live bootstrap onto a generic template.
        self._upload_dir(sandbox, self._app_dir, f"{_REMOTE_APP_DIR}/app")
        sandbox.exec(
            "pip",
            args=["install", "--quiet", "fastapi", "uvicorn", "jinja2",
                  "python-multipart", "itsdangerous"],
            timeout_ms=300_000,
        )

    def _launch_shop(self, sandbox, *, secret_key, mutations, faults) -> None:
        sandbox.env({
            "GT_DB_PATH": _REMOTE_DB,
            "GT_SECRET_KEY": secret_key,
            "GT_MUTATIONS": ",".join(mutations),
            "GT_FAULTS": ",".join(faults),
        })
        # Background the server; it must outlive this exec call.
        sandbox.exec(
            "bash",
            args=["-lc",
                  f"cd {_REMOTE_APP_DIR} && nohup uvicorn app.main:default_app --factory "
                  f"--host 0.0.0.0 --port {SHOP_PORT} >/tmp/shop.log 2>&1 &"],
            timeout_ms=30_000,
        )

    def _preview(self, sandbox) -> str:
        info = sandbox.preview_url(SHOP_PORT)
        url = info.get("url") if isinstance(info, dict) else str(info)
        if not url:
            raise RuntimeError(f"no preview url for port {SHOP_PORT}: {info!r}")
        return url

    def _await_ready(self, sandbox, base_url: str) -> None:
        import httpx

        deadline = time.monotonic() + self._ready_timeout_s
        last = None
        while time.monotonic() < deadline:
            try:
                r = httpx.get(urljoin(base_url.rstrip("/") + "/", "login"), timeout=5.0)
                if r.status_code < 500:
                    return
                last = r.status_code
            except Exception as e:  # not-yet-listening; keep polling
                last = e
            time.sleep(2.0)
        raise RuntimeError(f"shop never became ready at {base_url} (last={last!r})")

    # -- file transfer ---------------------------------------------------------

    def _upload(self, sandbox, local: Path, remote: str) -> None:
        import httpx

        info = sandbox.upload_url(remote)
        url = info["url"] if isinstance(info, dict) else info
        with open(local, "rb") as fh:
            httpx.put(url, content=fh.read(), timeout=60.0).raise_for_status()

    def _upload_dir(self, sandbox, local_dir: Path, remote_dir: str) -> None:
        for p in sorted(local_dir.rglob("*")):
            if p.is_file() and "__pycache__" not in p.parts:
                rel = p.relative_to(local_dir).as_posix()
                self._upload(sandbox, p, f"{remote_dir}/{rel}")

    def _download(self, sandbox, remote: str, local: Path) -> None:
        import httpx

        info = sandbox.download_url(remote)
        url = info["url"] if isinstance(info, dict) else info
        data = httpx.get(url, timeout=60.0)
        data.raise_for_status()
        Path(local).write_bytes(data.content)


class SolariBrowserFactory:
    """Live `BrowserFactory`: a Solari cloud browser session, one page per run."""

    def __init__(self, *, recording: bool = False, region: str = "us-west") -> None:
        from solari_browser import Solari  # lazy: live-only dep

        self._solari = Solari(_require_key(), region=region)
        self._recording = recording

    @contextmanager
    def open(self, base_url: str) -> Iterator[Browser]:
        session = self._solari.launch(recording=self._recording)
        try:
            page = session.new_page()
            # patchright's new_page may be sync or async depending on the build.
            if hasattr(page, "__await__"):
                import asyncio

                page = asyncio.get_event_loop().run_until_complete(page)
            yield SolariBrowser(page, base_url)
        finally:
            try:
                session.close()          # release the cloud session (CLAUDE.md)
            finally:
                self._solari.close()


# ------------------------------------------------------------------------- smoke CLI


def smoke(task_id: str = "empty_cart") -> int:
    """One live case end-to-end: boot a VM, drive the real LLM agent, pull the db, grade.

    Proves the whole round-trip for minimal spend. Needs SOLARI_API_KEY and (for the real
    brain) ANTHROPIC_API_KEY. Set GT_SOLARI_SNAPSHOT to the baked shop snapshot if you have
    one; otherwise it live-bootstraps onto GT_SOLARI_TEMPLATE.
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

    provisioner = SandboxProvisioner()
    browser_factory = SolariBrowserFactory(recording=True)
    run_fn = solari_run_fn(
        lambda c: LLMPolicy(),
        provisioner=provisioner,
        browser_factory=browser_factory,
    )

    print(f"[smoke] booting a Solari VM for task {task_id!r} ...")
    asyncio.get_event_loop().run_until_complete(run_fn(case, CLEAN, 0))
    g = grade(case)
    print(f"[smoke] graded {task_id}: {'PASS' if g.passed else 'FAIL'}")
    for cr in g.results:
        print(f"    {'ok ' if cr.passed else 'FAIL'} {cr.name}: {cr.detail}")
    return 0 if g.passed else 1


def main(argv: Optional[list[str]] = None) -> int:
    import sys

    argv = sys.argv[1:] if argv is None else argv
    task_id = argv[0] if argv else "empty_cart"
    return smoke(task_id)


if __name__ == "__main__":
    raise SystemExit(main())
