"""
Keyless tests for the C2 Solari seam (bench/solari.py). No network, no SOLARI_API_KEY:
the live classes import their SDKs lazily, so everything here drives fakes.

Two things to prove:
  1. SolariBrowser faithfully translates the agent's selector vocabulary onto a
     Playwright page (the only bench-isms are "name=x" and a bare "/path" click).
  2. solari_run_fn wires provision -> drive -> pull-the-db-back so the LOCAL grader
     sees the agent's real effects, and ALWAYS tears down (kill + close) in finally.
"""

from __future__ import annotations

import asyncio
from contextlib import contextmanager
from pathlib import Path

import pytest

from app.main import create_app
from bench.agent import Action, AgentActionError, ScriptedPolicy, TestClientBrowser
from bench.env import build_seed, prepare_case
from bench.grader import grade
from bench.run import CLEAN, Condition
from bench.solari import SolariBrowser, fresh_secret_key, solari_run_fn, _to_css
from bench.tasks import TASKS


# --------------------------------------------------------------- SolariBrowser

class _FakeLocator:
    def __init__(self, page, label):
        self._page, self._label = page, label

    @property
    def first(self):
        return self

    async def click(self):
        self._page.calls.append(("click_text", self._label))


class _FakePage:
    """Records the Playwright calls SolariBrowser makes; stands in for a cloud page."""

    def __init__(self, *, html="<html></html>", fill_raises=False):
        self.url = "http://shop/"
        self._html = html
        self._fill_raises = fill_raises
        self.calls: list[tuple] = []

    async def goto(self, url, wait_until=None):
        self.calls.append(("goto", url, wait_until))
        self.url = url

    async def content(self):
        return self._html

    async def fill(self, selector, value):
        if self._fill_raises:
            raise RuntimeError("element not found")
        self.calls.append(("fill", selector, value))

    async def click(self, selector):
        self.calls.append(("click_css", selector))

    def get_by_text(self, label):
        return _FakeLocator(self, label)


def _run(coro):
    return asyncio.run(coro)


def test_to_css_translates_only_the_bench_isms():
    assert _to_css("#qty") == "#qty"
    assert _to_css("name=qty") == "[name='qty']"
    assert _to_css("input.foo") == "input.foo"   # raw CSS passes through


def test_goto_resolves_relative_to_base():
    page = _FakePage()
    b = SolariBrowser(page, "https://preview.example/")
    _run(b.goto("/cart"))
    assert page.calls[0] == ("goto", "https://preview.example/cart", "domcontentloaded")


def test_fill_css_and_name_selectors():
    page = _FakePage()
    b = SolariBrowser(page, "https://p/")
    _run(b.fill("#qty", "3"))
    _run(b.fill("name=ship_address", "12 Elm St"))
    assert ("fill", "#qty", "3") in page.calls
    assert ("fill", "[name='ship_address']", "12 Elm St") in page.calls


def test_fill_failure_is_an_agent_mistake_not_infra():
    page = _FakePage(fill_raises=True)
    b = SolariBrowser(page, "https://p/")
    with pytest.raises(AgentActionError):
        _run(b.fill("#nope", "x"))


def test_click_text_matches_a_link_or_button():
    page = _FakePage()
    b = SolariBrowser(page, "https://p/")
    _run(b.click("text=Place order"))
    assert ("click_text", "Place order") in page.calls


def test_click_bare_path_is_a_navigation():
    page = _FakePage()
    b = SolariBrowser(page, "https://p/")
    _run(b.click("/orders"))
    assert ("goto", "https://p/orders", "domcontentloaded") in page.calls


def test_content_and_url_pass_through():
    page = _FakePage(html="<html>hi</html>")
    b = SolariBrowser(page, "https://p/")
    assert _run(b.content()) == "<html>hi</html>"
    page.url = "https://p/cart"
    assert b.url == "https://p/cart"


def test_fresh_secret_key_is_unique_and_nonempty():
    a, b = fresh_secret_key(), fresh_secret_key()
    assert a and b and a != b


# ------------------------------------------------------- solari_run_fn wiring

_LOGIN = [
    Action.goto("/login"),
    Action.fill("#email", "alice@example.com"),
    Action.fill("#password", "password123"),
    Action.click("text=Log in"),
]
_EMPTY_CART = _LOGIN + [
    Action.goto("/cart"),
    Action.click("text=Empty cart"),
    Action.stop("done"),
]


class _FakeSolari:
    """Both Provisioner and BrowserFactory, backed by an in-process TestClientBrowser
    over the SAME local db. There's no VM, so the "pull the db back" is identity — but
    the wiring (serve -> instruction base_url -> drive -> grade local db) and the
    always-tear-down invariant are exactly what we're proving. Records kill/close."""

    def __init__(self, *, browser_raises=False):
        self.served_db: Path | None = None
        self.secret_key: str | None = None
        self.mutations = None
        self.killed = False
        self.closed = False
        self._browser_raises = browser_raises

    @contextmanager
    def serve(self, db_path, *, secret_key, mutations=(), faults=()):
        self.served_db = Path(db_path)
        self.secret_key = secret_key
        self.mutations = mutations
        try:
            yield "http://fake-shop"          # the public preview URL stand-in
        finally:
            self.killed = True                # VM torn down (db is already local)

    @contextmanager
    def open(self, base_url):
        assert base_url == "http://fake-shop"  # the agent is driven at the served URL
        browser = TestClientBrowser(create_app(self.served_db))
        if self._browser_raises:
            browser = _ExplodingBrowser()
        try:
            yield browser
        finally:
            self.closed = True                 # cloud session released


class _ExplodingBrowser:
    """A transport that dies mid-run — an INFRA failure that must propagate (engine
    retries) while cleanup still runs."""

    url = "http://fake-shop/"

    async def content(self):
        raise RuntimeError("cloud browser died")

    async def goto(self, url):  # pragma: no cover - content() fails first
        raise RuntimeError("cloud browser died")

    async def fill(self, s, v):  # pragma: no cover
        raise RuntimeError("cloud browser died")

    async def click(self, s):  # pragma: no cover
        raise RuntimeError("cloud browser died")


def _empty_cart_case(tmp_path):
    task = next(t for t in TASKS if t.id == "empty_cart")
    seed = build_seed(tmp_path / "seed.db")
    return task, prepare_case(task, tmp_path / "case", seed)


def test_run_fn_drives_the_agent_and_leaves_a_gradeable_db(tmp_path):
    task, case = _empty_cart_case(tmp_path)
    fake = _FakeSolari()
    run_fn = solari_run_fn(
        lambda c: ScriptedPolicy(_EMPTY_CART),
        provisioner=fake,
        browser_factory=fake,
    )
    _run(run_fn(case, CLEAN, 0))

    # The agent's real effects are in the LOCAL db; the unchanged SQL grader reads them.
    assert grade(case).passed
    # A fresh signing key was minted (preview URLs are public).
    assert fake.secret_key and fake.secret_key != "dev-only-secret"
    # Serve + browser both torn down.
    assert fake.killed and fake.closed


def test_run_fn_passes_condition_mutations_through(tmp_path):
    _, case = _empty_cart_case(tmp_path)
    fake = _FakeSolari()
    run_fn = solari_run_fn(lambda c: ScriptedPolicy(_EMPTY_CART),
                           provisioner=fake, browser_factory=fake)
    _run(run_fn(case, Condition("visual_noise", mutations=("visual_noise",)), 0))
    assert fake.mutations == ("visual_noise",)


def test_infra_failure_propagates_but_still_tears_down(tmp_path):
    _, case = _empty_cart_case(tmp_path)
    fake = _FakeSolari(browser_raises=True)
    run_fn = solari_run_fn(lambda c: ScriptedPolicy(_EMPTY_CART),
                           provisioner=fake, browser_factory=fake)
    with pytest.raises(RuntimeError):
        _run(run_fn(case, CLEAN, 0))
    assert fake.killed and fake.closed     # finally ran on both context managers
