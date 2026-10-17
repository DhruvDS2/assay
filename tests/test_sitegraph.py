"""
Site knowledge-graph memory (A3 / v5) — offline, keyless tests.

Everything here runs against the real shop through `TestClientBrowser` and drives
it with `ScriptedPolicy`. No `anthropic`, no network, no API key. Two things get
proved:

  (a) a `MemoryAugmentedPolicy` *builds* the graph as it runs a clean site and
      later *consults* it — the same control it saw before is recalled, fresh;

  (b) the STALE-GRAPH HAZARD — a graph built on the clean site, then run against
      the same site under `rename_controls`, misleads the agent: the control it
      recalls ("Add to cart") is not on the mutated page ("Add to bag") anymore,
      so `consult` reports `stale`, and a credulous agent that trusts it acts on a
      control that no longer exists. This is measured against the A1 no-memory
      baseline (the bare `ScriptedPolicy`, which never has a graph to be misled
      by): memory helps on the clean site and hurts once the site mutates.
"""

from __future__ import annotations

import asyncio
from pathlib import Path

import pytest

from app.main import create_app
from app.mutations import create_mutated_app
from bench.agent import (
    Action,
    Observation,
    ScriptedPolicy,
    TestClientBrowser,
    run,
)
from bench.env import build_seed, prepare_case
from bench.grader import grade
from bench.sitegraph import (
    Element,
    MemoryAugmentedPolicy,
    SiteGraph,
    intent_for,
    page_key,
)
from bench.tasks import BUY_BLUE_SHIRT


# ------------------------------------------------------------------- fixtures


@pytest.fixture
def seed(tmp_path: Path) -> Path:
    return build_seed(tmp_path / "seed.db")


def _login() -> list:
    return [
        Action.goto("/login"),
        Action.fill("#email", "alice@example.com"),
        Action.fill("#password", "password123"),
        Action.click("text=Log in"),
    ]


# The buy-blue-shirt script, through the same forms a browser uses.
def _buy_blue_shirt_steps() -> list:
    return _login() + [
        Action.goto("/product/SHIRT-BLU"),
        Action.click("text=Add to cart"),
        Action.goto("/checkout"),
        Action.fill("#ship_address", "12 Elm St, Austin, TX 78701"),
        Action.click("text=Place order"),
    ]


async def _observe(browser: TestClientBrowser) -> Observation:
    from bench.agent import visible_text

    html = await browser.content()
    return Observation(url=browser.url, html=html, text=visible_text(html))


async def _drive_and_learn(policy: MemoryAugmentedPolicy, browser) -> None:
    """
    Drive the browser with a memory-augmented policy, feeding each post-action
    page back so the graph learns transitions — a tiny A3 harness. Mirrors the
    observe -> decide -> act loop but closes the memory edge after each act.
    """
    for _ in range(25):
        obs = await _observe(browser)
        action = policy.decide("buy one blue oxford shirt", obs, ())
        if action.kind == "stop":
            break
        try:
            await _execute(browser, action)
        except Exception:
            pass
        after = await _observe(browser)
        policy.observe_result(after)


async def _execute(browser, action: Action) -> None:
    if action.kind == "goto":
        await browser.goto(action.target)
    elif action.kind == "fill":
        await browser.fill(action.target, action.value)
    elif action.kind == "click":
        await browser.click(action.target)


# ------------------------------------------------- page-key / small surfaces


def test_page_key_normalises_url_to_path():
    assert page_key("http://shop/cart?x=1") == "/cart"
    assert page_key("/product/SHIRT-BLU") == "/product/SHIRT-BLU"
    assert page_key("/cart") == page_key("http://localhost:8000/cart")


def test_intent_for_maps_label_back_to_intent():
    assert intent_for("Add to cart") == "add_to_cart"
    assert intent_for("Place order") == "place_order"
    assert intent_for("nonsense label") is None


# --------------------------------------------------------- (a) build + consult


def test_build_registers_pages_and_their_elements(seed, tmp_path):
    """Observing pages during a clean run fills the graph with real elements."""
    case = prepare_case(BUY_BLUE_SHIRT, tmp_path / "wd", seed)
    browser = TestClientBrowser(create_app(case.db_path))
    graph = SiteGraph()

    async def walk():
        await browser.goto("/login")
        # Log in so product/cart pages render their controls.
        await browser.fill("#email", "alice@example.com")
        await browser.fill("#password", "password123")
        await browser.click("text=Log in")
        # Put a shirt in the cart first so /checkout actually renders (an empty
        # cart redirects /checkout -> /cart).
        await browser.goto("/product/SHIRT-BLU")
        graph.build(await _observe(browser))
        await browser.click("text=Add to cart")
        for path in ("/cart", "/checkout"):
            await browser.goto(path)
            graph.build(await _observe(browser))

    asyncio.run(walk())

    assert graph.has_page("/product/SHIRT-BLU")
    assert graph.has_page("/checkout")

    # The product page learned an "Add to cart" button on the clean site.
    product = graph.page("/product/SHIRT-BLU")
    labels = {e.label.lower() for e in product.elements}
    assert "add to cart" in labels
    # The checkout page learned a "Place order" button.
    checkout = graph.page("/checkout")
    assert any("place order" in e.label.lower() for e in checkout.elements)


def test_memory_policy_builds_then_consults_on_clean_site(seed, tmp_path):
    """
    A memory-augmented agent runs the clean task: it builds the graph as it goes,
    gets real memory hits (consulting a control it already saw), never goes stale,
    and the grader confirms the task passed from SQL.
    """
    case = prepare_case(BUY_BLUE_SHIRT, tmp_path / "wd", seed)
    browser = TestClientBrowser(create_app(case.db_path))
    policy = MemoryAugmentedPolicy(ScriptedPolicy(_buy_blue_shirt_steps()))

    asyncio.run(_drive_and_learn(policy, browser))

    # It consulted memory and found live (fresh) controls — never stale on clean.
    assert policy.memory_hits >= 1
    assert policy.stale_hits == 0

    # And it actually did the job: the grader reads success from the database.
    g = grade(case)
    assert g.passed, f"clean memory run failed checks: {g.failed_checks}"


def test_consult_recalls_a_fresh_control_on_the_clean_site(seed, tmp_path):
    """Direct consult: after building on the clean product page, recall is a fresh hit."""
    case = prepare_case(BUY_BLUE_SHIRT, tmp_path / "wd", seed)
    browser = TestClientBrowser(create_app(case.db_path))
    graph = SiteGraph()

    async def go():
        await browser.goto("/login")
        await browser.fill("#email", "alice@example.com")
        await browser.fill("#password", "password123")
        await browser.click("text=Log in")
        await browser.goto("/product/SHIRT-BLU")
        obs = await _observe(browser)
        graph.build(obs)
        return obs

    obs = asyncio.run(go())
    recall = graph.consult("add_to_cart", obs)

    assert recall.hit and not recall.stale
    assert recall.action.kind == "click"
    assert "add to cart" in recall.element.label.lower()


# ----------------------------------------------- (b) the STALE-GRAPH HAZARD


def _build_clean_graph(seed, tmp_path) -> tuple[SiteGraph, object]:
    """Build a graph on the CLEAN product page, then return it frozen (learn off later)."""
    case = prepare_case(BUY_BLUE_SHIRT, tmp_path / "wd_clean", seed)
    browser = TestClientBrowser(create_app(case.db_path))
    graph = SiteGraph()

    async def go():
        await browser.goto("/login")
        await browser.fill("#email", "alice@example.com")
        await browser.fill("#password", "password123")
        await browser.click("text=Log in")
        await browser.goto("/product/SHIRT-BLU")
        graph.build(await _observe(browser))

    asyncio.run(go())
    return graph, case


def test_clean_graph_goes_stale_against_renamed_controls(seed, tmp_path):
    """
    THE HAZARD. Build the graph on the clean site, then observe the SAME product
    page under `rename_controls` (where "Add to cart" -> "Add to bag"). The graph
    still recalls "Add to cart", a control the mutated page no longer has, so
    `consult` reports stale and points at a label that isn't there.
    """
    graph, _ = _build_clean_graph(seed, tmp_path)

    # Confirm the clean graph learned the original label.
    product = graph.page("/product/SHIRT-BLU")
    assert any(e.label.lower() == "add to cart" for e in product.elements)

    # Now render the SAME page under the renaming mutation, WITHOUT re-building
    # the graph (it's frozen from the clean run — the staleness).
    case = prepare_case(BUY_BLUE_SHIRT, tmp_path / "wd_mut", seed)
    mutated = create_mutated_app(case.db_path, mutations=("rename_controls",))
    browser = TestClientBrowser(mutated)

    async def mutated_obs():
        await browser.goto("/login")
        await browser.fill("#email", "alice@example.com")
        await browser.fill("#password", "password123")
        await browser.click("text=Log in")
        await browser.goto("/product/SHIRT-BLU")
        return await _observe(browser)

    obs = asyncio.run(mutated_obs())

    # The page now says "Add to bag", not "Add to cart".
    assert "Add to bag" in obs.html
    assert "Add to cart" not in obs.html

    recall = graph.consult("add_to_cart", obs)
    assert recall.hit, "graph should still have an opinion (that's why it's dangerous)"
    assert recall.stale, "recalled control is gone from the page -> stale hazard"
    assert "add to cart" in recall.element.label.lower()
    # The stale action targets the vanished label.
    assert recall.action.target == "text=Add to cart"


def test_credulous_agent_following_stale_memory_misfires(seed, tmp_path):
    """
    Memory HURTS once stale: a credulous policy (`trust_stale=True`) built on the
    clean site, run on the mutated page. Its inner policy correctly reads the
    page's REAL label ("Add to bag") — but memory overrides that with the recalled
    "Add to cart", which no longer exists, so the agent misfires (a recorded
    mistake). On the clean site that same memory would have matched and worked;
    stale, it actively steers a competent inner policy wrong.
    """
    graph, _ = _build_clean_graph(seed, tmp_path)

    case = prepare_case(BUY_BLUE_SHIRT, tmp_path / "wd_mut2", seed)
    mutated = create_mutated_app(case.db_path, mutations=("rename_controls",))
    browser = TestClientBrowser(mutated)

    # Frozen clean graph + trust it even when stale = the credulous agent. The
    # inner policy would have clicked the correct mutated label on its own.
    policy = MemoryAugmentedPolicy(
        ScriptedPolicy([Action.click("text=Add to bag")]),
        graph=graph,
        learn=False,          # freeze the graph: don't adapt to the mutation
        trust_stale=True,     # follow memory even when it's stale
    )

    async def act_on_product():
        await browser.goto("/login")
        await browser.fill("#email", "alice@example.com")
        await browser.fill("#password", "password123")
        await browser.click("text=Log in")
        await browser.goto("/product/SHIRT-BLU")
        obs = await _observe(browser)
        action = policy.decide("buy one blue oxford shirt", obs, ())
        # Executing the stale action is a mistake: no such button on the page.
        mistake = None
        try:
            await _execute(browser, action)
        except Exception as e:
            mistake = str(e)
        return action, mistake

    action, mistake = asyncio.run(act_on_product())

    # Memory steered it at the vanished control...
    assert policy.stale_hits == 1
    assert action.target == "text=Add to cart"
    # ...and acting on it misfired — the hazard made concrete.
    assert mistake is not None and "Add to cart" in mistake

    # Contrast: a SKEPTICAL policy (same stale graph, trust_stale=False) drops the
    # stale recall and keeps the inner's correct mutated label — no misfire. The
    # hazard is specifically trusting stale memory, not having it.
    skeptic = MemoryAugmentedPolicy(
        ScriptedPolicy([Action.click("text=Add to bag")]),
        graph=graph,
        learn=False,
        trust_stale=False,
    )

    async def skeptic_act():
        await browser.goto("/product/SHIRT-BLU")
        obs = await _observe(browser)
        a = skeptic.decide("buy one blue oxford shirt", obs, ())
        err = None
        try:
            await _execute(browser, a)
        except Exception as e:
            err = str(e)
        return a, err

    s_action, s_mistake = asyncio.run(skeptic_act())
    assert skeptic.stale_hits == 1            # it saw the staleness...
    assert s_action.target == "text=Add to bag"   # ...but did not follow it
    assert s_mistake is None


def test_a1_baseline_has_no_graph_to_be_misled(seed, tmp_path):
    """
    The A1 no-memory baseline: a bare `ScriptedPolicy` reads the mutated label off
    the page and never consults a graph, so there is nothing to go stale. It is
    the control the hazard is measured against — same mutated site, but memory off
    means no stale misfire. (Here the scripted baseline keys off the real label.)
    """
    case = prepare_case(BUY_BLUE_SHIRT, tmp_path / "wd_base", seed)
    mutated = create_mutated_app(case.db_path, mutations=("rename_controls",))
    browser = TestClientBrowser(mutated)

    # Baseline agent reads the page's ACTUAL (mutated) labels — no memory.
    steps = _login() + [
        Action.goto("/product/SHIRT-BLU"),
        Action.click("text=Add to bag"),       # the real, mutated label
        Action.goto("/checkout"),
        Action.fill("#ship_address", "12 Elm St, Austin, TX 78701"),
        Action.click("text=Complete purchase"),  # mutated "Place order"
    ]
    transcript = asyncio.run(
        run(BUY_BLUE_SHIRT.instruction("http://shop"), browser, ScriptedPolicy(steps))
    )

    # No memory, so no stale misfire: a clean run against the mutated labels.
    assert transcript.mistakes == []
    g = grade(case)
    assert g.passed, f"A1 baseline failed under mutation: {g.failed_checks}"
