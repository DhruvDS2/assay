"""
The agent harness drives the real shop, and tells the truth about what it did.

These tests prove three things:

    the loop can operate every part of the shop through the Browser abstraction
      (login, add-to-cart, quantities, multi-form pages, link-following, cancel),
      and the grader confirms it from SQL — the browser analogue of selfcheck;

    an agent *mistake* is recorded on its step and the run still returns
      (no exception) — so bench/run.py grades it as a finished-but-wrong run;

    an *infra* failure raises out of run() — so bench/run.py retries it.
"""

from __future__ import annotations

import asyncio
import re
from pathlib import Path

import pytest

from app.main import create_app
from bench.agent import (
    Action,
    AgentActionError,
    Observation,
    ScriptedPolicy,
    TestClientBrowser,
    Transcript,
    run,
    testclient_run_fn,
    visible_text,
)
from bench.env import build_seed, prepare_case
from bench.grader import grade
from bench.tasks import (
    BUY_BLUE_SHIRT,
    BUY_SOCKS_X3,
    BUY_TOTE_AND_MUG,
    CANCEL_LATEST_ORDER,
    EMPTY_CART,
    TASKS,
)


# ------------------------------------------------------------------ scripted runs
# Each task expressed as browser steps — the oracle, re-derived through fill/click
# instead of raw POSTs. Together they exercise every Browser feature.


def _login() -> list:
    return [
        Action.goto("/login"),
        Action.fill("#email", "alice@example.com"),
        Action.fill("#password", "password123"),
        Action.click("text=Log in"),
    ]


def _open_latest_order(obs: Observation) -> Action:
    href = re.search(r"/orders/\d+", obs.html).group(0)   # /orders is newest-first
    return Action.click(f"a[href='{href}']")


SCRIPTS = {
    BUY_BLUE_SHIRT.id: _login() + [
        Action.goto("/product/SHIRT-BLU"),
        Action.click("text=Add to cart"),
        Action.goto("/checkout"),
        Action.fill("#ship_address", "12 Elm St, Austin, TX 78701"),
        Action.click("text=Place order"),
    ],
    BUY_SOCKS_X3.id: _login() + [
        Action.goto("/product/SOCK-WOOL"),
        Action.fill("#qty", "3"),                 # exercises filling a number field
        Action.click("text=Add to cart"),
        Action.goto("/checkout"),
        Action.fill("#ship_address", "400 Main St, Dallas, TX 75201"),
        Action.click("text=Place order"),
    ],
    BUY_TOTE_AND_MUG.id: _login() + [
        Action.goto("/cart"),
        Action.click("text=Remove Black Cap"),    # disambiguates among many forms
        Action.goto("/product/TOTE-CNV"),
        Action.click("text=Add to cart"),
        Action.goto("/product/MUG-ENML"),
        Action.click("text=Add to cart"),
        Action.goto("/checkout"),
        Action.fill("#ship_address", "9 Oak Ave, Houston, TX 77002"),
        Action.click("text=Place order"),
    ],
    CANCEL_LATEST_ORDER.id: _login() + [
        Action.goto("/orders"),
        _open_latest_order,                        # reads the page to find the link
        Action.click("text=Cancel order"),
    ],
    EMPTY_CART.id: _login() + [
        Action.goto("/cart"),
        Action.click("text=Empty cart"),
    ],
}


@pytest.fixture
def seed(tmp_path: Path) -> Path:
    return build_seed(tmp_path / "seed.db")


def _drive(task, steps, tmp_path, seed) -> tuple[Transcript, "Grade"]:
    case = prepare_case(task, tmp_path / "wd", seed)
    browser = TestClientBrowser(create_app(case.db_path))
    transcript = asyncio.run(
        run(task.instruction("http://shop"), browser, ScriptedPolicy(steps))
    )
    return transcript, grade(case)


@pytest.mark.parametrize("task", TASKS, ids=lambda t: t.id)
def test_scripted_agent_passes_every_task_through_the_browser(task, tmp_path, seed):
    transcript, g = _drive(task, SCRIPTS[task.id], tmp_path, seed)
    assert g.passed, f"{task.id} failed checks: {g.failed_checks}"
    assert transcript.mistakes == []            # a clean run makes no mistakes


# ----------------------------------------------------------------- the invariant


def test_agent_mistake_is_recorded_not_raised(tmp_path, seed):
    """Clicking a button that isn't there is a mistake: recorded, run continues."""
    steps = _login() + [
        Action.goto("/product/SHIRT-BLU"),
        Action.click("text=Teleport to checkout"),   # no such button
        Action.click("text=Add to cart"),            # loop still gets here
    ]
    transcript, g = _drive(BUY_BLUE_SHIRT, steps, tmp_path, seed)

    bad = [s for s in transcript.steps if s.action.target == "text=Teleport to checkout"]
    assert len(bad) == 1 and bad[0].error is not None
    assert "Teleport" in bad[0].error
    # The run finished normally and the next action still ran (shirt is in cart),
    # but the task wasn't completed (never checked out) -> graded as a failure.
    assert not g.passed


def test_infra_failure_propagates_out_of_run():
    """A transport error is not an AgentActionError, so run() must not swallow it."""

    class DeadBrowser:
        url = "about:blank"

        async def content(self):
            return ""

        async def goto(self, url):
            raise ConnectionError("browser crashed")

        async def fill(self, s, v): ...
        async def click(self, s): ...

    policy = ScriptedPolicy([Action.goto("/login")])
    with pytest.raises(ConnectionError):
        asyncio.run(run("do the thing", DeadBrowser(), policy))


def test_stop_ends_the_loop_with_its_reason(tmp_path, seed):
    case = prepare_case(BUY_BLUE_SHIRT, tmp_path / "wd", seed)
    browser = TestClientBrowser(create_app(case.db_path))
    steps = [Action.goto("/login"), Action.stop("I think I'm done")]
    transcript = asyncio.run(run("x", browser, ScriptedPolicy(steps), max_steps=25))

    assert transcript.stopped_reason == "I think I'm done"
    assert transcript.n_steps == 2                 # goto, then stop (loop broke)
    assert transcript.steps[-1].action.kind == "stop"


def test_max_steps_is_honoured(tmp_path, seed):
    case = prepare_case(BUY_BLUE_SHIRT, tmp_path / "wd", seed)
    browser = TestClientBrowser(create_app(case.db_path))
    # A never-ending policy: keep reloading. The loop must stop at max_steps.
    policy = ScriptedPolicy([Action.goto("/login")] * 100)
    transcript = asyncio.run(run("x", browser, policy, max_steps=5))
    assert transcript.n_steps == 5
    assert transcript.stopped_reason == "max_steps"


# ---------------------------------------------------------------- small surfaces


def test_visible_text_strips_tags_and_collapses_space():
    assert visible_text("<h1>Hi</h1>\n<p>  there &amp; you </p>") == "Hi there & you"


def test_fill_on_a_missing_field_is_a_mistake(tmp_path, seed):
    case = prepare_case(BUY_BLUE_SHIRT, tmp_path / "wd", seed)
    browser = TestClientBrowser(create_app(case.db_path))
    asyncio.run(browser.goto("/login"))
    with pytest.raises(AgentActionError):
        asyncio.run(browser.fill("#nonexistent", "x"))


# ----------------------------------------------------- integration through run.py


def test_full_matrix_runs_through_the_agent_harness(tmp_path, seed):
    """bench/run.py + engine + agent together: the scripted agent sweeps the matrix."""
    from bench.run import DEFAULT_CONDITIONS, run_matrix

    run_fn = testclient_run_fn(lambda case: ScriptedPolicy(SCRIPTS[case.task.id]))
    summary, _ = asyncio.run(
        run_matrix(TASKS, DEFAULT_CONDITIONS, 1,
                   root=tmp_path, run_fn=run_fn, seed_path=seed)
    )
    assert summary.passed == summary.planned == len(TASKS)
    assert summary.skipped == 0
