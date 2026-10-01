"""
v2 graders: prove the cheap judges can be scored against SQL truth — and that a
screenshot judge's false success is exactly what the scoring catches.

The headline test stages the classic trap: the agent bought the look-alike White
shirt instead of the Blue one. SQL knows (`correct_items` fails). The DOM grader,
reading the order page, also catches it. But a screenshot judge that just sees
"Order #1 ... placed" declares victory — and `false_success_rate` records it.
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from app.main import create_app
from bench.env import prepare_case
from bench.graders import (
    DomGrader,
    LlmScreenshotGrader,
    SqlGrader,
    false_success_rate,
    grade_all,
    render_state,
)
from bench.tasks import BUY_BLUE_SHIRT, add, checkout, login


# ---- a fake screenshot judge: rules success/failure without a model or a key ----

class _Verdict:
    def __init__(self, success, reason):
        self.type = "tool_use"
        self.name = "verdict"
        self.input = {"success": success, "reason": reason}


class _Msg:
    def __init__(self, blocks):
        self.content = blocks


class _Messages:
    def __init__(self, success):
        self._success = success

    def create(self, **kwargs):
        return _Msg([_Verdict(self._success, "it says placed on the screen")])


class FakeJudge:
    """A credulous judge: it always says what you told it to."""

    def __init__(self, success):
        self.messages = _Messages(success)


def _case_with(seed, tmp_path, label, sku, address):
    case = prepare_case(BUY_BLUE_SHIRT, tmp_path / label, seed)
    with TestClient(create_app(case.db_path)) as c:
        login(c, "alice@example.com", "password123")
        add(c, sku)
        checkout(c, address)
    return case


def test_render_state_shows_the_order_after_checkout(seed, tmp_path):
    case = _case_with(seed, tmp_path, "blue", "SHIRT-BLU", "12 Elm St, Austin, TX 78701")
    state = render_state(case)
    assert "blue oxford shirt" in state["/orders/latest"].lower()
    assert "placed" in state["/orders/latest"].lower()


def test_sql_grader_matches_truth(seed, tmp_path):
    case = _case_with(seed, tmp_path, "blue", "SHIRT-BLU", "12 Elm St, Austin, TX 78701")
    v = SqlGrader().judge(case, render_state(case))
    assert v.grader == "sql" and v.passed


def test_dom_grader_passes_right_order_and_fails_the_lookalike(seed, tmp_path):
    good = _case_with(seed, tmp_path, "blue", "SHIRT-BLU", "12 Elm St, Austin, TX 78701")
    bad = _case_with(seed, tmp_path, "white", "SHIRT-WHT", "12 Elm St, Austin, TX 78701")
    assert DomGrader().judge(good, render_state(good)).passed
    assert not DomGrader().judge(bad, render_state(bad)).passed   # DOM reads the item name


def test_screenshot_false_success_is_measured(seed, tmp_path):
    """White shirt bought: SQL fails, DOM fails, a credulous screenshot judge passes."""
    bad = _case_with(seed, tmp_path, "white", "SHIRT-WHT", "12 Elm St, Austin, TX 78701")
    graders = [SqlGrader(), DomGrader(), LlmScreenshotGrader(client=FakeJudge(True))]
    row = grade_all(bad, graders)

    by = {v.grader: v for v in row}
    assert not by["sql"].passed            # truth: wrong item
    assert not by["dom"].passed            # DOM caught the wrong item
    assert by["screenshot"].passed         # judge fooled by "placed"

    scores = false_success_rate([row])
    assert scores["screenshot"].false_success == 1
    assert scores["screenshot"].false_success_rate == 1.0
    assert scores["dom"].false_success == 0         # DOM didn't lie here


def test_correct_order_has_no_false_successes(seed, tmp_path):
    good = _case_with(seed, tmp_path, "blue", "SHIRT-BLU", "12 Elm St, Austin, TX 78701")
    graders = [SqlGrader(), DomGrader(), LlmScreenshotGrader(client=FakeJudge(True))]
    scores = false_success_rate([grade_all(good, graders)])
    assert scores["screenshot"].false_success == 0
    assert scores["screenshot"].agree == 1
