"""
v2 grader-comparison pass: the population of correct/wrong runs across conditions
produces the disagreement table, and the headline false-successes are the ones we
can prove by hand — the side-effect tasks where the screen looks right but SQL
knows better.
"""

from __future__ import annotations

from bench.gradecompare import (
    DEFAULT_RUNS,
    Run,
    format_report,
    run_comparison,
)
from bench.graders import DomGrader, LlmScreenshotGrader, SqlGrader
from bench.run import CLEAN, seven_conditions

# Reuse the credulous fake judge from the graders test (no key, no network).
from tests.test_graders import FakeJudge


def test_correct_runs_pass_sql_everywhere(seed, tmp_path):
    """Every oracle run is SQL-green under every condition — the achievability floor."""
    correct = tuple(r for r in DEFAULT_RUNS if r.label == "correct")
    cells = run_comparison(correct, seven_conditions(), [SqlGrader()],
                           root=tmp_path, seed_path=seed)
    assert cells and all(c.by_grader()["sql"].passed for c in cells)


def test_dom_false_passes_the_side_effect_tasks(seed, tmp_path):
    """
    The keyless headline: on `cancel_latest_order`/wrong and `empty_cart`/wrong the
    DOM grader says PASS while SQL says FAIL — the screen looks right, the DB doesn't.
    """
    cells = run_comparison(DEFAULT_RUNS, (CLEAN,), [SqlGrader(), DomGrader()],
                           root=tmp_path, seed_path=seed)
    by_cell = {(c.task_id, c.label): c.by_grader() for c in cells}

    for task_id in ("cancel_latest_order", "empty_cart"):
        g = by_cell[(task_id, "wrong")]
        assert not g["sql"].passed, task_id          # truth: the agent got it wrong
        assert g["dom"].passed, task_id              # DOM is fooled by the screen


def test_dom_catches_the_wrong_item(seed, tmp_path):
    """Where the mistake is visible on the page (White shirt), DOM agrees with SQL."""
    cells = run_comparison(DEFAULT_RUNS, (CLEAN,), [SqlGrader(), DomGrader()],
                           root=tmp_path, seed_path=seed)
    g = {(c.task_id, c.label): c.by_grader() for c in cells}[("buy_blue_shirt", "wrong")]
    assert not g["sql"].passed and not g["dom"].passed   # both catch the wrong item


def test_screenshot_judge_false_success_is_scored(seed, tmp_path):
    """
    A credulous screenshot judge certifies the wrong-item buy as done; the pass
    records it as a false-success. (Proves the screenshot column wires through.)
    """
    graders = [SqlGrader(), DomGrader(), LlmScreenshotGrader(client=FakeJudge(True))]
    wrong_shirt = tuple(r for r in DEFAULT_RUNS
                        if r.task.id == "buy_blue_shirt" and r.label == "wrong")
    cells = run_comparison(wrong_shirt, (CLEAN,), graders, root=tmp_path, seed_path=seed)
    report = format_report(cells, screenshot_live=False)
    assert "FALSE-SUCCESS" in report
    assert "screenshot said PASS" in report
