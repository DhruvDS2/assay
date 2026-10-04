"""
B1 report: the HTML is self-contained, puts the worst cells first, and renders the
ground-truth trace (the named checks the database used) for each failure.
"""

from __future__ import annotations

import asyncio

from bench.report import _worst_first, render_report, write_report
from bench.run import fault_conditions, run_matrix
from bench.hardened import hardened_recovery_run_fn
from bench.tasks import BUY_BLUE_SHIRT


def _summary(seed, tmp_path):
    summary, _ = asyncio.run(
        run_matrix((BUY_BLUE_SHIRT,), fault_conditions(), trials=1,
                   root=tmp_path / "m", seed_path=seed,
                   run_fn=hardened_recovery_run_fn())
    )
    return summary


def test_report_is_self_contained_html(seed, tmp_path):
    html = render_report(_summary(seed, tmp_path))
    assert html.startswith("<!doctype html>")
    assert "<script" not in html.lower()          # no JS, no external assets
    assert "http://" not in html and "https://" not in html
    assert "buy_blue_shirt" in html


def test_worst_cells_sort_first(seed, tmp_path):
    cells = _summary(seed, tmp_path).cells
    ordered = _worst_first(cells)
    rates = [c.pass_rate if c.pass_rate is not None else -1.0 for c in ordered]
    assert rates == sorted(rates)                  # ascending: failures/skips on top


def test_failures_show_their_ground_truth_checks(seed, tmp_path):
    """A failed fault cell names the SQL check that caught it, not a vibe."""
    html = render_report(_summary(seed, tmp_path))
    # error_on_checkout places no order, so the side-effect check fails by name.
    assert "fault_error_on_checkout" in html
    assert "exactly_one_new_order" in html


def test_grader_section_renders_false_success(seed, tmp_path):
    from bench.graders import GraderScore

    scores = {"dom": GraderScore("dom", total=10, agree=8, false_success=2, false_failure=0)}
    html = render_report(_summary(seed, tmp_path), grader_scores=scores)
    assert "Cheaper judges" in html
    assert "2/10 (20%)" in html                    # the false-success cell


def test_write_report_creates_the_file(seed, tmp_path):
    out = write_report(_summary(seed, tmp_path), tmp_path / "sub" / "r.html")
    assert out.exists() and out.read_text().startswith("<!doctype html>")
