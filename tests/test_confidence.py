"""
B3: the Wilson confidence bands tell the truth at the extremes.

We check three things: (1) known Wilson bounds for small samples match values
computed by hand (the whole point is that 0/10 and 10/10 don't collapse to a
zero-width band); (2) more samples at the same rate tighten the band; and
(3) wrapping a real `GraderScore` (14 false-successes of 70) yields a band that
brackets the 0.20 point estimate.
"""

from __future__ import annotations

from bench.confidence import band, false_success_band, wilson_interval
from bench.graders import GraderScore

TOL = 1e-4


def test_zero_successes_band():
    """0/10 at 95%: lower pinned to 0, upper ~0.2775 — not the normal-approx [0, 0]."""
    lower, upper = wilson_interval(0, 10)
    assert lower == 0.0
    assert abs(upper - 0.277533) < TOL


def test_all_successes_band():
    """10/10 at 95%: upper pinned to 1, lower ~0.7225 — the mirror of 0/10."""
    lower, upper = wilson_interval(10, 10)
    assert upper == 1.0
    assert abs(lower - 0.722467) < TOL


def test_one_of_ten_band():
    """1/10 at 95%: an interior case with hand-computed asymmetric bounds."""
    lower, upper = wilson_interval(1, 10)
    assert abs(lower - 0.017876) < TOL
    assert abs(upper - 0.404150) < TOL


def test_total_zero_is_maximal_ignorance():
    """No samples => the band is the whole [0, 1]; the point defaults to 0."""
    b = band(0, 0)
    assert (b.lower, b.upper) == (0.0, 1.0)
    assert b.point == 0.0
    assert b.total == 0


def test_more_samples_tighten_the_band():
    """Same rate (10%), bigger n => strictly narrower interval (monotone)."""
    widths = [band(n // 10, n).width for n in (10, 100, 1000)]
    assert widths[0] > widths[1] > widths[2]


def test_point_sits_inside_the_band():
    """The observed rate always lies between the bounds."""
    for successes, total in [(0, 10), (1, 10), (5, 10), (10, 10), (14, 70)]:
        b = band(successes, total)
        assert b.lower <= b.point <= b.upper


def test_confidence_level_widens_band():
    """A higher confidence level is a wider interval for the same data."""
    narrow = band(14, 70, confidence=0.80)
    wide = band(14, 70, confidence=0.99)
    assert wide.width > narrow.width


def test_wraps_grader_score_and_brackets_020():
    """14/70 false-successes => a band whose point is 0.20 and brackets it."""
    score = GraderScore(grader="dom", total=70, agree=50, false_success=14, false_failure=6)
    b = false_success_band(score)
    assert abs(b.point - 0.20) < TOL
    assert b.point == score.false_success_rate
    assert b.lower < 0.20 < b.upper
    # Hand-computed Wilson bounds for 14/70 at 95%.
    assert abs(b.lower - 0.123047) < TOL
    assert abs(b.upper - 0.308166) < TOL


def test_rejects_out_of_range_successes():
    """More successes than trials is a programming error, not a 100%+ rate."""
    import pytest

    with pytest.raises(ValueError):
        wilson_interval(8, 5)
