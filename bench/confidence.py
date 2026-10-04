"""
B3: confidence bands for grader false-success rates.

v2 (`bench/graders.py`) reports a grader's false-success as a bare fraction —
`14/70 = 0.20`. A point estimate hides its own uncertainty: 14/70 and 2/10 are
both 0.20, but one is far better pinned down than the other, and a grader that
scores 0/70 is not *proven* perfect — it's "0, give or take". B3 turns each
point count into a **confidence band**: the point estimate plus a lower/upper
bound at a stated confidence (default 95%).

**Why Wilson, not the textbook normal approximation.** The normal-approx interval
(`p̂ ± z·√(p̂(1-p̂)/n)`) collapses to zero width at the extremes — exactly the
cells graders actually hit here. A grader that false-passes 0 of 70 cases, or all
70 of 70, gets `p̂(1-p̂) = 0`, so normal-approx reports the absurd band `[0, 0]`
or `[1, 1]`: "certainly perfect" / "certainly broken" from finite data. It also
produces bounds below 0 or above 1 for small counts. The Wilson score interval
stays inside `[0, 1]` and keeps a sensible, asymmetric width at `x = 0` and
`x = n` (0/70 → roughly `[0, 0.05]`, not `[0, 0]`), which is why it's the right
tool for a benchmark whose headline numbers live at the boundary.

Pure stdlib: `math` for the arithmetic and `statistics.NormalDist` for the
z-score. No numpy/scipy.
"""

from __future__ import annotations

import math
import statistics
from dataclasses import dataclass


@dataclass(frozen=True)
class ConfidenceBand:
    """A binomial proportion with a Wilson score interval around it.

    `point` is the observed rate (successes / total); `lower`/`upper` bracket it
    at `confidence`. All three are in [0, 1]. For `total == 0` the proportion is
    undefined, so the band is the whole interval [0, 1] — maximal ignorance.
    """

    point: float
    lower: float
    upper: float
    confidence: float
    successes: int
    total: int

    @property
    def width(self) -> float:
        """How wide the interval is — shrinks as evidence accumulates."""
        return self.upper - self.lower


def _clamp01(x: float) -> float:
    """Pin a bound into [0, 1] — guards floating-point overshoot at the extremes."""
    return 0.0 if x < 0.0 else 1.0 if x > 1.0 else x


def wilson_interval(successes: int, total: int, confidence: float = 0.95) -> tuple[float, float]:
    """Wilson score interval for a binomial proportion of `successes` in `total`.

    Returns (lower, upper), both clamped to [0, 1]. Correct at the edges:
    `total == 0` gives the uninformative [0, 1]; `successes == 0` gives a
    lower bound of exactly 0 with a positive upper bound; `successes == total`
    gives an upper bound of exactly 1 with a sub-1 lower bound.
    """
    if total <= 0:
        return (0.0, 1.0)
    if successes < 0 or successes > total:
        raise ValueError(f"successes {successes} out of range for total {total}")

    p = successes / total
    # Two-sided z: e.g. 1.959964 for 95%.
    z = statistics.NormalDist().inv_cdf((1.0 + confidence) / 2.0)
    z2 = z * z
    denom = 1.0 + z2 / total
    center = (p + z2 / (2.0 * total)) / denom
    margin = (z / denom) * math.sqrt(p * (1.0 - p) / total + z2 / (4.0 * total * total))
    return (_clamp01(center - margin), _clamp01(center + margin))


def band(successes: int, total: int, confidence: float = 0.95) -> ConfidenceBand:
    """Wrap a raw success/total count in a `ConfidenceBand` at `confidence`."""
    lower, upper = wilson_interval(successes, total, confidence)
    point = successes / total if total else 0.0
    return ConfidenceBand(point, lower, upper, confidence, successes, total)


def false_success_band(score, confidence: float = 0.95) -> ConfidenceBand:
    """Confidence band for a `bench.graders.GraderScore`'s false-success rate.

    `score` is read-only: we use `.false_success` and `.total` (and leave
    `.false_success_rate`, `.agree`, `.false_failure` to the caller). The band's
    `point` equals `score.false_success_rate`, now carried with its interval.
    """
    return band(score.false_success, score.total, confidence)
