"""
B3: tests for the human-review queue (deterministic, keyless).

Covers the four contract points:
- uncertain verdicts (straddling band or abstention) are enqueued
- confident verdicts bypass the queue (enqueue returns None, nothing stored)
- a resolved (human) verdict overrides the machine verdict
- dequeue ordering is widest-band-first, ties broken stably by enqueue order
"""

from __future__ import annotations

from bench.confidence import ConfidenceBand
from bench.graders import Verdict
from bench.review_queue import ReviewQueue, is_uncertain, straddles


# --------------------------------------------------------------- band helpers

def _band(lower: float, upper: float, point: float | None = None) -> ConfidenceBand:
    """A hand-built band for exercising the queue, independent of Wilson math."""
    pt = point if point is not None else (lower + upper) / 2
    return ConfidenceBand(
        point=pt, lower=lower, upper=upper, confidence=0.95, successes=0, total=10
    )


def _pass(grader: str = "screenshot") -> Verdict:
    return Verdict(grader, passed=True, detail="machine pass")


def _fail(grader: str = "screenshot") -> Verdict:
    return Verdict(grader, passed=False, detail="machine fail")


def _abstain(grader: str = "dom") -> Verdict:
    return Verdict(grader, passed=False, detail="no opinion", abstained=True)


# --------------------------------------------------------------- predicates

def test_straddles_only_when_threshold_strictly_inside():
    assert straddles(_band(0.3, 0.7), 0.5) is True
    # Touching the threshold at an edge is committed, not straddling.
    assert straddles(_band(0.5, 0.9), 0.5) is False
    assert straddles(_band(0.1, 0.5), 0.5) is False
    # Entirely on one side.
    assert straddles(_band(0.6, 0.9), 0.5) is False
    assert straddles(_band(0.05, 0.2), 0.5) is False


def test_is_uncertain_covers_straddle_and_abstention():
    assert is_uncertain(_pass(), _band(0.3, 0.7), 0.5) is True
    assert is_uncertain(_pass(), _band(0.6, 0.9), 0.5) is False
    # Abstention is uncertain regardless of band position.
    assert is_uncertain(_abstain(), _band(0.6, 0.9), 0.5) is True


# --------------------------------------------------------------- enqueue routing

def test_uncertain_verdict_is_enqueued():
    q = ReviewQueue(threshold=0.5)
    item = q.enqueue("case1", _pass(), _band(0.3, 0.7))
    assert item is not None
    assert len(q) == 1
    assert q.get("case1") is item
    assert q.peek() is item


def test_abstention_is_enqueued_even_with_confident_band():
    q = ReviewQueue(threshold=0.5)
    item = q.enqueue("case1", _abstain(), _band(0.8, 0.95))
    assert item is not None
    assert len(q) == 1


def test_confident_verdict_bypasses_queue():
    q = ReviewQueue(threshold=0.5)
    # Band entirely above the threshold -> confident -> bypass.
    assert q.enqueue("hi", _pass(), _band(0.6, 0.9)) is None
    # Band entirely below the threshold -> confident -> bypass.
    assert q.enqueue("lo", _fail(), _band(0.05, 0.2)) is None
    assert len(q) == 0
    assert q.get("hi") is None
    assert q.peek() is None
    assert q.dequeue() is None


# --------------------------------------------------------------- resolution override

def test_resolution_overrides_machine_verdict():
    q = ReviewQueue(threshold=0.5)
    q.enqueue("case1", _pass(), _band(0.3, 0.7))   # machine says PASS
    item = q.get("case1")
    assert item.final.passed is True               # before review: machine stands
    assert item.resolved is False

    human = Verdict("human", passed=False, detail="actually wrong")
    resolved = q.resolve("case1", human)

    assert resolved.resolved is True
    assert resolved.final is human                 # human overrides machine
    assert resolved.final.passed is False
    assert resolved.verdict.passed is True         # machine verdict preserved alongside
    # The stored item reflects the resolution too.
    assert q.get("case1").final.passed is False


def test_resolve_unknown_key_raises():
    q = ReviewQueue(threshold=0.5)
    try:
        q.resolve("nope", _pass("human"))
    except KeyError:
        pass
    else:
        raise AssertionError("expected KeyError for unknown key")


# --------------------------------------------------------------- ordering

def test_dequeue_is_widest_band_first():
    q = ReviewQueue(threshold=0.5)
    # All straddle 0.5; widths: narrow=0.2, mid=0.4, wide=0.8.
    q.enqueue("narrow", _pass(), _band(0.4, 0.6))
    q.enqueue("wide", _pass(), _band(0.1, 0.9))
    q.enqueue("mid", _pass(), _band(0.3, 0.7))

    assert q.dequeue().key == "wide"
    assert q.dequeue().key == "mid"
    assert q.dequeue().key == "narrow"
    assert q.dequeue() is None


def test_pending_order_matches_dequeue_order():
    q = ReviewQueue(threshold=0.5)
    q.enqueue("narrow", _pass(), _band(0.4, 0.6))
    q.enqueue("wide", _pass(), _band(0.1, 0.9))
    q.enqueue("mid", _pass(), _band(0.3, 0.7))
    assert [it.key for it in q.pending()] == ["wide", "mid", "narrow"]


def test_ties_broken_stably_by_enqueue_order():
    q = ReviewQueue(threshold=0.5)
    # Three identical-width straddling bands -> tie on width. Use the same
    # endpoints so widths are bit-for-bit equal and the tie-break is exercised.
    q.enqueue("first", _pass(), _band(0.3, 0.7))
    q.enqueue("second", _pass(), _band(0.3, 0.7))
    q.enqueue("third", _pass(), _band(0.3, 0.7))
    # All identical width; enqueue order decides.
    assert [it.key for it in q.pending()] == ["first", "second", "third"]
    assert q.dequeue().key == "first"
    assert q.dequeue().key == "second"
    assert q.dequeue().key == "third"


def test_resolved_items_drop_out_of_pending_but_still_overridable():
    q = ReviewQueue(threshold=0.5)
    q.enqueue("a", _pass(), _band(0.1, 0.9))
    q.enqueue("b", _pass(), _band(0.3, 0.7))
    q.resolve("a", Verdict("human", passed=False))
    # Resolved "a" no longer awaits review; "b" is next.
    assert [it.key for it in q.pending()] == ["b"]
    assert q.peek().key == "b"
    # But the resolution still stands and overrides.
    assert q.get("a").final.passed is False
