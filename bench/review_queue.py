"""
B3: a human-review queue for the uncertain grading tail.

v2 (`bench/graders.py`) gives each non-SQL grader a yes/no `Verdict`. B3's
`bench/confidence.py` wraps a grader's false-success *rate* in a Wilson
`ConfidenceBand`. But a confidence band is also the right tool for one verdict's
own uncertainty: a grader that passed a case backed by a wide band straddling the
decision threshold is a verdict we shouldn't trust — and shouldn't silently ship.

This module routes that uncertain tail to a human. A `ReviewQueue` takes grader
verdicts, each with its Wilson band, and a decision `threshold`:

- A verdict is **uncertain** if it *abstained*, or if its band **straddles** the
  threshold (lower < threshold < upper) — the evidence doesn't commit to one side.
- An uncertain verdict is **enqueued** for a human to resolve.
- A **confident** verdict (band entirely on one side of the threshold) **bypasses**
  the queue and stands as-is.

Humans review the *most* uncertain items first: the queue dequeues **widest band
first**, ties broken stably by enqueue order. A human `resolve` writes a final
verdict that **overrides** the machine's — the whole point of a review queue.

Pure stdlib. Reads `bench.graders.Verdict` and `bench.confidence.ConfidenceBand`
read-only; it never mutates or re-grades anything.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Optional

from bench.confidence import ConfidenceBand
from bench.graders import Verdict


def straddles(band: ConfidenceBand, threshold: float) -> bool:
    """True when the band brackets the threshold on both sides.

    `lower < threshold < upper` — the interval can't decide which side of the
    decision line the true rate is on. A band that merely *touches* the threshold
    (lower == threshold, or upper == threshold) is counted as committed, not
    straddling: its mass is on one side.
    """
    return band.lower < threshold < band.upper


def is_uncertain(verdict: Verdict, band: ConfidenceBand, threshold: float) -> bool:
    """A verdict is uncertain if it abstained or its band straddles the threshold."""
    return verdict.abstained or straddles(band, threshold)


@dataclass(frozen=True)
class ReviewItem:
    """One case's verdict awaiting (or past) human review.

    `key` identifies the case/grader pair so a human can resolve the right item.
    `verdict`/`band` are the machine's call and its uncertainty. `seq` is the
    enqueue order, kept for a stable tie-break. `resolution`, once set by
    `resolve`, is the human verdict that overrides `verdict`.
    """

    key: str
    verdict: Verdict
    band: ConfidenceBand
    seq: int
    resolution: Optional[Verdict] = None

    @property
    def resolved(self) -> bool:
        return self.resolution is not None

    @property
    def final(self) -> Verdict:
        """The verdict that stands: the human's if resolved, else the machine's."""
        return self.resolution if self.resolution is not None else self.verdict


class ReviewQueue:
    """Routes uncertain grader verdicts to humans; passes confident ones through.

    `enqueue` admits *only* uncertain items (confident verdicts return `None` and
    are left untouched). `dequeue` hands back the widest-band item first — the
    most uncertain reviewed first — breaking ties by enqueue order. `resolve`
    records a human verdict that overrides the machine's.
    """

    def __init__(self, threshold: float = 0.5) -> None:
        self.threshold = threshold
        self._items: dict[str, ReviewItem] = {}
        self._next_seq = 0

    # --------------------------------------------------------------- ingest
    def enqueue(self, key: str, verdict: Verdict, band: ConfidenceBand) -> Optional[ReviewItem]:
        """Admit `verdict` for human review iff it's uncertain under `threshold`.

        Confident verdicts bypass the queue: returns `None`, nothing is stored.
        Uncertain verdicts are stored and returned as a `ReviewItem`.
        """
        if not is_uncertain(verdict, band, self.threshold):
            return None
        item = ReviewItem(key=key, verdict=verdict, band=band, seq=self._next_seq)
        self._next_seq += 1
        self._items[key] = item
        return item

    # --------------------------------------------------------------- ordering
    def _order(self) -> list[ReviewItem]:
        """Pending items, widest band first, enqueue order breaking ties stably."""
        pending = [it for it in self._items.values() if not it.resolved]
        # Sort key: wider band first (so negate width), then earlier seq first.
        return sorted(pending, key=lambda it: (-it.band.width, it.seq))

    def pending(self) -> list[ReviewItem]:
        """All unresolved items in review order (widest band first)."""
        return self._order()

    def peek(self) -> Optional[ReviewItem]:
        """The next item a human would review, without removing it."""
        order = self._order()
        return order[0] if order else None

    def dequeue(self) -> Optional[ReviewItem]:
        """Remove and return the most-uncertain pending item, or `None` if empty.

        Dequeued items leave the queue entirely; resolve by `key` if needed, or
        resolve before dequeuing. Either way the item's final verdict is tracked
        only while it lives in the queue.
        """
        order = self._order()
        if not order:
            return None
        item = order[0]
        del self._items[item.key]
        return item

    # --------------------------------------------------------------- resolve
    def resolve(self, key: str, human_verdict: Verdict) -> ReviewItem:
        """Record a human verdict for `key`; it overrides the machine verdict.

        Returns the updated item. Raises `KeyError` if nothing is queued for `key`.
        """
        item = self._items.get(key)
        if item is None:
            raise KeyError(f"no queued item for key {key!r}")
        resolved = replace(item, resolution=human_verdict)
        self._items[key] = resolved
        return resolved

    # --------------------------------------------------------------- introspection
    def __len__(self) -> int:
        """Number of items currently in the queue (resolved or not)."""
        return len(self._items)

    def get(self, key: str) -> Optional[ReviewItem]:
        """The queued item for `key`, or `None`."""
        return self._items.get(key)
