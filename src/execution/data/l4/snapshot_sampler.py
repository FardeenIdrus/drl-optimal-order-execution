"""Stage L4-2/2d - sample the reconstructed book on a fixed clock grid.

Drives a :class:`~execution.data.l4.book_engine.BookEngine` over a timestamped event
stream (from :func:`~execution.data.l4.book_diffs_reader.attach_timestamps`) and emits
one snapshot per ``cadence`` interval of wall-clock time -- the book state as it stands
entering each grid boundary. The agent decides on a clock, not per event (~1,400 events/s
is untrainable), so this is where event-time collapses to the decision cadence; the fine
event stream still feeds calibration and order-flow features separately.

Because the reconstruction is a diff replay, the book near the touch is only correct
after a warm-up (pre-existing resting orders churn out). ``warmup_events`` suppresses
emission until that many events have been applied; in the full-month run the engine
carries across hour boundaries, so the warm-up is paid once at the start, not per hour.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Iterator, List, Optional, Tuple

from execution.data.l4.book_diffs_reader import BookEvent
from execution.data.l4.book_engine import BookEngine, Level

logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class Snapshot:
    """The sampled book at one grid boundary.

    Attributes:
        ts: grid-boundary timestamp (ns) this snapshot represents.
        best_bid, best_ask, mid, spread: top of book (``None`` if a side is empty).
        bids, asks: top-k ``(price, size)`` levels, best first.
    """

    ts: int
    best_bid: Optional[float]
    best_ask: Optional[float]
    mid: Optional[float]
    spread: Optional[float]
    bids: List[Level]
    asks: List[Level]


class SnapshotSampler:
    """Stateful sampler: drives a :class:`BookEngine` and emits snapshots on a grid.

    State (``next_boundary``, ``n_applied``) persists across :meth:`feed` calls, so a
    driver can feed one file at a time (carrying the book and the grid across hour/day
    boundaries) and checkpoint via :meth:`get_state` / :meth:`set_state` to resume a
    long run without replaying from the start.
    """

    def __init__(
        self,
        engine: BookEngine,
        cadence_ns: int,
        top_k: int = 20,
        warmup_events: int = 0,
    ) -> None:
        self.engine = engine
        self.cadence_ns = int(cadence_ns)
        self.top_k = int(top_k)
        self.warmup_events = int(warmup_events)
        self.next_boundary: Optional[int] = None
        self.n_applied = 0

    def feed(
        self, timestamped_events: Iterator[Tuple[Optional[int], BookEvent]]
    ) -> Iterator[Snapshot]:
        """Apply events and yield a :class:`Snapshot` at each grid boundary crossed.

        Boundaries are aligned to the absolute ``cadence_ns`` grid (reproducible, and
        consistent across files). A quiet gap spanning several boundaries emits one
        snapshot per boundary (the book is unchanged across them). ``None``-timestamp
        events (pre-anchor warm-up) are applied but never trigger emission.
        """
        cad = self.cadence_ns
        for ts, ev in timestamped_events:
            if ts is not None:
                if self.next_boundary is None:
                    self.next_boundary = (ts // cad + 1) * cad
                while ts >= self.next_boundary:
                    if self.n_applied >= self.warmup_events:
                        self.engine.drop_crossed()  # heal any stale crossed levels first
                        bb, ba, mid, spread = self.engine.top_of_book()
                        bids, asks = self.engine.snapshot(self.top_k)
                        yield Snapshot(self.next_boundary, bb, ba, mid, spread, bids, asks)
                    self.next_boundary += cad
            self.engine.apply(ev)
            self.n_applied += 1

    def get_state(self) -> dict:
        """Serialisable sampler state (the engine is checkpointed separately)."""
        return {"next_boundary": self.next_boundary, "n_applied": self.n_applied}

    def set_state(self, state: dict) -> None:
        self.next_boundary = state["next_boundary"]
        self.n_applied = int(state["n_applied"])


def iter_snapshots(
    timestamped_events: Iterator[Tuple[Optional[int], BookEvent]],
    engine: BookEngine,
    cadence_ns: int,
    top_k: int = 20,
    warmup_events: int = 0,
) -> Iterator[Snapshot]:
    """Convenience one-shot wrapper over :class:`SnapshotSampler` (see its docstring)."""
    return SnapshotSampler(engine, cadence_ns, top_k, warmup_events).feed(timestamped_events)
