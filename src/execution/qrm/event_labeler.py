"""Classify each book event as limit / cancel / market.

The QRM's three event types are: a **limit** order arriving (adds depth), a
**cancellation** (removes depth without a trade), and a **market** order (a trade that
consumes depth). The book-diff stream gives the mechanical change (new / remove /
update) but not *why* a removal happened, so we split it with the order-status
lifecycle:

    * ``new``    -> **limit** (an order was placed on the book).
    * ``update`` -> **market** (size only ever drops mid-life via a partial fill).
    * ``remove`` -> **market** if the order was fully filled (a trade cleared it),
                    else **cancel** (the common case; ~97% of removals).

"Fully filled" is taken from the order-status file (statusId ``filled`` with remaining
size 0); everything else removed is a cancel. This mirrors the reconstruction's removal
logic and its 100%-coverage check.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Set, Union

from execution.data.l4.book_diffs_reader import NEW, UPDATE, BookEvent
from execution.data.l4.orders_reader import STATUS_FILLED, read_records

logger = logging.getLogger(__name__)

# QRM event-type labels
LIMIT = "limit"
CANCEL = "cancel"
MARKET = "market"


def full_fill_oids(orders_path: Union[str, Path]) -> Set[int]:
    """Set of oids that were fully filled (statusId ``filled`` with remaining size 0).

    A ``remove`` for one of these was a trade (market order cleared the order); any other
    ``remove`` is a cancellation.
    """
    a = read_records(orders_path)
    mask = (a["statusId"] == STATUS_FILLED) & (a["sz"] == 0)
    return set(a["oid"][mask].tolist())


def classify(event: BookEvent, fully_filled: Set[int]) -> str:
    """Return the QRM event type (:data:`LIMIT` / :data:`CANCEL` / :data:`MARKET`).

    Book-diff-only classification (used for the event-mix sanity check). For the QRM
    *calibration*, use :func:`classify_for_calibration` instead: market orders are counted
    from the trades file, not inferred here (the book-diffs omit ~1/3 of fills).
    """
    if event.kind == NEW:
        return LIMIT
    if event.kind == UPDATE:
        return MARKET  # a mid-life size drop is a partial fill
    # REMOVE: a trade if the order fully filled, otherwise a cancellation
    return MARKET if event.oid in fully_filled else CANCEL


def classify_for_calibration(event: BookEvent, fully_filled: Set[int]):
    """Calibration classifier: :data:`LIMIT`, :data:`CANCEL`, or ``None`` (skip a fill).

    Fills (partial ``update`` or a fully-filled ``remove``) are skipped here and counted
    instead from the complete trades file, so market-order intensity isn't undercounted.
    Cancels and limit arrivals still come from the book-diffs. The event is still applied
    to the book by the caller regardless (a skipped fill still consumes depth).
    """
    if event.kind == NEW:
        return LIMIT
    if event.kind == UPDATE:
        return None  # partial fill -> counted from the trades file
    return None if event.oid in fully_filled else CANCEL
