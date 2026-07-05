"""Stage L4-1b - read the binary order-status file and build oid -> timestamp maps.

The Oxford order-status archive stores one gzip'd file of packed 54-byte C structs per
coin-hour (``<date>/<coin>_<HH>.data.gz``); see SCHEMA.md. Book-diff records carry no
timestamp, so we take event time from here: every accepted order has an ``open`` status
event whose ``ts`` (nanoseconds) is exactly when it was placed on the book. That gives a
dense ``oid -> open-time`` map used to time-stamp the book-diff stream (Stage L4-2).
"""
from __future__ import annotations

import gzip
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Union

import numpy as np

logger = logging.getLogger(__name__)

# Packed record layout (SCHEMA.md section 1). 54 bytes, little-endian.
RECORD_DTYPE = np.dtype(
    [
        ("ts", "<u8"), ("userId", "<u4"), ("isBuilder", "?"), ("statusId", "<u1"),
        ("isAsk", "?"), ("limitPx", "<u4"), ("sz", "<u4"), ("oid", "<u8"),
        ("timestampDiff", "<u4"), ("triggerCondition", "<i4"), ("triggered", "?"),
        ("isTrigger", "?"), ("hasChildren", "?"), ("isPositionTpsl", "?"),
        ("reduceOnly", "?"), ("orderTypeId", "<u1"), ("tifId", "<u1"),
        ("triggerPx", "<u4"), ("origSz", "<u4"),
    ]
)
assert RECORD_DTYPE.itemsize == 54

# statusId codes (SCHEMA.md statuses.csv).
STATUS_OPEN = 1
STATUS_CANCELED = 2
STATUS_FILLED = 5

# Every status that means a *resting* order has LEFT the book. Book-diffs reliably emit a
# `remove` only for plain `canceled`(2) and most fills; the other cancel variants below are
# largely absent from the diff stream, so they must be driven from the order-status file or
# the reconstruction leaks phantom orders. Verified on real data: {cancels} + {filled,sz==0}
# covers 100% of book-diff removals (nothing leaves the book without one of these).
# Excludes reject statuses (3,4,6,15,17) and `open`/`triggered`, which never remove resting
# depth. 7=reduceOnlyCxl 10=scheduledCxl 11=siblingFilledCxl 12=selfTradeCxl 13=marginCxl
# 14=vaultWithdrawalCxl 16=liquidatedCxl.
CANCEL_STATUSES = frozenset({2, 7, 10, 11, 12, 13, 14, 16})

_PX_POWERS = np.array([1, 1e1, 1e2, 1e3, 1e4, 1e5, 1e6, 1e7], dtype=np.float64)


def decode_price(encoded) -> np.ndarray:
    """Decode the uint32 fixed-point price/size encoding to float (SCHEMA.md).

    Top 3 bits are the decimal count, low 29 bits the integer value:
    ``decoded = (encoded & 0x1FFFFFFF) / 10 ** (encoded >> 29)``.
    """
    encoded = np.asarray(encoded, dtype=np.uint32)
    decimals = encoded >> 29
    value = encoded & 0x1FFFFFFF
    return value / _PX_POWERS[decimals]


def read_records(path: Union[str, Path]) -> np.ndarray:
    """Read one gzip'd order-status hour into a structured array.

    One hour is ~8M records (~0.4 GB transient); read whole, then discarded by the
    caller once the map is built. Raises if the file size is not a multiple of 54.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"order-status file not found: {path}")
    with gzip.open(path, "rb") as fh:
        buf = fh.read()
    if len(buf) % RECORD_DTYPE.itemsize != 0:
        raise ValueError(
            f"{path.name}: byte size {len(buf)} is not a multiple of {RECORD_DTYPE.itemsize}"
        )
    return np.frombuffer(buf, dtype=RECORD_DTYPE)


def build_open_ts_map(path: Union[str, Path]) -> Dict[int, int]:
    """Map ``oid -> open timestamp (ns)`` for every ``open`` event in one order hour.

    Used to time-stamp book-diff ``new`` events (each order's placement time). Cancel /
    fill times are not needed: ``new`` events are dense enough (hundreds per second) to
    serve as a monotonic time backbone for the whole diff stream (Stage L4-2 carries
    the last ``new`` timestamp forward onto the intervening remove/update events).
    """
    a = read_records(path)
    mask = a["statusId"] == STATUS_OPEN
    m = dict(zip(a["oid"][mask].tolist(), a["ts"][mask].tolist()))
    logger.info("open-ts map %s: %d opens of %d records", Path(path).name, len(m), len(a))
    return m


def build_open_and_removes(path: Union[str, Path]) -> Tuple[Dict[int, int], List[Tuple[int, int]]]:
    """Read one order-status hour once; return ``(open_ts_map, removes)``.

    ``removes`` is a time-sorted list of ``(ts_ns, oid)`` for every event that takes a
    resting order OFF the book: any cancel variant (:data:`CANCEL_STATUSES`) or a full
    fill (``filled`` with remaining size 0). The book-diff stream reliably emits removes
    only for plain cancels and most fills, so without this the reconstruction leaks
    phantom orders (chiefly from ``reduceOnlyCxl`` / ``siblingFilledCxl`` etc.) that
    accumulate and eventually cross the book. This set covers 100% of true removals, so a
    remove injected for an order the diffs already removed is a harmless orphan.
    """
    a = read_records(path)
    open_mask = a["statusId"] == STATUS_OPEN
    open_map = dict(zip(a["oid"][open_mask].tolist(), a["ts"][open_mask].tolist()))

    remove_mask = np.isin(a["statusId"], list(CANCEL_STATUSES)) | (
        (a["statusId"] == STATUS_FILLED) & (a["sz"] == 0)
    )
    rts, roid = a["ts"][remove_mask], a["oid"][remove_mask]
    order = np.argsort(rts, kind="stable")
    removes = list(zip(rts[order].tolist(), roid[order].tolist()))
    logger.info(
        "maps %s: %d opens, %d removes of %d records",
        Path(path).name, len(open_map), len(removes), len(a),
    )
    return open_map, removes
