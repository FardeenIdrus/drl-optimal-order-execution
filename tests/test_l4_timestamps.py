"""Unit tests for Stage L4-1b/2c: binary order-status reader and timestamp attachment.

Run with:  PYTHONPATH=src pytest tests/test_l4_timestamps.py
"""
import gzip

import numpy as np

from execution.data.l4.book_diffs_reader import (
    ASK,
    BID,
    NEW,
    REMOVE,
    UPDATE,
    BookEvent,
    attach_timestamps,
    inject_removes,
)
from execution.data.l4.book_engine import BookEngine
from execution.data.l4.orders_reader import (
    RECORD_DTYPE,
    STATUS_CANCELED,
    STATUS_FILLED,
    STATUS_OPEN,
    build_open_and_removes,
    build_open_ts_map,
    decode_price,
)


def _write_orders_gz(path, records):
    """records: list of dicts with ts/statusId/oid (other fields zero-filled)."""
    a = np.zeros(len(records), dtype=RECORD_DTYPE)
    for i, r in enumerate(records):
        for k, v in r.items():
            a[i][k] = v
    with gzip.open(path, "wb") as fh:
        fh.write(a.tobytes())


# ------------------------------------------------------------- binary reader
def test_decode_price_matches_schema_example():
    # SCHEMA.md worked example: $96,543.21 -> decimals=2, value=9654321 -> 0x40935031
    encoded = (2 << 29) | 9654321
    assert encoded == 0x40935031
    assert np.isclose(decode_price(encoded), 96543.21, atol=1e-6)
    # a size with 4 decimals: 0.5455 -> value 5455, decimals 4
    enc_sz = (4 << 29) | 5455
    assert np.isclose(decode_price(enc_sz), 0.5455, atol=1e-9)


def test_build_open_ts_map_keeps_only_opens(tmp_path):
    p = tmp_path / "btc_00.data.gz"
    _write_orders_gz(p, [
        {"ts": 1000, "oid": 10, "statusId": STATUS_OPEN},
        {"ts": 1500, "oid": 11, "statusId": STATUS_CANCELED},  # not an open -> excluded
        {"ts": 2000, "oid": 12, "statusId": STATUS_OPEN},
    ])
    m = build_open_ts_map(p)
    assert m == {10: 1000, 12: 2000}


def test_build_open_and_removes_covers_cancels_and_full_fills(tmp_path):
    p = tmp_path / "btc_00.data.gz"
    _write_orders_gz(p, [
        {"ts": 1000, "oid": 10, "statusId": STATUS_OPEN},
        {"ts": 1100, "oid": 11, "statusId": STATUS_OPEN},
        {"ts": 1200, "oid": 12, "statusId": STATUS_OPEN},
        {"ts": 2000, "oid": 10, "statusId": STATUS_FILLED, "sz": 0},   # full fill -> remove
        {"ts": 2100, "oid": 11, "statusId": 7},                        # reduceOnlyCxl -> remove
        {"ts": 2200, "oid": 12, "statusId": STATUS_FILLED, "sz": 5},   # partial (sz>0) -> NOT a remove
    ])
    open_map, removes = build_open_and_removes(p)
    assert open_map == {10: 1000, 11: 1100, 12: 1200}
    assert removes == [(2000, 10), (2100, 11)]  # full fill + reduceOnly cancel, time-sorted


def test_inject_removes_clears_a_leaked_cancel():
    # book-diffs place oid1 and oid2 but NEVER remove oid1 (a cancel variant they omit)
    tstream = [
        (1000, BookEvent(oid=1, side=BID, px=100.0, kind=NEW, sz=1.0)),
        (2000, BookEvent(oid=2, side=ASK, px=101.0, kind=NEW, sz=1.0)),
    ]
    removes = [(1500, 1)]  # order-status says oid1 left the book at 1500
    e = BookEngine()
    for _ts, ev in inject_removes(iter(tstream), removes):
        e.apply(ev)
    assert e.n_orders == 1                      # oid1 removed via the injected terminal event
    assert e.best_bid() is None                 # no phantom bid left
    assert e.best_ask() == 101.0


def test_retry_late_removes_clears_a_raced_cancel():
    # The sub-second race: a cancel whose status ts PRECEDES the order's `new` in the
    # diff-file ordering. The time-merged injection fires the remove first (orphan),
    # the `new` then rests forever. The end-of-hour retry must clear it.
    from execution.data.l4.reconstruct_month import retry_late_removes

    tstream = [
        (1000, BookEvent(oid=9, side=BID, px=99.0, kind=NEW, sz=1.0)),   # clock anchor
        (1200, BookEvent(oid=1, side=BID, px=100.0, kind=NEW, sz=1.0)),  # opens AFTER its cancel ts
    ]
    removes = [(1100, 1)]  # terminal status at 1100 < the new's arrival in the stream
    e = BookEngine()
    for _ts, ev in inject_removes(iter(tstream), removes):
        e.apply(ev)
    assert e.n_orphan_remove == 1 and e.best_bid() == 100.0   # the race: phantom rests
    assert retry_late_removes(e, removes) == 1                # retry clears it
    assert e.best_bid() == 99.0 and e.n_orders == 1
    # a second retry is a no-op: legit resting orders are untouched
    assert retry_late_removes(e, removes) == 0


def test_build_open_ts_map_rejects_bad_size(tmp_path):
    p = tmp_path / "bad.data.gz"
    with gzip.open(p, "wb") as fh:
        fh.write(b"\x00" * 55)  # not a multiple of 54
    try:
        build_open_ts_map(p)
        assert False, "expected ValueError on non-54-multiple file"
    except ValueError:
        pass


# ------------------------------------------------------------- timestamping
def _ev(oid, kind, side=BID, px=100.0, sz=1.0):
    return BookEvent(oid=oid, side=side, px=px, kind=kind, sz=sz)


def test_attach_timestamps_carry_forward():
    events = [
        _ev(1, NEW), _ev(1, REMOVE),            # new sets clock, remove inherits it
        _ev(2, NEW), _ev(2, UPDATE, sz=0.5),    # next new advances, update inherits
    ]
    ts_map = {1: 1000, 2: 2000}
    out = list(attach_timestamps(iter(events), ts_map))
    assert [t for t, _ in out] == [1000, 1000, 2000, 2000]


def test_attach_timestamps_clock_is_forward_only():
    # a 'new' with an earlier open ts must NOT move the clock backward (monotonic guarantee)
    events = [_ev(1, NEW), _ev(2, NEW), _ev(3, REMOVE)]
    ts_map = {1: 2000, 2: 1000}  # oid 2 opens "earlier" -> ignored for the clock
    out = list(attach_timestamps(iter(events), ts_map))
    assert [t for t, _ in out] == [2000, 2000, 2000]


def test_attach_timestamps_none_before_first_match_and_missing_oid():
    events = [
        _ev(99, REMOVE),   # before any matched new -> None
        _ev(5, NEW),       # oid missing from map -> does NOT advance clock (still None)
        _ev(1, NEW),       # matched -> clock = 1000
        _ev(5, REMOVE),    # inherits 1000
    ]
    ts_map = {1: 1000}
    out = list(attach_timestamps(iter(events), ts_map))
    assert [t for t, _ in out] == [None, None, 1000, 1000]
