"""Unit tests for Stage 3c: QRM intensity calibration helpers.

Run with:  PYTHONPATH=src pytest tests/test_qrm_calibrate.py
"""
import numpy as np

from execution.data.l4.book_diffs_reader import ASK, BID, NEW, REMOVE, TRADE, BookEvent
from execution.data.l4.book_engine import BookEngine
from execution.qrm.calibrate_intensities import (
    _depth_of,
    _topk_volumes,
    calibrate,
    intensities_from,
)
from execution.qrm.event_labeler import MARKET


def _mk_book():
    e = BookEngine()
    # asks at 101,102,103 ; bids at 100,99,98
    for oid, (side, px) in enumerate([(ASK, 101.0), (ASK, 102.0), (ASK, 103.0),
                                      (BID, 100.0), (BID, 99.0), (BID, 98.0)]):
        e.apply(BookEvent(oid=oid, side=side, px=px, kind=NEW, sz=1.0))
    return e


def test_depth_of_tick_distance():
    e = _mk_book()
    assert _depth_of(e, ASK, 101.0, 1.0, 5) == 1   # at best ask
    assert _depth_of(e, ASK, 103.0, 1.0, 5) == 3   # two ticks back
    assert _depth_of(e, ASK, 100.5, 1.0, 5) == 1   # inside spread -> depth 1
    assert _depth_of(e, ASK, 110.0, 1.0, 5) is None  # beyond K
    assert _depth_of(e, BID, 100.0, 1.0, 5) == 1   # at best bid
    assert _depth_of(e, BID, 98.0, 1.0, 5) == 3


def test_topk_volumes_reads_levels_from_best():
    e = _mk_book()
    assert _topk_volumes(e, ASK, 1.0, 3) == [1.0, 1.0, 1.0]
    assert _topk_volumes(e, BID, 1.0, 3) == [1.0, 1.0, 1.0]
    # a gap (no order at 104) reads as 0 volume at that depth
    e.apply(BookEvent(oid=99, side=ASK, px=105.0, kind=NEW, sz=2.0))
    assert _topk_volumes(e, ASK, 1.0, 5) == [1.0, 1.0, 1.0, 0.0, 2.0]


def test_intensities_from_divides_and_guards_zero():
    counts = np.zeros((1, 2, 2, 3))
    time_in = np.zeros((1, 2, 2))
    counts[0, 1, 1, 1] = 10.0   # 10 cancels at depth1, qbucket1, ask
    time_in[0, 1, 1] = 2.0      # over 2 seconds
    lam = intensities_from(counts, time_in)
    assert lam[0, 1, 1, 1] == 5.0            # 10/2
    assert lam[0, 0, 0, 0] == 0.0            # never-observed state -> 0, not nan/inf


def test_calibrate_counts_event_at_correct_depth():
    # pre-seed best ask at 101; then a cancel arrives at the best (depth 1)
    stream = [
        (0, BookEvent(oid=1, side=ASK, px=101.0, kind=NEW, sz=2.0)),   # empty book -> not counted
        (1_000_000_000, BookEvent(oid=2, side=ASK, px=101.0, kind=NEW, sz=2.0)),  # depth1 limit
        (2_000_000_000, BookEvent(oid=1, side=ASK, px=101.0, kind=REMOVE, sz=0.0)),  # depth1 cancel
    ]
    counts, time_in = calibrate(iter(stream), fully_filled=set(), aes=[2.0] * 3, K=3, Q=5, tick=1.0)
    assert counts[0, :, 1, 0].sum() == 1   # one limit at depth1, ask
    assert counts[0, :, 1, 1].sum() == 1   # one cancel at depth1, ask
    assert time_in.sum() > 0               # some time accumulated


def test_calibrate_counts_trade_as_market_without_touching_book():
    # a TRADE event is counted as market at the resting level, and does NOT alter the book
    stream = [
        (0, BookEvent(oid=1, side=ASK, px=101.0, kind=NEW, sz=2.0)),          # seed the book
        (1_000_000_000, BookEvent(oid=2, side=ASK, px=101.0, kind=NEW, sz=2.0)),
        (2_000_000_000, BookEvent(oid=-1, side=ASK, px=101.0, kind=TRADE, sz=0.0)),  # a trade
    ]
    counts, _ = calibrate(iter(stream), fully_filled=set(), aes=[2.0] * 3, K=3, Q=5, tick=1.0)
    from execution.qrm.calibrate_intensities import _TYPE_IDX
    assert counts[0, :, 1, _TYPE_IDX[MARKET]].sum() == 1   # trade booked as market at depth1 ask
    assert counts[:, :, :, _TYPE_IDX[MARKET]].sum() == 1   # exactly one market event
