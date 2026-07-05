"""Unit tests for Stage 3a: QRM event-type classification.

Run with:  PYTHONPATH=src pytest tests/test_qrm_event_labeler.py
"""
from execution.data.l4.book_diffs_reader import ASK, BID, NEW, REMOVE, UPDATE, BookEvent
from execution.qrm.event_labeler import (
    CANCEL,
    LIMIT,
    MARKET,
    classify,
    classify_for_calibration,
)


def _ev(kind, oid=1):
    return BookEvent(oid=oid, side=BID, px=100.0, kind=kind, sz=1.0)


def test_new_is_limit():
    assert classify(_ev(NEW), set()) == LIMIT


def test_update_is_market():
    assert classify(_ev(UPDATE), set()) == MARKET  # partial fill


def test_remove_is_market_if_fully_filled():
    assert classify(_ev(REMOVE, oid=7), {7}) == MARKET  # a trade cleared it


def test_remove_is_cancel_otherwise():
    assert classify(_ev(REMOVE, oid=7), set()) == CANCEL
    assert classify(_ev(REMOVE, oid=7), {99}) == CANCEL  # different oid filled, not this one


def test_classify_for_calibration_skips_fills():
    # calibration classifier: limit/cancel kept; fills -> None (counted from trades file)
    assert classify_for_calibration(_ev(NEW), set()) == LIMIT
    assert classify_for_calibration(_ev(UPDATE), set()) is None          # partial fill
    assert classify_for_calibration(_ev(REMOVE, oid=7), {7}) is None     # full fill
    assert classify_for_calibration(_ev(REMOVE, oid=7), set()) == CANCEL  # a cancel
