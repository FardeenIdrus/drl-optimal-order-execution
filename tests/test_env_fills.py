"""Unit tests for the ask-ladder fill engine (Stage 2 / Phase 2 env).

Run with:  PYTHONPATH=src .venv/bin/pytest tests/test_env_fills.py
"""
import numpy as np
import pytest

from execution.env.fills import Fill, walk_ask_ladder

PX = np.array([100.0, 101.0, 102.0, 103.0])
SZ = np.array([1.0, 2.0, 3.0, 4.0])  # total depth = 10


def test_partial_first_level():
    f = walk_ask_ladder(PX, SZ, 0.5)
    assert f.filled == pytest.approx(0.5)
    assert f.cash == pytest.approx(50.0)
    assert f.levels_walked == 1
    assert not f.exhausted
    assert f.vwap == pytest.approx(100.0)


def test_exact_level_boundary_stays_on_one_level():
    f = walk_ask_ladder(PX, SZ, 1.0)
    assert f.filled == pytest.approx(1.0)
    assert f.cash == pytest.approx(100.0)
    assert f.levels_walked == 1


def test_multi_level_walk():
    f = walk_ask_ladder(PX, SZ, 2.5)  # 1@100 + 1.5@101
    assert f.filled == pytest.approx(2.5)
    assert f.cash == pytest.approx(100.0 + 1.5 * 101.0)  # 251.5
    assert f.levels_walked == 2
    assert f.vwap == pytest.approx(251.5 / 2.5)
    assert not f.exhausted


def test_zero_quantity_is_noop():
    f = walk_ask_ladder(PX, SZ, 0.0)
    assert f == Fill(0.0, 0.0, 0, False)
    assert np.isnan(f.vwap)


def test_exhausts_book_when_qty_exceeds_depth():
    f = walk_ask_ladder(PX, SZ, 100.0)
    assert f.filled == pytest.approx(10.0)  # all available depth
    assert f.cash == pytest.approx(1 * 100 + 2 * 101 + 3 * 102 + 4 * 103)  # 1020
    assert f.levels_walked == 4
    assert f.exhausted


def test_skips_nan_price_and_nonpositive_size():
    px = np.array([100.0, np.nan, 102.0])
    sz = np.array([0.0, 5.0, 1.0])  # L0 zero size, L1 nan price -> both skipped
    f = walk_ask_ladder(px, sz, 0.5)
    assert f.cash == pytest.approx(51.0)  # 0.5 @ 102
    assert f.filled == pytest.approx(0.5)
    assert f.levels_walked == 1


def test_cost_is_monotonic_and_convex_in_quantity():
    qtys = [0.5, 1.0, 2.0, 3.0, 4.0, 6.0]
    cash = [walk_ask_ladder(PX, SZ, q).cash for q in qtys]
    assert all(b >= a for a, b in zip(cash, cash[1:]))          # more qty -> more cost
    vwaps = [walk_ask_ladder(PX, SZ, q).vwap for q in qtys]
    assert all(b >= a - 1e-9 for a, b in zip(vwaps, vwaps[1:]))  # avg price worsens with size


def test_shape_mismatch_raises():
    with pytest.raises(ValueError):
        walk_ask_ladder(np.array([100.0, 101.0]), np.array([1.0]), 1.0)
