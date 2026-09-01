"""Unit tests for Stage 4: feature definitions, gap handling, and no-look-ahead.

Run with:  PYTHONPATH=src pytest tests/test_features.py
"""
import math

import pandas as pd
import pytest

from execution.data import features as F

BASE = 1_700_000_000_000  # arbitrary ms epoch


def _minute_df(mid, spread, bid5, ask5, rv, ts=None):
    n = len(mid)
    if ts is None:
        ts = [BASE + i * F.MS_PER_MINUTE for i in range(n)]
    return pd.DataFrame({
        "ts": ts, "mid": mid, "mean_spread": spread,
        "mean_bid_sz_5": bid5, "mean_ask_sz_5": ask5, "realized_variance": rv,
    })


def test_feature_definitions_and_shift():
    mid = [100.0, 101.0, 102.0, 103.0, 104.0]
    spread = [1.0, 1.0, 1.0, 1.0, 1.0]
    bid5 = [3.0, 3.0, 3.0, 3.0, 3.0]
    ask5 = [1.0, 1.0, 1.0, 1.0, 1.0]
    rv = [1e-6, 1e-6, 1e-6, 1e-6, 1e-6]
    out = F.compute_features(_minute_df(mid, spread, bid5, ask5, rv),
                             imbalance_depth=5, return_lookback=2, vol_window=2)

    # Row 0 is all-NaN from the decision-time shift.
    assert not out.iloc[0]["feature_valid"]
    # Row t uses bar t-1: spread_bps[t] = spread[t-1]/mid[t-1]*1e4
    assert math.isclose(out.iloc[3]["spread_bps"], 1.0 / mid[2] * 1e4, rel_tol=1e-9)
    # imbalance = (3-1)/(3+1) = 0.5 ; ask_depth = 1.0
    assert math.isclose(out.iloc[3]["imbalance"], 0.5, rel_tol=1e-9)
    assert math.isclose(out.iloc[3]["ask_depth"], 1.0, rel_tol=1e-9)
    # recent_return[t] = log(mid[t-1]/mid[t-1-L]) ; L=2 -> log(mid[2]/mid[0])
    assert math.isclose(out.iloc[3]["recent_return"], math.log(mid[2] / mid[0]), rel_tol=1e-9)


def test_rolling_vol_sum_and_quiet_minute():
    # vol_window=2 ; rolling_vol at bar b = sqrt(rv[b-1]+rv[b]); shifted by 1.
    rv = [1e-6, 3e-6, 5e-6, 7e-6]
    base = _minute_df([100.0] * 4, [1.0] * 4, [2.0] * 4, [2.0] * 4, rv)
    out = F.compute_features(base, vol_window=2, return_lookback=1)
    # out row 2 = bar 1 = sqrt(rv[0]+rv[1]) = sqrt(4e-6) = 2e-3
    assert math.isclose(out.iloc[2]["rolling_vol"], math.sqrt(4e-6), rel_tol=1e-9)

    # A present minute with undefined RV (single snapshot) contributes 0, window stays valid.
    rv_quiet = [1e-6, float("nan"), 5e-6, 7e-6]
    out_q = F.compute_features(_minute_df([100.0]*4, [1.0]*4, [2.0]*4, [2.0]*4, rv_quiet),
                               vol_window=2, return_lookback=1)
    # out row 2 = bar 1 = sqrt(rv[0] + 0) = sqrt(1e-6)
    assert math.isclose(out_q.iloc[2]["rolling_vol"], math.sqrt(1e-6), rel_tol=1e-9)
    assert bool(out_q.iloc[2]["feature_valid"])


def test_gap_invalidates_windows():
    # Drop the minute at index 2 (timestamp gap) -> reindex creates a NaN row.
    ts = [BASE + i * F.MS_PER_MINUTE for i in (0, 1, 3, 4, 5)]
    df = _minute_df([100.0]*5, [1.0]*5, [2.0]*5, [2.0]*5, [1e-6]*5, ts=ts)
    out = F.compute_features(df, return_lookback=1, vol_window=2)
    # The grid is contiguous (6 minutes); the missing minute must be present=NaN/invalid.
    assert len(out) == 6
    assert not out["feature_valid"].all()  # some rows invalid around the gap
    # recent_return across the gap (uses the missing minute) must be NaN
    assert out["recent_return"].isna().any()


def test_no_look_ahead():
    mid = [100.0, 101.0, 102.0, 103.0, 104.0, 105.0]
    df = _minute_df(mid, [1.0]*6, [2.0]*6, [2.0]*6, [1e-6]*6)
    out1 = F.compute_features(df, return_lookback=1, vol_window=2)

    df2 = df.copy()
    df2.loc[3, "mean_spread"] = 99.0  # perturb bar at index 3
    out2 = F.compute_features(df2, return_lookback=1, vol_window=2)

    # Decision row 3 uses bar 2 -> unaffected by changing bar 3.
    assert out1.iloc[3]["spread_bps"] == out2.iloc[3]["spread_bps"]
    # Decision row 4 uses bar 3 -> must change.
    assert out1.iloc[4]["spread_bps"] != out2.iloc[4]["spread_bps"]


# --- finer-resolution (minute windows -> bars via bar_seconds) ---------------------

def test_minute_windows_convert_to_bars_at_10s():
    # At bar_seconds=10 there are 6 bars/min, so a 1-MINUTE window is a 6-BAR window.
    # Build a 10s-spaced table (14 bars) and check the return lookback spans 6 bars and
    # the vol window sums 6 bars -- i.e. the wall-clock window is held fixed, not the
    # bar count (which would be 1 bar at 60s).
    n = 14
    ts = [BASE + i * 10_000 for i in range(n)]                 # 10s spacing
    mid = [100.0 + i for i in range(n)]
    df = _minute_df(mid, [1.0]*n, [3.0]*n, [1.0]*n, [1e-6]*n, ts=ts)
    out = F.compute_features(df, return_lookback=1, vol_window=1, bar_seconds=10)

    assert len(out) == n                                        # grid spacing is 10s, not 60s
    # recent_return[t] = log(mid[t-1] / mid[t-1-6]); row 8 -> log(mid[7]/mid[1])
    assert math.isclose(out.iloc[8]["recent_return"], math.log(mid[7] / mid[1]), rel_tol=1e-9)
    # rolling_vol[t] = sqrt(sum of 6 bars rv[t-6..t-1]) = sqrt(6e-6)
    assert math.isclose(out.iloc[8]["rolling_vol"], math.sqrt(6e-6), rel_tol=1e-9)
    # a window needing 6 prior bars cannot be valid before bar 6 (+1 shift) -> row < 7 invalid
    assert not out.iloc[6]["feature_valid"] and bool(out.iloc[8]["feature_valid"])


def test_60s_path_unchanged_under_bar_seconds_default():
    # bar_seconds defaults to 60, so minute windows == bar counts exactly (no regression).
    mid = [100.0, 101.0, 102.0, 103.0, 104.0]
    df = _minute_df(mid, [1.0]*5, [3.0]*5, [1.0]*5, [1e-6]*5)
    a = F.compute_features(df, return_lookback=2, vol_window=2)
    b = F.compute_features(df, return_lookback=2, vol_window=2, bar_seconds=60)
    pd.testing.assert_frame_equal(a, b)


def test_features_reject_bad_bar_seconds():
    df = _minute_df([100.0]*3, [1.0]*3, [2.0]*3, [2.0]*3, [1e-6]*3)
    for bad in (7, 0, -10, 11):                                 # must divide 60
        with pytest.raises(ValueError):
            F.compute_features(df, bar_seconds=bad)
