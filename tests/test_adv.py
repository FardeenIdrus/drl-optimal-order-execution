"""Unit tests for ADV order-sizing helpers (Phase 5, Setup Step 1).

Covers the pure (no-network) pieces: candle parsing/validation, ADV computation,
date->ms, and the volatile-regime depth cross-check. The networked fetch is not
exercised here (it is a one-time cached pull).

Run with:  PYTHONPATH=src .venv/bin/pytest tests/test_adv.py
"""
import numpy as np
import pandas as pd
import pytest

from execution.data.adv import (
    compute_adv,
    crosscheck_volatile_depth,
    date_to_ms,
    parse_candles,
)


def _raw(ts_vol):
    return [{"t": t, "T": t + 86_399_999, "s": "BTC", "i": "1d",
             "o": "1", "c": "1", "h": "1", "l": "1", "v": str(v), "n": 10}
            for t, v in ts_vol]


def test_date_to_ms_utc_midnight():
    assert date_to_ms("2024-01-01") == 1_704_067_200_000


def test_parse_candles_valid():
    out = parse_candles(_raw([(1_704_067_200_000, 100.5), (1_704_153_600_000, 200.0)]))
    assert [c["date"] for c in out] == ["2024-01-01", "2024-01-02"]
    assert out[0]["volume_btc"] == pytest.approx(100.5)


@pytest.mark.parametrize("bad", [{}, "nope", [], [{"t": 1}], [{"v": "1"}]])
def test_parse_candles_rejects_malformed(bad):
    with pytest.raises(ValueError):
        parse_candles(bad)


def test_compute_adv_mean_and_provenance():
    adv = compute_adv(parse_candles(_raw([
        (1_704_067_200_000, 100.0), (1_704_153_600_000, 200.0), (1_704_240_000_000, 300.0)])))
    assert adv["adv_btc"] == pytest.approx(200.0)
    assert adv["n_days"] == 3
    assert adv["median_daily_btc"] == pytest.approx(200.0)
    assert adv["date_start"] == "2024-01-01" and adv["date_end"] == "2024-01-03"


def test_compute_adv_rejects_negative_volume():
    with pytest.raises(ValueError):
        compute_adv([{"date": "2024-01-01", "volume_btc": -1.0, "ts": 0}])


def _depth_df():
    # 1 volatile episode (depth 3/step) + 1 calm episode (depth 30/step) to drop.
    rows = []
    for ep, reg, d in [(0, "volatile", 1.0), (1, "calm", 10.0)]:
        for step in range(2):
            row = {"episode_id": ep, "regime": reg, "step": step}
            row.update({f"ask_sz_{i}": d for i in range(1, 4)})  # 3 levels -> depth 3*d
            rows.append(row)
    return pd.DataFrame(rows)


def test_crosscheck_excludes_calm_and_computes_structural_proxies():
    cc = crosscheck_volatile_depth(_depth_df(), [3.0, 12.0], n_steps=2, n_levels=3, top_n=2)
    # volatile per-step depth = 3 (calm's 30 excluded); cumulative per episode = 6.
    assert cc.loc[0, "vol_depth_median_btc"] == pytest.approx(3.0)
    # size 3 over 2 steps -> slice 1.5 fits depth 3; cum 6 >= 3.
    assert cc.loc[0, "slice_btc_per_min"] == pytest.approx(1.5)
    assert cc.loc[0, "pct_steps_slice_gt_depth"] == pytest.approx(0.0)
    assert cc.loc[0, "pct_episodes_cum_lt_order"] == pytest.approx(0.0)
    # size 12 -> slice 6 > depth 3 every step; cum 6 < 12 every episode.
    assert cc.loc[1, "pct_steps_slice_gt_depth"] == pytest.approx(100.0)
    assert cc.loc[1, "pct_episodes_cum_lt_order"] == pytest.approx(100.0)


def test_crosscheck_requires_volatile_rows():
    calm_only = _depth_df()
    calm_only = calm_only[calm_only["regime"] == "calm"]
    with pytest.raises(ValueError):
        crosscheck_volatile_depth(calm_only, [1.0], n_steps=2, n_levels=3)
