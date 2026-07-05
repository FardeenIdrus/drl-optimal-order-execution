"""Unit tests for Stage 5: episodes, realised vol, split, and train-only threshold.

Run with:  PYTHONPATH=src pytest tests/test_regimes.py
"""
import math

import numpy as np
import pandas as pd
import pytest

from execution.data import regimes as R

EP_MS = R.EPISODE_MS
BASE = (1_700_000_000_000 // EP_MS) * EP_MS  # align to a 30-min boundary


def _build(ep_rv, ep_valid=None):
    """One row per minute for len(ep_rv) clock-aligned 30-min episodes."""
    rows = []
    for e, rv in enumerate(ep_rv):
        valid = True if ep_valid is None else ep_valid[e]
        for i in range(R.EPISODE_MINUTES):
            rows.append((BASE + e * EP_MS + i * R.MS_PER_MINUTE, valid, rv))
    df = pd.DataFrame(rows, columns=["ts", "feature_valid", "realized_variance"])
    return df[["ts", "feature_valid"]], df[["ts", "realized_variance"]]


def test_build_episodes_count_and_realized_vol():
    feats, minute = _build([1e-6, 1e-6, 1e-6, 1e-6])
    eps = R.build_episodes(feats, minute)
    assert len(eps) == 4
    assert math.isclose(eps.iloc[0]["realized_vol"], math.sqrt(30 * 1e-6), rel_tol=1e-9)
    assert (eps["n_minutes"] == 30).all()


def test_invalid_episode_dropped():
    feats, minute = _build([1e-6, 1e-6, 1e-6], ep_valid=[True, False, True])
    eps = R.build_episodes(feats, minute)
    assert len(eps) == 2  # the all-invalid episode is dropped


def test_quiet_minute_rv_treated_as_zero():
    feats, minute = _build([1e-6])
    minute.loc[5, "realized_variance"] = np.nan  # one single-snapshot minute
    eps = R.build_episodes(feats, minute)
    assert math.isclose(eps.iloc[0]["realized_vol"], math.sqrt(29 * 1e-6), rel_tol=1e-9)


def test_split_is_chronological():
    feats, minute = _build([1e-6] * 10)
    eps = R.assign_split(R.build_episodes(feats, minute), test_frac=0.2, buffer_episodes=0)
    assert (eps["split"] == "train").sum() == 8 and (eps["split"] == "test").sum() == 2
    assert eps[eps.split == "train"]["start_ts"].max() < eps[eps.split == "test"]["start_ts"].min()


def test_threshold_uses_train_only():
    train_rv = [1e-6, 2e-6, 3e-6, 4e-6, 5e-6, 6e-6, 7e-6, 8e-6]
    eps_a = R.assign_split(R.build_episodes(*_build(train_rv + [100e-6, 200e-6])),
                           test_frac=0.2, buffer_episodes=0)
    eps_b = R.assign_split(R.build_episodes(*_build(train_rv + [1e-6, 1e-6])),
                           test_frac=0.2, buffer_episodes=0)
    _, thr_a = R.assign_regime(eps_a, n_regimes=2)
    _, thr_b = R.assign_regime(eps_b, n_regimes=2)
    # test episodes differ wildly, but the threshold (median of TRAIN vols) is identical
    assert math.isclose(thr_a["median"], thr_b["median"], rel_tol=1e-12)


def test_regime_labels_by_threshold():
    eps = R.assign_split(R.build_episodes(*_build([1e-6, 2e-6, 3e-6, 4e-6,
                                                   5e-6, 6e-6, 7e-6, 8e-6, 9e-6, 10e-6])),
                         test_frac=0.2, buffer_episodes=0)
    eps, thr = R.assign_regime(eps, n_regimes=2)
    assert ((eps["regime"] == "volatile") == (eps["realized_vol"] > thr["median"])).all()


# --- finer-resolution (180 bars / 30-min episode at 10s) ---------------------------

def _build_10s(ep_rv, bars_per_ep=180):
    """One row per 10s bar for len(ep_rv) clock-aligned 30-min episodes (180 bars each)."""
    rows = []
    for e, rv in enumerate(ep_rv):
        for i in range(bars_per_ep):
            rows.append((BASE + e * EP_MS + i * 10_000, True, rv))
    df = pd.DataFrame(rows, columns=["ts", "feature_valid", "realized_variance"])
    return df[["ts", "feature_valid"]], df[["ts", "realized_variance"]]


def test_build_episodes_at_10s_resolution():
    feats, minute = _build_10s([1e-6, 1e-6])
    eps = R.build_episodes(feats, minute, bar_seconds=10)
    assert len(eps) == 2
    assert (eps["n_bars"] == 180).all() and (eps["n_minutes"] == 30).all()
    # realised vol sums all 180 bars' RV over the same 30-min window
    assert math.isclose(eps.iloc[0]["realized_vol"], math.sqrt(180 * 1e-6), rel_tol=1e-9)
    # the 60s default expects 30 rows/episode -> the 10s data (180 rows/bucket) is rejected
    assert len(R.build_episodes(feats, minute, bar_seconds=60)) == 0


def test_build_episodes_rejects_bad_bar_seconds():
    feats, minute = _build([1e-6])
    for bad in (7, 0, 11):                                  # must divide 60
        with pytest.raises(ValueError):
            R.build_episodes(feats, minute, bar_seconds=bad)


def test_build_episodes_short_horizon_10min_at_10s():
    # Option-2 lever: a 10-minute horizon at 10s = 60 bars/episode (vs 180 at 30 min).
    EP10 = 10 * R.MS_PER_MINUTE                              # 10-min window
    base = (1_700_000_000_000 // EP10) * EP10
    rows = [(base + e * EP10 + i * 10_000, True, 1e-6) for e in range(2) for i in range(60)]
    df = pd.DataFrame(rows, columns=["ts", "feature_valid", "realized_variance"])
    eps = R.build_episodes(df[["ts", "feature_valid"]], df[["ts", "realized_variance"]],
                           episode_minutes=10, bar_seconds=10)
    assert len(eps) == 2
    assert (eps["n_bars"] == 60).all() and (eps["n_minutes"] == 10).all()
    # the default 30-min horizon would expect 180 bars -> the 60-bar windows are rejected
    assert len(R.build_episodes(df[["ts", "feature_valid"]], df[["ts", "realized_variance"]],
                                episode_minutes=30, bar_seconds=10)) == 0
