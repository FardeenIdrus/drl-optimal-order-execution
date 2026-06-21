"""Unit tests for Stage 5: episodes, realised vol, split, and train-only threshold.

Run with:  PYTHONPATH=src pytest tests/test_regimes.py
"""
import math

import numpy as np
import pandas as pd

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
