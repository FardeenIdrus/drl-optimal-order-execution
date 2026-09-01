"""Unit tests for the train-fit feature normalizer.

Run with:  PYTHONPATH=src .venv/bin/pytest tests/test_normalize.py
"""
import numpy as np
import pytest

from execution.env.normalize import FeatureNormalizer

NAMES = ("spread_bps", "imbalance", "recent_return", "rolling_vol", "ask_depth")


def test_standardize_only_features_are_zero_mean_unit_std():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(1000, 5))
    norm = FeatureNormalizer.fit(X, NAMES, log_features=())  # no log
    Z = norm.transform_many(X)
    assert np.allclose(Z.mean(axis=0), 0.0, atol=1e-9)
    assert np.allclose(Z.std(axis=0), 1.0, atol=1e-9)


def test_log_applied_only_to_log_features():
    norm = FeatureNormalizer.fit(np.ones((10, 5)) * 2.0, NAMES)
    assert norm.log_mask == (True, False, False, True, True)  # spread, vol, depth


def test_log_transform_compresses_heavy_tail():
    # ask_depth-like column with a huge outlier; others benign
    X = np.ones((100, 5))
    X[:, 4] = np.r_[np.full(99, 5.0), 1600.0]   # ask_depth: median ~5, one outlier 1600
    norm = FeatureNormalizer.fit(X, NAMES)
    z = norm.transform_many(X)[:, 4]
    # in log space the outlier is a few std out, not hundreds (as raw z-score would give)
    assert abs(z[-1]) < 12.0
    raw_z = (1600.0 - 5.0) / np.std(np.r_[np.full(99, 5.0), 1600.0])
    assert raw_z > 9.0  # raw would be ~10 std; log keeps it tame


def test_transform_matches_transform_many():
    rng = np.random.default_rng(1)
    X = np.abs(rng.normal(size=(50, 5))) + 0.1
    norm = FeatureNormalizer.fit(X, NAMES)
    one = norm.transform(X[7])
    many = norm.transform_many(X)[7]
    assert np.allclose(one, many)


def test_fit_is_train_only_test_not_recentered():
    # fit on train; a shifted "test" set must NOT come out zero-mean (no test-fit)
    train = np.zeros((100, 5)) + 1.0
    train += np.random.default_rng(2).normal(scale=0.1, size=(100, 5))
    norm = FeatureNormalizer.fit(train, NAMES, log_features=())
    test = np.zeros((100, 5)) + 5.0   # very different location
    zt = norm.transform_many(test)
    assert np.abs(zt.mean()) > 1.0     # large offset preserved -> train-fit, not test-fit


def test_json_round_trip(tmp_path):
    rng = np.random.default_rng(3)
    X = np.abs(rng.normal(size=(80, 5))) + 0.05
    norm = FeatureNormalizer.fit(X, NAMES)
    p = tmp_path / "norm.json"
    norm.to_json(p)
    back = FeatureNormalizer.from_json(p)
    assert back.feature_names == norm.feature_names
    assert back.log_mask == norm.log_mask
    assert np.allclose(norm.transform(X[0]), back.transform(X[0]))


def test_fit_rejects_non_finite():
    X = np.ones((10, 5))
    X[3, 2] = np.nan
    with pytest.raises(ValueError, match="non-finite"):
        FeatureNormalizer.fit(X, NAMES)
