"""Train-fit feature normalization for the RL observation (leakage-safe).

Normalization statistics are fit on the TRAIN split only and frozen, then applied
unchanged to train and test. This mirrors the Stage-5 train-median regime
threshold: no test-set information ever enters the transform, so the agent is
never implicitly told anything about the held-out period.

Per-feature scheme (chosen from the train distributions; skew in parentheses):
  * ``spread_bps`` (12), ``rolling_vol`` (5), ``ask_depth`` (33): ``log1p`` then
    standardize -- compresses the heavy right tail of these strictly-positive
    quantities so a handful of outliers (e.g. ask_depth up to ~1600 BTC vs a
    median ~5) cannot dominate the network input.
  * ``imbalance`` (~0), ``recent_return`` (~0): standardize only (signed,
    approximately symmetric, already well-scaled).
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

# Strictly-positive, heavy-tailed features that get a log1p before standardizing.
LOG_FEATURES: tuple[str, ...] = ("spread_bps", "rolling_vol", "ask_depth")
_ZERO_SCALE = 1e-12


@dataclass(frozen=True)
class FeatureNormalizer:
    """Frozen, train-fit per-feature transform: optional log1p, then z-score."""

    feature_names: tuple[str, ...]
    log_mask: tuple[bool, ...]
    center: np.ndarray   # (F,) post-log mean on TRAIN
    scale: np.ndarray    # (F,) post-log std on TRAIN (guarded against zero)

    @classmethod
    def fit(cls, features, feature_names, log_features: tuple[str, ...] = LOG_FEATURES
            ) -> "FeatureNormalizer":
        """Fit on a train feature matrix (any shape ending in F, e.g. (E,T,F))."""
        names = tuple(feature_names)
        mask = np.array([n in set(log_features) for n in names])
        X = np.asarray(features, dtype=np.float64).reshape(-1, len(names)).copy()
        if X.size == 0:
            raise ValueError("cannot fit FeatureNormalizer on empty features")
        if not np.isfinite(X).all():
            raise ValueError("train features contain non-finite values; refusing to fit")
        if mask.any():
            X[:, mask] = np.log1p(np.maximum(X[:, mask], 0.0))  # log-features are non-negative
        center = X.mean(axis=0)
        scale = X.std(axis=0)
        scale = np.where(scale < _ZERO_SCALE, 1.0, scale)  # guard constant features
        return cls(names, tuple(bool(b) for b in mask), center, scale)

    def transform(self, row) -> np.ndarray:
        """Transform one raw feature row (F,) -> normalized (F,)."""
        x = np.asarray(row, dtype=np.float64).copy()
        mask = np.array(self.log_mask)
        if mask.any():
            x[mask] = np.log1p(np.maximum(x[mask], 0.0))
        return (x - self.center) / self.scale

    def transform_many(self, rows) -> np.ndarray:
        """Transform a (N, F) matrix of raw features -> (N, F) normalized."""
        X = np.asarray(rows, dtype=np.float64).copy()
        mask = np.array(self.log_mask)
        if mask.any():
            X[:, mask] = np.log1p(np.maximum(X[:, mask], 0.0))
        return (X - self.center) / self.scale

    def to_json(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps({
            "feature_names": list(self.feature_names),
            "log_mask": list(self.log_mask),
            "center": self.center.tolist(),
            "scale": self.scale.tolist(),
        }, indent=2))

    @classmethod
    def from_json(cls, path: str | Path) -> "FeatureNormalizer":
        d = json.loads(Path(path).read_text())
        return cls(tuple(d["feature_names"]), tuple(d["log_mask"]),
                   np.asarray(d["center"], np.float64), np.asarray(d["scale"], np.float64))
