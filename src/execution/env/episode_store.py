"""Load frozen Stage-6 execution datasets into fast per-episode numpy arrays.

The environment touches the order book on every step of every episode, so the
parquet is reshaped once at construction into contiguous ``(episode, step, level)``
arrays and served from memory (no per-step disk reads or pandas lookups, which
would dominate training wall-clock).

This relies on the Stage-6 guarantee that rows are exactly ``n_steps`` per
episode, contiguous and step-ordered. That contract is *re-validated* here rather
than assumed: a violation raises instead of silently mis-shaping the data.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from execution.data.features import FEATURE_COLUMNS

N_LEVELS: int = 20
N_STEPS: int = 30


@dataclass(frozen=True)
class EpisodeStore:
    """In-memory, reshaped view of a train/test execution dataset.

    Array axes are ``E`` = episodes, ``T`` = steps/episode, ``L`` = book levels,
    ``F`` = features.
    """

    episode_ids: np.ndarray   # (E,)    int64
    regime: np.ndarray        # (E,)    str (object)
    mid: np.ndarray           # (E, T)  float64  -- arrival/decision mid per step
    ask_px: np.ndarray        # (E, T, L) float64 -- ascending ask prices
    ask_sz: np.ndarray        # (E, T, L) float64 -- ask sizes (BTC)
    features: np.ndarray      # (E, T, F) float64
    feature_names: tuple[str, ...]
    n_steps: int

    @property
    def n_episodes(self) -> int:
        return int(self.episode_ids.shape[0])

    @property
    def n_levels(self) -> int:
        return int(self.ask_px.shape[2])

    def chrono_split(self, val_frac: float) -> tuple["EpisodeStore", "EpisodeStore"]:
        """Carve a VALIDATION store chronologically from the end of this store.

        Episodes are held in chronological order (Stage-6: no shuffle, sorted by
        the chronological ``episode_id``), so the last ``val_frac`` fraction are
        the most recent -- adjacent to the test boundary -- and become validation;
        the earlier episodes are the training sub-split. The TEST set is a separate
        file and is untouched. Used by the generalization gate: train and
        gate on ``(train_sub, val)`` while TEST stays sealed.

        Returns ``(train_sub, val)``; each is non-empty for ``val_frac`` in (0, 1).
        """
        if not 0.0 < val_frac < 1.0:
            raise ValueError("val_frac must be in (0, 1)")
        n_val = max(1, min(self.n_episodes - 1, int(round(self.n_episodes * val_frac))))
        cut = self.n_episodes - n_val
        return self._episode_slice(slice(0, cut)), self._episode_slice(slice(cut, None))

    def _episode_slice(self, sl: slice) -> "EpisodeStore":
        """A new store over an episode-axis slice (read-only view of the arrays)."""
        return EpisodeStore(
            episode_ids=self.episode_ids[sl],
            regime=self.regime[sl],
            mid=self.mid[sl],
            ask_px=self.ask_px[sl],
            ask_sz=self.ask_sz[sl],
            features=self.features[sl],
            feature_names=self.feature_names,
            n_steps=self.n_steps,
        )

    @classmethod
    def from_parquet(
        cls,
        path: str | Path,
        *,
        n_levels: int = N_LEVELS,
        n_steps: int = N_STEPS,
        feature_names: tuple[str, ...] = tuple(FEATURE_COLUMNS),
    ) -> "EpisodeStore":
        """Build a store from a Stage-6 ``train.parquet`` / ``test.parquet``."""
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"dataset not found: {path}")

        ask_px_cols = [f"ask_px_{i}" for i in range(1, n_levels + 1)]
        ask_sz_cols = [f"ask_sz_{i}" for i in range(1, n_levels + 1)]
        needed = ["episode_id", "step", "regime", "mid", *ask_px_cols, *ask_sz_cols,
                  *feature_names]

        df = pd.read_parquet(path, columns=needed)
        df = df.sort_values(["episode_id", "step"], kind="stable").reset_index(drop=True)

        n = len(df)
        if n == 0 or n % n_steps != 0:
            raise ValueError(f"{path}: {n} rows is not a positive multiple of n_steps={n_steps}")
        n_ep = n // n_steps

        steps = df["step"].to_numpy().reshape(n_ep, n_steps)
        if not np.array_equal(steps, np.tile(np.arange(n_steps), (n_ep, 1))):
            raise ValueError(f"{path}: rows are not contiguous 0..{n_steps - 1} step blocks")
        eids = df["episode_id"].to_numpy().reshape(n_ep, n_steps)
        if not (eids == eids[:, :1]).all():
            raise ValueError(f"{path}: episode_id is not constant within a {n_steps}-step block")

        mid = df["mid"].to_numpy(np.float64).reshape(n_ep, n_steps)
        ask_px = df[ask_px_cols].to_numpy(np.float64).reshape(n_ep, n_steps, n_levels)
        ask_sz = df[ask_sz_cols].to_numpy(np.float64).reshape(n_ep, n_steps, n_levels)
        if feature_names:
            features = df[list(feature_names)].to_numpy(np.float64).reshape(
                n_ep, n_steps, len(feature_names))
        else:
            features = np.empty((n_ep, n_steps, 0), np.float64)
        regime = df["regime"].to_numpy().reshape(n_ep, n_steps)[:, 0]

        return cls(
            episode_ids=eids[:, 0].astype(np.int64),
            regime=regime,
            mid=mid,
            ask_px=ask_px,
            ask_sz=ask_sz,
            features=features,
            feature_names=tuple(feature_names),
            n_steps=n_steps,
        )
