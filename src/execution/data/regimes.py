"""Stage 5 - episode definition and volatility-regime tagging.

Defines execution episodes and labels each calm/volatile. Design choices are made for
defensibility, since the regime definition underpins the dissertation's contribution
(regime-dependent attribution):

  - Episodes are clock-aligned, NON-OVERLAPPING 30-minute windows (the execution
    horizon). Non-overlapping avoids inflating statistical significance in the
    regime comparison. Only fully-valid episodes are kept (all 30 minutes present and
    feature-valid) - never execute over fabricated/partial data.

  - The episode's volatility is realised volatility OVER the window,
    ``sqrt(sum of the 30 minutes' realised variance)`` (high-frequency,
    Andersen-Bollerslev style). It is ex-post and used ONLY as a label - the agent
    never sees it - and is deliberately distinct from the agent's trailing
    ``rolling_vol`` feature, which removes circularity.

  - Train/test is a chronological 80/20 split (no shuffle) with a one-episode buffer at
    the boundary. The calm/volatile threshold is the median realised vol of TRAIN
    episodes only, applied unchanged to test - so test data never informs the regime
    boundary (no leakage).

  - The early coverage gate reports regime x split counts: the contribution needs both
    regimes well represented among TEST episodes. A fixed train threshold assumes the
    vol distribution is roughly stationary; if the test period differs, test skews to
    one regime - the gate makes that explicit before any training.
"""
from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Dict, Tuple, Union

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

MS_PER_MINUTE = 60_000
EPISODE_MINUTES = 30
EPISODE_MS = EPISODE_MINUTES * MS_PER_MINUTE


def build_episodes(
    features_df: pd.DataFrame, minute_df: pd.DataFrame, episode_minutes: int = EPISODE_MINUTES,
    bar_seconds: int = 60,
) -> pd.DataFrame:
    """Build fully-valid, clock-aligned, non-overlapping episodes with realised vol.

    An episode is a clock-aligned ``episode_minutes``-minute window, which at
    ``bar_seconds`` resolution holds ``bars_per_episode = episode_minutes*60/bar_seconds``
    bars (30 at 60s, 180 at 10s). It is valid iff ALL of those bars are present and
    ``feature_valid``. Realised vol = sqrt(sum of the window's per-bar realised
    variance); single-snapshot bars (undefined variance) contribute 0. The window is
    the same wall-clock 30 minutes at any resolution; finer bars just sample its
    realised variance more granularly (the regime threshold is re-fit on train, so the
    calm/volatile split stays internally consistent per resolution).
    """
    if bar_seconds <= 0 or 60 % bar_seconds != 0:
        raise ValueError(f"bar_seconds must be a positive divisor of 60, got {bar_seconds}")
    bar_ms = bar_seconds * 1000
    bars_per_episode = episode_minutes * 60 // bar_seconds
    ep_ms = episode_minutes * MS_PER_MINUTE
    j = features_df[["ts", "feature_valid"]].merge(
        minute_df[["ts", "realized_variance"]], on="ts", how="left"
    )
    j["bucket"] = (j["ts"] // ep_ms) * ep_ms
    j["rv0"] = j["realized_variance"].fillna(0.0)

    agg = j.groupby("bucket").agg(
        n_rows=("ts", "size"),
        n_valid=("feature_valid", "sum"),
        rv_sum=("rv0", "sum"),
    )
    full_and_valid = (agg["n_rows"] == bars_per_episode) & (agg["n_valid"] == bars_per_episode)
    agg = agg[full_and_valid]

    eps = pd.DataFrame({
        "start_ts": agg.index.to_numpy(dtype="int64"),
        "end_ts": agg.index.to_numpy(dtype="int64") + ep_ms,  # exclusive
        "n_minutes": episode_minutes,    # wall-clock minutes (width-agnostic)
        "n_bars": bars_per_episode,      # rows/steps per episode at this resolution
        "realized_vol": np.sqrt(agg["rv_sum"].to_numpy()),
    }).sort_values("start_ts").reset_index(drop=True)
    return eps


def assign_split(eps: pd.DataFrame, test_frac: float = 0.2, buffer_episodes: int = 1) -> pd.DataFrame:
    """Chronological split (no shuffle): first 1-test_frac train, last test_frac test.

    Episodes are non-overlapping and time-ordered, so all train minutes precede all
    test minutes. A buffer of ``buffer_episodes`` episodes at the boundary is dropped
    for extra margin against any cross-boundary feature overlap.
    """
    eps = eps.sort_values("start_ts").reset_index(drop=True)
    n = len(eps)
    n_train = int(round((1 - test_frac) * n))
    split = np.array(["train"] * n, dtype=object)
    split[n_train:] = "test"
    eps = eps.assign(split=split)
    if buffer_episodes > 0:
        boundary = eps.index[eps["split"] == "test"][:buffer_episodes]
        eps = eps.drop(index=boundary).reset_index(drop=True)
    return eps


def assign_regime(eps: pd.DataFrame, n_regimes: int = 2) -> Tuple[pd.DataFrame, Dict[str, float]]:
    """Label each episode by regime using a threshold fixed on TRAIN episodes only."""
    train_vol = eps.loc[eps["split"] == "train", "realized_vol"]
    if n_regimes == 2:
        thr = float(train_vol.median())
        eps = eps.assign(regime=np.where(eps["realized_vol"] > thr, "volatile", "calm"))
        thresholds = {"median": thr}
    elif n_regimes == 3:
        q1, q2 = (float(x) for x in train_vol.quantile([1 / 3, 2 / 3]))
        eps = eps.assign(regime=pd.cut(eps["realized_vol"], [-np.inf, q1, q2, np.inf],
                                       labels=["calm", "normal", "volatile"]).astype(object))
        thresholds = {"q33": q1, "q67": q2}
    else:
        raise ValueError("n_regimes must be 2 or 3")
    return eps, thresholds


def regime_qa(eps: pd.DataFrame, thresholds: Dict[str, float]) -> Dict[str, object]:
    """Coverage gate + summary: regime x split counts and per-regime vol stats."""
    counts = (eps.groupby(["split", "regime"], observed=True).size()
              .unstack(fill_value=0).to_dict(orient="index"))
    vol_by_regime = {r: {"mean": round(float(g["realized_vol"].mean()), 6),
                         "median": round(float(g["realized_vol"].median()), 6),
                         "n": int(len(g))}
                     for r, g in eps.groupby("regime", observed=True)}
    test = eps[eps["split"] == "test"]
    test_counts = test.groupby("regime", observed=True).size().to_dict()
    min_test = min(test_counts.values()) if test_counts else 0
    return {
        "n_episodes": int(len(eps)),
        "thresholds": thresholds,
        "boundary_start_ts": int(eps[eps["split"] == "test"]["start_ts"].min()) if (eps["split"] == "test").any() else None,
        "counts_by_split_regime": counts,
        "vol_by_regime": vol_by_regime,
        "test_regime_counts": {str(k): int(v) for k, v in test_counts.items()},
        "min_test_regime_episodes": int(min_test),
    }


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    p = argparse.ArgumentParser(description="Stage 5 - episode definition + regime tagging")
    p.add_argument("--features-parquet", required=True)
    p.add_argument("--minute-parquet", required=True)
    p.add_argument("--out", required=True, help="output episodes parquet (scratch)")
    p.add_argument("--test-frac", type=float, default=0.2)
    p.add_argument("--buffer-episodes", type=int, default=1)
    p.add_argument("--n-regimes", type=int, default=2)
    p.add_argument("--bar-seconds", type=int, default=60,
                   help="bar width in seconds (default 60; finer e.g. 10). Sets bars/episode.")
    p.add_argument("--episode-minutes", type=int, default=30,
                   help="execution horizon in minutes (default 30; shorter e.g. 10). Sets the "
                        "episode window and bars/episode = episode_minutes*60/bar_seconds.")
    args = p.parse_args()

    features_df = pd.read_parquet(args.features_parquet, columns=["ts", "feature_valid"])
    minute_df = pd.read_parquet(args.minute_parquet, columns=["ts", "realized_variance"])

    eps = build_episodes(features_df, minute_df, episode_minutes=args.episode_minutes,
                         bar_seconds=args.bar_seconds)
    eps = assign_split(eps, test_frac=args.test_frac, buffer_episodes=args.buffer_episodes)
    eps, thresholds = assign_regime(eps, n_regimes=args.n_regimes)
    eps["episode_id"] = np.arange(len(eps))

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    eps.to_parquet(out, index=False)
    report = regime_qa(eps, thresholds)
    out.with_suffix(".qa.json").write_text(json.dumps(report, indent=2, default=str))

    logger.info("wrote %s (%d episodes)", out, len(eps))
    logger.info("thresholds: %s", thresholds)
    logger.info("regime x split counts: %s", report["counts_by_split_regime"])
    logger.info("TEST regime counts (coverage gate): %s | min=%d",
                report["test_regime_counts"], report["min_test_regime_episodes"])


if __name__ == "__main__":
    main()
