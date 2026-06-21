"""Stage 6 - materialise the train/test execution datasets.

Produces two self-contained, decision-time-aligned tables (train, test) that the RL
environment consumes directly: one row per (episode, step), carrying the agent's
observation features and the order book it trades against. Execution state (inventory,
time-remaining) is NOT here - the env adds it at runtime.

Alignment (no look-ahead), a uniform one-bar lag applied to BOTH sides:
  - observation features at decision minute t = features[t] (Stage 4 already aligned so
    these use only bars <= t-1);
  - fill book at decision minute t = the book as-of t = bar (t-1)'s end-of-minute book
    (minute[t-1]); a market order at the start of minute t executes against the book
    observable then, never minute t's own (future) closing book.

Train episodes strictly precede test episodes in time (Stage 5 chronological split +
buffer); this module asserts that boundary holds.
"""
from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Union

import pandas as pd

from execution.data import schema
from execution.data.regimes import EPISODE_MS, MS_PER_MINUTE

logger = logging.getLogger(__name__)

FEATURES: List[str] = ["spread_bps", "imbalance", "recent_return", "rolling_vol", "ask_depth"]
ASK_PX = schema.level_columns("ask", "px")
ASK_SZ = schema.level_columns("ask", "sz")
_FILL_BOOK = ASK_PX + ASK_SZ + ["bid_px_1", "bid_sz_1", "mid"]
_KEYS = ["episode_id", "step", "ts", "regime", "split"]


def materialise(
    features_df: pd.DataFrame, minute_df: pd.DataFrame, episodes_df: pd.DataFrame
) -> pd.DataFrame:
    """Assemble the decision-step table: features[t] + book as-of t (= minute[t-1])."""
    feats = features_df[["ts"] + FEATURES]

    # The book observed at decision t is the previous minute's end-of-minute book, so
    # relabel each book row forward by one minute.
    book = minute_df[["ts"] + _FILL_BOOK].copy()
    book["ts"] = book["ts"] + MS_PER_MINUTE

    dec = feats.merge(book, on="ts", how="inner")
    dec["bucket"] = (dec["ts"] // EPISODE_MS) * EPISODE_MS

    eps = episodes_df[["start_ts", "episode_id", "regime", "split"]].rename(
        columns={"start_ts": "bucket"}
    )
    dec = dec.merge(eps, on="bucket", how="inner")
    dec["step"] = ((dec["ts"] - dec["bucket"]) // MS_PER_MINUTE).astype(int)

    cols = _KEYS + FEATURES + ["mid", "bid_px_1", "bid_sz_1"] + ASK_PX + ASK_SZ
    return dec[cols].sort_values(["episode_id", "step"]).reset_index(drop=True)


def check_integrity(table: pd.DataFrame) -> Dict[str, object]:
    """Validate the materialised dataset; raise on must-hold invariants."""
    steps = table.groupby("episode_id")["step"].size()
    bad = steps[steps != 30]
    if len(bad):
        raise ValueError(f"{len(bad)} episodes do not have exactly 30 steps")

    train, test = table[table.split == "train"], table[table.split == "test"]
    if len(train) and len(test) and train["ts"].max() >= test["ts"].min():
        raise ValueError("train/test temporal overlap: max(train ts) >= min(test ts)")

    feat_book = FEATURES + ["mid", "bid_px_1"] + ASK_PX[:1] + ASK_SZ[:1]
    n_nan = int(table[feat_book].isna().sum().sum())
    ordering_ok = bool((table["bid_px_1"] < table["ask_px_1"]).all())

    return {
        "rows": int(len(table)),
        "episodes": int(table["episode_id"].nunique()),
        "train_rows": int(len(train)),
        "test_rows": int(len(test)),
        "steps_per_episode_ok": True,
        "train_before_test": True,
        "nan_in_core_fields": n_nan,
        "book_ordering_ok": ordering_ok,
        "counts_by_split_regime": (
            table.drop_duplicates("episode_id").groupby(["split", "regime"], observed=True)
            .size().unstack(fill_value=0).to_dict(orient="index")
        ),
    }


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    p = argparse.ArgumentParser(description="Stage 6 - materialise train/test datasets")
    p.add_argument("--features-parquet", required=True)
    p.add_argument("--minute-parquet", required=True)
    p.add_argument("--episodes-parquet", required=True)
    p.add_argument("--out-dir", required=True, help="output dir for train.parquet/test.parquet (scratch)")
    args = p.parse_args()

    features_df = pd.read_parquet(args.features_parquet, columns=["ts"] + FEATURES)
    minute_df = pd.read_parquet(args.minute_parquet, columns=["ts"] + _FILL_BOOK)
    episodes_df = pd.read_parquet(args.episodes_parquet,
                                  columns=["start_ts", "episode_id", "regime", "split"])

    table = materialise(features_df, minute_df, episodes_df)
    report = check_integrity(table)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    table[table.split == "train"].drop(columns="split").to_parquet(out_dir / "train.parquet", index=False)
    table[table.split == "test"].drop(columns="split").to_parquet(out_dir / "test.parquet", index=False)
    meta = {
        "alignment": "features[t] (uses bars <= t-1) + fill book as-of t (= minute[t-1]); uniform one-bar lag, no look-ahead",
        "observation_columns": FEATURES,
        "fill_book_columns": ASK_PX + ASK_SZ + ["bid_px_1", "bid_sz_1", "mid"],
        "execution_state": "inventory and time-remaining are added by the env at runtime, not stored here",
        **report,
    }
    (out_dir / "dataset_meta.json").write_text(json.dumps(meta, indent=2, default=str))

    logger.info("wrote train.parquet (%d rows) + test.parquet (%d rows) to %s",
                report["train_rows"], report["test_rows"], out_dir)
    for k in ("episodes", "train_before_test", "nan_in_core_fields", "book_ordering_ok",
              "counts_by_split_regime"):
        logger.info("  %s: %s", k, report[k])


if __name__ == "__main__":
    main()
