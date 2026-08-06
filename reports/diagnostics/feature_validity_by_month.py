#!/usr/bin/env python3
"""
Monthly feature validity for BOTH resolutions of the snapshot record.

WHY THIS EXISTS. The Data chapter reports that in November 2025 only a minority of the record
survives feature validation. The figures previously available for that statement (61% carrying
book data, 30.65% usable) are properties of the ONE-MINUTE feature store, while the same
paragraph counts episodes on the TEN-SECOND dataset. Quoting the two side by side without saying
which resolution each belongs to is the error class this project has caught repeatedly. This
script measures both at the same monthly granularity so the chapter can quote one resolution
throughout.

The two resolutions are PARALLEL, not nested: `configs/pipeline_10s.yaml` rebuilds stages 3-6
with bar_seconds = 10, re-resampling the raw snapshots. The ten-second store does not derive
from the one-minute bars, so no figure from one may be described as the other's basis.

TWO RATES PER MONTH, and they answer different questions:
  has_book   the share of the month's calendar bars that carry a book at all. What the venue
             supplied.
  usable     the share of the month's calendar bars for which all five variables compute. What
             survives the 30-minute volatility lookback and the 5-minute return lookback.
The gap between them is the cost of those lookbacks: a bar can carry a book and still fail
because the window behind it does not.

MEMORY. Each store is a single parquet of a few hundred megabytes. Read row group by row group,
two columns only, and accumulate counts; never load a whole file.

Sources: scratch_hyperliquid/features/btc_features_2024-2025.parquet      (1-minute)
         scratch_hyperliquid/features_10s/btc_features_2024-2025.parquet  (10-second)
Output:  scratch_hyperliquid/oxford_l4/feature_validity_by_month.json
"""
from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq

S = Path("/Users/fardeenidrus/Desktop/MSc Dissertation/scratch_hyperliquid")
OUT = S / "oxford_l4" / "feature_validity_by_month.json"

STORES = {
    "1-minute": {"path": S / "features" / "btc_features_2024-2025.parquet", "bar_seconds": 60},
    "10-second": {"path": S / "features_10s" / "btc_features_2024-2025.parquet",
                  "bar_seconds": 10},
}

# The figures the record already carries for the 1-minute store, checked rather than trusted.
RECORDED_1MIN = {"2025-11": {"has_book": 61.0, "usable": 30.65},
                 "train_low": 93.0, "train_high": 99.7, "overall_valid_pct": 94.58}
TRAIN_END = "2025-07"  # months up to and including this are training for every version


def scan(path: Path) -> pd.DataFrame:
    """Per-month bar counts: present on the calendar grid, carrying a book, and usable."""
    f = pq.ParquetFile(path)
    have = set(f.schema.names)
    assert {"ts", "feature_valid"} <= have, sorted(have)
    # A bar with no book has a null mid. That is what distinguishes "no data" from
    # "data present but the lookback window is incomplete".
    book_col = "mid" if "mid" in have else None
    cols = ["ts", "feature_valid"] + ([book_col] if book_col else [])

    rows = defaultdict(lambda: {"bars": 0, "has_book": 0, "usable": 0})
    for i in range(f.num_row_groups):
        t = f.read_row_group(i, columns=cols).to_pandas()
        # ts is epoch MILLISECONDS in every store; read as nanoseconds it lands in 1970.
        assert t["ts"].dtype.kind == "i", t["ts"].dtype
        ts = pd.to_datetime(t["ts"], unit="ms", utc=True)
        assert ts.min().year in (2024, 2025), ts.min()
        month = ts.dt.strftime("%Y-%m")
        valid = t["feature_valid"].astype(bool)
        book = t[book_col].notna() if book_col else valid
        g = pd.DataFrame({"month": month.values, "valid": valid.values, "book": book.values})
        for m, sub in g.groupby("month"):
            r = rows[m]
            r["bars"] += len(sub)
            r["has_book"] += int(sub["book"].sum())
            r["usable"] += int(sub["valid"].sum())
        del t, g

    d = pd.DataFrame(rows).T.reset_index().rename(columns={"index": "month"}).sort_values("month")
    d["has_book_pct"] = 100.0 * d["has_book"] / d["bars"]
    d["usable_pct"] = 100.0 * d["usable"] / d["bars"]
    return d.reset_index(drop=True)


def main() -> None:
    result = {"generated_by": "reports/diagnostics/feature_validity_by_month.py",
              "note": ("has_book = share of the month's calendar bars carrying a book; "
                       "usable = share for which all five variables compute. The two "
                       "resolutions are parallel rebuilds of the same raw snapshots."),
              "stores": {}}

    for label, spec in STORES.items():
        d = scan(spec["path"])
        train = d[d["month"] <= TRAIN_END]
        nov = d[d["month"] == "2025-11"]
        result["stores"][label] = {
            "path": str(spec["path"]),
            "bar_seconds": spec["bar_seconds"],
            "total_bars": int(d["bars"].sum()),
            "overall_usable_pct": round(100.0 * d["usable"].sum() / d["bars"].sum(), 2),
            "november_2025": {
                "bars": int(nov["bars"].iloc[0]),
                "has_book_pct": round(float(nov["has_book_pct"].iloc[0]), 2),
                "usable_pct": round(float(nov["usable_pct"].iloc[0]), 2),
            },
            "training_months_usable_pct": {
                "low": round(float(train["usable_pct"].min()), 2),
                "high": round(float(train["usable_pct"].max()), 2),
                "mean": round(float(train["usable_pct"].mean()), 2),
                "n_months": int(len(train)),
            },
            "by_month": [
                {"month": r.month, "bars": int(r.bars),
                 "has_book_pct": round(float(r.has_book_pct), 2),
                 "usable_pct": round(float(r.usable_pct), 2)}
                for r in d.itertuples()
            ],
        }

        print(f"\n=== {label} store ({spec['bar_seconds']}s bars) ===")
        print(f"  total bars {d['bars'].sum():,}   overall usable "
              f"{100.0 * d['usable'].sum() / d['bars'].sum():.2f}%")
        print(f"  NOVEMBER 2025: has book {nov['has_book_pct'].iloc[0]:.2f}%   "
              f"usable {nov['usable_pct'].iloc[0]:.2f}%")
        print(f"  training months usable: {train['usable_pct'].min():.2f}% to "
              f"{train['usable_pct'].max():.2f}%  (mean {train['usable_pct'].mean():.2f}%, "
              f"n={len(train)})")
        print("  by month:")
        for r in d.itertuples():
            print(f"    {r.month}  bars {int(r.bars):>9,}  has book {r.has_book_pct:6.2f}%  "
                  f"usable {r.usable_pct:6.2f}%")

    # Check the 1-minute figures against what the record already asserts.
    one = result["stores"]["1-minute"]
    checks = {
        "1min_november_has_book ~ 61%":
            abs(one["november_2025"]["has_book_pct"] - RECORDED_1MIN["2025-11"]["has_book"]) < 1.0,
        "1min_november_usable ~ 30.65%":
            abs(one["november_2025"]["usable_pct"] - RECORDED_1MIN["2025-11"]["usable"]) < 0.5,
        "1min_overall_valid ~ 94.58%":
            abs(one["overall_usable_pct"] - RECORDED_1MIN["overall_valid_pct"]) < 0.5,
    }
    result["checks_against_record"] = {k: bool(v) for k, v in checks.items()}
    print("\nCHECKS AGAINST THE RECORD (1-minute store)")
    for k, v in checks.items():
        print(f"  {'OK  ' if v else 'FAIL'} {k}")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(result, indent=2))
    print(f"\nwritten: {OUT}")


if __name__ == "__main__":
    main()
