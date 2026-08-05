"""Every measured quantity the Data chapter's six exhibits need, in one JSON.

WHY THIS EXISTS. Tables D1-D3 and Figures D1-D3 must contain no hand-typed number. This
script is the single point at which the chapter's numbers are read out of the parquets and
the manifest, so that a table builder is a formatter and never a source. Anything a builder
cannot find here is a number that was never measured, and that must surface as a failure
rather than as a plausible constant.

WHAT IT MEASURES, and why each piece is needed.

  1. Raw coverage per calendar date, in four mutually exclusive raw states -- complete,
     partial, missing, skipped -- read from the pull manifest rather than restated. Figure D1
     draws these as its lower layer.

  2. Episode contribution per calendar date, per dataset build. This is a DIFFERENT measure
     from raw coverage and the two disagree: a date can be complete by pull and contribute
     nothing, because the volatility window invalidates minutes whose 30-minute lookback
     overlaps a gap. Figure D1 draws this as its upper layer, and the disagreement is the
     point of the figure.

  3. Split boundaries per build, read from the parquets. The three builds do NOT share a
     boundary: the chronological split is applied at a fixed fraction of each build's own row
     count, and the builds have different row counts. Table D3 states all three.

  4. The episode-level volatility distribution with the train-median threshold, for Figure D2.
     The threshold was CHOSEN, not supplied by the data, and the figure has to show that.

  5. Depth by hour of day for December 2025, for Figure D3. It is measured on the SAME
     month as the traded volume, so the two panels can be read together.

  6. November 2025 episode counts by date and by hour, for the headline build. This answers
     whether the month's thin contribution is concentrated or spread, which changes what the
     thinness implies.

INTEGRITY. Coverage states are asserted to be mutually exclusive and to exhaust the calendar
span. Episode counts recovered per date are asserted to sum to each build's total. A failure
in either is raised, not warned.

Sources: scratch_hyperliquid/manifest/BTC_2024-01-01_2025-12-31_{coverage.json,manifest.csv}
         scratch_hyperliquid/{dataset,dataset_10s,dataset_10s_10min}/{train,test}.parquet
         scratch_hyperliquid/{episodes,episodes_10s,episodes_10s_10min}/btc_episodes_*.parquet
         scratch_hyperliquid/minute/btc_minute_2024-2025.parquet
Output:  scratch_hyperliquid/oxford_l4/data_chapter_measurements.json
"""
from __future__ import annotations

import json
from collections import defaultdict
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

S = Path("/Users/fardeenidrus/Desktop/MSc Dissertation/scratch_hyperliquid")
OUT = S / "oxford_l4" / "data_chapter_measurements.json"

MANIFEST_COV = S / "manifest" / "BTC_2024-01-01_2025-12-31_coverage.json"
MANIFEST_CSV = S / "manifest" / "BTC_2024-01-01_2025-12-31_manifest.csv"

BUILDS = {
    "dataset": {"label": "1-minute", "episodes_dir": "episodes"},
    "dataset_10s": {"label": "10-second (headline)", "episodes_dir": "episodes_10s"},
    "dataset_10s_10min": {"label": "10-second, 10-minute horizon",
                          "episodes_dir": "episodes_10s_10min"},
}


def daterange(start: str, end: str) -> list[str]:
    a = date.fromisoformat(start)
    b = date.fromisoformat(end)
    return [(a + timedelta(days=i)).isoformat() for i in range((b - a).days + 1)]


def raw_coverage() -> dict:
    cov = json.loads(MANIFEST_COV.read_text())
    span = daterange(cov["range"]["start"], cov["range"]["end"])

    skipped = set()
    for lo, hi in cov["skip_ranges"]:
        skipped |= set(daterange(lo, hi))
    missing = set(cov["days_missing"])
    partial = set(cov["days_partial"])

    man = pd.read_csv(MANIFEST_CSV)
    hours_present = man[man["present"].astype(str).str.lower() == "true"]
    hours_by_date = hours_present.groupby("date").size().to_dict()

    state = {}
    for d in span:
        if d in skipped:
            state[d] = "skipped"
        elif d in missing:
            state[d] = "missing"
        elif d in partial:
            state[d] = "partial"
        else:
            state[d] = "complete"

    counts = {k: sum(1 for v in state.values() if v == k)
              for k in ("complete", "partial", "missing", "skipped")}

    # The four states must partition the calendar span exactly.
    assert sum(counts.values()) == len(span), (counts, len(span))
    assert counts["complete"] == cov["days_fully_covered"], (counts, cov["days_fully_covered"])
    assert counts["skipped"] == cov["days_skipped"]
    assert counts["missing"] == len(cov["days_missing"])
    assert counts["partial"] == len(cov["days_partial"])

    return {
        "span_start": span[0],
        "span_end": span[-1],
        "span_days": len(span),
        "days_attempted": cov["days_in_range_excluding_skips"],
        "hours_present": cov["hours_present"],
        "hours_expected_attempted": cov["hours_expected"],
        "coverage_pct_attempted": cov["coverage_pct"],
        "coverage_pct_calendar": round(100.0 * cov["hours_present"] / (24 * len(span)), 2),
        "total_size_gib": cov["total_size_gib"],
        "n_raw_files": cov["hours_present"],
        "state_counts": counts,
        "state_by_date": state,
        "hours_by_partial_date": {d: int(hours_by_date.get(d, 0)) for d in sorted(partial)},
        "days_missing": sorted(missing),
        "skip_ranges": cov["skip_ranges"],
    }


def build_measures(name: str) -> dict:
    root = S / name
    tr = pq.read_table(root / "train.parquet", columns=["episode_id", "ts"]).to_pandas()
    te = pq.read_table(root / "test.parquet", columns=["episode_id", "ts"]).to_pandas()
    meta = json.loads((root / "dataset_meta.json").read_text())

    # ts is epoch MILLISECONDS in every build; read as nanoseconds it silently lands in 1970.
    assert tr["ts"].dtype.kind == "i", tr["ts"].dtype
    tr_ts = pd.to_datetime(tr["ts"], unit="ms", utc=True)
    te_ts = pd.to_datetime(te["ts"], unit="ms", utc=True)
    assert tr_ts.min().year == 2024, tr_ts.min()

    # One row per episode, stamped by its FIRST bar: an episode belongs to the date it starts.
    def per_episode(df: pd.DataFrame, ts: pd.Series) -> pd.DataFrame:
        g = pd.DataFrame({"episode_id": df["episode_id"].values, "ts": ts.values})
        return g.groupby("episode_id", as_index=False)["ts"].min()

    ep_tr = per_episode(tr, tr_ts)
    ep_te = per_episode(te, te_ts)

    n_ep_tr, n_ep_te = len(ep_tr), len(ep_te)
    assert n_ep_tr + n_ep_te == meta["episodes"], (n_ep_tr, n_ep_te, meta["episodes"])

    # Validation is the final 15% of train.parquet, carved chronologically.
    n_val = int(round(n_ep_tr * 0.15))
    n_train_only = n_ep_tr - n_val

    by_date_tr = ep_tr["ts"].dt.date.astype(str).value_counts().to_dict()
    by_date_te = ep_te["ts"].dt.date.astype(str).value_counts().to_dict()
    by_date_all = defaultdict(int)
    for d, n in list(by_date_tr.items()) + list(by_date_te.items()):
        by_date_all[d] += int(n)
    assert sum(by_date_all.values()) == meta["episodes"]

    # Evaluation window: first test date through the span end, inclusive of empty days.
    test_dates = sorted(by_date_te)
    eval_window = daterange(test_dates[0], "2025-12-31")
    contributing = set(test_dates)
    zero_days = [d for d in eval_window if d not in contributing]

    nov = {d: n for d, n in sorted(by_date_te.items()) if d.startswith("2025-11")}
    nov_total = sum(nov.values())

    # November's within-day placement, for the concentration probe.
    te_nov = ep_te[ep_te["ts"].dt.strftime("%Y-%m").eq("2025-11")]
    nov_by_hour = te_nov["ts"].dt.hour.value_counts().sort_index().to_dict()

    return {
        "label": BUILDS[name]["label"],
        "steps_per_episode": int(meta["rows"] / meta["episodes"]),
        "episodes_total": int(meta["episodes"]),
        "episodes_train_parquet": n_ep_tr,
        "episodes_train_only": n_train_only,
        "episodes_validation": n_val,
        "episodes_test": n_ep_te,
        "rows_total": int(meta["rows"]),
        "train_before_test": bool(meta["train_before_test"]),
        "train_last_ts": str(tr_ts.max()),
        "test_first_ts": str(te_ts.min()),
        "test_last_ts": str(te_ts.max()),
        "test_regime": meta["counts_by_split_regime"]["test"],
        "train_regime": meta["counts_by_split_regime"]["train"],
        "eval_window_days": len(eval_window),
        "eval_dates_contributing": len(contributing),
        "eval_dates_zero": len(zero_days),
        "eval_zero_dates": zero_days,
        "november_episodes": nov_total,
        "november_share_of_test": round(100.0 * nov_total / n_ep_te, 2),
        "november_by_date": nov,
        "november_by_hour": {str(k): int(v) for k, v in nov_by_hour.items()},
        "episodes_by_date": dict(sorted(by_date_all.items())),
        "test_episodes_by_date": {d: int(n) for d, n in sorted(by_date_te.items())},
    }


def volatility_distribution(episodes_dir: str, check_against_qa: bool) -> dict:
    """Each BUILD carries its own threshold, and the three differ.

    The rule is identical across builds -- the median of episode realised volatility over the
    training split -- but the builds cut episodes differently, so the median lands in a
    different place. Quoting one build's threshold as though it were the project's would be the
    same error as quoting one build's split boundary.
    """
    ep = pq.read_table(S / episodes_dir / "btc_episodes_2024-2025.parquet",
                       columns=["realized_vol", "split", "regime"]).to_pandas()

    train = ep[ep["split"] == "train"]["realized_vol"].to_numpy()
    test = ep[ep["split"] == "test"]["realized_vol"].to_numpy()
    thr = float(np.median(train))

    if check_against_qa:
        qa = json.loads((Path(__file__).resolve().parents[1] / "phase1_qa.json").read_text())
        recorded = qa["stage5_regimes"]["thresholds"]["median"]
        assert abs(thr - recorded) < 1e-12, (thr, recorded)

    # The labels in the file must be exactly what applying this threshold produces.
    assert int((train > thr).sum()) == int((ep[ep["split"] == "train"]["regime"] == "volatile").sum())
    assert int((test > thr).sum()) == int((ep[ep["split"] == "test"]["regime"] == "volatile").sum())

    counts = ep.groupby(["split", "regime"]).size().unstack(fill_value=0).to_dict()
    return {
        "threshold": thr,
        "threshold_definition": "median of episode realised volatility over the TRAIN split",
        "n_episodes": int(len(ep)),
        "train_vol": {"n": int(train.size), "min": float(train.min()),
                      "p05": float(np.percentile(train, 5)),
                      "median": float(np.median(train)),
                      "p95": float(np.percentile(train, 95)), "max": float(train.max())},
        "test_vol": {"n": int(test.size), "min": float(test.min()),
                     "p05": float(np.percentile(test, 5)),
                     "median": float(np.median(test)),
                     "p95": float(np.percentile(test, 95)), "max": float(test.max())},
        "counts_by_split_regime": {k: {kk: int(vv) for kk, vv in v.items()}
                                   for k, v in counts.items()},
        "test_volatile_pct": round(100.0 * (test > thr).mean(), 2),
        "train_volatile_pct": round(100.0 * (train > thr).mean(), 2),
        "hist_bins": 80,
        "train_hist": np.histogram(train, bins=80, range=(0.0, float(np.percentile(train, 99.5))))[0].tolist(),
        "test_hist": np.histogram(test, bins=80, range=(0.0, float(np.percentile(train, 99.5))))[0].tolist(),
        "hist_range": [0.0, float(np.percentile(train, 99.5))],
    }


def depth_by_hour() -> dict:
    """Resting depth by hour of day, measured on the SAME MONTH as the traded volume.

    The earlier version of this measurement used the two-year minute record. That made the
    depth panel and the volume panel cover different periods, so the two could not jointly
    support the claim that intraday variation in participation comes from volume rather than
    from thinner books. This version reads the December book reconstruction directly, at
    twenty levels each side on a half-second grid, so both panels describe the same 31 days.

    Two DISJOINT series are returned, not nested ones. The touch is the best bid plus the best
    ask. "Behind" is levels two to five on each side, which is the depth standing behind the
    touch and NOT including it. Plotting nested totals invites a reader to add them.
    """
    files = sorted((S / "oxford_l4" / "book_05s_v2").glob("*.parquet"))
    assert files, "December book grid not found"
    cols = ["ts", "bid_sz_1", "ask_sz_1"] + \
           [f"{side}_sz_{i}" for side in ("bid", "ask") for i in range(2, 6)]

    touch_sum = defaultdict(float)
    behind_sum = defaultdict(float)
    counts = defaultdict(int)
    n_rows = 0
    for f in files:
        d = pq.read_table(f, columns=cols).to_pandas()
        # ts on this grid is epoch NANOSECONDS, unlike the minute record's milliseconds.
        ts = pd.to_datetime(d["ts"], unit="ns", utc=True)
        assert ts.min().year == 2025 and ts.min().month == 12, ts.min()
        hour = ts.dt.hour.to_numpy()
        touch = (d["bid_sz_1"] + d["ask_sz_1"]).to_numpy()
        behind = sum(d[f"{s}_sz_{i}"] for s in ("bid", "ask") for i in range(2, 6)).to_numpy()
        for h in range(24):
            m = hour == h
            k = int(m.sum())
            if k:
                touch_sum[h] += float(np.nansum(touch[m]))
                behind_sum[h] += float(np.nansum(behind[m]))
                counts[h] += k
        n_rows += len(d)

    assert sum(counts.values()) == n_rows
    return {
        "source": "oxford_l4/book_05s_v2 (December 2025, half-second grid, 20 levels a side)",
        "period": "2025-12-01 to 2025-12-31",
        "n_grid_rows": n_rows,
        "n_days": len(files),
        "touch_btc": {str(h): round(touch_sum[h] / counts[h], 4) for h in range(24)},
        "behind_btc": {str(h): round(behind_sum[h] / counts[h], 4) for h in range(24)},
        "definitions": {
            "touch_btc": "best bid size + best ask size, BTC",
            "behind_btc": "levels 2-5 on both sides, BTC; excludes the touch",
        },
    }


def main() -> None:
    res = {
        "generated_by": "reports/diagnostics/data_chapter_measure.py",
        "raw_coverage": raw_coverage(),
        "builds": {name: build_measures(name) for name in BUILDS},
        "volatility": {name: volatility_distribution(cfg["episodes_dir"], name == "dataset")
                       for name, cfg in BUILDS.items()},
        "depth_by_hour": depth_by_hour(),
        "adv": json.loads((S / "adv" / "btc_adv.json").read_text()) | {"daily_volume_btc": "omitted, see source"},
    }
    OUT.write_text(json.dumps(res, indent=2))
    print(f"wrote {OUT}")
    rc = res["raw_coverage"]
    print(f"raw coverage: {rc['state_counts']}  "
          f"{rc['coverage_pct_attempted']}% of {rc['days_attempted']} attempted / "
          f"{rc['coverage_pct_calendar']}% of {rc['span_days']} calendar")
    for n, b in res["builds"].items():
        print(f"{n:20s} test {b['episodes_test']:>6,}  boundary {b['train_last_ts']} -> "
              f"{b['test_first_ts']}  zero-days {b['eval_dates_zero']}  Nov {b['november_episodes']}")


if __name__ == "__main__":
    main()
