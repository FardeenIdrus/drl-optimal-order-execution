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

  7. How the December per-order month was divided, for Table D3's Panel C. This is a THIRD
     kind of division and had no exhibit before 2026-08-06: the snapshot record is split by
     date, the simulator's episodes by random seed, and this month by HOUR. The split is
     chronological WITHIN each volatility regime, so calendar dates appear on both sides
     while no hour does; a table that implied a date cut would be wrong.

INTEGRITY. Coverage states are asserted to be mutually exclusive and to exhaust the calendar
span. Episode counts recovered per date are asserted to sum to each build's total. A failure
in either is raised, not warned.

Sources: scratch_hyperliquid/manifest/BTC_2024-01-01_2025-12-31_{coverage.json,manifest.csv}
         scratch_hyperliquid/{dataset,dataset_10s,dataset_10s_10min}/{train,test}.parquet
         scratch_hyperliquid/{episodes,episodes_10s,episodes_10s_10min}/btc_episodes_*.parquet
         scratch_hyperliquid/minute/btc_minute_2024-2025.parquet
         scratch_hyperliquid/oxford_l4/step3g/{regime_labels.parquet,regime_report.json}
         scratch_hyperliquid/oxford_l4/book_05s_v2/*.parquet
Output:  scratch_hyperliquid/oxford_l4/data_chapter_measurements.json
"""
from __future__ import annotations

import json
import re
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

    # THE BOUNDARY GAP, measured rather than asserted. Table D3 prints the last training bar
    # and the first test bar; between them sits one whole episode plus one bar, in every
    # version. That is a deliberate buffer so no episode straddles the split. Unexplained, the
    # gap reads as missing data, which is the opposite of what it is.
    steps = int(meta["rows"] / meta["episodes"])
    first_ep = tr[tr["episode_id"] == tr["episode_id"].iloc[0]]
    bar_seconds = float(np.median(np.diff(np.sort(first_ep["ts"].to_numpy()))) / 1000.0)
    gap_seconds = (te_ts.min() - tr_ts.max()).total_seconds()
    gap_bars = gap_seconds / bar_seconds
    assert abs(gap_bars - round(gap_bars)) < 1e-6, gap_bars
    buffer_episodes = (round(gap_bars) - 1) / steps
    assert buffer_episodes == 1.0, (name, gap_bars, steps, buffer_episodes)

    return {
        "label": BUILDS[name]["label"],
        "steps_per_episode": steps,
        "bar_seconds": bar_seconds,
        "boundary_gap_seconds": gap_seconds,
        "boundary_gap_bars": round(gap_bars),
        "boundary_buffer_episodes": buffer_episodes,
        "test_fraction_pct": round(100.0 * n_ep_te / meta["episodes"], 2),
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


def december_partition() -> dict:
    """How the December per-order month was divided, for Table D3 Panel C.

    A THIRD kind of division, on a different unit from the other two. The snapshot record
    labels each 30-minute EPISODE; this labels each HOUR, on the standard deviation of
    one-second mid-price changes within it. The rule is the same in principle -- the median,
    chosen and then applied -- but the two thresholds are not comparable and must never be
    printed as though they were.

    THE SPLIT IS NOT A DATE CUT, and describing it as one would be false. `step3g.label_regimes`
    takes the earliest CAL_FRACTION of each REGIME's hours to fit the simulator and holds the
    latest back. Calm and volatile hours interleave through the month, so calendar dates land on
    both sides -- while no hour does, which is what keeps the holdout clean.

    ONE HOUR OF 744 IS UNLABELLED, and it is the SAME exclusion section 3.3 already discloses,
    not a second one. The reconstruction starts at 00:12:29 on 1 December, leaving that hour
    below `MIN_SNAPSHOTS_PER_HOUR`. Measured here rather than asserted, because an earlier
    version of the chapter plan called it a separate warm-up exclusion and that was wrong.
    """
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
    from execution.qrm.step3g import (BLOCK_LEN, CAL_FRACTION, MIN_SNAPSHOTS_PER_HOUR,
                                      N_BLOCKS)

    d = pd.read_parquet(S / "oxford_l4" / "step3g" / "regime_labels.parquet")
    rep = json.loads((S / "oxford_l4" / "step3g" / "regime_report.json").read_text())

    grid = d.groupby(["regime", "split"]).size().unstack(fill_value=0)
    cells = {f"{r}/{s}": int(grid.loc[r, s]) for r in grid.index for s in grid.columns}
    assert cells == {k: int(v) for k, v in rep["hours"].items()}, (cells, rep["hours"])

    # The month's own calendar, so the shortfall is measured against what a full month holds.
    n_days = d["date"].nunique()
    hours_expected = 24 * n_days
    assert len(d) == sum(cells.values())

    # Dates on both sides of the split. Nine of them, and no hour is shared.
    cal_dates = set(d[d["split"] == "calibrate"]["date"])
    hold_dates = set(d[d["split"] == "holdout"]["date"])
    both = sorted(cal_dates & hold_dates)
    assert d.duplicated(subset=["date", "hour"]).sum() == 0

    # The unlabelled hour, counted from the grid itself rather than assumed.
    first_day = sorted(d["date"].unique())[0]
    bk = pq.read_table(S / "oxford_l4" / "book_05s_v2" / f"{first_day}.parquet",
                       columns=["ts"]).to_pandas()
    day0 = int(pd.Timestamp(f"{first_day[:4]}-{first_day[4:6]}-{first_day[6:]}", tz="UTC").value)
    hr = ((bk["ts"] - day0) // 3_600_000_000_000).astype(int)
    hour0_snapshots = int((hr == 0).sum())
    first_ts = str(pd.to_datetime(bk["ts"].min(), unit="ns", utc=True))
    assert hour0_snapshots < MIN_SNAPSHOTS_PER_HOUR, hour0_snapshots
    assert len(d) == hours_expected - 1, (len(d), hours_expected)

    # THE SPLIT IS NOT THE CONSUMPTION, and the plan conflated them until 2026-08-06.
    # 557 hours are AVAILABLE to fit; the fit of record drew five contiguous six-hour blocks
    # from each regime. Counted from the bundle the trainer actually loads
    # (`train_reactive.py:81` reads qrm_bundle_{regime}_b.npz, whose accumulators carry the
    # hour list), not inferred from the code's defaults: an earlier, superseded run sampled
    # 56 hours per regime by stride and would give the wrong figure.
    fitted = {}
    for reg in ("calm", "volatile"):
        acc = np.load(S / "oxford_l4" / "step3g" / f"accumulators_{reg}_b.npz",
                      allow_pickle=True)
        hrs = acc["hours"].tolist()
        assert len(hrs) == BLOCK_LEN * N_BLOCKS, (reg, len(hrs))
        # every fitting hour must be inside that regime's calibrate split
        cal = {(r.date, int(r.hour)) for r in
               d[(d.regime == reg) & (d.split == "calibrate")].itertuples()}
        assert all((h.split(":")[0], int(h.split(":")[1])) in cal for h in hrs), reg
        fitted[reg] = {"hours": len(hrs), "first": hrs[0], "last": hrs[-1]}

    return {
        "source": "oxford_l4/step3g/{regime_labels.parquet,regime_report.json}",
        "unit": "one calendar hour of the December per-order month",
        "fit_of_record": {
            "bundle": "oxford_l4/step3g/qrm_bundle_{calm,volatile}_b.npz",
            "loaded_by": "src/execution/qrm/train_reactive.py:81",
            "block_len_hours": int(BLOCK_LEN),
            "n_blocks_per_regime": int(N_BLOCKS),
            "hours_per_regime": fitted["calm"]["hours"],
            "hours_total": fitted["calm"]["hours"] + fitted["volatile"]["hours"],
            "by_regime": fitted,
            "note": ("contiguous blocks nearest each regime's median volatility, drawn from "
                     "the calibrating side only. The calibrating side holds 557 hours; the "
                     "fit consumed 60 of them."),
        },
        "period": f"{first_day} to {sorted(d['date'].unique())[-1]}",
        "days": int(n_days),
        "hours_expected": int(hours_expected),
        "hours_labelled": int(len(d)),
        "threshold_vol_1s": rep["threshold_vol_1s"],
        "threshold_definition": ("median over the month of the standard deviation of "
                                 "one-second mid-price changes within an hour, in USD"),
        "median_vol_calm": rep["median_vol"]["calm"],
        "median_vol_volatile": rep["median_vol"]["volatile"],
        "separation_ratio": rep["separation_ratio"],
        "calibrate_fraction": float(CAL_FRACTION),
        "cells": cells,
        "calm_total": cells["calm/calibrate"] + cells["calm/holdout"],
        "volatile_total": cells["volatile/calibrate"] + cells["volatile/holdout"],
        "calibrate_total": cells["calm/calibrate"] + cells["volatile/calibrate"],
        "holdout_total": cells["calm/holdout"] + cells["volatile/holdout"],
        "calibrate_date_span": [min(cal_dates), max(cal_dates)],
        "holdout_date_span": [min(hold_dates), max(hold_dates)],
        "dates_on_both_sides": both,
        "n_dates_on_both_sides": len(both),
        "unlabelled_hour": {
            "date": first_day, "hour": 0,
            "snapshots": hour0_snapshots,
            "snapshots_required": int(MIN_SNAPSHOTS_PER_HOUR),
            "minutes_present": round(hour0_snapshots / 120.0, 2),
            "reconstruction_first_timestamp": first_ts,
            "cause": ("the same 12.48-minute shortfall at the start of the reconstruction that "
                      "section 3.3 discloses; not a separate exclusion"),
        },
    }


def coverage_bound(build: str = "dataset_10s") -> dict:
    """What the days lost from the test period were like, for section 3.5.

    WHY THIS EXISTS. The Data chapter's limitations rest on one measured claim: the days that
    fell out of the test period were the BUSY ones, so the results generalise to a quieter
    market than the venue's. Until 2026-08-08 those figures existed only as prose in an internal
    log -- the one set of numbers in the chapter that no script produced. A limitations section
    whose own numbers cannot be re-derived is the wrong place to be sloppy.

    WHAT IT COMPARES. Every calendar date inside the test window, split into those that supplied
    at least one episode and those that supplied none, against the venue's own daily traded
    volume. The test is Mann-Whitney rather than a difference of means because daily volume is
    heavily right-skewed and one excluded day is the largest of the two years -- a mean would be
    dominated by it, which is the objection the test avoids.

    DIRECTION MATTERS AND IS ASSERTED. If the excluded days were the QUIET ones the bound would
    run the other way, so the sign is checked rather than assumed.
    """
    from scipy.stats import mannwhitneyu

    b = build_measures(build)
    adv = json.loads((S / "adv" / "btc_adv.json").read_text())
    daily = adv["daily_volume_btc"]

    excluded = [d for d in b["eval_zero_dates"] if d in daily]
    contributing = [d for d in sorted(b["test_episodes_by_date"]) if d in daily]
    assert len(excluded) == b["eval_dates_zero"], (len(excluded), b["eval_dates_zero"])
    assert not set(excluded) & set(contributing)

    ex = np.array([daily[d] for d in excluded], dtype=float)
    inc = np.array([daily[d] for d in contributing], dtype=float)
    med_ex, med_in = float(np.median(ex)), float(np.median(inc))
    u = mannwhitneyu(ex, inc, alternative="two-sided")
    assert med_ex > med_in, (med_ex, med_in)  # the excluded days are the BUSY ones

    biggest = max(daily, key=lambda d: daily[d])
    return {
        "build": build,
        "source": "adv/btc_adv.json daily volumes; excluded dates from this file's builds block",
        "n_excluded_days": len(excluded),
        "n_contributing_days": len(contributing),
        "median_excluded_btc": round(med_ex, 0),
        "median_contributing_btc": round(med_in, 0),
        "ratio": round(med_ex / med_in, 2),
        "mannwhitney_u": float(u.statistic),
        "mannwhitney_p": float(u.pvalue),
        "test": "Mann-Whitney U, two-sided; daily volume is right-skewed so a mean would be "
                "dominated by the single largest day",
        "largest_day_of_span": {"date": biggest, "btc": round(daily[biggest], 0),
                                "is_excluded": biggest in excluded},
        "excluded_dates": excluded,
        "bearing": ("Internal validity is untouched: every policy and benchmark scores the same "
                    "episodes under common random numbers. What is bounded is the market the "
                    "results generalise to."),
    }


def perorder_pipeline() -> dict:
    """The three December processing steps that no other exhibit covers.

    Table 3.1 already carries the SOURCES (what each stream holds, that the book changes have
    no clock, the identifier match, 31 of 31 days, 5,355,301 states). Table 3.4 carries the
    DIVISION into fitting and holdout hours. What sits nowhere is the processing between them:
    the warm-up, the phantom removal and the second-pass verification. This block measures
    those, so the table that states them types no number of its own.

    The warm-up count, the grid start and the per-day snapshot totals are read from the
    reconstruction's OWN run log, not from a note. The phantom count is the one value that
    cannot be re-derived here: it comes from the audit of 2026-07-04 registered at
    `qrm_step4_criteria.md:1690`, and is recorded with that provenance rather than recomputed.

    A trap this block exists partly to fix: the run log's `evicted` counter sums to 129,894
    across the month and is a DIFFERENT mechanism (guard eviction, the fix for the frozen-book
    bug). It is not the phantom count and must never be substituted for it.
    """
    log = S / "oxford_l4" / "book_05s_v2_run.log"
    assert log.exists(), f"reconstruction run log not found: {log}"
    txt = log.read_text()

    warmups = {int(x) for x in re.findall(r"warmup=(\d+)", txt)}
    assert len(warmups) == 1, f"more than one warm-up setting in the log: {warmups}"
    snaps = [int(x) for x in re.findall(r"done: (\d+) snapshots", txt)]
    evicted = [int(x) for x in re.findall(r"evicted=(\d+)", txt)]
    assert len(snaps) == 31, f"expected 31 logged days, found {len(snaps)}"

    grid = sorted((S / "oxford_l4" / "book_05s_v2").glob("*.parquet"))
    assert grid, "December book grid not found"
    first_ts = pd.read_parquet(grid[0], columns=["ts"])["ts"].min()
    start = pd.to_datetime(int(first_ts), unit="ns", utc=True)
    into_day = (start - start.normalize()).total_seconds()

    dp = december_partition()
    n_rows = sum(snaps)
    assert n_rows == depth_by_hour()["n_grid_rows"], "log total disagrees with the grid"

    return {
        "source": "oxford_l4/book_05s_v2_run.log + book_05s_v2/*.parquet",
        "days_logged": len(snaps),
        "n_grid_rows": n_rows,
        "warmup_events": warmups.pop(),
        "grid_start_utc": start.isoformat(),
        "grid_start_minutes_into_1_dec": round(into_day / 60.0, 2),
        "hours_expected": dp["hours_expected"],
        "hours_labelled": dp["hours_labelled"],
        "hours_unlabelled": dp["hours_expected"] - dp["hours_labelled"],
        "phantom_orders_removed": 4345,
        "phantom_provenance": ("audited 2026-07-04, registered at qrm_step4_criteria.md:1690; "
                               "a cancellation recorded before the order it cancels"),
        "evicted_total_NOT_phantoms": sum(evicted),
        "evicted_note": ("guard eviction, a different mechanism from the phantom removal. "
                         "NEVER substitute this for phantom_orders_removed."),
        "verification": ("a second pass advances the book event by event and samples it back to "
                         "the same half-second points; the two methods agree exactly"),
    }


def main() -> None:
    res = {
        "generated_by": "reports/diagnostics/data_chapter_measure.py",
        "raw_coverage": raw_coverage(),
        "builds": {name: build_measures(name) for name in BUILDS},
        "volatility": {name: volatility_distribution(cfg["episodes_dir"], name == "dataset")
                       for name, cfg in BUILDS.items()},
        "depth_by_hour": depth_by_hour(),
        "december_partition": december_partition(),
        "perorder_pipeline": perorder_pipeline(),
        "coverage_bound": coverage_bound(),
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
    cb = res["coverage_bound"]
    print(f"coverage bound: {cb['n_excluded_days']} excluded days, median "
          f"{cb['median_excluded_btc']:,.0f} vs {cb['median_contributing_btc']:,.0f} BTC, "
          f"ratio {cb['ratio']}x, Mann-Whitney p = {cb['mannwhitney_p']:.4g}; largest day "
          f"{cb['largest_day_of_span']['date']} excluded: {cb['largest_day_of_span']['is_excluded']}")
    dp = res["december_partition"]
    print(f"december: {dp['hours_labelled']}/{dp['hours_expected']} hours labelled  "
          f"{dp['cells']}  threshold {dp['threshold_vol_1s']:.4f}  "
          f"dates on both sides {dp['n_dates_on_both_sides']}  "
          f"unlabelled hour {dp['unlabelled_hour']['snapshots']}/"
          f"{dp['unlabelled_hour']['snapshots_required']} snapshots")


if __name__ == "__main__":
    main()
