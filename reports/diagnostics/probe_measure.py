#!/usr/bin/env python3
"""
Two measurements on the 10-second dataset's held-out test split, neither of which previously had
a script behind it.

(a) THE IMPUTED VOLATILE SHARE OF THE EXCLUDED DAYS.
    The project's results record carries the figure, 43.6%, but not its derivation: it says only
    "volume-matching them to the table above", and no .py file contained the quintile logic or
    the value. This script writes the method down and recomputes it, so the figure is
    reproducible on demand rather than asserted from a calculation that no longer exists.

    The method, stated explicitly:
      1. Take the 10-second dataset's held-out TEST split. Stamp each episode by the date of its
         FIRST bar, so an episode belongs to the day it starts.
      2. Join each in-set day to its traded volume from adv/btc_adv.json.
      3. Rank the in-set days into five equal-count volume quintiles (pandas qcut).
      4. Per quintile record: days, episodes, median daily volume, and the volatile share.
      5. Identify the EXCLUDED days: calendar dates inside the test span carrying zero episodes.
      6. Map each excluded day to the quintile its volume falls in, and take that quintile's
         volatile share as the day's predicted share. Average across excluded days.

    RECOVERED 2026-08-05, and steps 4 and 6 both needed recovering because the prose recorded
    neither. Two definitions of "volatile share" exist and they differ:
      DAY-MEAN      each day's volatile fraction, then averaged unweighted across the quintile's
                    days. This is what the record used. It reproduces the recorded quintile
                    table to the decimal: 4.9 / 18.5 / 31.7 / 48.6 / 52.9.
      EPISODE-WTD   volatile episodes divided by all episodes in the quintile:
                    4.8 / 15.4 / 31.5 / 46.6 / 50.2.
    Step 6 is an UNWEIGHTED mean over the 25 excluded days. With the day-mean quintile shares
    it returns exactly the recorded 43.6%; with the episode-weighted shares, 41.2%.

    A caution the record does not carry: the day-mean quintile table gives every day equal
    weight, so it does NOT reconcile with the test set's episode-weighted volatile share of
    29.6%. Reading the two side by side implies an inconsistency that is really a change of
    weighting. Both are computed and labelled here.

(b) NOVEMBER'S CONCENTRATION.
    November contributes 317 of the 10-second dataset's 6,474 test episodes (4.9%). Whether those
    317 spread thinly across the month or bunch into a few hours of a few days changes what can
    honestly be said about them, and the two readings had never been distinguished. This emits
    November's episode counts by date and by hour, plus concentration statistics.

Sources: scratch_hyperliquid/dataset_10s/{test.parquet,dataset_meta.json}
         scratch_hyperliquid/adv/btc_adv.json
Output:  scratch_hyperliquid/oxford_l4/probe_measurements.json
"""
from __future__ import annotations

import json
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

S = Path("/Users/fardeenidrus/Desktop/MSc Dissertation/scratch_hyperliquid")
BUILD = S / "dataset_10s"
ADV = S / "adv" / "btc_adv.json"
OUT = S / "oxford_l4" / "probe_measurements.json"

# Values the record asserts, checked rather than trusted.
RECORDED_QUINTILE_DAYS = [30, 29, 29, 29, 29]
RECORDED_VOLATILE_SHARES = [4.9, 18.5, 31.7, 48.6, 52.9]
RECORDED_EXCLUDED_DAYS = 25
RECORDED_IMPUTED_SHARE = 43.6
RECORDED_ACTUAL_SHARE = 29.6
RECORDED_NOV_EPISODES = 317
RECORDED_TEST_EPISODES = 6474


def daterange(a: date, b: date) -> list[str]:
    return [(a + timedelta(days=i)).isoformat() for i in range((b - a).days + 1)]


def load_episodes() -> pd.DataFrame:
    """One row per test episode: its start timestamp, its date, its hour, its regime."""
    t = pq.read_table(BUILD / "test.parquet",
                      columns=["episode_id", "ts", "regime"]).to_pandas()

    # ts is epoch MILLISECONDS in every build. Read as nanoseconds it lands silently in 1970.
    assert t["ts"].dtype.kind == "i", t["ts"].dtype
    ts = pd.to_datetime(t["ts"], unit="ms", utc=True)
    assert ts.min().year == 2025, ts.min()

    df = pd.DataFrame({"episode_id": t["episode_id"].values,
                       "ts": ts.values,
                       "regime": t["regime"].values})

    # An episode's regime label is constant within the episode; assert rather than assume.
    nun = df.groupby("episode_id")["regime"].nunique()
    assert (nun == 1).all(), f"{int((nun != 1).sum())} episodes carry mixed regime labels"

    ep = df.groupby("episode_id", as_index=False).agg(ts=("ts", "min"),
                                                      regime=("regime", "first"))
    ep["date"] = ep["ts"].dt.strftime("%Y-%m-%d")
    ep["hour"] = ep["ts"].dt.hour
    return ep


def volatile_flag(s: pd.Series) -> pd.Series:
    """Regime may be stored as a string or as an integer code. Resolve it once, explicitly."""
    if s.dtype.kind in "OU":
        vals = set(pd.unique(s))
        assert vals <= {"calm", "volatile"}, f"unexpected regime values: {vals}"
        return s.eq("volatile")
    vals = set(int(v) for v in pd.unique(s))
    assert vals <= {0, 1}, f"unexpected regime codes: {vals}"
    return s.astype(int).eq(1)  # 1 == volatile


def main() -> None:
    ep = load_episodes()
    ep["is_volatile"] = volatile_flag(ep["regime"])

    meta = json.loads((BUILD / "dataset_meta.json").read_text())
    adv = json.loads(ADV.read_text())
    daily = {k: float(v) for k, v in adv["daily_volume_btc"].items()}

    n_ep = len(ep)
    actual_volatile_share = 100.0 * ep["is_volatile"].mean()

    # ---- the test span, and the days inside it that carry no episodes -------------------
    span_start = date.fromisoformat(ep["date"].min())
    span_end = date.fromisoformat(ep["date"].max())
    span = daterange(span_start, span_end)
    present = sorted(ep["date"].unique())
    excluded = [d for d in span if d not in set(present)]

    # ---- step 2-4: quintiles over the in-set days ---------------------------------------
    by_day = (ep.groupby("date")
                .agg(episodes=("episode_id", "size"),
                     volatile=("is_volatile", "sum"))
                .reset_index())
    missing_vol = [d for d in by_day["date"] if d not in daily]
    assert not missing_vol, f"no traded volume recorded for {missing_vol}"
    by_day["volume_btc"] = by_day["date"].map(daily)
    by_day["quintile"] = pd.qcut(by_day["volume_btc"], 5, labels=[1, 2, 3, 4, 5])
    by_day["day_volatile_share_pct"] = 100.0 * by_day["volatile"] / by_day["episodes"]

    q = (by_day.groupby("quintile", observed=True)
               .agg(days=("date", "size"),
                    episodes=("episodes", "sum"),
                    volatile=("volatile", "sum"),
                    median_volume=("volume_btc", "median"),
                    min_volume=("volume_btc", "min"),
                    max_volume=("volume_btc", "max"),
                    day_mean_share_pct=("day_volatile_share_pct", "mean"))
               .reset_index())
    # Two definitions, and the record used the first. See the module docstring.
    q["episode_wtd_share_pct"] = 100.0 * q["volatile"] / q["episodes"]
    q["volatile_share_pct"] = q["day_mean_share_pct"]

    # ---- steps 5-6: volume-match the excluded days ---------------------------------------
    # Quintile edges come from qcut's own bins, recomputed on the in-set days.
    _, bins = pd.qcut(by_day["volume_btc"], 5, labels=[1, 2, 3, 4, 5], retbins=True)
    share_by_q = dict(zip(q["quintile"].astype(int), q["day_mean_share_pct"]))
    share_by_q_epwtd = dict(zip(q["quintile"].astype(int), q["episode_wtd_share_pct"]))

    exc_missing_vol = [d for d in excluded if d not in daily]
    exc_rows = []
    for d in excluded:
        if d not in daily:
            exc_rows.append({"date": d, "volume_btc": None, "quintile": None,
                             "predicted_volatile_share_pct": None,
                             "predicted_episode_wtd_pct": None})
            continue
        v = daily[d]
        # np.digitize with the interior edges: a volume above the top edge maps to quintile 5,
        # below the bottom edge to quintile 1. Both occur, because the excluded days were not
        # part of the ranking that produced the bins.
        k = int(np.digitize([v], bins[1:-1], right=True)[0]) + 1
        k = max(1, min(5, k))
        exc_rows.append({"date": d, "volume_btc": v, "quintile": k,
                         "predicted_volatile_share_pct": share_by_q[k],
                         "predicted_episode_wtd_pct": share_by_q_epwtd[k]})
    exc = pd.DataFrame(exc_rows)
    scored = exc.dropna(subset=["predicted_volatile_share_pct"])

    # Step 6 is an unweighted mean over the excluded days, under each quintile definition.
    variants = {
        "day-mean quintile shares (the recorded method)":
            float(scored["predicted_volatile_share_pct"].mean()),
        "episode-weighted quintile shares":
            float(scored["predicted_episode_wtd_pct"].mean()),
    }
    reproduces = {k: abs(v - RECORDED_IMPUTED_SHARE) < 0.05 for k, v in variants.items()}

    # ---- (b) November -------------------------------------------------------------
    nov = ep[ep["date"].str.startswith("2025-11")].copy()
    nov_by_date = nov.groupby("date").size().sort_index()
    nov_by_hour = nov.groupby("hour").size().reindex(range(24), fill_value=0)
    nov_days_in_month = len(daterange(date(2025, 11, 1), date(2025, 11, 30)))
    nov_days_present = int(nov_by_date.size)

    def top_share(counts: pd.Series, k: int) -> float:
        return 100.0 * counts.sort_values(ascending=False).head(k).sum() / counts.sum()

    def gini(counts: pd.Series, n_slots: int) -> float:
        """Gini over all slots, absent ones counted as zero. 0 = perfectly even."""
        x = np.zeros(n_slots, dtype=float)
        x[:len(counts)] = np.sort(counts.values)
        x = np.sort(x)
        n = len(x)
        if x.sum() == 0:
            return 0.0
        return float((2 * np.sum((np.arange(1, n + 1)) * x)) / (n * x.sum()) - (n + 1) / n)

    nov_stats = {
        "episodes": int(nov_by_date.sum()),
        "share_of_test_pct": round(100.0 * nov_by_date.sum() / n_ep, 2),
        "days_in_month": nov_days_in_month,
        "days_carrying_episodes": nov_days_present,
        "days_absent": nov_days_in_month - nov_days_present,
        "episodes_per_present_day_mean": round(float(nov_by_date.mean()), 1),
        "episodes_per_present_day_median": float(nov_by_date.median()),
        "episodes_per_present_day_min": int(nov_by_date.min()),
        "episodes_per_present_day_max": int(nov_by_date.max()),
        "top_3_days_share_pct": round(top_share(nov_by_date, 3), 1),
        "top_5_days_share_pct": round(top_share(nov_by_date, 5), 1),
        "gini_over_30_days": round(gini(nov_by_date, nov_days_in_month), 3),
        "hours_of_24_present": int((nov_by_hour > 0).sum()),
        "top_3_hours_share_pct": round(top_share(nov_by_hour[nov_by_hour > 0], 3), 1),
        "gini_over_24_hours": round(gini(nov_by_hour[nov_by_hour > 0], 24), 3),
        "by_date": {d: int(c) for d, c in nov_by_date.items()},
        "by_hour": {int(h): int(c) for h, c in nov_by_hour.items()},
    }

    # ---- checks against the record, printed rather than silently trusted -----------------
    checks = {
        "test_episodes_match_record": (n_ep, RECORDED_TEST_EPISODES, n_ep == RECORDED_TEST_EPISODES),
        "test_episodes_match_meta": (n_ep, meta.get("episodes"), None),
        "excluded_day_count_match": (len(excluded), RECORDED_EXCLUDED_DAYS,
                                     len(excluded) == RECORDED_EXCLUDED_DAYS),
        "quintile_days_match": (list(map(int, q["days"])), RECORDED_QUINTILE_DAYS,
                                list(map(int, q["days"])) == RECORDED_QUINTILE_DAYS),
        "quintile_shares_match": ([round(v, 1) for v in q["day_mean_share_pct"]],
                                  RECORDED_VOLATILE_SHARES,
                                  [round(v, 1) for v in q["day_mean_share_pct"]]
                                  == RECORDED_VOLATILE_SHARES),
        "actual_volatile_share_match": (round(actual_volatile_share, 1), RECORDED_ACTUAL_SHARE,
                                        abs(actual_volatile_share - RECORDED_ACTUAL_SHARE) < 0.05),
        "november_episodes_match": (nov_stats["episodes"], RECORDED_NOV_EPISODES,
                                    nov_stats["episodes"] == RECORDED_NOV_EPISODES),
        "imputed_share_reproduced": (variants, RECORDED_IMPUTED_SHARE, any(reproduces.values())),
    }

    result = {
        "generated_by": "reports/diagnostics/probe_measure.py",
        "dataset": "dataset_10s (10-second bars), held-out test split",
        "sources": {"episodes": str(BUILD / "test.parquet"),
                    "meta": str(BUILD / "dataset_meta.json"),
                    "daily_volume": str(ADV),
                    "volume_pulled_at": adv.get("pulled_at"),
                    "volume_source": adv.get("source")},
        "imputation": {
            "method": [
                "stamp each test episode by the date of its first bar",
                "join in-set days to daily traded volume from adv/btc_adv.json",
                "rank in-set days into five equal-count volume quintiles (pandas qcut)",
                "per quintile: days, episodes, volatile share of episodes, median volume",
                "excluded days = calendar dates in the test span carrying zero episodes",
                "map each excluded day into a quintile by its volume; take that quintile's "
                "volatile share; average across excluded days",
            ],
            "test_episodes": n_ep,
            "test_span": {"start": span[0], "end": span[-1], "calendar_days": len(span)},
            "days_present": len(present),
            "days_excluded": len(excluded),
            "actual_volatile_share_pct": round(actual_volatile_share, 2),
            "quintile_bin_edges_btc": [round(float(b), 2) for b in bins],
            "quintiles": [
                {"quintile": int(r.quintile), "days": int(r.days), "episodes": int(r.episodes),
                 "volatile_episodes": int(r.volatile),
                 "day_mean_share_pct": round(float(r.day_mean_share_pct), 2),
                 "episode_wtd_share_pct": round(float(r.episode_wtd_share_pct), 2),
                 "median_volume_btc": round(float(r.median_volume), 2),
                 "min_volume_btc": round(float(r.min_volume), 2),
                 "max_volume_btc": round(float(r.max_volume), 2)}
                for r in q.itertuples()
            ],
            "excluded_days": [
                {"date": r["date"],
                 "volume_btc": None if r["volume_btc"] is None else round(r["volume_btc"], 2),
                 "quintile": r["quintile"],
                 "predicted_volatile_share_pct": (
                     None if r["predicted_volatile_share_pct"] is None
                     else round(r["predicted_volatile_share_pct"], 2)),
                 "predicted_episode_wtd_pct": (
                     None if r["predicted_episode_wtd_pct"] is None
                     else round(r["predicted_episode_wtd_pct"], 2))}
                for r in exc_rows
            ],
            "excluded_days_without_volume": exc_missing_vol,
            "excluded_median_volume_btc": round(float(scored["volume_btc"].median()), 2),
            "in_set_median_volume_btc": round(float(by_day["volume_btc"].median()), 2),
            "imputed_share_variants_pct": {k: round(v, 2) for k, v in variants.items()},
            "variant_reproducing_record": [k for k, v in reproduces.items() if v] or None,
        },
        "november": nov_stats,
        "checks_against_record": {k: {"computed": v[0], "recorded": v[1], "match": v[2]}
                                  for k, v in checks.items()},
    }

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(result, indent=2, default=str))

    # ---- console report ------------------------------------------------------------------
    print(f"\nTEST SET  {BUILD.name}  episodes={n_ep}  span {span[0]} to {span[-1]} "
          f"({len(span)} calendar days)")
    print(f"  days with episodes {len(present)}   days excluded {len(excluded)}")
    print(f"  actual volatile share {actual_volatile_share:.2f}%")

    print("\n(b) VOLUME QUINTILES OVER THE IN-SET DAYS")
    print(f"  {'Q':<3}{'days':>6}{'episodes':>10}{'day-mean %':>12}{'ep-wtd %':>10}"
          f"{'median vol':>13}{'range':>26}")
    for r in q.itertuples():
        print(f"  {int(r.quintile):<3}{int(r.days):>6}{int(r.episodes):>10}"
              f"{r.day_mean_share_pct:>11.1f}%{r.episode_wtd_share_pct:>9.1f}%"
              f"{r.median_volume:>13,.0f}"
              f"{f'{r.min_volume:,.0f} - {r.max_volume:,.0f}':>26}")
    print("  day-mean = each day's volatile fraction, averaged unweighted across the quintile's")
    print("  days. THIS IS THE RECORDED DEFINITION. It gives every day equal weight, so it does")
    print(f"  NOT reconcile with the test set's episode-weighted {actual_volatile_share:.1f}%.")

    print(f"\n  excluded days: {len(excluded)}, median volume "
          f"{scored['volume_btc'].median():,.0f} against in-set median "
          f"{by_day['volume_btc'].median():,.0f}")
    print("  imputed volatile share of the excluded days (unweighted mean over the 25 days):")
    for k, v in variants.items():
        mark = f"  <-- reproduces the recorded {RECORDED_IMPUTED_SHARE}%" if reproduces[k] else ""
        print(f"    {v:>6.2f}%   using {k}{mark}")

    print("\n  EXCLUDED DAYS IN FULL")
    for r in exc_rows:
        v = "no volume" if r["volume_btc"] is None else f"{r['volume_btc']:>10,.0f}"
        k = "-" if r["quintile"] is None else r["quintile"]
        p = ("-" if r["predicted_volatile_share_pct"] is None
             else f"{r['predicted_volatile_share_pct']:.1f}%")
        print(f"    {r['date']}  volume {v}  quintile {k}  predicted volatile {p}")

    print(f"\n(c) NOVEMBER 2025 — {nov_stats['episodes']} episodes, "
          f"{nov_stats['share_of_test_pct']}% of the test set")
    print(f"  days carrying episodes {nov_stats['days_carrying_episodes']} of "
          f"{nov_stats['days_in_month']}   absent {nov_stats['days_absent']}")
    print(f"  per present day: mean {nov_stats['episodes_per_present_day_mean']}, "
          f"median {nov_stats['episodes_per_present_day_median']}, "
          f"min {nov_stats['episodes_per_present_day_min']}, "
          f"max {nov_stats['episodes_per_present_day_max']}")
    print(f"  busiest 3 days hold {nov_stats['top_3_days_share_pct']}%, "
          f"busiest 5 hold {nov_stats['top_5_days_share_pct']}%   "
          f"Gini over 30 days {nov_stats['gini_over_30_days']}")
    print(f"  hours of 24 present {nov_stats['hours_of_24_present']}   "
          f"busiest 3 hours hold {nov_stats['top_3_hours_share_pct']}%   "
          f"Gini over hours {nov_stats['gini_over_24_hours']}")
    print("\n  BY DATE")
    for d, c in nov_by_date.items():
        print(f"    {d}  {int(c):>4}")
    print("\n  BY HOUR (UTC)")
    for h, c in nov_by_hour.items():
        print(f"    {int(h):02d}:00  {int(c):>4}")

    print("\nCHECKS AGAINST THE RECORD")
    for k, (comp, rec, ok) in checks.items():
        flag = "n/a " if ok is None else ("OK  " if ok else "FAIL")
        if isinstance(comp, dict):
            comp = {kk: round(vv, 2) for kk, vv in comp.items()}
        print(f"  {flag} {k}: computed {comp}  recorded {rec}")

    print(f"\nwritten: {OUT}")


if __name__ == "__main__":
    main()
