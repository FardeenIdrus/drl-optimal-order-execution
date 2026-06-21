"""Stage 0 - manifest & availability.

Builds the ground-truth record of which hourly L2 files actually exist for a coin
over a date range, so later stages pull from a verified list rather than assuming
contiguous coverage. Output is two files in an out dir (kept OUTSIDE the repo):

    <coin>_<start>_<end>_manifest.csv   one row per (date, hour): present + size + key
    <coin>_<start>_<end>_coverage.json  summary: missing days, partial days, totals

Usage:
    PYTHONPATH=src python -m execution.data.manifest \
        --coin BTC --start 2024-01-01 --end 2025-12-31 --out <scratch>/manifest
"""
from __future__ import annotations

import argparse
import csv
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import date, timedelta
from pathlib import Path
from typing import Dict, List, Tuple

from execution.data.sources import hyperliquid_l2 as hl

# Known gap weeks in the Hyperliquid archive (excluded by request); inclusive ranges.
DEFAULT_SKIP_RANGES: List[Tuple[str, str]] = [
    ("2025-10-17", "2025-10-22"),
    ("2025-11-18", "2025-11-24"),
]
HOURS_PER_DAY = 24


def _daterange(start: date, end: date):
    d = start
    while d <= end:
        yield d
        d += timedelta(days=1)


def _in_skip(d: date, skip: List[Tuple[date, date]]) -> bool:
    return any(lo <= d <= hi for lo, hi in skip)


def build_manifest(
    start: str,
    end: str,
    coin: str = "BTC",
    skip_ranges: List[Tuple[str, str]] | None = None,
    workers: int = 16,
) -> Tuple[List[dict], dict]:
    """List availability for every (date, hour) in range, skipping excluded weeks.

    Returns ``(rows, coverage)`` where rows is one dict per (date, hour) and coverage
    is the summary. Pure listing; downloads nothing.
    """
    start_d, end_d = date.fromisoformat(start), date.fromisoformat(end)
    skip = [(date.fromisoformat(a), date.fromisoformat(b))
            for a, b in (skip_ranges if skip_ranges is not None else DEFAULT_SKIP_RANGES)]

    dates = [d for d in _daterange(start_d, end_d) if not _in_skip(d, skip)]
    n_skipped_days = sum(1 for d in _daterange(start_d, end_d) if _in_skip(d, skip))

    # List each date in parallel (one S3 list per date -> {hour: size}).
    found: Dict[str, Dict[int, int]] = {}
    errors: Dict[str, str] = {}
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futs = {ex.submit(hl.list_coin_hours, d.strftime("%Y%m%d"), coin): d for d in dates}
        for fut in as_completed(futs):
            d = futs[fut]
            try:
                found[d.isoformat()] = fut.result()
            except Exception as e:  # record, do not abort the whole manifest
                errors[d.isoformat()] = str(e)

    # Flatten to one row per (date, hour).
    rows: List[dict] = []
    for d in dates:
        ds = d.isoformat()
        hours = found.get(ds, {})
        for h in range(HOURS_PER_DAY):
            present = h in hours
            rows.append({
                "date": ds,
                "hour": h,
                "present": present,
                "size_bytes": hours.get(h, 0),
                "key": hl.object_key(d.strftime("%Y%m%d"), h, coin) if present else "",
            })

    present_rows = [r for r in rows if r["present"]]
    days_with_any = {r["date"] for r in present_rows}
    full_days = {ds for ds in days_with_any
                 if sum(1 for r in present_rows if r["date"] == ds) == HOURS_PER_DAY}
    expected_days = [d.isoformat() for d in dates]
    missing_days = [ds for ds in expected_days if ds not in days_with_any]
    partial_days = sorted(days_with_any - full_days)
    total_bytes = sum(r["size_bytes"] for r in present_rows)

    coverage = {
        "coin": coin,
        "range": {"start": start, "end": end},
        "skip_ranges": skip_ranges if skip_ranges is not None else DEFAULT_SKIP_RANGES,
        "days_in_range_excluding_skips": len(dates),
        "days_skipped": n_skipped_days,
        "days_with_data": len(days_with_any),
        "days_fully_covered": len(full_days),
        "days_missing": missing_days,
        "days_partial": partial_days,
        "hours_present": len(present_rows),
        "hours_expected": len(dates) * HOURS_PER_DAY,
        "coverage_pct": round(100 * len(present_rows) / (len(dates) * HOURS_PER_DAY), 2),
        "total_size_bytes": total_bytes,
        "total_size_gib": round(total_bytes / 1024**3, 2),
        "list_errors": errors,
    }
    return rows, coverage


def write_outputs(rows: List[dict], coverage: dict, out_dir: str, coin: str,
                  start: str, end: str) -> Tuple[Path, Path]:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    stem = f"{coin}_{start}_{end}"
    csv_path = out / f"{stem}_manifest.csv"
    json_path = out / f"{stem}_coverage.json"
    with csv_path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["date", "hour", "present", "size_bytes", "key"])
        w.writeheader()
        w.writerows(rows)
    with json_path.open("w") as f:
        json.dump(coverage, f, indent=2)
    return csv_path, json_path


def main() -> None:
    p = argparse.ArgumentParser(description="Stage 0 - L2 availability manifest")
    p.add_argument("--coin", default="BTC")
    p.add_argument("--start", required=True, help="YYYY-MM-DD")
    p.add_argument("--end", required=True, help="YYYY-MM-DD")
    p.add_argument("--out", required=True, help="output dir (keep outside the repo)")
    p.add_argument("--workers", type=int, default=16)
    args = p.parse_args()

    rows, coverage = build_manifest(args.start, args.end, args.coin, workers=args.workers)
    csv_path, json_path = write_outputs(rows, coverage, args.out, args.coin, args.start, args.end)

    c = coverage
    print(f"manifest: {csv_path}")
    print(f"coverage: {json_path}")
    print(f"  days in range (excl. skips): {c['days_in_range_excluding_skips']} "
          f"(+{c['days_skipped']} skipped)")
    print(f"  days with data: {c['days_with_data']} | fully covered: {c['days_fully_covered']} "
          f"| missing: {len(c['days_missing'])} | partial: {len(c['days_partial'])}")
    print(f"  hours: {c['hours_present']}/{c['hours_expected']} ({c['coverage_pct']}%)")
    print(f"  total download size: {c['total_size_gib']} GiB compressed")
    if c["list_errors"]:
        print(f"  WARNING: {len(c['list_errors'])} dates failed to list")
    if c["days_missing"]:
        print(f"  missing days: {c['days_missing'][:10]}{' ...' if len(c['days_missing'])>10 else ''}")


if __name__ == "__main__":
    main()
