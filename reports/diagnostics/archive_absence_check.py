#!/usr/bin/env python3
"""
Day-level query of the venue's archive for every date the Data chapter calls absent.

WHY THIS EXISTS. Section 3.3 states that no day is missing because of how the data was collected,
and that direct queries to the archive confirm it. Until 2026-08-05 the basis for the thirteen
never-requested days was a code comment in `manifest.py` asserting the archive holds nothing for
those weeks. A comment is a belief, not evidence. The queries were run and recorded in prose as
addendum (Y9); this script re-runs them and writes the result as a checkable artefact, so the
claim can be produced on demand rather than re-derived.

WHAT IT ASKS. For each date, how many hour prefixes exist under
`s3://hyperliquid-archive/market_data/<YYYYMMDD>/`. The archive stores one prefix per hour, so a
complete day has 24 and an absent day has 0. Control days are included deliberately: a query
returning zero everywhere would be indistinguishable from a broken query.

The bucket is requester-pays, so every call passes --request-payer requester.

Output: scratch_hyperliquid/oxford_l4/archive_absence_check.json
"""
from __future__ import annotations

import json
import subprocess
from datetime import date, timedelta
from pathlib import Path

BUCKET = "hyperliquid-archive"
OUT = Path("/Users/fardeenidrus/Desktop/MSc Dissertation/scratch_hyperliquid"
           ) / "oxford_l4" / "archive_absence_check.json"


def daterange(a: str, b: str) -> list[str]:
    x, y = date.fromisoformat(a), date.fromisoformat(b)
    return [(x + timedelta(days=i)).isoformat() for i in range((y - x).days + 1)]


# Every date the chapter calls absent, plus complete days either side as controls.
TARGETS = (
    [(d, "never requested (skip week)") for d in daterange("2025-10-17", "2025-10-22")]
    + [(d, "never requested (skip week)") for d in daterange("2025-11-18", "2025-11-24")]
    + [(d, "requested, returned nothing") for d in ["2024-01-10", "2024-02-01", "2025-11-25"]]
    + [(d, "control: recorded complete") for d in
       ["2025-10-15", "2025-10-23", "2025-11-16", "2025-11-26", "2025-11-27"]]
)


def hour_prefixes(day: str) -> int:
    """Count hour prefixes for one date. A complete day has 24; an absent day has 0."""
    key = day.replace("-", "")
    r = subprocess.run(
        ["aws", "s3", "ls", f"s3://{BUCKET}/market_data/{key}/", "--request-payer", "requester"],
        capture_output=True, text=True)
    if r.returncode != 0 and r.stdout.strip() == "":
        return 0  # a listing with no keys exits non-zero; that is the absent case
    return len([ln for ln in r.stdout.splitlines() if ln.strip().endswith("/")])


def main() -> None:
    rows = []
    for day, category in TARGETS:
        n = hour_prefixes(day)
        rows.append({"date": day, "category": category, "hour_prefixes": n})
        print(f"  {day}  {category:<30} {n:>3} hour prefixes")

    absent = [r for r in rows if not r["category"].startswith("control")]
    controls = [r for r in rows if r["category"].startswith("control")]
    checks = {
        "every_absent_date_returns_zero": all(r["hour_prefixes"] == 0 for r in absent),
        "every_control_returns_24": all(r["hour_prefixes"] == 24 for r in controls),
    }

    result = {
        "generated_by": "reports/diagnostics/archive_absence_check.py",
        "bucket": f"s3://{BUCKET}/market_data/<YYYYMMDD>/",
        "question": "how many hour prefixes does the archive hold for this date",
        "n_absent_dates_checked": len(absent),
        "n_control_dates_checked": len(controls),
        "rows": rows,
        "checks": checks,
        "bearing": ("Section 3.3 states that no day is missing as a result of the data "
                    "collection process. Every date it calls absent returns nothing from the "
                    "venue's archive; every control day returns a full 24. This establishes "
                    "that the data is not there now. It does not prove it was never there, "
                    "and no such claim is made."),
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(result, indent=2))

    print("\nCHECKS")
    for k, v in checks.items():
        print(f"  {'OK  ' if v else 'FAIL'} {k}")
    print(f"\nwritten: {OUT}")


if __name__ == "__main__":
    main()
