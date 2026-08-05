"""December 2025 traded volume for BTC, by date and by hour of day.

WHY THIS EXISTS. Figure D3 grounds every order size used in the dissertation against the
venue's actual liquidity, and the participation contrast it carries -- a 25 BTC order is
0.083% of a day but a much larger share of the five minutes it executes in -- cannot be
stated without an intraday volume profile. No such profile existed: the only volume artefact
in the project is a DAILY series from the /info API (adv/btc_adv.json), which cannot resolve
hour of day.

WHAT IT MEASURES. Every BTC trade printed in the venue's own trade stream for December 2025,
summed by UTC date and by UTC hour of day. The hour comes from each record's own timestamp,
not from the filename, because the archive's hourly files are not aligned to the hour they
are named for (a file named 14 opens with a record stamped 13:59:59).

INTEGRITY. The stream is multi-asset. Only records whose coin field is exactly BTC are
counted, and the count of skipped non-BTC records is reported so that a filter failure would
be visible rather than silent. Each trade is one printed record, so summing sz counts the
traded quantity once, not once per counterparty.

Source: scratch_hyperliquid/oxford_l4/trades_extract/<YYYYMMDD>/<H>.gz
Output: scratch_hyperliquid/oxford_l4/trade_volume_202512.json
"""
from __future__ import annotations

import gzip
import json
from collections import defaultdict
from pathlib import Path

ROOT = Path("/Users/fardeenidrus/Desktop/MSc Dissertation/scratch_hyperliquid/oxford_l4")
SRC = ROOT / "trades_extract"
OUT = ROOT / "trade_volume_202512.json"

BTC_PREFIX = '{"coin":"BTC"'
SZ_KEY = '"sz":"'
TIME_KEY = '"time":"'


def main() -> None:
    by_date: dict[str, float] = defaultdict(float)
    by_hour: dict[int, float] = defaultdict(float)
    trades_by_hour: dict[int, int] = defaultdict(int)
    n_btc = 0
    n_other = 0
    n_files = 0
    truncated: list[dict] = []
    by_date_hour: dict[tuple[str, int], float] = defaultdict(float)

    for day_dir in sorted(SRC.iterdir()):
        if not day_dir.is_dir():
            continue
        for f in sorted(day_dir.iterdir()):
            if f.suffix != ".gz":
                continue
            n_files += 1
            # A truncated archive member is a property of the venue's archive, not of this
            # read. Keep the records that decompressed cleanly and record the file, so that
            # a partial hour is visible in the output rather than silently averaged away.
            try:
                with gzip.open(f, "rt") as fh:
                    for line in fh:
                        if not line.startswith(BTC_PREFIX):
                            n_other += 1
                            continue
                        i = line.find(SZ_KEY)
                        j = line.find('"', i + len(SZ_KEY))
                        sz = float(line[i + len(SZ_KEY):j])
                        t = line.find(TIME_KEY)
                        stamp = line[t + len(TIME_KEY):t + len(TIME_KEY) + 13]
                        date = stamp[:10]
                        hour = int(stamp[11:13])
                        by_date[date] += sz
                        by_hour[hour] += sz
                        trades_by_hour[hour] += 1
                        by_date_hour[(date, hour)] += sz
                        n_btc += 1
            except (EOFError, OSError) as exc:
                truncated.append({"file": str(f.relative_to(SRC)), "error": type(exc).__name__})

    dates = sorted(by_date)
    daily = [by_date[d] for d in dates]

    # The archive's hourly folders spill either side of the month: a file named for the first
    # hour of 1 January carries records stamped 31 December, and vice versa. Every December
    # statistic below is therefore restricted to dates inside December, and the spill dates are
    # reported separately rather than averaged in. Averaging them in would understate the daily
    # mean by counting two partial days as whole ones.
    dec = sorted(d for d in dates if d.startswith("2025-12"))
    dec_daily = [by_date[d] for d in dec]
    dec_hour = defaultdict(float)
    for (d, h), v in by_date_hour.items():
        if d.startswith("2025-12"):
            dec_hour[h] += v
    dec_total = sum(dec_daily)

    out = {
        "december_only": {
            "n_dates": len(dec),
            "date_first": dec[0],
            "date_last": dec[-1],
            "total_btc": dec_total,
            "mean_daily_btc": dec_total / len(dec),
            "median_daily_btc": sorted(dec_daily)[len(dec_daily) // 2],
            "min_daily_btc": min(dec_daily),
            "max_daily_btc": max(dec_daily),
            "by_date_btc": {d: by_date[d] for d in dec},
            "by_hour_of_day_btc": {str(h): dec_hour[h] for h in range(24)},
            "mean_btc_per_hour_of_day": {str(h): dec_hour[h] / len(dec) for h in range(24)},
        },
        "spill_dates_excluded": {d: by_date[d] for d in dates if not d.startswith("2025-12")},
        "source": str(SRC),
        "n_files_read": n_files,
        "n_files_truncated": len(truncated),
        "truncated_files": truncated,
        "n_btc_trades": n_btc,
        "n_non_btc_records_skipped": n_other,
        "n_dates": len(dates),
        "date_first": dates[0],
        "date_last": dates[-1],
        "total_btc": sum(daily),
        "mean_daily_btc": sum(daily) / len(daily),
        "median_daily_btc": sorted(daily)[len(daily) // 2],
        "min_daily_btc": min(daily),
        "max_daily_btc": max(daily),
        "by_date_btc": {d: by_date[d] for d in dates},
        "by_hour_of_day_btc": {str(h): by_hour[h] for h in range(24)},
        "by_hour_of_day_trades": {str(h): trades_by_hour[h] for h in range(24)},
    }
    OUT.write_text(json.dumps(out, indent=2))
    print(f"wrote {OUT}")
    print(f"files {n_files}  BTC trades {n_btc:,}  non-BTC skipped {n_other:,}")
    print(f"all dates seen {len(dates)} ({dates[0]} to {dates[-1]}); "
          f"spill excluded: {out['spill_dates_excluded']}")
    d = out["december_only"]
    print(f"DECEMBER {d['n_dates']} days: mean {d['mean_daily_btc']:,.1f} BTC/day  "
          f"median {d['median_daily_btc']:,.1f}  "
          f"min {d['min_daily_btc']:,.1f}  max {d['max_daily_btc']:,.1f}")
    if truncated:
        print(f"TRUNCATED ARCHIVE MEMBERS: {truncated}")


if __name__ == "__main__":
    main()
