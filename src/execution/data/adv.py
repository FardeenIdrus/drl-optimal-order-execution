"""Average daily volume (ADV) for order sizing, from the Hyperliquid candle API.

Daily (``1d``) candles are free from the Hyperliquid ``info`` endpoint and go back
years (only *minute* volume is capped), so ADV in BTC is obtainable. The meta-order
is sized as a small percentage of ADV -- the academic standard, since the
square-root impact law is expressed in ``Q / V`` terms (order size over daily
volume). The chosen size is cross-checked against typical **volatile-regime** book
depth (:func:`crosscheck_volatile_depth`) so per-step slices fit the thinner books
and the deadline residual stays rare.

The candle ``v`` field is volume in the **base coin** (BTC), not quote (USD): a
BTC day shows ``v`` on the order of 1e4, which at a ~1e5 USD price is ~1e9 USD of
notional -- so ``v`` is BTC. :func:`compute_adv` averages it directly.

This is a one-time network pull cached to ``<scratch>/adv/btc_adv.json``; nothing
downstream calls the network. Run:

    PYTHONPATH=src .venv/bin/python -m execution.data.adv \
        --scratch-root "<scratch_hyperliquid>" --config configs/pipeline.yaml
"""
from __future__ import annotations

import argparse
import json
import logging
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

logger = logging.getLogger(__name__)

INFO_URL = "https://api.hyperliquid.xyz/info"
DAY_MS = 86_400_000
N_LEVELS = 20  # full visible ask ladder in the dataset (executable depth)


# --------------------------------------------------------------------- helpers
def date_to_ms(date_str: str) -> int:
    """UTC midnight of a ``YYYY-MM-DD`` string, in epoch milliseconds."""
    dt = datetime.strptime(date_str, "%Y-%m-%d").replace(tzinfo=timezone.utc)
    return int(dt.timestamp() * 1000)


def parse_candles(raw: Any) -> list[dict]:
    """Validate a raw candle-snapshot response into ``[{date, volume_btc, ts}]``.

    Pure (no network) so it is unit-testable. Raises ``ValueError`` on a
    malformed payload (not a list, empty, or a candle missing the volume field),
    per the no-silent-failures rule.
    """
    if not isinstance(raw, list):
        raise ValueError(f"candle response is not a list: {type(raw).__name__}")
    if not raw:
        raise ValueError("candle response is empty")
    out: list[dict] = []
    for c in raw:
        if not isinstance(c, dict) or "v" not in c or "t" not in c:
            raise ValueError(f"malformed candle (missing 't'/'v'): {c!r}")
        ts = int(c["t"])
        out.append({
            "ts": ts,
            "date": datetime.fromtimestamp(ts / 1000, tz=timezone.utc).strftime("%Y-%m-%d"),
            "volume_btc": float(c["v"]),
        })
    return out


def compute_adv(candles: list[dict]) -> dict:
    """Average daily volume (BTC) and provenance from parsed daily candles."""
    if not candles:
        raise ValueError("no candles to average")
    vols = np.asarray([c["volume_btc"] for c in candles], dtype=float)
    if not np.all(np.isfinite(vols)) or np.any(vols < 0):
        raise ValueError("daily volumes contain non-finite or negative values")
    dates = [c["date"] for c in candles]
    return {
        "adv_btc": float(vols.mean()),
        "n_days": int(vols.size),
        "median_daily_btc": float(np.median(vols)),
        "min_daily_btc": float(vols.min()),
        "max_daily_btc": float(vols.max()),
        "date_start": min(dates),
        "date_end": max(dates),
    }


def fetch_daily_candles(
    coin: str,
    start: str,
    end: str,
    *,
    base_url: str = INFO_URL,
    timeout: float = 30.0,
    chunk_days: int = 300,
) -> list[dict]:
    """Pull ``1d`` candles for ``coin`` over ``[start, end]`` (inclusive dates).

    Requests are chunked (``chunk_days``) and de-duplicated by open-time so the
    pull is robust to any server-side result cap; the daily granularity over a
    couple of years is well within limits regardless.
    """
    start_ms, end_ms = date_to_ms(start), date_to_ms(end) + DAY_MS - 1
    by_ts: dict[int, dict] = {}
    win_start = start_ms
    while win_start <= end_ms:
        win_end = min(win_start + chunk_days * DAY_MS, end_ms)
        body = json.dumps({
            "type": "candleSnapshot",
            "req": {"coin": coin, "interval": "1d", "startTime": win_start, "endTime": win_end},
        }).encode()
        req = urllib.request.Request(
            base_url, data=body, headers={"Content-Type": "application/json"})
        logger.info("candle pull %s..%s",
                    datetime.fromtimestamp(win_start / 1000, tz=timezone.utc).date(),
                    datetime.fromtimestamp(win_end / 1000, tz=timezone.utc).date())
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            raw = json.load(resp)
        for c in parse_candles(raw):
            by_ts[c["ts"]] = c
        win_start = win_end + 1
    candles = [by_ts[t] for t in sorted(by_ts)]
    logger.info("fetched %d distinct daily candles", len(candles))
    return candles


def crosscheck_volatile_depth(
    df: pd.DataFrame,
    sizes_btc: list[float],
    *,
    n_steps: int = 30,
    n_levels: int = N_LEVELS,
    top_n: int = 5,
) -> pd.DataFrame:
    """Cross-check candidate order sizes against **volatile-regime** book depth.

    For the volatile episodes only (the thinner books -- the binding case), this
    measures whether each candidate order is workable within depth over the
    horizon. ``df`` is the per-step dataset frame (columns ``regime`` and
    ``ask_sz_1..ask_sz_{n_levels}``); execution walks all ``n_levels`` so full
    visible depth is the executability measure (``top_n`` depth is reported too
    for context, since it is the ``ask_depth`` feature).

    Returns one row per candidate size with: the TWAP-paced per-minute slice, the
    median/p10 of volatile per-step full depth, the share of volatile steps whose
    slice exceeds that step's visible depth, and the share of volatile *episodes*
    whose cumulative 30-step depth is below the order (a loose lower-bound proxy:
    a paced scheduler that cannot borrow future depth realizes a higher residual
    rate, so measure the realized frequency directly in eval).
    """
    vol = df[df["regime"] == "volatile"]
    if vol.empty:
        raise ValueError("no volatile-regime rows in dataset")
    full_cols = [f"ask_sz_{i}" for i in range(1, n_levels + 1)]
    top_cols = [f"ask_sz_{i}" for i in range(1, top_n + 1)]
    per_step_full = vol[full_cols].to_numpy(dtype=float)
    per_step_depth = np.nansum(per_step_full, axis=1)          # (rows,) full depth
    per_step_top = np.nansum(vol[top_cols].to_numpy(dtype=float), axis=1)
    # cumulative depth per volatile episode (sum of per-step full depth over its steps)
    cum_by_ep = vol.assign(_d=per_step_depth).groupby("episode_id")["_d"].sum().to_numpy()

    med, p10 = float(np.median(per_step_depth)), float(np.percentile(per_step_depth, 10))
    med_top = float(np.median(per_step_top))
    rows = []
    for size in sizes_btc:
        slice_btc = size / n_steps
        rows.append({
            "size_btc": round(size, 4),
            "slice_btc_per_min": round(slice_btc, 4),
            "vol_depth_median_btc": round(med, 3),
            "vol_depth_p10_btc": round(p10, 3),
            "vol_top5_depth_median_btc": round(med_top, 3),
            "pct_steps_slice_gt_depth": round(100.0 * np.mean(slice_btc > per_step_depth), 3),
            "pct_episodes_cum_lt_order": round(100.0 * np.mean(cum_by_ep < size), 3),
        })
    return pd.DataFrame(rows)


# ------------------------------------------------------------------------ CLI
def pull_and_cache(coin: str, start: str, end: str, out_path: Path,
                   *, base_url: str = INFO_URL) -> dict:
    """Fetch daily candles, compute ADV, and write the cached JSON record."""
    candles = fetch_daily_candles(coin, start, end, base_url=base_url)
    adv = compute_adv(candles)
    record = {
        "coin": coin,
        "interval": "1d",
        "window": {"start": start, "end": end},
        "pulled_at": datetime.now(tz=timezone.utc).isoformat(),
        "source": base_url,
        **adv,
        "daily_volume_btc": {c["date"]: c["volume_btc"] for c in candles},
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(record, indent=2))
    logger.info("wrote ADV record -> %s (adv=%.1f BTC over %d days)",
                out_path, adv["adv_btc"], adv["n_days"])
    return record


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    ap = argparse.ArgumentParser(description="Pull Hyperliquid ADV and size the meta-order")
    ap.add_argument("--scratch-root", required=True, help="path to scratch_hyperliquid/")
    ap.add_argument("--config", default="configs/pipeline.yaml", help="for coin/start/end")
    ap.add_argument("--coin", default=None, help="override config coin")
    ap.add_argument("--pct-grid", default="0.1,0.25,0.5,1.0",
                    help="candidate order sizes as %% of ADV (comma-separated)")
    ap.add_argument("--refresh", action="store_true", help="re-pull even if cached")
    args = ap.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text())
    coin = args.coin or cfg["coin"]
    start, end = cfg["start"], cfg["end"]
    scratch = Path(args.scratch_root)
    out_path = scratch / "adv" / f"{coin.lower()}_adv.json"

    if out_path.exists() and not args.refresh:
        record = json.loads(out_path.read_text())
        logger.info("using cached ADV %s (%.1f BTC); --refresh to re-pull",
                    out_path, record["adv_btc"])
    else:
        record = pull_and_cache(coin, start, end, out_path)

    adv = record["adv_btc"]
    pct_grid = [float(p) for p in args.pct_grid.split(",")]
    sizes = [pct / 100.0 * adv for pct in pct_grid]
    print(f"\nADV ({coin}) = {adv:,.1f} BTC  "
          f"[{record['n_days']} days {record['window']['start']}..{record['window']['end']}, "
          f"median {record['median_daily_btc']:,.0f}]")
    print("Candidate sizes (% of ADV -> BTC):")
    for pct, sz in zip(pct_grid, sizes):
        print(f"  {pct:>5.2f}%  ->  {sz:8.2f} BTC")

    ds = scratch / "dataset" / "train.parquet"
    cols = ["regime", "episode_id"] + [f"ask_sz_{i}" for i in range(1, N_LEVELS + 1)]
    df = pd.read_parquet(ds, columns=cols)
    cc = crosscheck_volatile_depth(df, sizes, n_steps=cfg["regimes"]["episode_minutes"])
    cc.insert(0, "pct_adv", pct_grid)
    print("\nVolatile-regime depth cross-check (executability over the 30-min horizon):")
    print(cc.to_string(index=False))
    print("\nReading: a workable primary has slice << volatile depth, "
          "pct_steps_slice_gt_depth ~ 0, and pct_episodes_cum_lt_order ~ 0.")


if __name__ == "__main__":
    main()
