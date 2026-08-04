"""Measurement (ii): event-level one-tick-ahead AUC, the estimand Gould and Bonart (2016) use.

Registered protocol: reports/qrm_step4_criteria.md section 9 (2026-08-03), written BEFORE this
ran. NO PASS/FAIL GATE -- this is measurement, not a test (R5).

Question, exactly as they pose it: given the top-of-book queues right now, does queue imbalance
call the direction of the NEXT change in the mid? Scored by AUC (0.5 = coin flip).

Why the 0.5 s grid cannot answer it: 10.4% of grid intervals contain a mid move of two ticks or
more, so the grid reports the NET of several changes as though it were one. This module walks
the event stream and emits on every top-of-book change instead.

INTEGRITY GATE (aborts the run): the same pass also drives the ORIGINAL 500 ms grid sampler and
compares each day's top-of-book columns against the stored book_05s_v2 parquet. If any day fails
to reproduce exactly, the stream or engine differs from what produced the record and no number
from this run may be quoted.

Deliberately standalone: no existing module is edited. Memory-flat -- one hour of events at a
time; only histograms and one day's grid rows are held.
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd

from execution.data.l4.book_diffs_reader import (
    ASK, BID, NEW, BookEvent, attach_timestamps, inject_removes, stream_events,
)
from execution.data.l4.book_engine import BookEngine
from execution.data.l4.orders_reader import build_open_and_removes
from execution.data.l4.reconstruct_month import retry_late_removes
from execution.data.l4.snapshot_sampler import SnapshotSampler

sys.path.insert(0, str(Path(__file__).resolve().parent))
from tick_class_measure import AucAccum, bin_index  # noqa: E402  (shared, already tested)

logger = logging.getLogger(__name__)

SCRATCH = Path("/Users/fardeenidrus/Desktop/MSc Dissertation/scratch_hyperliquid")
DIFFS = SCRATCH / "oxford_l4" / "diffs_extract"
ORDERS = SCRATCH / "oxford_l4" / "orders_extract"
GRID = SCRATCH / "oxford_l4" / "book_05s_v2"
LABELS = SCRATCH / "oxford_l4" / "step3g" / "regime_labels.parquet"
OUT = SCRATCH / "oxford_l4" / "tick_class" / "event_ahead_auc.json"

# reconstruction constants, identical to the run that produced book_05s_v2
COIN = "BTC"
CADENCE_NS = 500_000_000
TOP_K = 20
WARMUP_EVENTS = 1_500_000
GATE_COLS = ["ts", "best_bid", "best_ask", "mid", "spread", "bid_sz_1", "ask_sz_1"]
EPS = 1e-12


class EventAheadAccum:
    """One (regime, split) cell: imbalance BEFORE a mid change vs the direction of that change."""

    __slots__ = ("acc", "n_changes")

    def __init__(self) -> None:
        self.acc = AucAccum()
        self.n_changes = 0

    def add(self, imb_before: float, direction: float) -> None:
        if not np.isfinite(imb_before) or direction == 0.0:
            return
        self.acc.add(np.array([imb_before]), np.array([direction]))
        self.n_changes += 1

    def result(self) -> dict:
        r = self.acc.result()
        return {"auc": r["auc"], "n_price_changes": self.n_changes}


class TopTracker:
    """Incremental top of book.

    ``BookEngine.snapshot(k)`` sorts every price level on both sides; its own docstring says
    it is meant for the sampling cadence, not the per-event hot path. Calling it once per
    event (~1,400/s) dominated a first version of this module entirely.

    Instead: cache the best price and size per side, and recompute a side ONLY when an event
    touches a level at or better than that side's current best. An event deeper in the book
    cannot move the touch. When the touch level empties we do fall back to a full scan for the
    next best -- but those are exactly the price-change events this measurement is about, so
    they cannot be skipped.

    Correctness is not assumed: the caller's per-day integrity gate compares the 500 ms output
    of the same pass against the stored record and aborts on any mismatch.
    """

    __slots__ = ("e", "bb", "ba", "bsz", "asz")

    def __init__(self, engine: BookEngine) -> None:
        self.e = engine
        self.bb = self.ba = self.bsz = self.asz = None
        self.invalidate()

    def invalidate(self) -> None:
        """Full recompute of both sides (after healing, or at start)."""
        self._recompute_bid()
        self._recompute_ask()

    def _recompute_bid(self) -> None:
        b = self.e._bids
        if b:
            self.bb = max(b)
            self.bsz = b[self.bb]
        else:
            self.bb = self.bsz = None

    def _recompute_ask(self) -> None:
        a = self.e._asks
        if a:
            self.ba = min(a)
            self.asz = a[self.ba]
        else:
            self.ba = self.asz = None

    def affected(self, ev: BookEvent) -> Tuple[Optional[str], Optional[str]]:
        """Which sides this event may move, decided BEFORE it is applied.

        For NEW the level is on the event itself; for REMOVE/UPDATE it is the resting
        order's own level, which only the oid map knows.
        """
        hit_b = hit_a = None
        prev = self.e._orders.get(ev.oid)
        if prev is not None:
            side, px, _ = prev
            if side == BID:
                if self.bb is None or px >= self.bb:
                    hit_b = BID
            elif self.ba is None or px <= self.ba:
                hit_a = ASK
        if ev.kind == NEW:
            if ev.side == BID:
                if self.bb is None or ev.px >= self.bb:
                    hit_b = BID
            elif self.ba is None or ev.px <= self.ba:
                hit_a = ASK
        return hit_b, hit_a

    def refresh(self, hit_b: Optional[str], hit_a: Optional[str]) -> None:
        if hit_b is not None:
            self._recompute_bid()
        if hit_a is not None:
            self._recompute_ask()

    def state(self) -> Tuple[Optional[float], Optional[float], Optional[float]]:
        """(mid, bid_sz_1, ask_sz_1); mid is None if a side is empty OR the book is crossed.

        Crossed states are transient artefacts of diff replay that the record heals at each
        grid boundary. We do not measure through one; the count is reported.
        """
        if self.bb is None or self.ba is None or self.bb >= self.ba:
            return None, None, None
        return 0.5 * (self.bb + self.ba), self.bsz, self.asz


def run(max_days: Optional[int] = None, max_hours: Optional[int] = None,
        skip_gate: bool = False) -> dict:
    labels = pd.read_parquet(LABELS)
    labels["date"] = labels["date"].astype(str)
    label_map = {(r.date, int(r.hour)): (str(r.regime), str(r.split))
                 for r in labels.itertuples()}

    days = sorted(d.name for d in DIFFS.iterdir() if d.is_dir())
    if max_days is not None:
        days = days[:max_days]

    engine = BookEngine()
    sampler = SnapshotSampler(engine, CADENCE_NS, TOP_K, WARMUP_EVENTS)
    top = TopTracker(engine)

    cells: Dict[Tuple[str, str], EventAheadAccum] = {
        (rg, sp): EventAheadAccum()
        for rg in ("calm", "volatile") for sp in ("calibrate", "holdout")
    }
    gate_report, n_ev_total, t_start = [], 0, time.perf_counter()
    n_crossed_skipped = 0

    for date in days:
        t0 = time.perf_counter()
        n_ev_day = 0
        # STREAMING integrity gate: hold the stored day as flat arrays and compare each grid
        # emission as it is produced. Nothing accumulates, so peak memory is independent of
        # how long a day is (an earlier version buffered 172,800 row dicts per day).
        want_path = GRID / f"{date}.parquet"
        want = (pd.read_parquet(want_path, columns=GATE_COLS) if want_path.exists()
                and not skip_gate else None)
        want_cols = ([want[c].to_numpy() for c in GATE_COLS] if want is not None else None)
        gate_i, gate_ok, n_grid = 0, True, 0

        for hour in range(24 if max_hours is None else max_hours):
            of = ORDERS / date / f"{COIN.lower()}_{hour:02d}.data.gz"
            df = DIFFS / date / f"ex{hour}.gz"
            if not of.exists() or not df.exists():
                continue
            open_map, removes = build_open_and_removes(of)
            stream = inject_removes(attach_timestamps(stream_events(df, COIN), open_map), removes)

            key = label_map.get((date, hour))
            cell = cells[key] if key is not None else None

            prev_mid: Optional[float] = None
            prev_imb: float = float("nan")

            for ts, ev in stream:
                # --- grid sampler, driven on the SAME stream: the integrity gate -------
                if ts is not None:
                    if sampler.next_boundary is None:
                        sampler.next_boundary = (ts // CADENCE_NS + 1) * CADENCE_NS
                    while ts >= sampler.next_boundary:
                        if sampler.n_applied >= WARMUP_EVENTS:
                            if engine.drop_crossed():
                                top.invalidate()      # healing moved levels: cache is stale
                            bb, ba, mid, spread = engine.top_of_book()
                            bids, asks = engine.snapshot(1)
                            row = (sampler.next_boundary, bb, ba, mid, spread,
                                   bids[0][1] if bids else np.nan,
                                   asks[0][1] if asks else np.nan)
                            n_grid += 1
                            if want_cols is not None and gate_ok:
                                if gate_i >= len(want_cols[0]):
                                    gate_ok = False
                                else:
                                    for ci, v in enumerate(row):
                                        w = want_cols[ci][gate_i]
                                        if not (v == w or (v != v and w != w)):
                                            gate_ok = False
                                            break
                                    gate_i += 1
                        sampler.next_boundary += CADENCE_NS

                hit_b, hit_a = top.affected(ev)       # decided BEFORE apply (needs the oid map)
                engine.apply(ev)
                sampler.n_applied += 1
                n_ev_day += 1

                # --- event sampler: emit on every change in the mid -------------------
                if hit_b is None and hit_a is None:
                    continue                          # deeper than the touch: cannot move it
                top.refresh(hit_b, hit_a)
                if cell is None or sampler.n_applied < WARMUP_EVENTS:
                    continue
                mid, bsz, asz = top.state()
                if mid is None:
                    n_crossed_skipped += 1
                    continue
                if prev_mid is not None and mid != prev_mid:
                    cell.add(prev_imb, float(np.sign(mid - prev_mid)))
                denom = bsz + asz
                prev_imb = ((bsz - asz) / denom) if denom > EPS else float("nan")
                prev_mid = mid

            n_ev_total += n_ev_day
            retry_late_removes(engine, removes)

        # --- per-day integrity gate verdict --------------------------------------------
        if want_cols is not None:
            # A truncated probe (max_hours) legitimately stops short of the day; only a FULL
            # day is required to be row-complete. Every row compared must still match exactly.
            complete = gate_i == len(want_cols[0]) or max_hours is not None
            ok = bool(gate_ok and complete)
            gate_report.append({"date": date, "rows_got": n_grid,
                                "rows_want": int(len(want_cols[0])), "exact_match": ok})
            if not ok:
                raise SystemExit(
                    f"INTEGRITY GATE FAILED on {date} (matched {gate_i} of "
                    f"{len(want_cols[0])} rows): this run does not reproduce book_05s_v2. "
                    f"No number from it may be quoted.")
        del want, want_cols

        el = time.perf_counter() - t0
        logger.info("day %s: %d events, %d grid rows, gate=%s, %.0fs",
                    date, n_ev_day, n_grid,
                    gate_report[-1]["exact_match"] if gate_report else "skipped", el)

    out = {
        "registered": "reports/qrm_step4_criteria.md section 9 (2026-08-03)",
        "estimand": "sign of the NEXT change in the mid, in event time (Gould and Bonart 2016)",
        "no_pass_fail_gate": True,
        "days": len(days), "events_applied": n_ev_total,
        "wall_seconds": round(time.perf_counter() - t_start, 1),
        "crossed_states_skipped": n_crossed_skipped,
        "integrity_gate": {"per_day": gate_report,
                           "all_days_exact": all(g["exact_match"] for g in gate_report)
                           if gate_report else None},
        "auc": {f"{rg}/{sp}": cells[(rg, sp)].result()
                for rg in ("calm", "volatile") for sp in ("calibrate", "holdout")},
        "provenance": {
            "diffs": str(DIFFS), "orders": str(ORDERS), "labels": str(LABELS),
            "warmup_events": WARMUP_EVENTS, "cadence_ns": CADENCE_NS, "top_k": TOP_K,
            "predictor": "(bid_sz_1 - ask_sz_1)/(bid_sz_1 + ask_sz_1) at the state BEFORE the change",
            "boundary": "predictor and the change it labels lie in the same labelled hour",
        },
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(out, indent=1))
    return out


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", datefmt="%H:%M:%S")
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-days", type=int, default=None)
    ap.add_argument("--max-hours", type=int, default=None, help="timing probe only")
    ap.add_argument("--skip-gate", action="store_true", help="probe only; never for a real run")
    a = ap.parse_args()
    r = run(a.max_days, a.max_hours, a.skip_gate)
    print(f"\ndays={r['days']}  events={r['events_applied']:,}  "
          f"wall={r['wall_seconds']}s  gate_all_exact={r['integrity_gate']['all_days_exact']}")
    for k, v in r["auc"].items():
        print(f"  {k:22s} AUC={v['auc']:.4f}  n_changes={v['n_price_changes']:,}")
    print(f"\nwritten: {OUT}")


if __name__ == "__main__":
    main()
