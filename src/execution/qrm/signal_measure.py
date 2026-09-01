"""Measured-signal extension, Phase A1: measure the order-flow-imbalance to future-return
relationship in the real December 2025 data (per regime, calibrate/holdout chronological
split), plus the perfect-foresight stop/go bound.

Registered protocol: the dated addendum in reports/qrm_step4_criteria.md. All constants below are registered; nothing is
tunable from the command line by design.

Candidates (computed causally on the 0.5 s reconstructed book grid):
  S1  trade-flow imbalance: signed market-order volume over the trailing 1 s window
      (buy-aggressor positive), normalised by its trailing rolling std.
  S2  top-of-book depth imbalance: (bid_sz_1 - ask_sz_1) / (bid_sz_1 + ask_sz_1),
      same normalisation treatment.

Outputs one JSON with every regression (both candidates, both regimes, both splits, all
horizons), the held-out confirmation verdicts, the registered selection outcome, the
perfect-foresight bound, and full provenance. Streams day by day; RAM-flat.
"""
from __future__ import annotations

import argparse
import json
import logging
import math
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

from execution.data.l4.book_diffs_reader import ASK
from execution.data.l4.trades_reader import read_trades

logger = logging.getLogger(__name__)

# ----------------------------------------------------------------- registered constants
GRID_MS = 500                    # book grid cadence
S1_WINDOW_BINS = 2               # trailing 1 s of signed trade flow (two 0.5 s bins)
NORM_BINS = 600                  # rolling causal std window: 5 minutes of bins
NORM_MIN_BINS = 120              # warm-up: signal undefined for the first minute of a day
HORIZONS_S: Dict[str, int] = {   # horizon label -> forward steps on the 0.5 s grid
    "0.5": 1, "1": 2, "2": 4, "5": 10, "10": 20, "30": 60, "60": 120,
}
SELECT_HORIZON = "1"             # selection rule: holdout R^2 at 1 s, volatile regime
SELECT_TIE_REL = 0.10            # ties within 10% relative go to S1
CONFIRM_P = 0.01                 # held-out confirmation: p < 0.01 ...
CONFIRM_FRACTION = 0.5           # ... same sign, >= half the calibrate-split magnitude
DECISION_STEPS = 2               # decisions every 1 s (two grid steps), as in the env
EPISODE_DECISIONS = 300          # 5-minute episode at 1 s decisions (primary setting)
FAST_PACE = 2.0                  # foresight-bound schedule paces (clip of the action set)
MATERIALITY_BPS = 0.05           # registered stop/go floor
EPS = 1e-12
MIN_STD = 1e-9                   # below this the signal is undefined (amendment 1a)


# ----------------------------------------------------------------- pure helpers (tested)
def bin_signed_trades(grid_ts: np.ndarray, trades: List[Tuple[int, str, float, float]]
                      ) -> np.ndarray:
    """Signed market-order volume per grid bin. Bin i covers (grid_ts[i-1], grid_ts[i]];
    trades at or before grid_ts[0] fall into bin 0. Buy-aggressor (resting side ASK) is
    positive; sell-aggressor negative."""
    out = np.zeros(len(grid_ts), dtype=np.float64)
    if not trades:
        return out
    ts = np.array([t[0] for t in trades], dtype=np.int64)
    signed = np.array([t[3] if t[1] == ASK else -t[3] for t in trades], dtype=np.float64)
    idx = np.searchsorted(grid_ts, ts, side="left")
    idx = np.clip(idx, 0, len(grid_ts) - 1)
    np.add.at(out, idx, signed)
    return out


def trailing_sum(raw: np.ndarray, window: int) -> np.ndarray:
    """Causal trailing sum over `window` bins (inclusive of the current bin)."""
    c = np.cumsum(raw)
    out = c.copy()
    out[window:] = c[window:] - c[:-window]
    return out


def causal_normalise(raw: np.ndarray, norm_bins: int = NORM_BINS,
                     min_bins: int = NORM_MIN_BINS) -> np.ndarray:
    """raw_t / (rolling std of raw over the PRIOR norm_bins values, excluding t).
    NaN during the warm-up (fewer than min_bins prior values). Strictly causal."""
    s = pd.Series(raw)
    prior_std = s.rolling(norm_bins, min_periods=min_bins).std().shift(1).to_numpy()
    out = np.full(len(raw), np.nan)
    ok = np.isfinite(prior_std) & (prior_std > MIN_STD)
    out[ok] = raw[ok] / prior_std[ok]
    return out


def depth_imbalance(bid_sz1: np.ndarray, ask_sz1: np.ndarray) -> np.ndarray:
    denom = bid_sz1 + ask_sz1
    with np.errstate(divide="ignore", invalid="ignore"):
        raw = (bid_sz1 - ask_sz1) / denom
    raw[~np.isfinite(raw)] = 0.0
    return raw


class RegAccum:
    """Streaming simple-regression accumulator: y = a + b x."""

    __slots__ = ("n", "sx", "sy", "sxx", "syy", "sxy")

    def __init__(self) -> None:
        self.n = 0.0
        self.sx = self.sy = self.sxx = self.syy = self.sxy = 0.0

    def add(self, x: np.ndarray, y: np.ndarray) -> None:
        m = np.isfinite(x) & np.isfinite(y)
        x, y = x[m], y[m]
        self.n += len(x)
        self.sx += float(x.sum())
        self.sy += float(y.sum())
        self.sxx += float((x * x).sum())
        self.syy += float((y * y).sum())
        self.sxy += float((x * y).sum())

    def stats(self) -> Dict[str, float]:
        n = self.n
        if n < 3:
            return {"n": int(n), "slope": float("nan"), "r2": float("nan"),
                    "p": float("nan")}
        vx = self.sxx - self.sx * self.sx / n
        vy = self.syy - self.sy * self.sy / n
        cov = self.sxy - self.sx * self.sy / n
        if vx <= 0 or vy <= 0:
            return {"n": int(n), "slope": 0.0, "r2": 0.0, "p": 1.0}
        slope = cov / vx
        r = cov / math.sqrt(vx * vy)
        r = max(min(r, 1.0), -1.0)
        r2 = r * r
        if r2 >= 1.0:
            p = 0.0
        else:
            t = r * math.sqrt((n - 2) / (1.0 - r2))
            # normal approximation is exact enough at these n (>= thousands)
            p = 2.0 * (1.0 - 0.5 * (1.0 + math.erf(abs(t) / math.sqrt(2.0))))
        return {"n": int(n), "slope": slope, "r2": r2, "p": p}


def foresight_bound(dec_mids: np.ndarray, dec_signals: np.ndarray, slope_1s: float,
                    n_decisions: int = EPISODE_DECISIONS, fast: float = FAST_PACE
                    ) -> List[float]:
    """Per-episode-window cost advantage (bps, positive = cheaper than TWAP) of the
    registered clairvoyant-bound schedule on real mid paths, impact ignored.

    Schedule: pace `fast` when the predicted move (slope_1s * signal) is adverse for a
    buyer (price predicted to rise), else pace 0, subject to completing: once remaining
    inventory equals max feasible buying for the intervals left, buying is forced at
    `fast`. Windows are non-overlapping blocks of n_decisions decisions."""
    advantages: List[float] = []
    n_windows = len(dec_mids) // n_decisions
    unit = 1.0 / n_decisions
    for w in range(n_windows):
        mids = dec_mids[w * n_decisions:(w + 1) * n_decisions]
        sigs = dec_signals[w * n_decisions:(w + 1) * n_decisions]
        if not np.all(np.isfinite(mids)) or not np.all(np.isfinite(sigs)):
            advantages.append(float("nan"))
            continue
        remaining = 1.0
        cost = 0.0
        for i in range(n_decisions):
            left = n_decisions - i
            max_later = (left - 1) * fast * unit
            forced = remaining - max_later > 1e-12
            predicted_rise = slope_1s * sigs[i] > 0.0
            if forced or predicted_rise:
                q = min(fast * unit, remaining)
                # never buy so much that the remainder is unspendable (paces >= 0 ok)
                cost += q * mids[i]
                remaining -= q
            if remaining <= 1e-12:
                remaining = 0.0
                # done early: stop (TWAP still averages the window)
                if i < n_decisions - 1:
                    pass
        if remaining > 1e-9:      # infeasible window (should not happen)
            advantages.append(float("nan"))
            continue
        twap_cost = float(np.mean(mids))
        arrival = float(mids[0])
        advantages.append((twap_cost - cost) / arrival * 1e4)
    return advantages


# ----------------------------------------------------------------- the measurement run
def run_measurement(scratch: Path, out_path: Path) -> Dict:
    book_dir = scratch / "book_05s_v2"
    trades_dir = scratch / "trades_extract"
    labels_path = scratch / "step3g" / "regime_labels.parquet"
    labels = pd.read_parquet(labels_path)
    labels["date"] = labels["date"].astype(str)

    accums: Dict[Tuple[str, str, str, str], RegAccum] = {}
    for cand in ("S1", "S2"):
        for regime in ("calm", "volatile"):
            for split in ("calibrate", "holdout"):
                for h in HORIZONS_S:
                    accums[(cand, regime, split, h)] = RegAccum()

    # collected per (regime): holdout decision-time mids + signals per candidate,
    # for the foresight bound (built after selection)
    bound_store: Dict[Tuple[str, str], List[np.ndarray]] = {}
    hours_used: Dict[Tuple[str, str], int] = {}

    for date, day_labels in labels.groupby("date", sort=True):
        book_file = book_dir / f"{date}.parquet"
        if not book_file.exists():
            logger.warning("missing book day %s; skipping %d labelled hours",
                           date, len(day_labels))
            continue
        day = pd.read_parquet(
            book_file, columns=["ts", "mid", "bid_sz_1", "ask_sz_1"])
        grid_ts = day["ts"].to_numpy(dtype=np.int64)
        mid = day["mid"].to_numpy(dtype=np.float64)

        # raw candidate series over the FULL day (normalisation needs continuity)
        raw_s2 = depth_imbalance(day["bid_sz_1"].to_numpy(np.float64),
                                 day["ask_sz_1"].to_numpy(np.float64))
        flow = np.zeros(len(grid_ts))
        for hour in range(24):
            tf = trades_dir / date / f"{hour}.gz"
            if tf.exists():
                trades = read_trades(tf)
                flow += bin_signed_trades(grid_ts, trades)
        raw_s1 = trailing_sum(flow, S1_WINDOW_BINS)
        # amendment 1b: S2 is bounded and scale-free by construction; used raw
        sig = {"S1": causal_normalise(raw_s1), "S2": raw_s2}

        # forward returns in bps at every horizon (within-day; tail NaN)
        fwd = {}
        for h, steps in HORIZONS_S.items():
            r = np.full(len(mid), np.nan)
            r[:-steps] = (mid[steps:] - mid[:-steps]) / mid[:-steps] * 1e4
            fwd[h] = r

        hour_of = pd.to_datetime(grid_ts, utc=True).hour.to_numpy()
        for _, row in day_labels.iterrows():
            hmask = hour_of == int(row["hour"])
            key = (str(row["regime"]), str(row["split"]))
            hours_used[key] = hours_used.get(key, 0) + 1
            for cand in ("S1", "S2"):
                x = sig[cand][hmask]
                for h in HORIZONS_S:
                    accums[(cand, key[0], key[1], h)].add(x, fwd[h][hmask])
            if key[1] == "holdout":
                didx = np.where(hmask)[0][::DECISION_STEPS]
                for cand in ("S1", "S2"):
                    bound_store.setdefault((key[0], cand), []).append(
                        np.stack([mid[didx], sig[cand][didx]]))

    results = {c: {r: {s: {h: accums[(c, r, s, h)].stats() for h in HORIZONS_S}
                       for s in ("calibrate", "holdout")}
                   for r in ("calm", "volatile")}
               for c in ("S1", "S2")}

    # held-out confirmation per candidate/regime at every horizon
    confirmation = {}
    for cand in ("S1", "S2"):
        confirmation[cand] = {}
        for regime in ("calm", "volatile"):
            per_h = {}
            for h in HORIZONS_S:
                cal = results[cand][regime]["calibrate"][h]
                hold = results[cand][regime]["holdout"][h]
                ok = (np.isfinite(hold["slope"]) and np.isfinite(cal["slope"])
                      and hold["p"] < CONFIRM_P
                      and np.sign(hold["slope"]) == np.sign(cal["slope"])
                      and abs(hold["slope"]) >= CONFIRM_FRACTION * abs(cal["slope"]))
                per_h[h] = bool(ok)
            confirmation[cand][regime] = per_h

    # registered selection rule
    r2 = {c: results[c]["volatile"]["holdout"][SELECT_HORIZON]["r2"] for c in ("S1", "S2")}
    if (not np.isfinite(r2["S2"])) or r2["S1"] <= 0 and r2["S2"] <= 0:
        winner = "S1"
    elif abs(r2["S1"] - r2["S2"]) <= SELECT_TIE_REL * max(r2["S1"], r2["S2"], EPS):
        winner = "S1"
    else:
        winner = "S1" if r2["S1"] > r2["S2"] else "S2"

    # perfect-foresight bound on holdout, per regime, for the selected candidate,
    # using the CALIBRATE-split 1 s slope as the predictor (registered)
    bound = {}
    for regime in ("calm", "volatile"):
        slope = results[winner][regime]["calibrate"][SELECT_HORIZON]["slope"]
        chunks = bound_store.get((regime, winner), [])
        real: List[float] = []
        placebo: List[float] = []
        for arr in chunks:
            real.extend(foresight_bound(arr[0], arr[1], slope))
            # amendment 1c: identical rule on the circularly shifted signal — same
            # statistics, no price alignment; drift and deferral mechanics cancel
            placebo.extend(foresight_bound(
                arr[0], np.roll(arr[1], EPISODE_DECISIONS), slope))
        pairs = [(r, pl) for r, pl in zip(real, placebo)
                 if np.isfinite(r) and np.isfinite(pl)]
        corrected = [r - pl for r, pl in pairs]
        bound[regime] = {
            "slope_used_bps_per_sigma": slope,
            "n_windows": len(corrected),
            "mean_raw_advantage_bps": float(np.mean([r for r, _ in pairs])) if pairs else float("nan"),
            "mean_placebo_advantage_bps": float(np.mean([pl for _, pl in pairs])) if pairs else float("nan"),
            "mean_corrected_advantage_bps": float(np.mean(corrected)) if corrected else float("nan"),
            "p95_corrected_advantage_bps": float(np.percentile(corrected, 95)) if corrected else float("nan"),
        }

    verdict = {}
    for regime in ("calm", "volatile"):
        b = bound[regime]["mean_corrected_advantage_bps"]
        conf = confirmation[winner][regime][SELECT_HORIZON]
        verdict[regime] = bool(np.isfinite(b) and b >= MATERIALITY_BPS and conf)
    stop_go = {"threshold_bps": MATERIALITY_BPS, "per_regime_pass": verdict,
               "PROCEED": bool(any(verdict.values()))}

    out = {
        "registered": {
            "grid_ms": GRID_MS, "s1_window_bins": S1_WINDOW_BINS,
            "norm_bins": NORM_BINS, "norm_min_bins": NORM_MIN_BINS,
            "horizons_s": HORIZONS_S, "select_horizon_s": SELECT_HORIZON,
            "select_tie_rel": SELECT_TIE_REL, "confirm_p": CONFIRM_P,
            "confirm_fraction": CONFIRM_FRACTION,
            "split": "step3g chronological calibrate/holdout (registered 2026-07-XX)",
            "bound": ("clairvoyant pace {0,2} schedule, impact ignored, calibrate "
                      "slope; PLACEBO-CORRECTED per amendment 1 (real minus "
                      "circularly-shifted-signal advantage, per window)"),
            "amendment_1": ("2026-07-22: NaN below MIN_STD instead of eps-division; "
                            "S2 raw; placebo-corrected bound"),
            "materiality_bps": MATERIALITY_BPS,
        },
        "provenance": {
            "book_dir": str(book_dir), "trades_dir": str(trades_dir),
            "labels": str(labels_path),
            "hours_used": {f"{k[0]}/{k[1]}": v for k, v in sorted(hours_used.items())},
        },
        "results": results,
        "confirmation": confirmation,
        "selection": {"winner": winner, "holdout_r2_at_select_horizon_volatile": r2},
        "bound": bound,
        "stop_go": stop_go,
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, indent=1))
    logger.info("wrote %s ; winner=%s ; PROCEED=%s", out_path, winner,
                stop_go["PROCEED"])
    return out


# ---------------------------------------------------------- Phase A2: endogenous baseline
A2_EPISODES = 1200               # per regime (registered)
A2_INTERVALS = 600               # post-warm-up 0.5 s intervals per episode
A2_SEED_BASE = 30_000_000        # diagnostic-only seed range, disjoint from all blocks


def sample_episode(env, seed: int, n_intervals: int = A2_INTERVALS
                   ) -> Tuple[np.ndarray, np.ndarray]:
    """Background-only episode: record (mid, sim-S2) BEFORE each interval advance.

    Sim S2 mirrors the real definition: first non-empty level per side, sizes in BTC
    via the bundle's per-level unit sizes; NaN while a side is swept."""
    env.reset(seed=seed)
    ep = env._ep
    K = env.K
    aes = np.asarray(env.bundle.aes, dtype=np.float64)
    mids = np.empty(n_intervals)
    s2 = np.empty(n_intervals)
    for i in range(n_intervals):
        mids[i] = ep.p_mid
        bi, ai = env._best_slots(ep)
        if bi >= K or ai >= K:
            s2[i] = np.nan
        else:
            b = float(ep.state[bi]) * aes[bi]
            a = float(ep.state[K + ai]) * aes[ai]
            s2[i] = (b - a) / (b + a) if (b + a) > 0 else np.nan
        env._run_interval(track_flow=False)
    return mids, s2


def accumulate_episode(accums: Dict[str, RegAccum], mids: np.ndarray, s2: np.ndarray
                       ) -> None:
    """Within-episode forward returns at every horizon into the accumulators."""
    for h, steps in HORIZONS_S.items():
        if steps >= len(mids):
            continue
        ret = (mids[steps:] - mids[:-steps]) / mids[:-steps] * 1e4
        accums[h].add(s2[:-steps], ret)


def run_endogenous(scratch: Path, out_path: Path) -> Dict:
    from execution.qrm.step4_gates import _env  # heavy import kept local

    measurement = json.loads((scratch / "signal" / "measurement.json").read_text())
    winner = measurement["selection"]["winner"]
    endo = {}
    for regime in ("calm", "volatile"):
        env = _env(scratch, regime, 25.0)
        accums = {h: RegAccum() for h in HORIZONS_S}
        for i in range(A2_EPISODES):
            mids, s2 = sample_episode(env, A2_SEED_BASE + i)
            accumulate_episode(accums, mids, s2)
            if (i + 1) % 200 == 0:
                logger.info("%s: %d/%d episodes", regime, i + 1, A2_EPISODES)
        endo[regime] = {h: accums[h].stats() for h in HORIZONS_S}

    residual = {}
    stop = {}
    for regime in ("calm", "volatile"):
        residual[regime] = {}
        for h in HORIZONS_S:
            real = measurement["results"][winner][regime]["calibrate"][h]["slope"]
            sim = endo[regime][h]["slope"]
            residual[regime][h] = {"real_calibrate": real, "endogenous": sim,
                                   "residual": real - sim}
        r1 = residual[regime][SELECT_HORIZON]
        stop[regime] = bool(r1["endogenous"] > r1["real_calibrate"])

    out = {
        "registered": {"episodes_per_regime": A2_EPISODES,
                       "intervals_per_episode": A2_INTERVALS,
                       "seed_base": A2_SEED_BASE, "signal": winner,
                       "residual": "real calibrate slope minus endogenous slope"},
        "endogenous": endo,
        "residual": residual,
        "stop_rule_endogenous_exceeds_real_at_1s": stop,
        "STOP": bool(any(stop.values())),
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, indent=1))
    logger.info("wrote %s ; STOP=%s", out_path, out["STOP"])
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--scratch", required=True, type=Path)
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--mode", choices=["real", "endogenous"], default="real")
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    if args.mode == "real":
        out = args.out or (args.scratch / "signal" / "measurement.json")
        run_measurement(args.scratch, out)
    else:
        out = args.out or (args.scratch / "signal" / "endogenous_baseline.json")
        run_endogenous(args.scratch, out)


if __name__ == "__main__":
    main()
