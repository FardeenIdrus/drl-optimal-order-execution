"""Stage 3g driver — regime-conditional QRM calibration (calm vs volatile).

Builds TWO calibrated simulators from the validated v2 reconstruction: one from the
month's calm hours, one from its volatile hours. Pre-registered rules (locked before
calibration ran; amendments 2026-07-04):

* Regime label = hourly realised volatility (std of 1 s mid changes, the gate's M1
  statistic) from the v2 reconstruction, split at the month MEDIAN — mirroring the L2
  track's methodology. The rule is fixed here, before any label is computed.
* EVERYTHING is calibrated per regime — the queue-event rates (quiet-spell
  conditioned, same guard method as 3f), book shapes, AES, the volume anchor, AND the
  exogenous move process (most of the calm/volatile difference lives in the moves).
* Chronological split WITHIN each regime: the earliest 75 % of that regime's hours
  calibrate, the latest 25 % are held out (no look-ahead).
* Pre-committed checks: regime separation (volatile/calm median-vol ratio; overlap),
  ample hours per regime per split, and the 3f coverage rule per regime.

Subcommands: ``label`` (this file's Step 1-2: label + separation report).
"""
from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd

from execution.qrm.step3f import _day0_ns

logger = logging.getLogger(__name__)

MIN_SNAPSHOTS_PER_HOUR = 6000  # >= 50 min of 0.5 s snapshots to trust an hour's label
CAL_FRACTION = 0.75            # earliest 75 % of each regime's hours calibrate
TICK_3G = 1.0                  # BTC tick (as in step3f)


def hourly_vol(book_dir: Path) -> pd.DataFrame:
    """Hourly realised volatility (std of 1 s mid changes, $) for the whole month."""
    rows = []
    for p in sorted(book_dir.glob("*.parquet")):
        df = pd.read_parquet(p, columns=["ts", "mid"])
        hr = ((df.ts - _day0_ns(p.stem)) // 3_600_000_000_000).astype(int)
        for h, g in df.groupby(hr):
            if h < 0 or h > 23 or len(g) < MIN_SNAPSHOTS_PER_HOUR:
                continue
            m = g.mid.to_numpy()[::2]
            rows.append({"date": p.stem, "hour": int(h),
                         "vol_1s": float(np.std(np.diff(m)))})
    out = pd.DataFrame(rows).sort_values(["date", "hour"]).reset_index(drop=True)
    logger.info("hourly vol: %d usable hours across %d days",
                len(out), out.date.nunique())
    return out


def label_regimes(hv: pd.DataFrame) -> pd.DataFrame:
    """Median-split labels + chronological calibrate/holdout split per regime."""
    threshold = float(hv.vol_1s.median())
    hv = hv.copy()
    hv["regime"] = np.where(hv.vol_1s >= threshold, "volatile", "calm")
    hv["split"] = "holdout"
    for reg, g in hv.groupby("regime"):
        n_cal = int(len(g) * CAL_FRACTION)
        hv.loc[g.index[:n_cal], "split"] = "calibrate"  # rows already chronological
    hv.attrs["threshold"] = threshold
    return hv


def separation_report(hv: pd.DataFrame) -> dict:
    """Pre-committed checks: is the split real, and is there enough data?"""
    calm, vol = hv[hv.regime == "calm"], hv[hv.regime == "volatile"]
    ratio = float(vol.vol_1s.median() / calm.vol_1s.median())
    # overlap: how much of each regime's mass crosses the other's quartile
    overlap = float((vol.vol_1s < calm.vol_1s.quantile(0.75)).mean())
    counts = hv.groupby(["regime", "split"]).size().to_dict()
    return {
        "threshold_vol_1s": hv.attrs["threshold"],
        "median_vol": {"calm": float(calm.vol_1s.median()),
                       "volatile": float(vol.vol_1s.median())},
        "separation_ratio": ratio,
        "volatile_mass_below_calm_p75": overlap,
        "hours": {f"{r}/{s}": int(n) for (r, s), n in counts.items()},
        "calm_p05_p95": [float(calm.vol_1s.quantile(q)) for q in (0.05, 0.95)],
        "volatile_p05_p95": [float(vol.vol_1s.quantile(q)) for q in (0.05, 0.95)],
    }


def cmd_label(args: argparse.Namespace) -> None:
    book_dir = Path(args.book_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(exist_ok=True)
    hv = label_regimes(hourly_vol(book_dir))
    rep = separation_report(hv)
    hv.to_parquet(out_dir / "regime_labels.parquet", index=False)
    (out_dir / "regime_report.json").write_text(json.dumps(rep, indent=2))
    print(json.dumps(rep, indent=2))


# ------------------------------------------------- per-regime calibration
BLOCK_LEN = 6   # contiguous-block mode: hours per block
N_BLOCKS = 5    # blocks per regime, chosen closest to the regime's median vol


def select_blocks(labels: pd.DataFrame, regime: str,
                  block_len: int = BLOCK_LEN, n_blocks: int = N_BLOCKS) -> list:
    """Contiguous blocks of ``block_len`` hours near the regime's MEDIAN volatility.

    Mechanism this serves (design decision, 2026-07-06): one stationary rate
    table blended from scattered heterogeneous hours is smoother than any real hour
    (too wide vs quiet hours, too tight vs burst hours). Calibrating on contiguous,
    median-typical stretches gives a simulator OF a typical hour of the regime. Rule
    (fixed before running): candidate = any run of ``block_len`` consecutive clock
    hours all labelled (regime, calibrate); score = |median hourly vol of the block −
    regime median|; pick the ``n_blocks`` best non-overlapping blocks. All candidate
    hours are in the calibrate split, so every block precedes the holdout.
    """
    g = labels[(labels.regime == regime) & (labels.split == "calibrate")].copy()
    med = float(g.vol_1s.median())
    t = pd.to_datetime(g.date, format="%Y%m%d").map(pd.Timestamp.toordinal) * 24 + g.hour
    g["t"] = t.astype(int)
    by_t = g.set_index("t")
    cands = []
    for t0 in sorted(g.t):
        window = [t0 + i for i in range(block_len)]
        if all(w in by_t.index for w in window):
            vols = by_t.loc[window, "vol_1s"]
            cands.append((abs(float(vols.median()) - med), t0))
    cands.sort()
    chosen: list = []
    for _score, t0 in cands:
        if all(abs(t0 - c) >= block_len for c in chosen):
            chosen.append(t0)
            if len(chosen) == n_blocks:
                break
    blocks = []
    for t0 in sorted(chosen):
        rows = by_t.loc[[t0 + i for i in range(block_len)]]
        blocks.append([(r.date, int(r.hour)) for r in rows.itertuples()])
    logger.info("%s: %d blocks of %dh chosen (median-vol targets): %s",
                regime, len(blocks), block_len, [b[0] for b in blocks])
    return blocks


def _block_stream(scratch: Path, block: list, coin: str, with_trades: bool):
    """One timestamped stream across a contiguous block's (date, hour) pairs."""
    from execution.qrm.step3f import chained_stream

    for date, hour in block:
        yield from chained_stream(scratch, date, [hour], coin, with_trades)


def sample_hours(labels: pd.DataFrame, regime: str, split: str, max_hours: int) -> list:
    """Every Nth hour of a regime's split, chronologically (spreads coverage across
    the month; every sampled calibration hour still precedes every holdout hour of
    that regime by construction of the chronological split)."""
    g = labels[(labels.regime == regime) & (labels.split == split)]
    step = max(1, int(np.ceil(len(g) / max_hours)))
    rows = g.iloc[::step]
    logger.info("%s/%s: sampled %d of %d hours (every %d)",
                regime, split, len(rows), len(g), step)
    return [(r.date, int(r.hour)) for r in rows.itertuples()]


def walk_regime_hours(
    scratch: Path, units: list, coin: str, aes, K: int, Q: int, guard_ns: int
):
    """Quiet-spell calibration over walk units (each a list of (date, hour)): one
    fresh engine + warm-up per unit (a contiguous block pays warm-up once; scattered
    hours pay it each — the 3f cold-start discipline either way); accumulators and
    same-clock stats are summed across units."""
    from execution.qrm.quiet_spell import QuietStats, calibrate_quiet
    from execution.qrm.step3f import fully_filled_set

    counts_sum = time_sum = None
    qs_sum = QuietStats()
    for unit in units:
        stream = _block_stream(scratch, unit, coin, with_trades=True)
        ff = set()
        for date, hour in unit:
            ff |= fully_filled_set(scratch, date, [hour], coin)
        counts, time_in, qs = calibrate_quiet(stream, ff, aes, K, Q, TICK_3G, guard_ns)
        counts_sum = counts if counts_sum is None else counts_sum + counts
        time_sum = time_in if time_sum is None else time_sum + time_in
        for f in ("quiet_s", "total_s", "quiet_trade_btc", "total_trade_btc",
                  "ask1_zero_s", "n_moves", "n_events_counted"):
            setattr(qs_sum, f, getattr(qs_sum, f) + getattr(qs, f))
    return counts_sum, time_sum, qs_sum


def walk_regime_burst(scratch: Path, units: list, coin: str, K: int):
    """Burst profiles per walk unit, summed (arrivals, exposure, sizes, q1 profile)."""
    from execution.qrm.quiet_spell import measure_burst

    prof_sum = None
    for unit in units:
        stream = _block_stream(scratch, unit, coin, with_trades=False)
        prof = measure_burst(stream, K=K, tick=TICK_3G)
        if prof_sum is None:
            prof_sum = prof
        else:
            prof_sum.arrivals_inner += prof.arrivals_inner
            prof_sum.exposure_s += prof.exposure_s
            prof_sum.size_sums += prof.size_sums
            prof_sum.size_counts += prof.size_counts
            prof_sum.q1_seconds += prof.q1_seconds
            prof_sum.n_moves += prof.n_moves
            prof_sum.total_s += prof.total_s
    return prof_sum


def cmd_calibrate_regime(args: argparse.Namespace) -> None:
    from execution.qrm.assemble import assemble
    from execution.qrm.quiet_spell import choose_q, decide_guard, quiet_aes

    scratch = Path(args.scratch)
    out_dir = Path(args.out_dir)
    labels = pd.read_parquet(out_dir / "regime_labels.parquet")
    if args.mode == "blocks":
        blocks = select_blocks(labels, args.regime)
        hours = [h for b in blocks for h in b]
        units = blocks          # one walker per contiguous block
    else:
        hours = sample_hours(labels, args.regime, "calibrate", args.max_hours)
        units = [[h] for h in hours]  # one walker per hour

    prof = walk_regime_burst(scratch, units, args.coin, args.K)
    confirmed, guard_ns = decide_guard(prof)
    aes = quiet_aes(prof, guard_ns)
    Q = choose_q(prof.q1_seconds)
    if not confirmed:
        logger.warning("%s: burst refuted -> unguarded calibration", args.regime)

    counts, time_in, qs = walk_regime_hours(
        scratch, units, args.coin, aes, args.K, Q, guard_ns)
    lam_cells = time_in > 0
    thin = (counts.sum(axis=3) < 100) & lam_cells
    thin_exposure = float(time_in[thin].sum() / max(time_in.sum(), 1e-9))
    target = qs.market_units_target(float(aes[0]))
    bundle = assemble(counts, time_in, aes, TICK_3G, invariant="empirical",
                      market_target_units_per_s=target)
    bundle.save(str(out_dir / f"qrm_bundle_{args.regime}{args.suffix}.npz"))
    np.savez(out_dir / f"accumulators_{args.regime}{args.suffix}.npz",
             counts=counts, time_in=time_in, aes=aes, Q=Q, guard_ns=guard_ns,
             quiet_s=qs.quiet_s, total_s=qs.total_s,
             quiet_trade_btc=qs.quiet_trade_btc, ask1_zero_s=qs.ask1_zero_s,
             hours=np.array([f"{d}:{h:02d}" for d, h in hours]))
    report = {
        "regime": args.regime, "mode": args.mode, "K": args.K, "Q": Q,
        "n_hours": len(hours), "n_walk_units": len(units),
        "guard_ms": guard_ns / 1e6, "burst_confirmed": bool(confirmed),
        "steady_per_s": prof.steady_rate(), "first_bin_per_s": float(prof.rates()[0]),
        "aes": aes.tolist(), "quiet_s": qs.quiet_s, "total_s": qs.total_s,
        "quiet_fraction": qs.quiet_s / max(qs.total_s, 1e-9),
        "events_counted": qs.n_events_counted,
        "market_anchor_units_per_s": target,
        "ask1_zero_uncond": qs.ask1_zero_s / max(qs.total_s, 1e-9),
        "thin_cell_exposure_share": thin_exposure,
        "coverage_rule_triggered": thin_exposure > 0.10,
    }
    (out_dir / f"calibration_report_{args.regime}{args.suffix}.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


def _pooled_hour_stats(book_dir: Path, hours: list) -> dict:
    """Pooled real statistics over a set of (date, hour): per-hour 1 s mid deltas and
    0.5 s tick deltas concatenated (never differencing across hour boundaries)."""
    from execution.qrm.step3f import _hour_slice

    d1s, d05, spreads = [], [], []
    first_mid = None
    by_date: dict = {}
    for date, hour in hours:
        by_date.setdefault(date, []).append(hour)
    for date, hs in by_date.items():
        for hour in hs:
            g = _hour_slice(book_dir, date, [hour])
            m = g.mid.to_numpy()
            if first_mid is None and len(m):
                first_mid = float(m[0])
            d1s.append(np.diff(m[::2]))
            d05.append(np.diff(m))
            spreads.append(g.spread.to_numpy())
    return {
        "vol_1s": float(np.std(np.concatenate(d1s))),
        "spread_ticks": float(np.concatenate(spreads).mean() / TICK_3G),
        "tick_deltas_05s": np.concatenate(d05) / TICK_3G,
        "first_mid": first_mid,
    }


def _move_process_from_deltas(deltas: np.ndarray):
    """MoveProcess from per-0.5s tick deltas (same binning as exo_ref_sim)."""
    from execution.qrm.exo_ref_sim import MAX_ABS_MOVE, MoveProcess

    # remediation I7: same explicit binning rule as exo_ref_sim (|d|<0.75 -> 0,
    # else round half away from zero) — np.rint's half-to-even distorted the mix.
    mag = np.abs(np.asarray(deltas, float))
    k = np.where(mag < 0.75, 0.0, np.floor(mag + 0.5) * np.sign(deltas))
    d = np.clip(k, -MAX_ABS_MOVE, MAX_ABS_MOVE).astype(int)
    support = np.arange(-MAX_ABS_MOVE, MAX_ABS_MOVE + 1)
    cnt = np.bincount(d + MAX_ABS_MOVE, minlength=len(support)).astype(float)
    return MoveProcess(interval_s=0.5, moves=support, probs=cnt / cnt.sum())


def cmd_gate_regime(args: argparse.Namespace) -> None:
    from execution.qrm.assemble import load_bundle
    from execution.qrm.exo_ref_sim import run_exo_qrm

    scratch = Path(args.scratch)
    out_dir = Path(args.out_dir)
    book_dir = Path(args.book_dir)
    tol = json.loads(Path(args.tolerances).read_text())["tolerances"]
    labels = pd.read_parquet(out_dir / "regime_labels.parquet")
    acc = np.load(out_dir / f"accumulators_{args.regime}{args.suffix}.npz")
    calrep = json.loads((out_dir / f"calibration_report_{args.regime}{args.suffix}.json").read_text())
    bundle = load_bundle(str(out_dir / f"qrm_bundle_{args.regime}{args.suffix}.npz"))
    aes1 = float(acc["aes"][0])
    cal_hours = [(h.split(":")[0], int(h.split(":")[1])) for h in acc["hours"].tolist()]

    cal = _pooled_hour_stats(book_dir, cal_hours)
    mp = _move_process_from_deltas(cal["tick_deltas_05s"])
    hold_hours = sample_hours(labels, args.regime, "holdout", args.max_hours)
    hold = _pooled_hour_stats(book_dir, hold_hours)

    by_type = acc["counts"].sum(axis=(0, 1, 2))
    m4_limit = float(by_type[0] / by_type.sum())
    m4_cancel = float(by_type[1] / by_type.sum())
    anchor_btc = calrep["market_anchor_units_per_s"] * aes1

    sims = []
    for s in range(args.n_seeds):
        r = run_exo_qrm(bundle, mp, horizon_s=args.horizon_s,
                        initial_price=cal["first_mid"], seed=s)
        sims.append({
            "seed": s, "vol_1s": r["vol_1s"], "spread_ticks": r["spread_ticks_mean"],
            "q1_zero": r["q1_zero_frac"], "limit_share": r["event_mix"]["limit"],
            "cancel_share": r["event_mix"]["cancel"],
            "market_btc_per_s": r["event_mix"]["market"] * r["n_events"] / args.horizon_s * aes1,
            "stall_frac": r["sampler_stall_frac"],
        })
        logger.info("%s seed %d: %s", args.regime, s,
                    {k: round(float(v), 4) for k, v in sims[-1].items() if k != "seed"})
    sdf = pd.DataFrame(sims)

    def chk(name, sim, real, rel, per_seed):
        lo, hi = real * (1 - rel), real * (1 + rel)
        n_wide = int(((per_seed >= real * (1 - 1.5 * rel)) & (per_seed <= real * (1 + 1.5 * rel))).sum())
        return {"metric": name, "sim_mean": float(sim), "real": float(real),
                "band": [float(lo), float(hi)], "mean_pass": bool(lo <= sim <= hi),
                "seeds_in_1p5x_band": n_wide,
                "stability_pass": bool(n_wide >= args.n_seeds - 2)}
    checks = [
        chk("M1_vol_1s", sdf.vol_1s.mean(), cal["vol_1s"], tol["T_vol_1s"], sdf.vol_1s),
        chk("M2_spread", sdf.spread_ticks.mean(), cal["spread_ticks"], tol["T_spread_ticks"], sdf.spread_ticks),
        chk("M3_inner_empty", sdf.q1_zero.mean(), calrep["ask1_zero_uncond"], tol["T_p_wide"], sdf.q1_zero),
    ]
    for nm, real, key in [("M4_limit_share", m4_limit, "limit_share"),
                          ("M4_cancel_share", m4_cancel, "cancel_share")]:
        mean = float(sdf[key].mean())
        checks.append({"metric": nm, "sim_mean": mean, "real": real,
                       "band": [real - 0.05, real + 0.05],
                       "mean_pass": bool(abs(mean - real) <= 0.05),
                       "seeds_in_1p5x_band": int((abs(sdf[key] - real) <= 0.075).sum()),
                       "stability_pass": bool((abs(sdf[key] - real) <= 0.075).sum() >= args.n_seeds - 2)})
    m5 = float(sdf.market_btc_per_s.mean())
    checks.append({"metric": "M5a_market_btc_per_s_vs_anchor", "sim_mean": m5,
                   "real": anchor_btc, "band": [0.8 * anchor_btc, 1.2 * anchor_btc],
                   "mean_pass": bool(0.8 * anchor_btc <= m5 <= 1.2 * anchor_btc),
                   "seeds_in_1p5x_band": int(((sdf.market_btc_per_s >= 0.7 * anchor_btc)
                                              & (sdf.market_btc_per_s <= 1.3 * anchor_btc)).sum()),
                   "stability_pass": True})
    report = {
        "regime": args.regime, "protocol": "3f Revision-1 fidelity design, per regime",
        "n_calibration_hours": len(cal_hours), "n_seeds": args.n_seeds,
        "checks": checks,
        "transfer_report_holdout_not_gated": {
            "n_holdout_hours_sampled": len(hold_hours),
            "vol_1s": {"sim": float(sdf.vol_1s.mean()), "real_holdout": hold["vol_1s"]},
            "spread_ticks": {"sim": float(sdf.spread_ticks.mean()),
                             "real_holdout": hold["spread_ticks"]},
        },
        "reported_not_gated": {
            "sampler_stall_frac_mean": float(sdf.stall_frac.mean()),
            "stall_validity_concern": bool(sdf.stall_frac.mean() > 0.01),
        },
        "per_seed": sims,
        "all_pass": bool(all(c["mean_pass"] and c["stability_pass"] for c in checks)),
    }
    (out_dir / f"gate_verdict_{args.regime}{args.suffix}.json").write_text(json.dumps(report, indent=2))
    print(json.dumps({k: report[k] for k in ("regime", "checks", "all_pass")}, indent=2))


def _tilt_to_mean(k: np.ndarray, pr: np.ndarray, target: float = 0.0):
    """theta such that the exponentially tilted distribution p'_k proportional to
    pr*exp(-theta*k) has mean == target. mean(theta) is strictly decreasing in theta,
    so bracket [lo, hi] with mean(lo) > target > mean(hi) and bisect. Works for either
    sign of the initial mean and any target. Returns (achieved_mean, tilted_probs).

    Used for zero-mean centering (target=0) and for the CB1 drift neutralisation
    (target = small negative, so the exo table cancels the endogenous background drift)."""
    k = k.astype(float)
    def mean_at(theta):
        w = pr * np.exp(-theta * np.clip(k, -50, 50))
        w = w / w.sum()
        return float((w * k).sum()), w
    lo, hi = -1e-6, 1e-6
    while mean_at(lo)[0] <= target:
        lo *= 2.0
        if lo < -60:
            break
    while mean_at(hi)[0] >= target:
        hi *= 2.0
        if hi > 60:
            break
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        m, _ = mean_at(mid)
        if m > target:
            lo = mid
        else:
            hi = mid
    return mean_at(0.5 * (lo + hi))


def _tilt_to_zero_mean(k: np.ndarray, pr: np.ndarray):
    """Back-compat wrapper: tilt to zero mean (see _tilt_to_mean)."""
    return _tilt_to_mean(k, pr, 0.0)


def cmd_center_move_process(args) -> None:
    """Remediation I1 (user-approved 2026-07-07): remove the built-in drift.

    The measured per-0.5s jump tables inherited December's realized direction
    (calm +0.0043, volatile +0.0149 ticks/interval -> +$2.6/+$9.0 expected per
    300s episode), which hands any BUYER a mechanical front-loading edge and
    confounds "execution skill" with "the month went up". Fix: EXPONENTIAL TILT
    p'_k = p_k * exp(-theta*k) / Z with theta chosen so the mean is zero — the
    minimum-relative-entropy distribution satisfying the zero-drift constraint
    (Esscher transform; the least-informative change that removes direction).
    Writes move_process_{regime}_centered.npz; verifies |mean| < 1e-9 and
    reports the variance change (expected < 2%)."""
    from pathlib import Path
    out_dir = Path(args.out_dir)
    for regime in ("calm", "volatile"):
        d = np.load(out_dir / f"move_process_{regime}.npz")
        k = d["moves"].astype(float)
        pr = d["probs"].astype(float)
        mean0 = float((pr * k).sum())
        std0 = float(np.sqrt((pr * k * k).sum() - mean0 ** 2))

        m1, w = _tilt_to_zero_mean(k, pr)
        theta = float("nan")  # recorded implicitly by the saved probs
        std1 = float(np.sqrt((w * k * k).sum() - m1 ** 2))
        np.savez(out_dir / f"move_process_{regime}_centered.npz",
                 interval_s=float(d["interval_s"]), moves=d["moves"], probs=w,
                 mean_before=mean0, std_before=std0)
        logger.info("%s centered: mean %+.5f -> %+.1e | std %.4f -> %.4f (%+.2f%%)",
                    regime, mean0, m1, std0, std1, 100 * (std1 - std0) / std0)
        assert abs(m1) < 1e-9


def cmd_deconvolve_endo(args) -> None:
    """Remediation I6 (follow-up to R2): remove double-counted background motion.

    The exogenous table was measured from ALL real mid changes; with the engine's
    endogenous (queue-depletion) moves now captured (I6 fix), background price motion
    is generated twice. Rule, fixed in writing before this calibration ran: if the
    endogenous share of move variance exceeds 5%, subtract the MEASURED endogenous
    per-interval move distribution from the exogenous table (rate subtraction on the
    affected bins), floor at zero, renormalise into the 0 bin, and re-center.
    Measured 2026-07-07: calm 11.5% (deconvolve), volatile 3.7% (document only).
    """
    from pathlib import Path
    from execution.qrm.reactive_env import ReactiveQRMEnv
    out_dir = Path(args.out_dir)
    regime = args.regime
    env = ReactiveQRMEnv(str(out_dir / f"qrm_bundle_{regime}_b.npz"),
                         str(out_dir / f"move_process_{regime}_centered.npz"),
                         order_btc=25.0)
    counts = {}
    n_int = 0
    for seed in range(args.n_episodes):
        env.reset(seed=seed)
        ep = env._ep
        for _ in range(600):
            before = ep.p_ref
            env._run_interval(track_flow=False)
            dm_exo = int(ep.move_path[ep.move_idx - 1])
            d_endo = int(round((ep.p_ref - before) / env.tick)) - dm_exo
            if d_endo != 0:
                counts[d_endo] = counts.get(d_endo, 0) + 1
            n_int += 1
    d = np.load(out_dir / f"move_process_{regime}_centered.npz")
    k = d["moves"].astype(int)
    pr = d["probs"].astype(float).copy()
    z = int(np.where(k == 0)[0][0])
    moved = 0.0
    for tick_move, c in sorted(counts.items()):
        p_endo = c / n_int
        idx = z + tick_move
        take = min(pr[idx], p_endo)
        pr[idx] -= take
        moved += take
        logger.info("%s: endo P(%+d)=%.4f -> subtracted %.4f from exo bin", regime,
                    tick_move, p_endo, take)
    pr[z] += moved
    # re-center by exponential tilt (robust solver — the first version's bracket
    # inverted for negative means and exploded the tail; caught by the variance check)
    _, w = _tilt_to_zero_mean(k.astype(float), pr)
    np.savez(out_dir / f"move_process_{regime}_centered.npz",
             interval_s=float(d["interval_s"]), moves=d["moves"], probs=w,
             deconvolved_endo_share=moved)
    logger.info("%s: exo table deconvolved (%.4f mass moved to 0-bin), re-centered, saved",
                regime, moved)


# dedicated seed block for drift neutralisation + the fairness gate, disjoint from the
# train (k*1e7), curve (1e6), judgement (5e6) and confirmation (9e6) blocks.
FAIRNESS_SEED0 = 3_000_000


def _measure_bg_drift(env, n, seed0=FAIRNESS_SEED0):
    """Mean background p_ref drift (ticks/episode) under a do-nothing policy. The agent
    never trades until the deadline and the deadline force-buy does NOT move p_ref, so
    this isolates the pure background drift the agent faces."""
    tick = env.tick
    d = np.zeros(n)
    for j in range(n):
        env.reset(seed=seed0 + j)
        p0 = env._ep.p_ref
        done = False
        while not done:
            _o, _r, done, _i = env.step(0)
        d[j] = (env._ep.p_ref - p0) / tick
    return d


def _measure_pace_gradient(env, n, seed0=FAIRNESS_SEED0):
    """cost(constant m-pace) - cost(1.0x adaptive-TWAP), bps, CRN-paired per seed. A field
    that rewards front-loading (residual drift) shows monotone faster=cheaper. Returns
    {m: (mean_bps, t, mean_deadline_residual_frac)}."""
    from execution.qrm.reactive_env import ACTIONS  # noqa: PLC0415
    mults = [0.5, 0.8, 1.0, 1.2, 1.5, 2.0]
    idxs = [ACTIONS.index(m) for m in mults]
    cost = {m: np.zeros(n) for m in mults}
    resid = {m: np.zeros(n) for m in mults}
    for m, ai in zip(mults, idxs):
        for j in range(n):
            env.reset(seed=seed0 + j)
            tot = 0.0
            done = False
            info = {}
            while not done:
                _o, r, done, info = env.step(ai)
                tot += r
            cost[m][j] = -tot
            resid[m][j] = info["deadline_residual_btc"] / env.order_btc
    base = cost[1.0]
    out = {}
    for m in mults:
        diff = cost[m] - base
        se = diff.std(ddof=1) / np.sqrt(n)
        out[m] = (float(diff.mean()), float(diff.mean() / se) if se > 0 else 0.0,
                  float(resid[m].mean()))
    return out


def cmd_neutralize_drift(args) -> None:
    """Audit CB1 fix (2026-07-08): centering the EXO table to zero mean did not zero the
    REALISED background drift, because the endogenous queue dynamics (a -60/-75% bid/ask
    calibrated-rate asymmetry inherited from December's rising month) carry their own
    upward mean (~+9 ticks/ep calm, +5 volatile). That drift mechanically rewards
    front-loading and confounds impact skill. Fix (approach A): tilt the exo table to a
    small NEGATIVE mean that cancels the realised drift, calibrated empirically by Newton
    iteration on the measured background drift. Book depth (impact) and variance are left
    essentially unchanged. Overwrites move_process_{regime}_centered.npz and backs up the
    drifty original once. Run AFTER center-move-process + deconvolve-endo (the last step)."""
    from pathlib import Path
    from execution.qrm.reactive_env import ReactiveQRMEnv, INTERVALS_PER_DECISION
    from execution.qrm.exo_ref_sim import MoveProcess
    out_dir = Path(args.out_dir)
    regime = args.regime
    mp_path = out_dir / f"move_process_{regime}_centered.npz"
    backup = out_dir / f"move_process_{regime}_centered_DRIFTY_backup.npz"
    # always calibrate from the ORIGINAL drifty distribution so re-runs are idempotent
    # (exponential tilting is not; re-tilting an already-tilted table would compound it).
    base_src = backup if backup.exists() else mp_path
    d0 = np.load(base_src)
    k = d0["moves"].astype(int)
    pr0 = d0["probs"].astype(float)
    interval_s = float(d0["interval_s"])
    env = ReactiveQRMEnv(str(out_dir / f"qrm_bundle_{regime}_b.npz"), str(mp_path),
                         order_btc=25.0)
    n_int = env.n_steps * INTERVALS_PER_DECISION
    var0 = float((pr0 * k * k).sum() - (pr0 * k).sum() ** 2)

    target = 0.0
    probs = pr0.copy()
    drift_before = None
    mean = float("nan")
    for it in range(args.max_iters):
        _, probs = _tilt_to_mean(k.astype(float), pr0, target)
        env.move_process = MoveProcess(interval_s=interval_s, moves=k, probs=probs)
        d = _measure_bg_drift(env, args.n_cal, args.seed0)
        mean, se = float(d.mean()), float(d.std(ddof=1) / np.sqrt(args.n_cal))
        if drift_before is None:
            drift_before = mean
        logger.info("%s it%d: exo_mean %+.6f/int -> drift %+.3f ticks/ep (SE %.2f, t %+.2f)",
                    regime, it, target, mean, se, mean / se)
        if abs(mean) < 0.5 or abs(mean) < 1.5 * se:
            break
        target -= mean / n_int          # per-interval Newton step (slope ~= 1)
    var1 = float((probs * k * k).sum() - (probs * k).sum() ** 2)

    if not backup.exists():
        np.savez(backup, interval_s=interval_s, moves=k, probs=pr0)
        logger.info("%s: backed up drifty exo table -> %s", regime, backup.name)
    np.savez(mp_path, interval_s=interval_s, moves=k, probs=probs,
             exo_mean_per_interval=float(target), realized_drift_before=float(drift_before),
             realized_drift_after=float(mean), drift_neutralized=True,
             calibration_seed0=int(args.seed0))
    logger.info("%s: NEUTRALISED. exo_mean %+.6f/int | drift %+.2f -> %+.2f ticks/ep | "
                "var %.4f -> %.4f (%+.2f%%) | overwrote %s (backup kept)", regime, target,
                drift_before, mean, var0, var1, 100 * (var1 - var0) / var0, mp_path.name)


def cmd_fairness_gate(args) -> None:
    """Audit CB1 acceptance test (the power the underpowered R3 gate lacked). On the
    neutralised env asserts: (a) background p_ref drift is statistically zero (|t| < 2),
    and (b) no ORDER-COMPLETING constant-pace policy gets a material AND significant
    front-loading advantage over adaptive-TWAP (not [mean <= -0.02 bps and t <= -2.5]).
    Writes fairness_verdict_{regime}.json."""
    from pathlib import Path
    from execution.qrm.reactive_env import ReactiveQRMEnv
    out_dir = Path(args.out_dir)
    regime = args.regime
    env = ReactiveQRMEnv(str(out_dir / f"qrm_bundle_{regime}_b.npz"),
                         str(out_dir / f"move_process_{regime}_centered.npz"), order_btc=25.0)
    # drift is UNPAIRED and noisy (esp. volatile) -> needs a large sample for power;
    # the pace gradient is CRN-PAIRED and low-noise -> a smaller sample already resolves it.
    d = _measure_bg_drift(env, args.n_drift, args.seed0)
    drift_mean = float(d.mean())
    drift_se = float(d.std(ddof=1) / np.sqrt(args.n_drift))
    drift_t = drift_mean / drift_se
    drift_pass = abs(drift_t) < 2.0

    grad = _measure_pace_gradient(env, args.n_grad, args.seed0)
    rows = []
    grad_pass = True
    for m in sorted(grad):
        mean, t, resid = grad[m]
        completing = resid < 0.02
        material_sig = bool(completing and mean <= -0.02 and t <= -2.5)
        if material_sig:
            grad_pass = False
        rows.append({"mult": m, "cost_vs_twap_bps": round(mean, 4), "t": round(t, 2),
                     "deadline_resid_frac": round(resid, 4), "completing": completing,
                     "material_sig_advantage": material_sig})
    verdict = {
        "regime": regime, "n_drift": args.n_drift, "n_grad": args.n_grad,
        "seed0": args.seed0,
        "background_drift_ticks_per_ep": round(drift_mean, 3),
        "background_drift_t": round(drift_t, 2), "drift_pass": bool(drift_pass),
        "pace_gradient": rows, "gradient_pass": bool(grad_pass),
        "fairness_pass": bool(drift_pass and grad_pass),
    }
    (out_dir / f"fairness_verdict_{regime}.json").write_text(json.dumps(verdict, indent=2))
    print(json.dumps(verdict, indent=2))


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", datefmt="%H:%M:%S")
    ap = argparse.ArgumentParser(description="Stage 3g regime-conditional calibration")
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("label")
    p.add_argument("--book-dir", required=True)
    p.add_argument("--out-dir", required=True)
    p.set_defaults(fn=cmd_label)
    p = sub.add_parser("calibrate-regime")
    p.add_argument("--scratch", required=True)
    p.add_argument("--out-dir", required=True)
    p.add_argument("--regime", required=True, choices=["calm", "volatile"])
    p.add_argument("--coin", default="BTC")
    p.add_argument("--K", type=int, default=10)
    p.add_argument("--max-hours", type=int, default=60)
    p.add_argument("--mode", choices=["scattered", "blocks"], default="scattered")
    p.add_argument("--suffix", default="")
    p.set_defaults(fn=cmd_calibrate_regime)
    p = sub.add_parser("gate-regime")
    p.add_argument("--scratch", required=True)
    p.add_argument("--out-dir", required=True)
    p.add_argument("--book-dir", required=True)
    p.add_argument("--tolerances", required=True)
    p.add_argument("--regime", required=True, choices=["calm", "volatile"])
    p.add_argument("--max-hours", type=int, default=60)
    p.add_argument("--n-seeds", type=int, default=8)
    p.add_argument("--horizon-s", type=float, default=600.0)
    p.add_argument("--suffix", default="")
    p.set_defaults(fn=cmd_gate_regime)
    p = sub.add_parser("center-move-process")
    p.add_argument("--out-dir", required=True)
    p.set_defaults(fn=cmd_center_move_process)
    p = sub.add_parser("deconvolve-endo")
    p.add_argument("--out-dir", required=True)
    p.add_argument("--regime", required=True, choices=["calm", "volatile"])
    p.add_argument("--n-episodes", type=int, default=20)
    p.set_defaults(fn=cmd_deconvolve_endo)
    p = sub.add_parser("neutralize-drift")
    p.add_argument("--out-dir", required=True)
    p.add_argument("--regime", required=True, choices=["calm", "volatile"])
    p.add_argument("--n-cal", type=int, default=2000)
    p.add_argument("--max-iters", type=int, default=4)
    p.add_argument("--seed0", type=int, default=FAIRNESS_SEED0,
                   help="episode block to calibrate the drift on; use 5_000_000 for the EVAL block")
    p.set_defaults(fn=cmd_neutralize_drift)
    p = sub.add_parser("fairness-gate")
    p.add_argument("--out-dir", required=True)
    p.add_argument("--regime", required=True, choices=["calm", "volatile"])
    p.add_argument("--n-drift", type=int, default=8000)
    p.add_argument("--n-grad", type=int, default=3000)
    p.add_argument("--seed0", type=int, default=FAIRNESS_SEED0,
                   help="episode block to gate on; use 5_000_000 to gate the EVAL block")
    p.set_defaults(fn=cmd_fairness_gate)
    args = ap.parse_args()
    args.fn(args)


if __name__ == "__main__":
    main()
