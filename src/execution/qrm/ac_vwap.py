"""Almgren-Chriss reference + VWAP scoping + risk-return frontier (criteria §8 Amendment A3).

Everything here is DESCRIPTIVE comparator machinery: no edge claims, no pass/fail beyond
the implementation gates. Calibration provenance, the kappa*T grid, the gates, and the
seed-coherence rule are all fixed in the A3 registration BEFORE any evaluation.

Subcommands:
    calibrate    -> signal/ac_calibration.json (sigma from move process, eta from G2,
                    gamma=0 justified by G1; full provenance recorded)
    gates        -> signal/gates/ac_vwap_gates.json (GATE-AC1/AC2/V1/V2; must ALL pass
                    before any comparator number is examined)
    eval         -> step5_comparators/{injected_dev,base_5e6}.json + per-episode .npz
                    (AC family, oracle VWAP, TWAPs; agents re-evaluated with per-episode
                    arrays on the dev seeds -- means must reproduce judgement.json)

    PYTHONPATH=src .venv/bin/python -m execution.qrm.ac_vwap <subcommand> --scratch <oxford_l4>
"""
from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import numpy as np

from execution.qrm.reactive_baselines import (ACTIONS, adaptive_twap, make_fixed_twap,
                                              run_episodes)
from execution.qrm.reactive_env import ReactiveQRMEnv

logger = logging.getLogger(__name__)

A_ONE = ACTIONS.index(1.0)
KAPPA_T_GRID = (0.0, 1.0, 2.0, 4.0)      # registered (A3): dimensionless urgency
REGIMES = ("calm", "volatile")


# ----------------------------------------------------------------- env factory
def make_env(scratch: Path, regime: str, inject: bool) -> ReactiveQRMEnv:
    kw = {}
    if inject:
        sol = json.loads((scratch / "signal" / "kernel_solution.json").read_text())
        kw = dict(signal_injection=True, signal_kernel=sol["regimes"][regime]["kernel"])
    return ReactiveQRMEnv(
        str(scratch / "step3g" / f"qrm_bundle_{regime}_b.npz"),
        str(scratch / "step3g" / f"move_process_{regime}_centered.npz"),
        order_btc=25.0, n_steps=300, **kw)


# ----------------------------------------------------------------- calibration
def calibrate(scratch: Path) -> dict:
    """sigma from the move process; eta from G2's SIM small-size slope; gamma=0 by G1."""
    gates = json.loads((scratch / "signal" / "gates" / "sigext_gates_v4c_PASS.json").read_text())
    out = {"registered": "criteria section 8 Amendment A3", "regimes": {}}
    for regime in REGIMES:
        mp = np.load(scratch / "step3g" / f"move_process_{regime}_centered.npz")
        m, p = mp["moves"].astype(float), mp["probs"]
        var_int = float((p * m ** 2).sum() - (p * m).sum() ** 2)     # ticks^2 / 0.5 s
        sigma_bps = float(np.sqrt(2.0 * var_int) * 0.1)              # bps / sqrt(s)
        sim = gates["G2_cost_vs_size_rev1"]["regimes"][regime]["sim_bps"]
        eta = float(sim["2"] - sim["1"])                             # bps per unit, 1->2u
        g1 = gates["G1_reaction_lever_rev1"]["regimes"][regime]
        out["regimes"][regime] = {
            "sigma_bps_per_sqrt_s": sigma_bps,
            "eta_bps_per_unit": eta,
            "eta_source": "G2 sim cost-vs-size, small-size slope (1->2 units)",
            "eta_concavity_disclosure": {k: sim[k] for k in ("1", "2", "5", "10")},
            "gamma": 0.0,
            "gamma_justification": {
                "probe_bps_t1": g1["probe_bps_t1"], "probe_bps_t30": g1["probe_bps_t30"],
                "reading": "impact fully decays by 30 s (probe goes negative)"},
            "implied_lambda_by_kappaT": {
                str(kT): (eta * (kT / 300.0) ** 2 / sigma_bps ** 2 if sigma_bps > 0 else None)
                for kT in KAPPA_T_GRID},
        }
    path = scratch / "signal" / "ac_calibration.json"
    path.write_text(json.dumps(out, indent=1))
    logger.info("wrote %s", path)
    return out


# ------------------------------------------------------------------- policies
def _remaining_fraction(kappa_T: float, T: int) -> np.ndarray:
    """r_j for j=0..T: remaining fraction at the START of step j (r_0=1, r_T=0)."""
    j = np.arange(T + 1, dtype=float)
    if kappa_T == 0.0:
        return 1.0 - j / T
    k = kappa_T / T
    return np.sinh(k * (T - j)) / np.sinh(kappa_T)


def make_schedule_policy(targets_remaining: np.ndarray, order_btc: float):
    """Pace-multiple policy delivering an arbitrary schedule as a RATE REQUEST (the
    registered emulation pattern, identical to make_fixed_twap): each step requests the
    schedule's own slice X*(r_j - r_{j+1}) via the nearest pace multiple, and the env's
    internal carry accumulator handles fill granularity. NO line-correction: fills land
    in whole queue units (~0.54 BTC vs ~0.083 BTC slices), so remaining_btc sits above
    the smooth line between fills BY DESIGN; correcting against that phantom backlog
    over-trades catastrophically (the first gate run failed on exactly this -- 4% action
    agreement -- and this docstring is the audit trail of the fix)."""
    slices = np.asarray(targets_remaining, dtype=float)
    if slices.ndim == 1 and len(slices) and not np.isclose(slices.sum(), order_btc):
        # a remaining-schedule was passed: convert to per-step slices
        slices = order_btc * (slices[:-1] - slices[1:])

    def policy(env: ReactiveQRMEnv, _obs) -> int:
        ep = env._ep
        j = ep.step_idx
        steps_left = env.n_steps - j
        adaptive_pace = ep.remaining_btc / max(steps_left, 1)
        if adaptive_pace <= 1e-12:
            return 0
        want = slices[min(j, len(slices) - 1)] / adaptive_pace
        return int(np.argmin([abs(a - want) for a in ACTIONS]))
    return policy


def make_ac(kappa_T: float, order_btc: float = 25.0, T: int = 300):
    if kappa_T == 0.0:
        # exact uniform slices: byte-identical arithmetic to make_fixed_twap (no
        # float noise from differencing the schedule; a knife-edge tie in the unit
        # tests caught exactly this)
        return make_schedule_policy(np.full(T, order_btc / T), order_btc)
    return make_schedule_policy(_remaining_fraction(kappa_T, T), order_btc)


def make_vwap_expected(order_btc: float = 25.0, T: int = 300):
    """Independent construction from the weight definition w_j = E[vol_j]/sum. The
    calibrated rate tensor has no time axis -> E[vol_j] is constant -> w_j = 1/T."""
    return make_schedule_policy(np.full(T, order_btc / T), order_btc)


# --------------------------------------------------- episode runners (recording)
def run_recording(env: ReactiveQRMEnv, policy, seeds, record_volume=False):
    """run_episodes + per-episode action sequences (+ per-step background volume)."""
    costs = np.zeros(len(seeds))
    acts = []
    vols = []
    for i, seed in enumerate(seeds):
        obs = env.reset(seed=seed)
        done, tot, a_seq = False, 0.0, []
        while not done:
            a = policy(env, obs)
            a_seq.append(a)
            obs, r, done, info = env.step(a)
            tot += r
        costs[i] = -tot
        acts.append(np.array(a_seq, dtype=np.int8))
        if record_volume:
            vols.append(np.array(env._ep.flow_market_units, dtype=np.int32))
    return costs, acts, vols


def run_oracle_vwap(env: ReactiveQRMEnv, seeds, vol_profiles, order_btc: float = 25.0):
    """Pass 2 of the oracle: trade proportional to the recorded (pass-1, neutral-policy)
    realized volume profile for the same seed. Look-ahead reference, labelled."""
    costs = np.zeros(len(seeds))
    for i, seed in enumerate(seeds):
        v = vol_profiles[i].astype(float)
        w = v / v.sum() if v.sum() > 0 else np.full(len(v), 1.0 / len(v))
        pol = make_schedule_policy(order_btc * w, order_btc)
        obs = env.reset(seed=seed)
        done, tot = False, 0.0
        while not done:
            obs, r, done, info = env.step(pol(env, obs))
            tot += r
        costs[i] = -tot
    return costs


# ----------------------------------------------------------------------- gates
def cmd_gates(scratch: Path, n_ac1: int, n_v: int) -> None:
    from scipy.stats import wilcoxon
    out = {"registered": "criteria section 8 Amendment A3 (gates fixed before running)"}
    seeds_ac1 = list(range(18_000_000, 18_000_000 + n_ac1))
    seeds_v = list(range(18_000_000, 18_000_000 + n_v))

    # GATE-AC2 first (pure trajectory property, no episodes)
    fl = []
    for kT in KAPPA_T_GRID:
        r = _remaining_fraction(kT, 300)
        fl.append(float(1.0 - r[150]))          # executed by halftime
    out["GATE_AC2"] = {"executed_by_halftime": dict(zip(map(str, KAPPA_T_GRID), fl)),
                       "pass": bool(all(fl[i] < fl[i + 1] for i in range(len(fl) - 1)))}

    for regime in REGIMES:
        env = make_env(scratch, regime, inject=True)
        # GATE-AC1: kappaT=0 vs adaptive TWAP
        c_ada, a_ada, v_ada = run_recording(env, adaptive_twap, seeds_ac1, record_volume=True)
        c_ac0, a_ac0, _ = run_recording(env, make_ac(0.0), seeds_ac1)
        agree = float(np.mean([np.mean(x == y) for x, y in zip(a_ac0, a_ada)]))
        d = c_ac0 - c_ada
        out[f"GATE_AC1_{regime}"] = {
            "n": n_ac1, "action_agreement": agree,
            "paired_mean_diff_bps": float(d.mean()),
            "se": float(d.std(ddof=1) / np.sqrt(n_ac1)),
            "pass": bool(agree > 0.95 and abs(d.mean()) < 0.02)}
        # GATE-V1: independent expected-VWAP == fixed-TWAP emulation, byte identity
        _c_ve, a_ve, _ = run_recording(env, make_vwap_expected(), seeds_v)
        _c_fx, a_fx, _ = run_recording(env, make_fixed_twap(env), seeds_v)
        ident = bool(all(np.array_equal(x, y) for x, y in zip(a_ve, a_fx)))
        agree_ada = float(np.mean([np.mean(x == y) for x, y in
                                   zip(a_ve, a_ada[:n_v])]))
        out[f"GATE_V1_{regime}"] = {"n": n_v, "byte_identical_to_fixed": ident,
                                    "agreement_vs_adaptive": agree_ada, "pass": ident}
        # GATE-V2: oracle beats adaptive (subset sanity)
        c_orc = run_oracle_vwap(env, seeds_v, v_ada[:n_v])
        d2 = c_orc - c_ada[:n_v]
        out[f"GATE_V2_{regime}"] = {
            "n": n_v, "oracle_minus_adaptive_bps": float(d2.mean()),
            "wilcoxon_p": float(wilcoxon(d2[d2 != 0]).pvalue),
            "pass": bool(d2.mean() < 0)}
        logger.info("%s gates: AC1 agree %.3f diff %+.4f | V1 ident %s | V2 %+.4f",
                    regime, agree, d.mean(), ident, d2.mean())
    out["all_pass"] = bool(all(v.get("pass") for k, v in out.items()
                               if isinstance(v, dict) and "pass" in v))
    path = scratch / "signal" / "gates" / "ac_vwap_gates.json"
    path.write_text(json.dumps(out, indent=1))
    logger.info("wrote %s ; ALL_PASS=%s", path, out["all_pass"])


# ------------------------------------------------------------------ evaluation
def cmd_eval(scratch: Path, which: str, n_eval: int) -> None:
    """Comparator suite per the seed-coherence rule. Saves per-episode arrays."""
    from scipy.stats import wilcoxon
    inject = which == "injected_dev"
    seed0 = 18_000_000 if inject else 5_000_000
    seeds = list(range(seed0, seed0 + n_eval))
    outdir = scratch / "step5_comparators"
    outdir.mkdir(exist_ok=True)
    report = {"which": which, "seed0": seed0, "n": n_eval, "inject": inject,
              "registered": "criteria section 8 Amendment A3", "regimes": {}}
    per_ep = {}
    for regime in REGIMES:
        env = make_env(scratch, regime, inject=inject)
        c_ada, _a, v_ada = run_recording(env, adaptive_twap, seeds, record_volume=True)
        rows, arrs = {}, {"adaptive": c_ada}
        arrs["fixed"] = run_episodes(env, make_fixed_twap(env), seeds)["cost_bps"]
        for kT in KAPPA_T_GRID:
            arrs[f"ac_kT{kT:g}"] = run_recording(env, make_ac(kT), seeds)[0]
        arrs["vwap_oracle"] = run_oracle_vwap(env, seeds, v_ada)
        for name, c in arrs.items():
            d = c - c_ada
            nz = d[d != 0]
            rows[name] = {"mean_cost_bps": float(c.mean()),
                          "std_cost_bps": float(c.std(ddof=1)),
                          "mean_vs_adaptive_bps": float(d.mean()),
                          "wilcoxon_p_vs_adaptive": (float(wilcoxon(nz).pvalue)
                                                     if len(nz) > 10 else 1.0)}
            logger.info("%s/%s %s: mean %+.4f std %.3f vs-ada %+.4f", which, regime,
                        name, c.mean(), c.std(ddof=1), d.mean())
        report["regimes"][regime] = rows
        per_ep[regime] = arrs
        np.savez_compressed(outdir / f"{which}_{regime}.npz", **per_ep[regime])
        (outdir / f"{which}.json").write_text(json.dumps(report, indent=1))
    logger.info("eval %s done", which)


def cmd_eval_agents(scratch: Path, n_eval: int) -> None:
    """Re-evaluate the 10 base PPO models on the dev seeds SAVING per-episode arrays
    (frontier needs variance). Integrity: means must reproduce judgement.json."""
    from stable_baselines3 import PPO
    seeds = list(range(18_000_000, 18_000_000 + n_eval))
    dev = json.loads((scratch / "step5_signal_dev" / "judgement.json").read_text())
    ref = {r["run"]: r["mean_vs_adaptive_bps"] for r in dev["per_run"]}
    outdir = scratch / "step5_comparators"
    outdir.mkdir(exist_ok=True)
    res = {}
    for regime in REGIMES:
        env = make_env(scratch, regime, inject=True)
        c_ada = run_episodes(env, adaptive_twap, seeds)["cost_bps"]
        res[f"adaptive_{regime}"] = c_ada
        for s in range(5):
            run = f"ppo_{regime}_s{s}"
            model = PPO.load(str(scratch / "runs_signal_phaseD" / run / "model.zip"),
                             device="cpu")
            def pol(env_, obs, _m=model):
                a, _ = _m.predict(obs, deterministic=True)
                return int(a)
            c = run_episodes(env, pol, seeds)["cost_bps"]
            res[run] = c
            diff = float((c - c_ada).mean())
            ok = abs(diff - ref[run]) < 1e-9
            logger.info("agent %s: vs-ada %+.4f (judgement %+.4f) integrity %s",
                        run, diff, ref[run], "EXACT" if ok else "MISMATCH")
    np.savez_compressed(outdir / "agents_dev_per_episode.npz", **res)
    logger.info("agents per-episode arrays saved")


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s",
                        datefmt="%H:%M:%S")
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["calibrate", "gates", "eval", "eval-agents"])
    ap.add_argument("--scratch", required=True, type=Path)
    ap.add_argument("--which", choices=["injected_dev", "base_5e6"], default="injected_dev")
    ap.add_argument("--n-eval", type=int, default=2000)
    ap.add_argument("--n-ac1", type=int, default=2000)
    ap.add_argument("--n-v", type=int, default=500)
    a = ap.parse_args()
    if a.cmd == "calibrate":
        calibrate(a.scratch)
    elif a.cmd == "gates":
        cmd_gates(a.scratch, a.n_ac1, a.n_v)
    elif a.cmd == "eval":
        cmd_eval(a.scratch, a.which, a.n_eval)
    else:
        cmd_eval_agents(a.scratch, a.n_eval)


if __name__ == "__main__":
    main()
