"""Amendment A2 Test 2, CONFIRMATION stage — ONE SHOT on virgin block 21e6.
Tuned coefficients from the dev-block grid (registered selection rule):
calm c*=1.25, volatile c*=2.0. n=2000, CRN, vs BOTH TWAPs. Predictions
(registered): mean <= -0.05 bps, Wilcoxon p<0.01, both benchmarks, both regimes.
The value reported here IS the ceiling of record. 21e6 SPENT after this run."""
import json, sys
from pathlib import Path
import numpy as np
from scipy.stats import wilcoxon

sys.path.insert(0, "/Users/fardeenidrus/Desktop/MSc Dissertation/code/drl-optimal-order-execution/src")
from execution.qrm.step5_judgement import _core
from execution.qrm.reactive_baselines import adaptive_twap, make_fixed_twap, run_episodes
from execution.qrm.reactive_env import ACTIONS, WARMUP_INTERVALS

SCRATCH = Path("/Users/fardeenidrus/Desktop/MSc Dissertation/scratch_hyperliquid/oxford_l4")
OUT = SCRATCH / "step5_signal_ceiling21e6"
SEED0, N = 21_000_000, 2000
C_STAR = {"calm": 1.25, "volatile": 2.0}


def make_tilt(c):
    def pol(env, _obs):
        ep = env._ep
        if ep is None or ep.s2_path is None:
            return ACTIONS.index(1.0)
        k = min(max(ep.move_idx - WARMUP_INTERVALS, 0), len(ep.s2_path) - 1)
        v = ep.s2_path[k]
        s = float(v) if np.isfinite(v) else env.signal_mean
        want = min(max(1.0 + c * s, 0.0), 2.0)
        return int(np.argmin([abs(a - want) for a in ACTIONS]))
    return pol


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    seeds = list(range(SEED0, SEED0 + N))
    report = {"seed0": SEED0, "n": N, "c_star": C_STAR,
              "registered": "criteria section 8 Amendment A2 Test 2 (one shot)",
              "regimes": {}}
    for regime in ("calm", "volatile"):
        env = _core(SCRATCH, regime, 25.0, 300, inject=True)
        ada = run_episodes(env, adaptive_twap, seeds)
        fix = run_episodes(env, make_fixed_twap(env), seeds)
        tun = run_episodes(env, make_tilt(C_STAR[regime]), seeds)
        row = {"c_star": C_STAR[regime],
               "executed_frac": float(tun["executed_frac"].mean())}
        for bname, base in (("adaptive", ada), ("fixed", fix)):
            d = tun["cost_bps"] - base["cost_bps"]
            nz = d[d != 0]
            row[f"vs_{bname}"] = {
                "mean_diff_bps": float(d.mean()),
                "se": float(d.std(ddof=1) / np.sqrt(N)),
                "wilcoxon_p": float(wilcoxon(nz).pvalue) if len(nz) > 10 else 1.0,
                "frac_cheaper": float((d < 0).mean())}
        report["regimes"][regime] = row
        print("CEILING", regime, json.dumps(row), flush=True)
        (OUT / "ceiling_confirmation.json").write_text(json.dumps(report, indent=1))
    print("CEILING DONE", flush=True)


if __name__ == "__main__":
    main()
