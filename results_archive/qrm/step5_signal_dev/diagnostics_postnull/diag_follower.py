"""Post-null diagnostic D2: how much is the injected signal worth to a strategy
that reads it PERFECTLY? CRN-paired vs adaptive TWAP on the judge's exact dev
seeds (18e6, n=2000). Three exploiters:
  follower   : registered secondary benchmark (pace nearest 1 + s2)
  half       : pace nearest 1 + 0.5*s2 (gentler tilt, diagnostic-only)
  bangbang   : s2 > +1sd -> 2.0x ; s2 < -1sd -> 0.0x ; else 1.0x (diagnostic-only)
Diagnostic only — no frozen rule touched; results to step5_signal_dev/diagnostics_postnull/.
"""
import json, sys
from pathlib import Path
import numpy as np
from scipy.stats import wilcoxon

sys.path.insert(0, "/Users/fardeenidrus/Desktop/MSc Dissertation/code/drl-optimal-order-execution/src")
from execution.qrm.step5_judgement import _core
from execution.qrm.reactive_baselines import adaptive_twap, signal_follower, run_episodes
from execution.qrm.reactive_env import ACTIONS, WARMUP_INTERVALS

SCRATCH = Path("/Users/fardeenidrus/Desktop/MSc Dissertation/scratch_hyperliquid/oxford_l4")
OUT = SCRATCH / "step5_signal_dev" / "diagnostics_postnull"
SEED0, N = 18_000_000, 2000


def _sig(env):
    ep = env._ep
    if ep is None or ep.s2_path is None:
        return 0.0
    k = min(max(ep.move_idx - WARMUP_INTERVALS, 0), len(ep.s2_path) - 1)
    v = ep.s2_path[k]
    return float(v) if np.isfinite(v) else env.signal_mean


def half_tilt(env, _obs):
    want = min(max(1.0 + 0.5 * _sig(env), 0.0), 2.0)
    return int(np.argmin([abs(a - want) for a in ACTIONS]))


def bangbang(env, _obs):
    s = _sig(env)
    if s > 1.0:
        return len(ACTIONS) - 1
    if s < -1.0:
        return 0
    return ACTIONS.index(1.0)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    seeds = list(range(SEED0, SEED0 + N))
    report = {"seed0": SEED0, "n": N, "note": "diagnostic only; judge's dev seeds", "regimes": {}}
    for regime in ("calm", "volatile"):
        env = _core(SCRATCH, regime, 25.0, 300, inject=True)
        base = run_episodes(env, adaptive_twap, seeds)
        rows = {}
        for name, pol in (("follower", signal_follower), ("half", half_tilt),
                          ("bangbang", bangbang)):
            r = run_episodes(env, pol, seeds)
            d = r["cost_bps"] - base["cost_bps"]
            nz = d[d != 0]
            p = float(wilcoxon(nz).pvalue) if len(nz) > 10 else 1.0
            rows[name] = {"mean_diff_bps": float(d.mean()),
                          "se": float(d.std(ddof=1) / np.sqrt(N)),
                          "wilcoxon_p": p,
                          "frac_cheaper": float((d < 0).mean()),
                          "executed_frac": float(r["executed_frac"].mean())}
            print(regime, name, json.dumps(rows[name]), flush=True)
        report["regimes"][regime] = {"adaptive_mean_cost_bps": float(base["cost_bps"].mean()),
                                     "exploiters": rows}
        (OUT / "diag_signal_follower.json").write_text(json.dumps(report, indent=1))
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
