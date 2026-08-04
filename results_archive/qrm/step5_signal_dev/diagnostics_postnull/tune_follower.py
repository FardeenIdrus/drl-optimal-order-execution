"""Amendment A2 Test 2, TUNING stage (dev block 18e6, exploration by design).
Family pace = nearest-grid(1 + c*s), c in the registered 7-value grid, per regime;
select c* minimising mean CRN-paired diff vs adaptive TWAP, n=2000.
The dev-tuned value is NEVER the reported ceiling (winner's-curse-exposed by design);
confirmation happens once on 21e6. Usage: tune_follower.py <regime>"""
import json, sys
from pathlib import Path
import numpy as np
from scipy.stats import wilcoxon

sys.path.insert(0, "/Users/fardeenidrus/Desktop/MSc Dissertation/code/drl-optimal-order-execution/src")
from execution.qrm.step5_judgement import _core
from execution.qrm.reactive_baselines import adaptive_twap, run_episodes
from execution.qrm.reactive_env import ACTIONS, WARMUP_INTERVALS

SCRATCH = Path("/Users/fardeenidrus/Desktop/MSc Dissertation/scratch_hyperliquid/oxford_l4")
OUT = SCRATCH / "step5_signal_dev" / "diagnostics_postnull"
SEED0, N = 18_000_000, 2000
GRID = (0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 2.0)   # registered (A2); includes c=1


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
    regime = sys.argv[1]
    OUT.mkdir(parents=True, exist_ok=True)
    seeds = list(range(SEED0, SEED0 + N))
    env = _core(SCRATCH, regime, 25.0, 300, inject=True)
    base = run_episodes(env, adaptive_twap, seeds)
    rows = {}
    for c in GRID:
        r = run_episodes(env, make_tilt(c), seeds)
        d = r["cost_bps"] - base["cost_bps"]
        nz = d[d != 0]
        rows[str(c)] = {"mean_diff_bps": float(d.mean()),
                        "se": float(d.std(ddof=1) / np.sqrt(N)),
                        "wilcoxon_p": float(wilcoxon(nz).pvalue) if len(nz) > 10 else 1.0,
                        "executed_frac": float(r["executed_frac"].mean())}
        print(f"TUNE {regime} c={c}: {rows[str(c)]['mean_diff_bps']:+.4f} bps "
              f"(se {rows[str(c)]['se']:.4f})", flush=True)
    c_star = min(GRID, key=lambda c: rows[str(c)]["mean_diff_bps"])
    out = {"regime": regime, "seed0": SEED0, "n": N, "grid": list(GRID),
           "rows": rows, "c_star": c_star,
           "dev_tuned_value_bps": rows[str(c_star)]["mean_diff_bps"],
           "note": "dev-tuned value is selection-biased BY DESIGN; ceiling = 21e6 confirmation"}
    (OUT / f"tune_follower_{regime}.json").write_text(json.dumps(out, indent=1))
    print(f"TUNE {regime} DONE: c*={c_star} dev value "
          f"{rows[str(c_star)]['mean_diff_bps']:+.4f} bps", flush=True)


if __name__ == "__main__":
    main()
