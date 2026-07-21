"""DQN Q-value flatness diagnostic (argument bank §N5; first run 2026-07-16).

For every DQN probe agent in the two 2.5-min cells (25 BTC = all-collapsed vs
5 BTC = mostly valid), roll 5 deterministic episodes on the dev block and record,
per state: the Q-values over the 7 pace actions. Reports per agent:
  - mean within-state Q spread (best minus worst action) — "can the value net
    tell the actions apart at all?"
  - share of states where argmax = action 0 ("trade nothing")
Read-only; deterministic (fixed episode seeds 5,000,000-5,000,004).

Run:  PYTHONPATH=src .venv/bin/python reports/diagnostics/dqn_q_flatness.py
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from stable_baselines3 import DQN

from execution.qrm.reactive_env import ACTIONS
from execution.qrm.step5_judgement import _core

S = Path("/Users/fardeenidrus/Desktop/MSc Dissertation/scratch_hyperliquid/oxford_l4")
SEEDS = range(5_000_000, 5_000_005)


def diagnose(run_dir: Path, btc: float, env_steps: int, regime: str) -> dict:
    model = DQN.load(str(run_dir / "model.zip"), device="cpu")
    env = _core(S, regime, btc, env_steps)
    qs = []
    for seed in SEEDS:
        obs = env.reset(seed=seed)
        done = False
        while not done:
            with torch.no_grad():
                q = model.q_net(torch.as_tensor(obs, dtype=torch.float32).unsqueeze(0)).numpy()[0]
            qs.append(q)
            a, _ = model.predict(obs, deterministic=True)
            obs, _, done, _ = env.step(int(a))
    qs = np.array(qs)
    return {
        "run": run_dir.name,
        "n_states": len(qs),
        "mean_q_spread": float((qs.max(1) - qs.min(1)).mean()),
        "argmax0_share": float((qs.argmax(1) == 0).mean()),
    }


def main() -> None:
    import json
    audits = {}
    for cell in ("b5h150", "b25h150"):
        audits.update({e["run"]: e["valid"] for e in json.load(
            open(S / f"step5_dqnprobe_{cell}" / "behaviour_audit.json"))})
    print(f"{'run':<30} {'audit':>10} {'Q spread':>10} {'argmax=0 %':>11}")
    rows = []
    for cell, btc in (("b25h150", 25.0), ("b5h150", 5.0)):
        for d in sorted((S / f"runs_dqnprobe_{cell}").iterdir()):
            if not d.is_dir():
                continue
            regime = d.name.split("_")[1]
            r = diagnose(d, btc, 150, regime)
            r["valid"] = audits[d.name]
            rows.append(r)
            print(f"{r['run']:<30} {'valid' if r['valid'] else 'COLLAPSED':>10} "
                  f"{r['mean_q_spread']:>10.5f} {100 * r['argmax0_share']:>10.1f}%")
    coll = [r for r in rows if not r["valid"]]
    val = [r for r in rows if r["valid"]]
    print(f"\nCOLLAPSED (n={len(coll)}): mean Q-spread {np.mean([r['mean_q_spread'] for r in coll]):.5f}, "
          f"mean argmax=0 share {100 * np.mean([r['argmax0_share'] for r in coll]):.1f}%")
    print(f"VALID     (n={len(val)}): mean Q-spread {np.mean([r['mean_q_spread'] for r in val]):.5f}, "
          f"mean argmax=0 share {100 * np.mean([r['argmax0_share'] for r in val]):.1f}%")
    print("\nACTIONS:", ACTIONS)
    out = Path(__file__).with_suffix(".json")
    out.write_text(json.dumps({"seeds": [int(s) for s in SEEDS], "rows": rows}, indent=2))
    print("wrote", out)


if __name__ == "__main__":
    main()
