"""Post-null diagnostic D1: do the trained PPO policies RESPOND to the signal
feature (obs index 27)? Collect real obs from the injected env, then sweep the
signal feature over a grid holding everything else fixed; report the mean chosen
pace multiple per signal value. Flat line = agent ignores the signal."""
import sys
from pathlib import Path
import numpy as np

sys.path.insert(0, "/Users/fardeenidrus/Desktop/MSc Dissertation/code/drl-optimal-order-execution/src")
from execution.qrm.step5_judgement import _core
from execution.qrm.reactive_env import ACTIONS

SCRATCH = Path("/Users/fardeenidrus/Desktop/MSc Dissertation/scratch_hyperliquid/oxford_l4")
RUNS = SCRATCH / "runs_signal_phaseD"
SWEEP = [-3.0, -2.0, -1.0, 0.0, 1.0, 2.0, 3.0]

from stable_baselines3 import PPO

for run in ("ppo_calm_s1", "ppo_calm_s0", "ppo_volatile_s0", "ppo_volatile_s2"):
    regime = "volatile" if "volatile" in run else "calm"
    model = PPO.load(str(RUNS / run / "model.zip"), device="cpu")
    env = _core(SCRATCH, regime, 25.0, 300, inject=True)
    bank = []
    for i in range(6):                       # 6 episodes, on-policy obs
        obs = env.reset(seed=77_000_000 + i)
        done = False
        while not done:
            bank.append(obs.copy())
            a, _ = model.predict(obs, deterministic=True)
            obs, _r, done, _ = env.step(int(a))
    bank = np.array(bank[:: max(1, len(bank) // 400)])   # ~400 states
    print(f"{run}: {len(bank)} states | signal feature observed "
          f"std {bank[:, 27].std():.3f} range [{bank[:, 27].min():.2f}, {bank[:, 27].max():.2f}]")
    line = []
    for s in SWEEP:
        b = bank.copy()
        b[:, 27] = s
        acts, _ = model.predict(b, deterministic=True)
        mults = np.array([ACTIONS[int(a)] for a in np.atleast_1d(acts)])
        line.append(mults.mean())
    print("   mean pace multiple over signal sweep " +
          "  ".join(f"{s:+.0f}sd:{m:.3f}" for s, m in zip(SWEEP, line)))
    print(f"   pace response (∂pace/∂sd, +3 vs -3): {(line[-1] - line[0]) / 6.0:+.4f}")
