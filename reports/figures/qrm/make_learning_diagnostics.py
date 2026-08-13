"""Learning diagnostics from Stable-Baselines3 training logs: did the agents actually learn?

ORIGIN. The supervisor asked whether the agents were learning at all, and the honest answer at
the time was that we could not say: every campaign trained with no logger, so value loss,
explained variance and episode reward were never recorded and cannot be recovered from a saved
model. The ten base PPO agents and ten base DQN agents were therefore RETRAINED with logging
enabled, and each re-run was gated on reproducing its original final curve value exactly. All
twenty passed, so these trajectories belong to the agents of record rather than to
statistically-similar look-alikes.

WHAT IS PLOTTED, AND WHY IT DIFFERS BY ALGORITHM. SB3 records different quantities for
policy-gradient and value-based methods, and plotting the wrong one would be meaningless:

  PPO   rollout/ep_rew_mean        average reward per episode
        train/value_loss           the critic's regression error
        train/explained_variance   SEE THE WARNING BELOW -- this is NOT a clean measure

  DQN   rollout/ep_rew_mean        average reward per episode
        train/loss                 the TD error -- DQN's value error
        (DQN has NO explained_variance: that quantity comes from advantage estimation and
         exists only for PPO/A2C. Plotting it for a value-based agent would be a category
         error, so the TD loss is shown in its place and the panel says so.)

A TRAP THAT MUST NOT REACH THE REPORT. SB3's `train/explained_variance` is computed as
`explained_variance(rollout_buffer.values, rollout_buffer.returns)`, and in
`common/buffers.py` the target is built as `returns = advantages + values` -- i.e. the value
function is scored against a quantity CONSTRUCTED FROM ITS OWN PREDICTIONS via GAE
bootstrapping. It is optimistically biased by construction, and it is measured on training
rollouts under the stochastic exploring policy.

The project's post-hoc diagnostic (`diag_learning.py`) computes something different and honest:
true discounted return-to-go accumulated from realised rewards, under the DETERMINISTIC policy,
on evaluation episodes. No self-reference.

The two therefore disagree, and the disagreement is expected, not a contradiction:
SB3 training-time reports roughly +0.11 to +0.21; the honest post-hoc measure reports -0.004
under the original observation and +0.405 once the missing arrival-price reference is added.
**Cite the post-hoc figure for value-function quality. The training-time series is evidence
that the optimiser is running, and nothing more.** Putting both in the report without this
distinction would read as a contradiction, or worse, as citing the flattering one.

READING GUIDE. The question is not "does the loss go down" -- a loss can fall while the agent
gets no better at the task. The question that matters is whether AVERAGE EPISODE REWARD
improves. It is the left column, and it is flat.

Sources (absolute):
  /Users/fardeenidrus/Desktop/MSc Dissertation/scratch_hyperliquid/oxford_l4/runs_signal_logged/*/progress.csv
  /Users/fardeenidrus/Desktop/MSc Dissertation/scratch_hyperliquid/oxford_l4/runs_signal_logged_dqn/*/progress.csv
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

S = Path("/Users/fardeenidrus/Desktop/MSc Dissertation/scratch_hyperliquid/oxford_l4")
OUT = Path(__file__).resolve().parent
BLUE, RED, INK, MUTED = "#1f77b4", "#d62728", "#222222", "#666666"
RCOL = {"calm": BLUE, "volatile": RED}

plt.rcParams.update({"font.size": 10, "axes.titlesize": 11, "axes.labelsize": 10,
                     "figure.dpi": 150, "savefig.bbox": "tight",
                     "axes.spines.top": False, "axes.spines.right": False})


def load(runs_dir: str, algo: str) -> dict[str, pd.DataFrame]:
    out = {}
    for d in sorted((S / runs_dir).iterdir()):
        p = d / "progress.csv"
        if d.is_dir() and p.exists() and d.name.startswith(algo):
            out[d.name] = pd.read_csv(p)
    return out


def smooth(y: np.ndarray, w: int = 25) -> np.ndarray:
    """Rolling mean. Episode reward is extremely noisy per rollout; the raw series is also
    drawn faintly so the smoothing is visible rather than hidden."""
    s = pd.Series(y).rolling(w, min_periods=1, center=True).mean()
    return s.to_numpy()


def panel(ax, runs, xcol, ycol, title, ylab, logy=False, hline=None, hlabel=None):
    n = 0
    for name, df in runs.items():
        if ycol not in df.columns:
            continue
        d = df[[xcol, ycol]].dropna()
        if d.empty:
            continue
        regime = "calm" if "calm" in name else "volatile"
        x = d[xcol].to_numpy() / 1e6
        y = d[ycol].to_numpy()
        ax.plot(x, y, color=RCOL[regime], lw=0.6, alpha=0.12, zorder=1)
        ax.plot(x, smooth(y), color=RCOL[regime], lw=1.4, alpha=0.85, zorder=3)
        n += 1
    if hline is not None:
        ax.axhline(hline, color=INK, lw=1.1, ls="--", zorder=2)
        if hlabel:
            ax.annotate(hlabel, xy=(0.99, hline), xycoords=("axes fraction", "data"),
                        ha="right", va="bottom", fontsize=7.5, color=INK)
    if logy:
        ax.set_yscale("log")
    ax.set_xlabel("training steps (millions)")
    ax.set_ylabel(ylab)
    ax.set_title(title, fontsize=10.5)
    return n


def main() -> None:
    ppo = load("runs_signal_logged", "ppo")
    dqn = load("runs_signal_logged_dqn", "dqn")
    print(f"loaded {len(ppo)} PPO and {len(dqn)} DQN trajectories")

    fig, axes = plt.subplots(2, 3, figsize=(14.0, 7.4))

    # ---- PPO row -------------------------------------------------------------------
    panel(axes[0, 0], ppo, "time/total_timesteps", "rollout/ep_rew_mean",
          "PPO — average reward per episode", "mean episode reward")
    panel(axes[0, 1], ppo, "time/total_timesteps", "train/value_loss",
          "PPO — value error (critic regression loss)", "value loss", logy=True)
    n = panel(axes[0, 2], ppo, "time/total_timesteps", "train/explained_variance",
              "PPO — training-time explained variance\n(optimistically biased — see caption)",
              "explained variance", hline=0.0, hlabel="explains nothing")
    axes[0, 2].set_ylim(-1.0, 1.0)
    axes[0, 2].annotate("This is the library's own metric, scored against a\n"
                        "target built from the value function itself\n"
                        "(returns = advantages + values). Measured\n"
                        "honestly against realised return-to-go under\n"
                        "the deterministic policy, the same critics\n"
                        "explain -0.004 of the variance.",
                        xy=(0.03, 0.03), xycoords="axes fraction", ha="left", va="bottom",
                        fontsize=7.2, color=MUTED)

    # ---- DQN row -------------------------------------------------------------------
    panel(axes[1, 0], dqn, "time/total_timesteps", "rollout/ep_rew_mean",
          "DQN — average reward per episode", "mean episode reward")
    panel(axes[1, 1], dqn, "time/total_timesteps", "train/loss",
          "DQN — value error (temporal-difference loss)", "TD loss", logy=True)
    ax = axes[1, 2]
    panel(ax, dqn, "time/total_timesteps", "rollout/exploration_rate",
          "DQN — exploration rate (annealed)", "epsilon")
    ax.annotate("DQN has no explained-variance analogue:\nthat quantity comes from advantage\n"
                "estimation and exists only for PPO.\nIts value error is the TD loss, centre.",
                xy=(0.97, 0.60), xycoords="axes fraction", ha="right", va="top",
                fontsize=8, color=MUTED)

    handles = [Line2D([], [], color=BLUE, lw=2, label="calm regime (5 seeds)"),
               Line2D([], [], color=RED, lw=2, label="volatile regime (5 seeds)"),
               Line2D([], [], color=MUTED, lw=0.8, alpha=0.4, label="raw per-update series"),
               Line2D([], [], color=INK, lw=1.1, ls="--", label="critic explains nothing")]
    fig.legend(handles=handles, loc="lower center", ncol=4, frameon=False,
               bbox_to_anchor=(0.5, -0.04))
    # TITLE RULE, applied across S11/F23/F26/F27: name the SETTING and the FINDING, and never
    # a reference point a first-time reader cannot resolve. "The original agents" is meaningless
    # to an examiner meeting the figure once. Which inputs these agents had belongs in the
    # caption, not the title -- see the manifest's caption obligation for F23.
    fig.suptitle("In the injected market, average episode reward does not improve over the "
                 "full training budget", y=1.005, fontsize=12)
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(OUT / f"fig23_learning_diagnostics.{ext}")
    plt.close(fig)
    print(f"  wrote {OUT}/fig23_learning_diagnostics.pdf/.png")

    # ---- numbers for the caption and the write-up ----------------------------------
    print("\n--- summary for the write-up (first vs last 10% of training) ---")
    for label, runs, vcol in (("PPO", ppo, "train/explained_variance"),
                              ("DQN", dqn, "train/loss")):
        for regime in ("calm", "volatile"):
            rs = {k: v for k, v in runs.items() if regime in k}
            first, last, rew0, rew1 = [], [], [], []
            for df in rs.values():
                d = df[["time/total_timesteps", vcol, "rollout/ep_rew_mean"]].dropna()
                k = max(1, len(d) // 10)
                first.append(d[vcol].iloc[:k].mean()); last.append(d[vcol].iloc[-k:].mean())
                rew0.append(d["rollout/ep_rew_mean"].iloc[:k].mean())
                rew1.append(d["rollout/ep_rew_mean"].iloc[-k:].mean())
            print(f"  {label} {regime:<9} {vcol.split('/')[-1]:<19} "
                  f"{np.mean(first):+.4f} -> {np.mean(last):+.4f}   "
                  f"ep_rew {np.mean(rew0):+.4f} -> {np.mean(rew1):+.4f}")


if __name__ == "__main__":
    main()
