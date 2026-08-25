"""L2 (frozen-replay) track figures for the LaTeX report -> PDF + PNG.

All numbers are read from the run artifacts themselves (meta.json per agent;
size_sweep_results.json for the benchmark sweep). Every panel is VALIDATION data and is
labelled as such: the sealed test set has not been evaluated yet (l2_test_protocol.md).

Run:  .venv/bin/python reports/figures/l2/make_l2_figures.py
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

L2 = Path("/Users/fardeenidrus/Desktop/MSc Dissertation/scratch_hyperliquid")
OUT = Path(__file__).resolve().parent
plt.rcParams.update({
    "font.size": 10, "axes.titlesize": 11, "axes.labelsize": 10,
    "figure.dpi": 150, "savefig.bbox": "tight",
    "axes.spines.top": False, "axes.spines.right": False,
})
BLUE, RED = "#1f77b4", "#d62728"

PANELS = [
    ("1-minute bars\n30-min deadline", "runs",
     [("ppo", "96.57"), ("ppo", "193.13"), ("dqn", "96.57"), ("dqn", "193.13")]),
    ("10-second bars\n30-min deadline", "runs_10s",
     [("ppo", "96.57"), ("ppo", "193.13"), ("ppo", "386.27"),
      ("dqn", "96.57"), ("dqn", "193.13"), ("dqn", "386.27")]),
    ("10-second bars\n10-min deadline", "runs_10s_10min",
     [("ppo", "96.57"), ("ppo", "193.13"), ("dqn", "96.57"), ("dqn", "193.13")]),
]


def arm_values(dirname: str, algo: str, size: str):
    vals, flagged = [], []
    for s in range(5):
        m = json.load(open(L2 / dirname / f"{algo}_size{size}_seed{s}" / "meta.json"))
        vals.append(m["val_vs_twap_final"])
        flagged.append(m["val_residual_freq_final"] > 0.10)
    return np.array(vals), np.array(flagged)


def save(fig, name: str):
    fig.savefig(OUT / f"{name}.pdf")
    fig.savefig(OUT / f"{name}.png")
    plt.close(fig)
    print(f"wrote {name}.pdf/.png")


# ---------------------------------------------------------------- L1
def fig_l1_three_lever():
    """L1: the frozen-replay null across all three design levers (VALIDATION data)."""
    fig, axes = plt.subplots(1, 3, figsize=(11.5, 4.0), sharey=True,
                             gridspec_kw={"width_ratios": [4, 6, 4]})
    for ax, (title, dirname, arms) in zip(axes, PANELS):
        for i, (algo, size) in enumerate(arms):
            vals, flagged = arm_values(dirname, algo, size)
            color = BLUE if algo == "ppo" else RED
            x = np.full(5, i) + np.linspace(-0.16, 0.16, 5)
            for xv, v, fl in zip(x, vals, flagged):
                ax.scatter(xv, v, s=26, marker="o" if algo == "ppo" else "s",
                           facecolors=color if fl else "none", edgecolors=color,
                           linewidths=1.1, zorder=2)
            ax.plot(i, vals.mean(), "D", ms=7, color="k", zorder=3)
        ax.axhline(0, color="k", lw=0.9)
        ax.set_xticks(range(len(arms)))
        ax.set_xticklabels([f"{a.upper()}\n{s}" for a, s in arms], fontsize=7.6)
        ax.set_title(title, fontsize=9.5)
        ax.set_xlim(-0.6, len(arms) - 0.4)
    axes[0].set_ylabel("cost vs TWAP (bps)\nVALIDATION data")
    # Descriptive, and free of banned terms: "frozen replay" (the ruling is recorded order
    # books), "track" and "arm" (the noun is configuration), plus "not yet used", which was a
    # chronology and is now false -- the test period has been used.
    fig.suptitle("Recorded order books: cost against TWAP by bar resolution, order size\n"
                 "and deadline, on validation data (all 70 agents)",
                 y=1.09)
    handles = [
        Line2D([], [], marker="o", ls="", markerfacecolor="none", markeredgecolor=BLUE,
               ms=7, label="PPO seed"),
        Line2D([], [], marker="s", ls="", markerfacecolor="none", markeredgecolor=RED,
               ms=7, label="DQN seed"),
        Line2D([], [], marker="s", ls="", markerfacecolor=RED, markeredgecolor=RED,
               ms=7, label="filled = leaned on forced deadline buy (>10% of episodes)"),
        Line2D([], [], marker="D", ls="", color="k", ms=7, label="configuration mean"),
    ]
    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, -0.04),
               ncol=4, frameon=False, fontsize=8.5)
    save(fig, "frozen_three_lever_null")


# ---------------------------------------------------------------- L2 sweep
def fig_l2_size_sweep():
    """L2 sweep: benchmark execution cost grows with order size; the trained agents'
    advantage over TWAP does not appear at any trained size (VALIDATION data)."""
    sweep = json.load(open(L2 / "size_sweep_results.json"))
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10.2, 3.9))
    for entry, ls in zip(sweep, ["-", "--"]):
        rows = entry["rows"]
        pct = [r["pct"] for r in rows]
        ax1.plot(pct, [r["twap_is"] for r in rows], ls + "o", color=BLUE, ms=5,
                 label=f"TWAP cost ({entry['label']})")
        ax1.plot(pct, [r["instant_is"] for r in rows], ls + "s", color=RED, ms=5,
                 label=f"immediate full execution ({entry['label']})")
    ax1.set_xscale("log")
    ax1.set_xticks([0.1, 0.25, 0.5, 1, 2, 5])
    ax1.set_xticklabels(["0.1", "0.25", "0.5", "1", "2", "5"])
    ax1.set_xlabel("order size (% of average daily volume)")
    ax1.set_ylabel("implementation shortfall (bps)")
    ax1.set_title("benchmark cost grows with order size", fontsize=10)
    ax1.legend(fontsize=7.5, frameon=False)

    from matplotlib.ticker import FixedLocator, NullLocator
    sizes = [("96.57", 0.5), ("193.13", 1.0), ("386.27", 2.0)]
    YMAX = 0.55
    for algo, color, marker, off in [("ppo", BLUE, "o", -0.06), ("dqn", RED, "s", 0.06)]:
        xs, means = [], []
        for size, pct in sizes:
            vals, flagged = arm_values("runs_10s", algo, size)
            ok = vals[~flagged]
            x = pct * (1 + off)
            ax2.plot([x, x], [ok.min(), ok.max()], color=color, lw=1.1, alpha=0.6)
            for v in vals[flagged]:
                if v > YMAX:
                    ax2.annotate(f"one collapsed seed\nat {v:+.1f} (off scale)", xy=(x, YMAX - 0.02),
                                 xytext=(x * 1.10, YMAX - 0.24), fontsize=7, color=color,
                                 ha="left", arrowprops=dict(arrowstyle="->", color=color, lw=0.8))
                else:
                    ax2.scatter(x, v, s=30, marker=marker, facecolors=color,
                                edgecolors=color, zorder=3)
            xs.append(x); means.append(ok.mean())
        ax2.plot(xs, means, marker + "-", color=color, ms=7,
                 label=f"{algo.upper()} (mean over audit-clean seeds)")
    ax2.axhline(0, color="k", lw=0.9)
    ax2.set_xscale("log")
    ax2.set_ylim(-0.35, YMAX)
    ax2.xaxis.set_major_locator(FixedLocator([0.5, 1.0, 2.0]))
    ax2.xaxis.set_minor_locator(NullLocator())
    ax2.set_xticklabels(["0.5", "1.0", "2.0"])
    ax2.set_xlabel("order size (% of average daily volume)")
    ax2.set_ylabel("agent cost vs TWAP (bps)")
    ax2.set_title("the trained agents' advantage does not appear\nat any size (10-second bars, VALIDATION data)",
                  fontsize=10)
    ax2.legend(fontsize=8, frameon=False)
    fig.suptitle("The size lever: execution gets harder as orders grow, but no exploitable gap opens",
                 y=1.04)
    save(fig, "frozen_size_sweep")


if __name__ == "__main__":
    fig_l1_three_lever()
    fig_l2_size_sweep()
    print("L2 FIGURES DONE ->", OUT)
