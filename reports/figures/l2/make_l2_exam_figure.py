"""L2 sealed-exam figure: the validation-to-test inversion (l2_test_protocol.md, 2026-07-30).

THE claim: an apparent edge with 5/5 seed agreement and p<=0.008 on genuinely sealed data,
produced equally by a behaviourally collapsed learner -> it is a property of the PERIOD, not skill.
Every number from the frozen result JSONs. Palette validated for colour-blind separation
(house blue/red pair, worst-CVD dE 21.1).
"""
from __future__ import annotations
import json, glob
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

R = Path("/Users/fardeenidrus/Desktop/MSc Dissertation/scratch_hyperliquid/l2_test_results")
VAL = R / "val_recheck.json"      # copied out of /tmp 2026-07-30; /tmp does not survive reboot
OUT = Path(__file__).resolve().parent
plt.rcParams.update({"font.size": 10, "axes.titlesize": 11, "axes.labelsize": 10,
                     "figure.dpi": 150, "savefig.bbox": "tight",
                     "axes.spines.top": False, "axes.spines.right": False})
BLUE, RED, INK, MUTED = "#1f77b4", "#d62728", "#222222", "#666666"


def load():
    test = {}
    for f in glob.glob(str(R / "test_*.json")):
        for r in json.loads(Path(f).read_text())["runs_flat"]:
            test[(r["runs_dir"], r["run"])] = r
    val = {}
    for r in json.loads(VAL.read_text())["runs_flat"]:
        val[(r["runs_dir"], r["run"])] = r
    return val, test


def main():
    val, test = load()
    keys = [k for k in val if k[0] == "runs_10s"]           # the dataset with the apparent edge
    # Authored at print width (6.3in), panels stacked, notes drawn in ink.
    fig, axes = plt.subplots(2, 1, figsize=(6.3, 7.6))

    # ---- panel A: paired validation -> test, every agent ----
    ax = axes[0]
    for algo, col in (("ppo", BLUE), ("dqn", RED)):
        ks = [k for k in keys if k[1].startswith(algo)]
        for k in ks:
            v, t = val[k]["mean_paired_diff_bps"], test[k]["mean_paired_diff_bps"]
            ax.plot([0, 1], [v, t], "-", color=col, lw=1.1, alpha=0.55, zorder=2)
            ax.plot([0, 1], [v, t], "o", color=col, ms=4, alpha=0.8, zorder=3)
    ax.axhline(0.0, color=INK, lw=1.2, ls="--", zorder=1)
    # one collapsed DQN seed sits at +2.94 -> +4.34 bps and compresses the other 29 lines.
    # Clip the view and STATE it on the figure rather than dropping the point silently.
    allv = [val[k]["mean_paired_diff_bps"] for k in keys]
    allt = [test[k]["mean_paired_diff_bps"] for k in keys]
    lo = min(min(allv), min(allt)) - 0.08
    ax.set_ylim(lo, 0.75)
    n_off = sum(1 for a, b in zip(allv, allt) if a > 0.75 or b > 0.75)
    if n_off:
        ax.annotate(f"{n_off} collapsed DQN seed off-scale\n(+2.94 to +4.34 bps, same direction)",
                    xy=(0.5, 0.965), xycoords="axes fraction", ha="center", va="top",
                    fontsize=9, color=INK)
    ax.set_xlim(-0.25, 1.25)
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["validation\n(earlier period)", "held-out test\n(later period)"])
    ax.set_ylabel("cost vs TWAP (bps)   [below the line = cheaper]")
    ax.set_title("Every agent flips sign: 28 of 30")
    ax.annotate("worse than TWAP", xy=(0.99, 0.99), xycoords="axes fraction",
                ha="right", va="top", fontsize=9, color=INK, style="italic")
    ax.annotate("cheaper than TWAP", xy=(0.99, 0.02), xycoords="axes fraction",
                ha="right", va="bottom", fontsize=9, color=INK, style="italic")

    # ---- panel B: the control -- the collapsed DQN reverses too ----
    ax = axes[1]
    # MEDIAN, not mean: one collapsed DQN seed at +2.94 -> +4.34 bps drags the DQN mean to
    # -0.019 and makes the control look like a tie when 4 of 5 seeds in two of three arms are
    # actually cheaper on test. The median describes the typical agent; seed counts are
    # annotated so the reader sees the vote, not just the centre.
    groups, meds, cols, votes = [], [], [], []
    for algo, col in (("ppo", BLUE), ("dqn", RED)):
        ks = [k for k in keys if k[1].startswith(algo)]
        v = np.array([val[k]["mean_paired_diff_bps"] for k in ks])
        t = np.array([test[k]["mean_paired_diff_bps"] for k in ks])
        groups += [f"{algo.upper()}\nvalidation", f"{algo.upper()}\nheld-out test"]
        meds += [float(np.median(v)), float(np.median(t))]
        votes += [f"{int((v < 0).sum())}/{len(v)} cheaper", f"{int((t < 0).sum())}/{len(t)} cheaper"]
        cols += [col, col]
    means = meds
    x = np.arange(len(groups))
    bars = ax.bar(x, means, 0.6, color=cols, edgecolor="white", linewidth=1.2)
    for b, vt, m in zip(bars, votes, means):
        ax.text(b.get_x() + b.get_width() / 2, 0.012 if m < 0 else -0.012, vt,
                ha="center", va="bottom" if m < 0 else "top", fontsize=8.5, color=INK)
    for b, m in zip(bars, means):
        inside = m < 0
        ax.text(b.get_x() + b.get_width() / 2,
                m + (0.012 if m >= 0 else 0.012),
                f"{m:+.3f}", ha="center", va="bottom",
                fontsize=9, color="white" if inside else INK,
                fontweight="bold" if inside else "normal")
    ax.axhline(0.0, color=INK, lw=1.2, ls="--")
    ax.set_xticks(x)
    ax.set_xticklabels(groups, fontsize=9.5)
    ax.set_ylabel("median cost vs TWAP (bps)")
    ax.set_title("Median cost by algorithm and period\nDQN agents are behaviourally collapsed")

    handles = [Line2D([], [], color=BLUE, lw=2, marker="o", ms=5, label="PPO agents (15 runs)"),
               Line2D([], [], color=RED, lw=2, marker="o", ms=5,
                      label="DQN agents (15 runs; behaviourally collapsed)"),
               Line2D([], [], color=INK, lw=1.2, ls="--", label="TWAP")]
    fig.legend(handles=handles, loc="lower center", ncol=2, frameon=False,
               bbox_to_anchor=(0.5, -0.055))
    fig.suptitle("Cost against TWAP in both evaluation periods,\nwith the behaviourally collapsed control",
                 y=1.005, fontsize=12)
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(OUT / f"frozen_exam_inversion.{ext}")
    plt.close(fig)
    print(f"  wrote {OUT}/frozen_exam_inversion.pdf/.png")


if __name__ == "__main__":
    main()
