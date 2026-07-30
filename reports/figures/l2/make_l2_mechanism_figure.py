"""L2 inversion MECHANISM figure (l2_test_protocol.md mechanism addendum, 2026-07-30).

THE claim: the sealed test period's apparent edge is a PACING exposure, not skill. Three
panels, left to right, are the three stages of the argument:

  A  A fixed rule that cannot learn anything shows the whole inversion. The premium to
     deviating from TWAP reverses sign and is monotone in the size of the deviation.
  B  Each agent's effective front-loading dose is a property of its weights, so it is
     unchanged between periods (corr 0.999) -- while its cost vs TWAP flips.
  C  Decomposition: 95% of the observed val->test shift is the pacing term; the residual is
     half the materiality threshold.

Every number from the frozen diagnostic JSONs beside the exam results. Palette = the house
blue/red pair already CVD-validated for this track, plus Okabe-Ito green/orange for the
decomposition bars.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

R = Path("/Users/fardeenidrus/Desktop/MSc Dissertation/scratch_hyperliquid/l2_test_results")
OUT = Path(__file__).resolve().parent
BLUE, RED, INK, MUTED = "#1f77b4", "#d62728", "#222222", "#666666"
OI_GREEN, OI_ORANGE = "#009E73", "#E69F00"
DATASET = "runs_10s"          # the dataset whose agents inverted; the exam's decisive arm set

plt.rcParams.update({"font.size": 10, "axes.titlesize": 11, "axes.labelsize": 10,
                     "figure.dpi": 150, "savefig.bbox": "tight",
                     "axes.spines.top": False, "axes.spines.right": False})


def main() -> None:
    s2 = json.loads((R / "l2_inversion_stage2.json").read_text())[DATASET]
    s3 = json.loads((R / "l2_inversion_stage3.json").read_text())
    val3, test3 = s3["validation"], s3["test"]
    runs = sorted(val3)

    fig, axes = plt.subplots(1, 3, figsize=(14.2, 4.5))

    # ---------- A: the probe -- a rule that cannot learn shows the whole inversion ----------
    ax = axes[0]
    ms = sorted((float(k) for k in s2["validation"]["grid"]), key=float)
    for split, col, mk in (("validation", BLUE, "o"), ("test", RED, "s")):
        g = s2[split]["grid"]
        y = np.array([g[f"{m:g}"]["mean_diff_bps"] for m in ms])
        e = np.array([g[f"{m:g}"]["se_bps"] for m in ms])
        ax.errorbar(ms, y, yerr=e, fmt=mk + "-", color=col, lw=1.8, ms=6, capsize=3,
                    markeredgecolor="white", markeredgewidth=0.8, zorder=3)
    ax.axhline(0.0, color=INK, lw=1.2, ls="--", zorder=1)
    ax.axhspan(-0.05, 0.05, color=MUTED, alpha=0.13, zorder=0)
    ax.axvline(1.0, color=MUTED, lw=1.0, ls=":", zorder=1)
    ax.annotate("delay", xy=(0.5, ax.get_ylim()[1]), xytext=(0, -4),
                textcoords="offset points", ha="center", va="top", fontsize=8, color=MUTED)
    ax.annotate("front-load", xy=(2.0, ax.get_ylim()[1]), xytext=(0, -4),
                textcoords="offset points", ha="center", va="top", fontsize=8, color=MUTED)
    ax.set_xlabel("fixed pace, as a multiple of TWAP\n(1.0 reproduces TWAP exactly)")
    ax.set_ylabel("cost vs TWAP (bps)   [below = cheaper]")
    ax.set_title("A.  No learning involved:\nthe premium to deviating reverses sign",
                 fontsize=10.5)

    # ---------- B: dose is a property of the policy; cost is a property of the period ------
    ax = axes[1]
    bv = np.array([val3[k]["beta_frontload_dose"] for k in runs])
    bt = np.array([test3[k]["beta_frontload_dose"] for k in runs])
    mv = np.array([val3[k]["mean_diff_bps"] for k in runs])
    mt = np.array([test3[k]["mean_diff_bps"] for k in runs])
    keep = np.abs(mv) < 1.0            # the collapsed seed sits at +2.94/+4.34; see annotation
    # COLOUR MEANS PERIOD IN EVERY PANEL (blue = validation, red = sealed test). Algorithm is
    # carried by line style, never by colour -- reusing the same two colours for "period" in
    # one panel and "algorithm" in another would make the figure misread at a glance.
    for x, y0, y1, k in zip(bv[keep], mv[keep], mt[keep], np.array(runs)[keep]):
        ls = "-" if k.startswith("ppo") else (0, (2.2, 1.4))
        ax.plot([x, x], [y0, y1], ls=ls, color=MUTED, lw=1.0, alpha=0.75, zorder=2)
        ax.plot([x], [y0], "o", color=BLUE, ms=4.8, alpha=0.9, mec="white", mew=0.6, zorder=3)
        ax.plot([x], [y1], "s", color=RED, ms=4.8, alpha=0.9, mec="white", mew=0.6, zorder=3)
    ax.axhline(0.0, color=INK, lw=1.2, ls="--", zorder=1)
    ax.axvline(0.0, color=MUTED, lw=1.0, ls=":", zorder=1)
    ax.axhspan(-0.05, 0.05, color=MUTED, alpha=0.13, zorder=0)
    ax.set_xlabel("the agent's front-loading dose\n[negative = it delays instead]")
    ax.set_ylabel("cost vs TWAP (bps)")
    ax.set_title("B.  Dose predicts the direction of the flip,\nagent by agent", fontsize=10.5)
    n_off = int((~keep).sum())
    note = (f"dose is a property of the weights, so it does not move\n"
            f"between periods: corr(validation, test) = "
            f"{np.corrcoef(bv, bt)[0, 1]:.3f}\nonly the cost flips")
    if n_off:
        note += f"\n{n_off} collapsed seed off-scale (+2.94 / +4.34 bps)"
    ax.annotate(note, xy=(0.02, 0.02), xycoords="axes fraction", fontsize=7.4,
                color=MUTED, va="bottom")

    # ---------- C: the decomposition ----------
    ax = axes[2]
    pv = val3[runs[0]]["probe_mean_bps"]
    pt = test3[runs[0]]["probe_mean_bps"]
    shift = float((mt - mv).mean())
    pacing = float((bt * pt - bv * pv).mean())
    resid = shift - pacing
    bars = ax.bar([0, 1, 2], [shift, pacing, resid], 0.62,
                  color=[INK, OI_GREEN, OI_ORANGE], edgecolor="white", linewidth=1.2)
    for b, v in zip(bars, [shift, pacing, resid]):
        ax.text(b.get_x() + b.get_width() / 2, v - 0.018, f"{v:+.3f}",
                ha="center", va="top", fontsize=9.5, color=INK, fontweight="bold")
    ax.axhline(0.0, color=INK, lw=1.2, ls="--")
    ax.axhspan(-0.05, 0.05, color=MUTED, alpha=0.13, zorder=0)
    ax.set_xticks([0, 1, 2])
    ax.set_xticklabels(["observed\nval -> test shift", "explained by\npacing",
                        "residual\n(not pacing)"], fontsize=8.5)
    ax.set_ylabel("mean shift in cost vs TWAP (bps)")
    ax.set_ylim(min(shift, pacing) * 1.42, 0.13)
    ax.annotate(f"{100 * pacing / shift:.0f}% of the inversion is\na pacing exposure",
                xy=(1, min(shift, pacing) * 1.30), ha="center", va="bottom",
                fontsize=8.6, color=OI_GREEN, fontweight="bold")
    ax.annotate("inside the materiality band", xy=(2, 0.062), ha="center", va="bottom",
                fontsize=7.4, color=MUTED)
    ax.set_title("C.  95% of the apparent edge is\nexposure, not skill", fontsize=10.5)

    handles = [
        Line2D([], [], color=BLUE, marker="o", lw=1.8, label="validation period"),
        Line2D([], [], color=RED, marker="s", lw=1.8, label="sealed test period"),
        Line2D([], [], color=MUTED, lw=1.0, ls="-", label="PPO agent (panel B)"),
        Line2D([], [], color=MUTED, lw=1.0, ls=(0, (2.2, 1.4)), label="DQN agent (panel B)"),
        Line2D([], [], color=INK, lw=1.2, ls="--", label="TWAP"),
        Patch(facecolor=MUTED, alpha=0.13, label="+/-0.05 bps materiality band"),
    ]
    fig.legend(handles=handles, loc="lower center", ncol=6, frameon=False,
               bbox_to_anchor=(0.5, -0.11), fontsize=8.6)
    fig.suptitle("The sealed period's apparent edge is a pacing exposure any fixed rule "
                 "would have collected", y=1.02, fontsize=11.5)
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(OUT / f"frozen_inversion_mechanism.{ext}")
    plt.close(fig)
    print(f"  wrote {OUT}/frozen_inversion_mechanism.pdf/.png")
    print(f"    shift {shift:+.4f} = pacing {pacing:+.4f} + residual {resid:+.4f} bps "
          f"({100 * pacing / shift:.0f}% explained)")


if __name__ == "__main__":
    main()
