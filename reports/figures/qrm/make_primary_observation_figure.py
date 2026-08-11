"""F26 -- the observation specification tested on the PRIMARY (base) reacting-market track.

WHY THIS EXISTS, AND WHAT IT FIXES. S11 shows the same test on the INJECTED track: the critic
goes from explaining nothing to explaining well, and the cost does not move. That figure is
often read as covering the whole study. It does not. It covers the third environment. The
primary track -- the one carrying the dissertation's central claim -- had its COST verdict
reported in live-doc addendum (T) and its critic NEVER MEASURED. So the mechanism was
established where the claim is weakest and assumed where the claim is strongest.

This figure closes that, deliberately mirroring S11's layout so the two can be read as a pair:

  LEFT   the mechanism -- critic explained variance per run, original vs with the added feature
  RIGHT  the verdict   -- cost against adaptive TWAP, original vs with the added feature

DQN APPEARS IN THE RIGHT PANEL ONLY, and its absence on the left is not an omission. Explained
variance of a critic is a policy-gradient quantity fitted against returns; DQN has no such
object, so the measurement is undefined rather than merely missing. Its A4.3 arm was 10/10
behaviourally invalid, which is itself the finding and is annotated on the panel.

Sources (absolute):
  .../oxford_l4/diagnostics_primary/diag_learning_primary.json   (reports/diagnostics/diag_learning_primary.py)
  .../oxford_l4/step5_v3/{judgement,behaviour_audit}.json         campaign of record
  .../oxford_l4/step5_primary_v3_obsfix/{judgement,behaviour_audit}.json   Amendment A4.3
"""
from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

S = Path("/Users/fardeenidrus/Desktop/MSc Dissertation/scratch_hyperliquid/oxford_l4")
OUT = Path(__file__).resolve().parent
OI_VERM, MUTED, INK = "#D55E00", "#666666", "#222222"

plt.rcParams.update({"font.size": 10, "axes.titlesize": 11, "axes.labelsize": 10,
                     "figure.dpi": 150, "savefig.bbox": "tight",
                     "axes.spines.top": False, "axes.spines.right": False})


def _load(dirname: str):
    j = json.loads((S / dirname / "judgement.json").read_text())
    a = {e["run"]: e for e in json.loads((S / dirname / "behaviour_audit.json").read_text())}
    return j, a


def _is_base_run(run: str, seed: int) -> bool:
    """step5_v3 holds tuning variants alongside the base runs; keep only the untagged ones."""
    return re.search(rf"_s{seed}$", run) is not None


def main() -> None:
    diag = json.loads((S / "diagnostics_primary" / "diag_learning_primary.json").read_text())
    dm = {(d["arm"], d["run"]): d for d in diag}
    jo, ao = _load("step5_v3")
    jn, an = _load("step5_primary_v3_obsfix")

    fig, axes = plt.subplots(1, 2, figsize=(10.2, 4.2))

    # ------------------------------------------------------------------ left: mechanism
    ax = axes[0]
    runs = [f"ppo_{r}_s{s}" for r in ("calm", "volatile") for s in range(5)]
    for i, run in enumerate(runs):
        o = dm[("original", run)]["critic"]["explained_variance"]
        n = dm[("obsfix", run)]["critic"]["explained_variance"]
        ax.plot([o, n], [i, i], "-", color=MUTED, lw=1.0, zorder=1)
        ax.plot(o, i, "o", color=MUTED, mfc="white", ms=6, zorder=3)
        ax.plot(n, i, "o", color=OI_VERM, ms=6, zorder=3)
    ax.axvline(0.0, color=INK, lw=1.0, ls="--", zorder=0)
    ax.set_yticks(np.arange(len(runs)))
    ax.set_yticklabels([r.replace("ppo_", "").replace("_", " ") for r in runs], fontsize=8)
    ax.set_xlabel("critic explained variance   [higher = the value function predicts]")
    ax.set_title("The mechanism: the critic becomes learnable")
    om = np.mean([dm[("original", r)]["critic"]["explained_variance"] for r in runs])
    nm = np.mean([dm[("obsfix", r)]["critic"]["explained_variance"] for r in runs])
    ax.set_ylim(-0.9, len(runs) - 0.3)
    ax.annotate(f"mean {om:+.3f}  $\\rightarrow$  {nm:+.3f}", xy=(0.97, 0.03),
                xycoords="axes fraction", ha="right", va="bottom", fontsize=8.5, color=INK)

    # -------------------------------------------------------------------- right: verdict
    ax = axes[1]
    labels, om_, nm_, on_, nn_ = [], [], [], [], []
    for algo in ("ppo", "dqn"):
        for regime in ("calm", "volatile"):
            o = [r["mean_vs_adaptive_bps"] for r in jo["per_run"]
                 if r["algo"] == algo and r["regime"] == regime
                 and _is_base_run(r["run"], r["seed"]) and ao[r["run"]]["valid"]]
            n = [r["mean_vs_adaptive_bps"] for r in jn["per_run"]
                 if r["algo"] == algo and r["regime"] == regime and an[r["run"]]["valid"]]
            labels.append(f"{algo.upper()}\n{regime}")
            om_.append(np.mean(o) if o else np.nan); on_.append(len(o))
            nm_.append(np.mean(n) if n else np.nan); nn_.append(len(n))
    x = np.arange(len(labels)); w = 0.34
    b1 = ax.bar(x - w / 2, np.nan_to_num(om_), w, color=MUTED, edgecolor="white", linewidth=1.0)
    b2 = ax.bar(x + w / 2, np.nan_to_num(nm_), w, color=OI_VERM, edgecolor="white",
                linewidth=1.0)
    # Labels go ABOVE every bar whichever way it points, on two staggered rows and one line
    # each. Two-line labels at a common height overlapped their neighbour, and negative labels
    # placed underneath collided with the x tick labels. The valid-seed counts move into the
    # tick labels for the same reason.
    for row, (bars, means) in enumerate(((b1, om_), (b2, nm_))):
        for b, m in zip(bars, means):
            txt = "none valid" if np.isnan(m) else f"{m:+.4f}"
            top = max(0.0, 0.0 if np.isnan(m) else m)
            ax.text(b.get_x() + b.get_width() / 2, top + 0.005 + 0.017 * row, txt,
                    ha="center", va="bottom", fontsize=7.2,
                    color=MUTED if row == 0 else OI_VERM)
    ax.axhline(0.0, color=MUTED, lw=1.0, ls="--")
    ax.axhspan(-0.05, 0.05, color=MUTED, alpha=0.12, zorder=0)
    ax.set_xticks(x)
    ax.set_xticklabels([f"{lab}\nvalid {a}/5 $\\rightarrow$ {b}/5"
                        for lab, a, b in zip(labels, on_, nn_)], fontsize=8.5)
    ax.set_ylabel("cost vs adaptive TWAP (bps)")
    ax.set_title("The verdict: performance is unchanged")
    lo = min([v for v in om_ + nm_ if not np.isnan(v)] + [-0.05])
    hi = max([v for v in om_ + nm_ if not np.isnan(v)] + [0.05])
    ax.set_ylim(lo - 0.018, hi + 0.072)

    handles = [Line2D([], [], color=MUTED, marker="o", mfc="white", ls="none", ms=6,
                      label="original observation (27 inputs)"),
               Line2D([], [], color=OI_VERM, marker="o", ls="none", ms=6,
                      label="with price-vs-arrival (28 inputs)"),
               Patch(facecolor=MUTED, alpha=0.12, label="+/-0.05 bps materiality band")]
    fig.legend(handles=handles, loc="lower center", ncol=3, frameon=False,
               bbox_to_anchor=(0.5, -0.13))
    fig.suptitle("The primary reacting market: making the value function learnable does not "
                 "make the agents competitive", y=1.02, fontsize=11)
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(OUT / f"fig26_primary_observation.{ext}")
    plt.close(fig)
    print(f"wrote {OUT}/fig26_primary_observation.pdf/.png")
    print(f"  critic EV, primary track: mean {om:+.4f} -> {nm:+.4f}")
    for lab, a, b, ka, kb in zip(labels, om_, nm_, on_, nn_):
        print(f"  {lab.replace(chr(10),' '):16s} {a:+.4f} ({ka}/5) -> {b:+.4f} ({kb}/5)")


if __name__ == "__main__":
    main()
