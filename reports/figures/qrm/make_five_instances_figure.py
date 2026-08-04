"""Five candidate edges, and what each returned on a different block of data.

WHY THIS FIGURE EXISTS. Five times in this study a measured cost difference changed materially
when the same policies were re-evaluated on a different block. Reported separately -- and until
now they were reported in four different sections -- each reads as an unlucky candidate. Placed
on one axis they are the study's methodological finding: an out-of-sample block, agreement
across independent seeds, and a small p-value are jointly compatible with an effect that
belongs to the evaluation period rather than to the policy.

THE FRAMING, AND A CORRECTION MADE WHILE BUILDING THIS. A first version framed every instance as
"an edge found on one block that died on another". That is wrong for instance 4, which runs the
OTHER WAY: those arms are more expensive than TWAP on validation (+0.074 bps) and cheaper on the
sealed block (-0.322 bps). Block luck does not only destroy an apparent edge, it can manufacture
one -- and instance 4 is the case where it manufactured an edge that passed a genuinely sealed
exam. The unified framing is therefore "the same policies, two blocks", and the finding is that
the measured difference tracks the block rather than the policy.

WHY TWO PANELS. Instances 1, 2, 3 and 5 all sit within +/-0.07 bps; instance 4 moves by 0.4 bps.
On one axis the first four would be invisible. They are also different in kind -- four candidates
that evaporated, against one that was created -- so they are separated rather than squashed.

BLOCK VOCABULARY IS CARRIED ON THE FIGURE, not left to the caption. This study distinguishes a
DEVELOPMENT block (screening, re-usable), a RESERVE block (held out but re-usable) and a SEALED
block (used once, then spent). Three of the four left-panel instances were re-tested on a sealed
block; the grid cells used a reserve block and no sealed block was spent on them. Flattening
that distinction would overstate the evidence, so each row states which kind it used.

THE CONTROL. The right panel carries what converts instance 4 from a puzzle into evidence:
agents independently diagnosed as behaviourally collapsed -- incapable of having learned
anything -- earn the same apparent edge on the same data. A learner that has stopped learning
cannot acquire skill, so what changed between the periods was the market, not the policies.

All values verified against source 2026-08-01; see the paths below.

Sources (absolute):
  .../oxford_l4/RESULTS_MANIFEST.md                      block types, instances 1-3
  .../oxford_l4/step5_selection_v3/judgement.json        instances 1-2, first-block values
  .../oxford_l4/step5_esc_*, step5_xblock_*              instance 3
  .../l2_test_results/{val_recheck,test_runs_10s}.json   instance 4 + control
  .../oxford_l4/step5_a5_armA_sameagents/                instance 5
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

OUT = Path(__file__).resolve().parent
FIRST, FRESH = "#d62728", "#1f77b4"
INK, MUTED, GRID = "#222222", "#666666", "#cccccc"
MATERIALITY = 0.05

plt.rcParams.update({"font.size": 9.5, "axes.titlesize": 10.5, "axes.labelsize": 10,
                     "figure.dpi": 150, "savefig.bbox": "tight",
                     "axes.spines.top": False, "axes.spines.right": False})

# Verified against the sources listed above on 2026-08-01.
EVAPORATED = [
    dict(label="1. Selected champion\nchosen from 98 tuning runs",
         first=-0.0628, fresh=-0.0023, block="sealed",
         note="dev $p=0.004$, 5/5 valid  →  sealed $p=0.38$, 3/5 cheaper"),
    dict(label="2. Alternative champion\nsecond selection rule",
         first=-0.0562, fresh=-0.0022, block="sealed",
         note="dev $p=0.005$, 5/5 valid  →  sealed $p=0.39$, 2/5 cheaper"),
    dict(label="3. Grid cell\none of two calm triggers",
         first=-0.0609, fresh=+0.0177, block="reserve",
         note="dev $p=0.0001$  →  reserve $p=0.96$, sign reversed"),
    dict(label="5. Strongest candidate\ncorrected observation",
         first=-0.0583, fresh=+0.0009, block="sealed",
         note="dev $p=0.0001$, 5/5 cheaper, CI below zero  →  sealed $p=0.55$, 1/5"),
]

# Instance 4: validation -> sealed exam, every behaviour-valid arm on the affected dataset.
MANUFACTURED = [("PPO, 96.6 BTC", 0.1779, -0.4416, False),
                ("PPO, 193 BTC", 0.0740, -0.3221, False),
                ("PPO, 386 BTC", 0.2010, -0.4508, False),
                ("DQN, 96.6 BTC", 0.2220, -0.4893, True),
                ("DQN, 193 BTC", 0.1419, -0.2369, True),
                ("DQN, 386 BTC", 0.2696, -0.5105, True)]


def main() -> None:
    fig, (axL, axR) = plt.subplots(1, 2, figsize=(14.4, 6.4),
                                   gridspec_kw={"width_ratios": [1.0, 1.0]})

    # ---------------- left: four candidates that evaporated -----------------------
    y = np.arange(len(EVAPORATED))[::-1]
    h = 0.32
    for xb in (-MATERIALITY, MATERIALITY):
        axL.axvline(xb, color=MUTED, lw=1.0, ls=":", zorder=1)
    axL.axvline(0, color=INK, lw=1.2, zorder=2)
    for i, ins in enumerate(EVAPORATED):
        yy = y[i]
        axL.barh(yy + h / 2, ins["first"], height=h, color=FIRST, alpha=0.85, zorder=3)
        axL.barh(yy - h / 2, ins["fresh"], height=h, color=FRESH, alpha=0.85, zorder=3)
        axL.annotate(f"{ins['note']}   [{ins['block'].upper()} block]",
                     xy=(0.015, yy - h - 0.16), xycoords=("axes fraction", "data"),
                     ha="left", va="center", fontsize=7.0, color=MUTED)
    axL.set_yticks(y)
    axL.set_yticklabels([d["label"] for d in EVAPORATED], fontsize=8.8)
    axL.set_xlabel("cost difference against adaptive TWAP (bps)      ← cheaper", labelpad=8)
    axL.set_title("Four candidates that evaporated.\n"
                  "Each cleared its bar on the block it was found on.",
                  loc="left", fontsize=10.5)
    axL.grid(axis="x", color=GRID, lw=0.5, zorder=0)
    axL.set_axisbelow(True)
    axL.margins(y=0.20)

    # ---------------- right: instance 4, an edge that was manufactured ------------
    yy = np.arange(len(MANUFACTURED))[::-1]
    axR.axvspan(-MATERIALITY, MATERIALITY, color="#999999", alpha=0.13, zorder=0)
    axR.axvline(0, color=INK, lw=1.2, zorder=2)
    for i, (name, val, test, collapsed) in enumerate(MANUFACTURED):
        row = yy[i]
        axR.barh(row + h / 2, val, height=h, color=FIRST, alpha=0.85, zorder=3)
        axR.barh(row - h / 2, test, height=h, color=FRESH, alpha=0.85, zorder=3)
    axR.set_yticks(yy)
    axR.set_yticklabels([f"{n}{' *' if c else ''}" for n, _, _, c in MANUFACTURED],
                        fontsize=8.8)
    axR.set_xlabel("cost difference against adaptive TWAP (bps)      ← cheaper", labelpad=8)
    axR.set_title("4. An edge that was manufactured. Every arm loses on\n"
                  "validation and wins on the sealed block — including the controls.",
                  loc="left", fontsize=10.5)
    axR.grid(axis="x", color=GRID, lw=0.5, zorder=0)
    axR.set_axisbelow(True)
    axR.margins(y=0.16)

    handles = [Patch(facecolor=FIRST, alpha=0.85,
                     label="first block (development, or validation for instance 4)"),
               Patch(facecolor=FRESH, alpha=0.85, label="fresh block (sealed, or reserve)"),
               Patch(facecolor="#999999", alpha=0.3,
                     label="±0.05 bps pre-registered materiality threshold")]
    fig.text(0.52, 0.115,
             "*  independently diagnosed as behaviourally collapsed: these agents defer execution until the "
             "deadline forces it.\n   A learner that has stopped learning cannot acquire skill, so what changed "
             "between the two periods was the\n   market, not the policies.",
             ha="left", va="top", fontsize=7.6, color=MUTED)
    fig.legend(handles=handles, loc="lower center", ncol=3, frameon=False,
               bbox_to_anchor=(0.5, -0.055), fontsize=9)
    fig.suptitle("The measured edge tracks the evaluation block, not the policy: "
                 "an out-of-sample block, unanimous seeds and a small $p$ are not sufficient",
                 y=1.0, fontsize=11.8)
    fig.tight_layout(rect=[0, 0.135, 1, 0.965])
    for ext in ("pdf", "png"):
        fig.savefig(OUT / f"fig25_five_instances.{ext}")
    plt.close(fig)
    print(f"wrote {OUT}/fig25_five_instances.pdf/.png")
    for d in EVAPORATED:
        print(f"  {d['label'].splitlines()[0]:<28} {d['first']:+.4f} -> {d['fresh']:+.4f}  "
              f"[{d['block']}]")
    for n, v, t, c in MANUFACTURED:
        print(f"  instance 4 {n:<17} {v:+.4f} -> {t:+.4f}  "
              f"{'[collapsed control]' if c else ''}")


if __name__ == "__main__":
    main()
