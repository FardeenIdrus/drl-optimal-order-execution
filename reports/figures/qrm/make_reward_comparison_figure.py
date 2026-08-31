"""F27 -- average reward per episode, broken critic against working critic, on both tracks.

THE QUESTION THIS ANSWERS, and it is the supervisor's, in his words: "average reward per episode
should trend upward during training". F23 answers it for the ORIGINAL agents only -- the ten
whose critic explains nothing -- so the obvious reply was always "and what does it look like once
the critic is fixed?". Until 2026-08-11 that could not be answered: the three campaigns carrying
a working critic all trained with `log_learning: false` and no reward series existed for any of
them. Thirty agents were retrained with logging and every one reproduced its original's final
curve value at exactly zero delta, so these curves belong to the agents of record.

WHAT IS PLOTTED. Four series, two per panel:

  PRIMARY track   runs_primary_v3_logged         27 inputs, critic explains -0.003
                  runs_primary_v3_obsfix_logged  28 inputs, critic explains +0.422
  INJECTED track  runs_signal_logged             28 inputs, critic explains -0.004
                  runs_signal_obsfix_logged      29 inputs, critic explains +0.405

The primary track is drawn FIRST (left) because it carries the dissertation's central claim.

REWARD SIGN, VERIFIED IN CODE, because drawing it upside down would invert the whole reading.
`reactive_env.py:406`: reward = -is_usd / (arrival_mid * order_btc) * 1e4 -- negative
implementation shortfall in bps of arrival notional. **HIGHER IS BETTER.** A falling curve is an
agent getting more expensive, not cheaper. The axis label says so on the figure.

WHY MEANS AND BANDS RATHER THAN TWENTY LINES. Ten seeds per arm, two arms per panel, is forty
curves; drawn individually they are unreadable and the comparison that matters -- between arms --
disappears. Each arm is therefore the across-seed mean with a +-1 sd band, and the per-run
trend statistics are printed to the console and belong in the caption.

COMPARABILITY, stated because it bounds what the figure may be used for. Reward is comparable
BETWEEN the two arms inside one panel: same environment, same reward function, the only
difference is one observation feature. It is NOT comparable BETWEEN panels, because the base and
injected environments generate different price paths. No cross-panel claim is made or licensed.

Sources (absolute):
  .../scratch_hyperliquid/oxford_l4/runs_primary_v3_logged/ppo_*/progress.csv
  .../scratch_hyperliquid/oxford_l4/runs_primary_v3_obsfix_logged/ppo_*/progress.csv
  .../scratch_hyperliquid/oxford_l4/runs_signal_logged/ppo_*/progress.csv
  .../scratch_hyperliquid/oxford_l4/runs_signal_obsfix_logged/ppo_*/progress.csv
Gate record: .../scratch_hyperliquid/oxford_l4/logged_rerun_gate.json
"""
from __future__ import annotations

import glob
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

S = Path("/Users/fardeenidrus/Desktop/MSc Dissertation/scratch_hyperliquid/oxford_l4")
OUT = Path(__file__).resolve().parent
OI_VERM, MUTED, INK = "#D55E00", "#666666", "#222222"
SMOOTH = 25          # rolling window over logged rows; the raw per-rollout series is very noisy

PANELS = [
    ("The primary reacting market", "runs_primary_v3_logged", "runs_primary_v3_obsfix_logged",
     "27 inputs", "28 inputs"),
    ("The injected-signal market", "runs_signal_logged", "runs_signal_obsfix_logged",
     "28 inputs", "29 inputs"),
]

plt.rcParams.update({"font.size": 10, "axes.titlesize": 11, "axes.labelsize": 10,
                     "figure.dpi": 150, "savefig.bbox": "tight",
                     "axes.spines.top": False, "axes.spines.right": False})


def load_arm(dirname: str) -> tuple[np.ndarray, np.ndarray, list[dict]]:
    """Returns (steps_in_millions, matrix[seed, step], per-run trend stats)."""
    series, stats_rows = [], []
    for f in sorted(glob.glob(str(S / dirname / "ppo_*" / "progress.csv"))):
        d = pd.read_csv(f)[["time/total_timesteps", "rollout/ep_rew_mean"]].dropna()
        x = d.iloc[:, 0].to_numpy() / 1e6
        y = d.iloc[:, 1].to_numpy()
        r = stats.linregress(x, y)
        k = max(1, len(y) // 10)
        stats_rows.append({
            "run": Path(f).parent.name, "slope": r.slope, "p": r.pvalue,
            "first10": y[:k].mean(), "last10": y[-k:].mean(),
        })
        series.append(pd.Series(y).rolling(SMOOTH, min_periods=1, center=True).mean().to_numpy())
        steps = x
    n = min(len(s) for s in series)
    return steps[:n], np.vstack([s[:n] for s in series]), stats_rows


def main() -> None:
    # Authored at the printed width (6.3in, \textwidth), panels stacked, so every
    # font renders at its stated size (point sizes are absolute; resizing requires retuning every font and margin).
    fig, axes = plt.subplots(2, 1, figsize=(6.3, 7.2), sharey=True)
    console = []

    for ax, (title, dir_old, dir_new, lab_old, lab_new) in zip(axes, PANELS):
        # The two fitted trends are nearly identical -- the finding -- so they are given
        # INTERLEAVING dash patterns. Drawn with the same pattern, the second simply erases
        # the first and the figure appears to show one arm.
        for dirname, colour, lab, dash in ((dir_old, MUTED, lab_old, (7, 4)),
                                           (dir_new, OI_VERM, lab_new, (2, 4))):
            x, M, rows = load_arm(dirname)
            m, sd = M.mean(axis=0), M.std(axis=0, ddof=1)
            ax.fill_between(x, m - sd, m + sd, color=colour, alpha=0.09, linewidth=0, zorder=1)
            # The two arms lie almost on top of each other -- which IS the finding -- so the
            # mean curves are drawn semi-transparent or the lower one is simply invisible.
            ax.plot(x, m, color=colour, lw=1.3, alpha=0.70, zorder=3)
            # The straight fit carries the claim. Without it the downward drift is asserted in
            # an annotation and cannot be checked against the picture.
            fit = stats.linregress(x, m)
            ax.plot(x, fit.intercept + fit.slope * x, color=colour, lw=2.6,
                    dashes=dash, zorder=4)

            sl = np.array([r["slope"] for r in rows])
            up = sum(r["slope"] > 0 and r["p"] < 0.05 for r in rows)
            dn = sum(r["slope"] < 0 and r["p"] < 0.05 for r in rows)
            console.append((title, dirname, lab, len(rows), sl.mean(), up, dn,
                            np.mean([r["first10"] for r in rows]),
                            np.mean([r["last10"] for r in rows])))

        ax.axhline(0.0, color=INK, lw=0.8, ls=":", zorder=0)
        ax.set_xlabel("training steps (millions)")
        ax.set_title(title, fontsize=11.5)

    for ax in axes:
        ax.set_ylabel("average reward per episode\n[higher is cheaper execution]")

    # The claim of the figure, stated on the figure rather than left to the caption.
    for ax, (title, *_rest) in zip(axes, PANELS):
        rows = [c for c in console if c[0] == title]
        a, b = rows[0], rows[1]
        ax.annotate(f"mean trend  {a[4]:+.3f}  $\\rightarrow$  {b[4]:+.3f}  reward per Mstep",
                    xy=(0.5, 0.03), xycoords="axes fraction", ha="center", va="bottom",
                    fontsize=8.2, color=INK)

    handles = [Line2D([], [], color=MUTED, lw=2,
                      label="without the arrival price in the inputs"),
               Line2D([], [], color=OI_VERM, lw=2,
                      label="with the arrival price in the inputs"),
               Line2D([], [], color=INK, lw=2.6, dashes=(4, 3), label="fitted trend (dashed)"),
               Line2D([], [], color=MUTED, lw=6, alpha=0.20,
                      label="+/- 1 sd across 10 seeds")]
    fig.legend(handles=handles, loc="lower center", ncol=2, frameon=False,
               bbox_to_anchor=(0.5, -0.07))
    fig.suptitle("Average episode reward does not improve, with or\n"
                 "without the arrival price in the agent's inputs", y=1.005, fontsize=12.5)
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(OUT / f"fig27_reward_comparison.{ext}")
    plt.close(fig)
    print(f"wrote {OUT}/fig27_reward_comparison.pdf/.png\n")

    print(f"{'panel':30s} {'arm':32s} {'n':>3} {'slope/Mstep':>12} {'sigUP':>6} "
          f"{'sigDOWN':>8} {'first10%':>9} {'last10%':>9}")
    for c in console:
        print(f"{c[0]:30s} {c[1]:32s} {c[3]:3d} {c[4]:+12.4f} {c[5]:6d} {c[6]:8d} "
              f"{c[7]:+9.3f} {c[8]:+9.3f}")


if __name__ == "__main__":
    main()
