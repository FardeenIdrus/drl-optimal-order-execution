"""Figures D1, D2 and D3 for the Data chapter. Every value is read, none is typed.

FIGURE D1 -- coverage and contribution, two layers.
  The lower layer is RAW coverage: for each of the 731 calendar days, whether the pull got a
  complete day, a partial day, nothing at all, or whether the day was never attempted because
  the venue's archive has no data for that week. The upper layer is EPISODE CONTRIBUTION for
  the headline build: whether that date supplied any usable episode.
  The two layers exist because they disagree, and the disagreement is the figure's content.
  A date can be complete in the lower layer and empty in the upper one, because a minute whose
  30-minute volatility window overlaps a gap is invalidated rather than interpolated. Drawing
  one layer only would let "complete" read as "contributing", which it is not.

FIGURE D2 -- regime labelling.
  The distribution of episode realised volatility with the threshold marked. The threshold is
  the MEDIAN OF THE TRAINING SPLIT, applied unchanged to the sealed test set. It was chosen;
  the data did not supply it, and a figure that showed only the resulting counts would hide
  that. Because the test period is quieter than the training period, the same fixed threshold
  yields a test set that is not half volatile, and the figure shows the consequence next to
  the choice that produced it.

FIGURE D3 -- liquidity and participation.
  Panel (a) is resting depth by hour of day across the full recorded-book span. Panel (b) is
  traded volume by hour of day for the per-order month, which is the only period for which
  trade-level data exists. The two panels therefore cover different periods and that is
  labelled on each. The panel that matters for the argument is (b): it converts an order size
  expressed as a share of a DAY into a share of the FIVE MINUTES it is actually executed in.

Sources: scratch_hyperliquid/oxford_l4/data_chapter_measurements.json
         scratch_hyperliquid/oxford_l4/trade_volume_202512.json
         scratch_hyperliquid/adv/btc_adv.json
Output:  reports/figures/data/figD1_coverage.{pdf,png} and figD2, figD3
"""
from __future__ import annotations

import json
from datetime import date, timedelta
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch

S = Path("/Users/fardeenidrus/Desktop/MSc Dissertation/scratch_hyperliquid")
OUT = Path(__file__).resolve().parent

M = json.loads((S / "oxford_l4" / "data_chapter_measurements.json").read_text())
TV = json.loads((S / "oxford_l4" / "trade_volume_202512.json").read_text())
ADV = json.loads((S / "adv" / "btc_adv.json").read_text())

RC = M["raw_coverage"]
HEAD = M["builds"]["dataset_10s"]

plt.rcParams.update({
    "font.size": 8,
    "axes.titlesize": 9,
    "axes.labelsize": 8,
    "legend.fontsize": 7,
    "xtick.labelsize": 7,
    "ytick.labelsize": 7,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "figure.dpi": 150,
})

C_COMPLETE = "#2b6ca3"
C_PARTIAL = "#7fb2d6"
C_MISSING = "#c0392b"
C_SKIPPED = "#bdbdbd"
C_ZERO = "#e67e22"


def save(fig, stem: str) -> None:
    for ext in ("pdf", "png"):
        p = OUT / f"{stem}.{ext}"
        fig.savefig(p, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {OUT / (stem + '.pdf')} and .png")


def _dates(a: str, b: str) -> list[date]:
    x, y = date.fromisoformat(a), date.fromisoformat(b)
    return [x + timedelta(days=i) for i in range((y - x).days + 1)]


# --------------------------------------------------------------------------- Figure D1

def fig_d1() -> None:
    span = _dates(RC["span_start"], RC["span_end"])
    state = RC["state_by_date"]
    contrib = HEAD["test_episodes_by_date"]
    all_contrib = HEAD["episodes_by_date"]

    fig, ax = plt.subplots(figsize=(7.4, 2.5))

    colour = {"complete": C_COMPLETE, "partial": C_PARTIAL,
              "missing": C_MISSING, "skipped": C_SKIPPED}
    xs = [mdates.date2num(d) for d in span]
    cs = [colour[state[d.isoformat()]] for d in span]
    ax.bar(xs, [1.0] * len(span), width=1.0, color=cs, linewidth=0, align="edge")

    # Upper layer: dates inside the evaluation window that supply no episode at all.
    # The track is drawn full-width in pale grey FIRST. Without it the row is empty for 96%
    # of the span and its axis label reads as floating text beside blank paper.
    test_start = date.fromisoformat(HEAD["test_first_ts"][:10])
    ax.bar(xs, [0.34] * len(span), width=1.0, bottom=1.06, color="#f2f2f2",
           linewidth=0, align="edge", zorder=1)
    zero_x = [mdates.date2num(d) for d in span
              if d >= test_start and all_contrib.get(d.isoformat(), 0) == 0]
    ax.bar(zero_x, [0.34] * len(zero_x), width=1.0, bottom=1.06, color=C_ZERO,
           linewidth=0, align="edge", zorder=2)

    bx = mdates.date2num(test_start)
    ax.axvline(bx, color="black", linewidth=1.1, zorder=5)
    # The dataset qualification lives in the caption. Naming it here would forward-reference
    # the three dataset versions, which the reader does not meet until section 3.4.
    ax.annotate(f"evaluation period begins {HEAD['test_first_ts'][:10]}",
                xy=(bx, 1.44), xytext=(bx - 330, 1.66), fontsize=7,
                arrowprops=dict(arrowstyle="->", lw=0.7))
    ax.text(0.0, 1.0, "(a)", transform=ax.transAxes, fontsize=9, fontweight="bold",
            ha="left", va="top")

    ax.set_ylim(0, 1.9)
    ax.set_yticks([0.5, 1.23])
    ax.set_yticklabels(["what was\ncollected", "supplied no\nepisode"])
    ax.set_xlim(xs[0], xs[-1] + 1)
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b\n%Y"))
    ax.spines["left"].set_visible(False)
    ax.tick_params(axis="y", length=0)

    sc = RC["state_counts"]
    handles = [
        Patch(color=C_COMPLETE, label=f"complete ({sc['complete']})"),
        Patch(color=C_PARTIAL, label=f"partial ({sc['partial']})"),
        Patch(color=C_MISSING, label=f"missing ({sc['missing']})"),
        Patch(color=C_SKIPPED, label=f"not attempted ({sc['skipped']})"),
        Patch(color=C_ZERO, label=f"in test window, no episodes ({HEAD['eval_dates_zero']})"),
    ]
    ax.legend(handles=handles, ncol=5, loc="upper center", bbox_to_anchor=(0.5, -0.28),
              frameon=False, columnspacing=1.1, handlelength=1.2)
    save(fig, "figD1_coverage")

    # An inset would crowd the band, so November is drawn as its own panel.
    #
    # REBUILT 2026-08-05. The first version drew episode counts as a STEP LINE over a
    # full-height coloured background. Two defects: a step implies continuity between points,
    # but daily counts are discrete; and the line competed with the colour for the same space,
    # so neither encoding read cleanly. Now the counts are BARS and the coverage colour is a
    # thin strip beneath the axis. The panel's payload then reads directly: a run of absent
    # bars sitting above unbroken blue is exactly "collected in full, no episodes".
    nov = _dates("2025-11-01", "2025-11-30")
    fig, ax = plt.subplots(figsize=(7.4, 1.9))
    nx = np.array([mdates.date2num(d) for d in nov], dtype=float)
    eps = [contrib.get(d.isoformat(), 0) for d in nov]
    top = max(eps)

    # Strip height is set in data units so the bars keep the rest of the panel.
    strip_h = top * 0.13
    ax.bar(nx, [strip_h] * len(nov), width=1.0, align="edge", linewidth=0, bottom=-strip_h,
           color=[colour[state[d.isoformat()]] for d in nov], zorder=1)
    ax.axhline(0.0, color="0.35", linewidth=0.6, zorder=3)
    # Bars are a neutral dark grey, NOT the coverage blue: the two encodings must not merge.
    ax.bar(nx + 0.5, eps, width=0.72, color="#3a3a3a", linewidth=0, zorder=2)

    ax.set_ylabel("episodes\ncontributed", fontsize=7)
    ax.set_ylim(-strip_h, top * 1.42)
    ax.set_yticks([0, 20, 40])
    ax.text(nx[0] - 0.6, -strip_h / 2, "collected", fontsize=6.5, ha="right", va="center")

    # Name the payload on the panel: a reader who does not notice the absent bars over solid
    # blue has missed the only thing this panel is for.
    #
    # A SPAN BRACKET, not an arrow. What is being labelled is an ABSENCE of bars, so an arrow
    # has nothing to land on and reads as floating text. The bracket names the range itself.
    # 5 to 14 November are the ten days the collection obtained in full that yield no episode;
    # 17 November also yields none but is partial, so it is outside the "in full" claim.
    b0 = mdates.date2num(date(2025, 11, 5))
    b1 = mdates.date2num(date(2025, 11, 15))
    yb, yt = top * 0.13, top * 0.21
    ax.plot([b0, b0, b1, b1], [yb, yt, yt, yb], color="0.25", lw=0.8,
            solid_joinstyle="miter", zorder=4)
    ax.text((b0 + b1) / 2, yt + top * 0.04, "collected in full, no episodes",
            fontsize=6.5, ha="center", va="bottom", color="0.15")
    ax.set_xlim(nx[0], nx[-1] + 1)
    ax.xaxis.set_major_locator(mdates.DayLocator(interval=3))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%d"))
    ax.set_xlabel("November 2025 (day of month)")
    ax.spines["left"].set_visible(False)
    ax.tick_params(axis="y", length=0)
    ax.set_title(f"(b)  {HEAD['november_episodes']} of "
                 f"{HEAD['episodes_test']:,} evaluation episodes "
                 f"({HEAD['november_share_of_test']}%), from "
                 f"{len(HEAD['november_by_date'])} of November's 30 days",
                 loc="left", fontsize=8)
    save(fig, "figD1b_november")


# --------------------------------------------------------------------------- Figure D2

def fig_d2() -> None:
    # Each dataset version computes its own threshold by the same rule and lands in a different
    # place, so the figure names the version it draws and states the other two.
    # "Sealed" is NOT used here: section 3.4 defines that term and this figure appears in it,
    # so the figure would be using the word before the reader meets it. Same fix as F-D1(b).
    V = M["volatility"]["dataset_10s"]
    others = {k: v["threshold"] for k, v in M["volatility"].items() if k != "dataset_10s"}
    lo, hi = V["hist_range"]
    edges = np.linspace(lo, hi, V["hist_bins"] + 1)
    centres = 0.5 * (edges[:-1] + edges[1:])
    width = edges[1] - edges[0]
    thr = V["threshold"]

    fig, ax = plt.subplots(figsize=(6.4, 2.6))
    tr = np.array(V["train_hist"], dtype=float)
    te = np.array(V["test_hist"], dtype=float)
    ax.bar(centres, tr / tr.sum(), width=width, color=C_COMPLETE, alpha=0.75,
           label=f"training episodes (n={V['train_vol']['n']:,})", linewidth=0)
    ax.step(centres, te / te.sum(), where="mid", color=C_MISSING, linewidth=1.3,
            label=f"evaluation episodes (n={V['test_vol']['n']:,})")
    ax.axvline(thr, color="black", linestyle="--", linewidth=1.1)
    ax.annotate(f"threshold {thr:.5f}\nchosen as the median of the TRAINING split,\n"
                f"then applied unchanged to the evaluation set",
                xy=(thr, ax.get_ylim()[1] * 0.74),
                xytext=(thr * 1.6, ax.get_ylim()[1] * 0.60), fontsize=7,
                arrowprops=dict(arrowstyle="->", lw=0.7))

    ax.set_xlabel("episode realised volatility")
    ax.set_ylabel("share of episodes")
    ax.set_xlim(lo, hi)
    ax.legend(frameon=False, loc="upper right")
    cbs = V["counts_by_split_regime"]
    ax.set_title(
        f"10-second dataset. Training is {V['train_volatile_pct']}% volatile by "
        f"construction; the evaluation set is {V['test_volatile_pct']}% "
        f"({cbs['calm']['test']:,} calm, {cbs['volatile']['test']:,} volatile)",
        loc="left")
    # Plain names, not folder names. "dataset_10s_10min" is an internal path and must not
    # appear in a report-facing figure; the labels match \Cref{tab:d3-partitions}.
    plain = {"dataset": "1-minute", "dataset_10s_10min": "10-second, shorter episode"}
    ax.annotate("same rule, other versions: "
                + ", ".join(f"{plain.get(k, k)} {v:.5f}"
                            for k, v in sorted(others.items(), key=lambda kv: plain.get(kv[0], kv[0]))),
                xy=(0.5, -0.30), xycoords="axes fraction", ha="center", fontsize=6.5,
                color="0.35")
    save(fig, "figD2_regimes")


# --------------------------------------------------------------------------- Figure D3

def fig_d3() -> None:
    D = M["depth_by_hour"]
    hours = np.arange(24)
    # Two DISJOINT series. Nested totals invite a reader to add them.
    touch = np.array([D["touch_btc"][str(h)] for h in hours])
    behind = np.array([D["behind_btc"][str(h)] for h in hours])

    dec = TV["december_only"]
    vol_h = np.array([dec["mean_btc_per_hour_of_day"][str(h)] for h in hours])

    fig, (a1, a2) = plt.subplots(1, 2, figsize=(7.4, 3.1))

    a1.plot(hours, touch, marker="s", ms=3, color=C_MISSING, label="at the best prices")
    a1.plot(hours, behind, marker="o", ms=3, color=C_COMPLETE, label="at levels 2 to 5")
    a1.set_xlabel("hour of day (UTC)")
    a1.set_ylabel("resting volume (BTC)")
    a1.set_xticks(range(0, 24, 4))
    a1.set_ylim(0, max(touch.max(), behind.max()) * 1.35)
    a1.legend(frameon=False, loc="upper left")
    a1.set_title(f"(a) resting volume, December 2025\n"
                 f"varies {touch.max() / touch.min():.1f}$\\times$ at the best prices",
                 loc="left")

    a2.bar(hours, vol_h, color=C_COMPLETE, width=0.8, linewidth=0)
    a2.set_xlabel("hour of day (UTC)")
    a2.set_ylabel("traded volume (BTC per hour)")
    a2.set_xticks(range(0, 24, 4))
    a2.set_title(f"(b) traded volume, December 2025\n"
                 f"varies {vol_h.max() / vol_h.min():.1f}$\\times$ over the same hours",
                 loc="left")

    # The participation contrast: the primary order against the volume of the five minutes it
    # is executed in, at the quietest and the busiest hour of the day.
    prim = 25.0
    five_min_lo = vol_h.min() / 12.0
    five_min_hi = vol_h.max() / 12.0
    # Headroom above the tallest bar, so the box never sits on the data.
    a2.set_ylim(0, vol_h.max() * 1.42)
    a2.annotate(
        f"25 BTC = {100 * prim / dec['mean_daily_btc']:.3f}% of a day,\n"
        f"but {100 * prim / five_min_hi:.0f}% to {100 * prim / five_min_lo:.0f}% of the "
        f"five minutes it is executed in",
        xy=(0.5, 0.985), xycoords="axes fraction", ha="center", va="top", fontsize=7,
        bbox=dict(boxstyle="round,pad=0.35", fc="white", ec="0.7", lw=0.6))

    fig.tight_layout()
    save(fig, "figD3_liquidity")
    print(f"  participation: 25 BTC is {100 * prim / five_min_hi:.1f}% to "
          f"{100 * prim / five_min_lo:.1f}% of a five-minute window")


def main() -> None:
    fig_d1()
    fig_d2()
    fig_d3()


if __name__ == "__main__":
    main()
