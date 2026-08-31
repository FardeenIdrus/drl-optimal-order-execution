"""Results-section figures for the LaTeX report.

Generates PDF (vector, for \\includegraphics) + PNG (preview) into reports/figures/
from the EXISTING sealed result files only (no new evaluation). Every number is read
from the source-of-record JSONs archived under results_archive/.

Style rules applied throughout (v2, after user review):
  - legends OUTSIDE the axes (below), never overlapping data;
  - seeds drawn as small open circles in their own column, clearly separated from the
    filled mean +- 95% CI marker (no random jitter; deterministic offsets);
  - no log-axis minor-tick label clutter; plain numbers on all axes;
  - every shaded band gets an explicit legend entry.

Run:  .venv/bin/python reports/figures/make_figures.py
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
from matplotlib.ticker import FixedLocator, NullLocator

S = Path("/Users/fardeenidrus/Desktop/MSc Dissertation/scratch_hyperliquid/oxford_l4")
OUT = Path(__file__).resolve().parent
plt.rcParams.update({
    "font.size": 10, "axes.titlesize": 11, "axes.labelsize": 10,
    "figure.dpi": 150, "savefig.bbox": "tight",
    "axes.spines.top": False, "axes.spines.right": False,
})
BPS = "cost vs adaptive-TWAP (bps)"
BLUE, RED = "#1f77b4", "#d62728"


def _load(dirname: str):
    j = json.load(open(S / dirname / "judgement.json"))
    a = {e["run"]: e for e in json.load(open(S / dirname / "behaviour_audit.json"))}
    return j, a


def _tag_of(run: str, seed: int) -> str:
    m = re.search(rf"_s{seed}(_.+)?$", run)
    return (m.group(1) or "") if m else ""


def _seed_means(j, audit, regime: str, tag: str, algo: str = "ppo", valid_only=True):
    out = []
    for r in j["per_run"]:
        if r["algo"] != algo or r["regime"] != regime:
            continue
        if _tag_of(r["run"], r["seed"]) != tag:
            continue
        if valid_only and not audit[r["run"]]["valid"]:
            continue
        out.append(r["mean_vs_adaptive_bps"])
    return np.array(out)


def _ci95(x: np.ndarray) -> float:
    from scipy.stats import t as tdist
    n = len(x)
    if n < 2:
        return 0.0
    return float(tdist.ppf(0.975, n - 1) * x.std(ddof=1) / np.sqrt(n))


def _seed_offsets(n: int, width: float = 0.05) -> np.ndarray:
    """Deterministic, symmetric vertical-strip offsets for n seed dots."""
    if n == 1:
        return np.zeros(1)
    return np.linspace(-width, width, n)



def _benchmark_hline(ax, x_frac: float = 0.985, ha: str = "right", va: str = "bottom"):
    """Zero line = the adaptive-TWAP benchmark (paired-difference space), labelled."""
    from matplotlib.transforms import blended_transform_factory
    ax.axhline(0, color="k", lw=0.9)
    tr = blended_transform_factory(ax.transAxes, ax.transData)
    ax.text(x_frac, 0, "adaptive-TWAP\nbenchmark", transform=tr, fontsize=7.5,
            color="k", ha=ha, va=va, linespacing=1.1)


def _benchmark_vline(ax, y_frac: float = 0.02):
    """Vertical variant for horizontal (forest) layouts."""
    from matplotlib.transforms import blended_transform_factory
    ax.axvline(0, color="k", lw=0.9)
    tr = blended_transform_factory(ax.transData, ax.transAxes)
    ax.text(0.002, y_frac, "adaptive-TWAP benchmark", transform=tr, fontsize=7.5,
            color="k", ha="left", va="bottom", rotation=90)


def save(fig, name: str):
    fig.savefig(OUT / f"{name}.pdf")
    fig.savefig(OUT / f"{name}.png")
    plt.close(fig)
    print(f"wrote {name}.pdf/.png")


# ---------------------------------------------------------------- figure 1
def fig_dev_vs_sealed():
    """Headline: development edge vs sealed out-of-sample, both champions."""
    sel, sa = _load("step5_selection_v3")
    c3a, a3a = _load("step5_confirm_v3a")
    c1b, a1b = _load("step5_confirm_v1b")

    groups = [
        ("faster-learning agent\n(learning rate 1e-3)",
         _seed_means(sel, sa, "volatile", "_v3a"),
         _seed_means(c3a, a3a, "volatile", "_v3a")),
        ("bigger-network agent\n(128 x 128)",
         _seed_means(sel, sa, "volatile", "_v1b"),
         _seed_means(c1b, a1b, "volatile", "_v1b")),
    ]
    fig, ax = plt.subplots(figsize=(6.4, 3.8))
    centers = []
    for i, (name, dev, sealed) in enumerate(groups):
        base = i * 2.0
        for k, (vals, color) in enumerate([(dev, BLUE), (sealed, RED)]):
            xc = base + (-0.35 if k == 0 else 0.35)
            # seeds: open circles in their own column, left of the mean marker
            ax.scatter(np.full(len(vals), xc - 0.14) + _seed_offsets(len(vals)),
                       vals, s=22, facecolors="none", edgecolors=color,
                       linewidths=1.1, zorder=2)
            # mean +- 95% CI: filled marker, clearly separated
            ax.errorbar(xc + 0.10, vals.mean(), yerr=_ci95(vals), fmt="D", ms=7,
                        color=color, capsize=5, elinewidth=1.6, zorder=3)
        centers.append(base)
    _benchmark_hline(ax, x_frac=0.99, ha="right")
    ax.set_xticks(centers)
    ax.set_xticklabels([g[0] for g in groups])
    ax.set_xlim(-1.1, centers[-1] + 1.75)   # extra right margin for the benchmark label
    ax.set_ylabel(BPS)
    ax.set_title("The apparent edge does not survive out-of-sample testing")
    handles = [
        Line2D([], [], marker="D", ls="", color=BLUE, ms=7,
               label="development block: mean ± 95% CI (in-sample selection)"),
        Line2D([], [], marker="D", ls="", color=RED, ms=7,
               label="confirmation block: mean ± 95% CI (opened once, fresh seeds)"),
        Line2D([], [], marker="o", ls="", markerfacecolor="none",
               markeredgecolor="grey", ms=6, label="individual seeds (5 per condition)"),
    ]
    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, -0.02),
               ncol=2, frameon=False, fontsize=8.5)
    save(fig, "fig1_dev_vs_sealed")


# ---------------------------------------------------------------- figure 2
def fig_size_response():
    """Size ladder: pooled edge vs order size, per regime (5-min horizon)."""
    sel, sa = _load("step5_selection_v3")
    cells = {5.0: _load("step5_sweep_b5"), 12.5: _load("step5_sweep_b12"),
             50.0: _load("step5_sweep_b50")}
    tags = {5.0: "_v3aB5", 12.5: "_v3aB12", 50.0: "_v3aB50"}
    sizes = [5.0, 12.5, 25.0, 50.0]

    fig, ax = plt.subplots(figsize=(6.2, 3.8))
    for regime, color, marker in [("volatile", RED, "o"), ("calm", BLUE, "s")]:
        ys, es = [], []
        for size in sizes:
            if size == 25.0:
                vals = _seed_means(sel, sa, regime, "_v3a")
            else:
                j, a = cells[size]
                vals = _seed_means(j, a, regime, tags[size])
            ys.append(vals.mean()); es.append(_ci95(vals))
        # equal categorical spacing: positions 0..3, labelled with the true sizes
        ax.errorbar(range(len(sizes)), ys, yerr=es, fmt=f"-{marker}", capsize=4,
                    color=color, label=f"{regime} regime", ms=6, elinewidth=1.4)
    _benchmark_hline(ax, x_frac=1.02, ha="left", va="center")
    ax.set_xticks(range(len(sizes)))
    ax.set_xticklabels(["5", "12.5", "25", "50"])
    # 25 BTC = the development size; noted in the LaTeX caption, marked lightly here:
    ax.axvline(2, color="grey", lw=0.7, ls=":", alpha=0.6, zorder=1)
    ax.set_xlabel("order size (BTC)   —   dotted line marks the development size (25)")
    ax.set_ylabel(BPS)
    ax.set_title("The apparent edge does not scale with order size\n"
                 "(a genuine impact-management edge should grow with size)")
    fig.legend(loc="upper center", bbox_to_anchor=(0.5, -0.02), ncol=2,
               frameon=False, fontsize=9)
    save(fig, "fig2_size_response")


# ---------------------------------------------------------------- figure 3
def fig_forest_selection():
    """Forest plot: every tuned variant (volatile), mean +- 95% CI, dev block."""
    sel, sa = _load("step5_selection_v3")
    NAMES = [("_v3a", "faster learning (lr 1e-3)"), ("_w3a", "combo: net 128 + reward x100"),
             ("_v3b", "slower learning (lr 1e-4)"), ("_v1b", "bigger network 128"),
             ("_v6", "reward x100 @ 10M steps"), ("_v5", "longer rollout"),
             ("_v4a", "no exploration bonus"), ("_v2", "reward x100"),
             ("_v6b", "slower-lr @ 10M steps"), ("_w3b", "combo: slow-lr + reward"),
             ("_v4b", "more exploration"), ("_v1a", "bigger network 64")]
    from matplotlib.transforms import blended_transform_factory
    fig, ax = plt.subplots(figsize=(6.6, 4.8))
    ys, ticklabels = [], []
    for i, (tag, label) in enumerate(NAMES):
        vals = _seed_means(sel, sa, "volatile", tag)
        if len(vals) == 0:
            continue
        y = len(NAMES) - i
        # descriptive whiskers: full seed range (mixed n makes t-CIs misleading here;
        # exact across-seed statistics are in the results table)
        ax.plot([vals.min(), vals.max()], [y, y], color="grey", lw=1.2,
                alpha=0.7, zorder=1)
        ax.scatter(vals, np.full(len(vals), y), s=20, facecolors="white",
                   edgecolors="grey", linewidths=1.0, zorder=2)
        ax.plot(vals.mean(), y, "D", ms=6.5, color=BLUE, zorder=3)
        ys.append(y); ticklabels.append(f"{label}  (n={len(vals)})")
    ax.axvline(0, color="k", lw=0.9)
    ax.axvline(-0.05, color="grey", lw=1.0, ls="--")
    # floor label ABOVE the axes (blended transform: x in data coords, y in axes coords)
    tr = blended_transform_factory(ax.transData, ax.transAxes)
    ax.text(-0.05, 1.015, "materiality threshold (-0.05)", transform=tr,
            ha="center", va="bottom", fontsize=8, color="grey")
    ax.text(0.0, 1.015, "adaptive-TWAP benchmark", transform=tr,
            ha="center", va="bottom", fontsize=8, color="k")
    ax.set_yticks(ys)
    ax.set_yticklabels(ticklabels, fontsize=8.5)
    ax.set_ylim(min(ys) - 0.7, max(ys) + 0.7)
    ax.set_xlabel(BPS + "  —  development block")
    ax.set_title("Tuning screen, volatile regime: a consistent small in-sample advantage\n"
                 "across variants (later shown to be shared evaluation-block luck)",
                 pad=18)
    handles = [
        Line2D([], [], marker="D", ls="", color=BLUE, ms=6.5, label="mean over seeds"),
        Line2D([], [], marker="o", ls="-", color="grey", markerfacecolor="white",
               markeredgecolor="grey", ms=6, lw=1.2,
               label="individual seeds (whisker = seed range)"),
    ]
    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, -0.02),
               ncol=2, frameon=False, fontsize=8.5)
    save(fig, "fig3_forest_tuning_volatile")


# ---------------------------------------------------------------- figure 4
def fig_three_block():
    """Same configs measured on three different evaluation blocks: sign flips."""
    def curve_final(path_pattern, seeds):
        vals = []
        for s in seeds:
            c = json.loads((S / path_pattern.format(s=s) / "curve.json").read_text())
            vals.append(c[-1]["mean_diff_bps"])
        return float(np.mean(vals))

    sel, sa = _load("step5_selection_v3")
    c3a, a3a = _load("step5_confirm_v3a")
    c1b, a1b = _load("step5_confirm_v1b")
    base_dev = _seed_means(*_load("step5_v3"), "volatile", "")

    # (label, monitor_1e6, dev_5e6, sealed-or-None)  [sealed blocks differ: v3a 9e6, v1b 13e6]
    rows = [
        ("base PPO", curve_final("runs_primary_v3/ppo_volatile_s{s}", range(5)),
         float(base_dev.mean()), None),
        ("faster-learning (selected)", curve_final("runs_tuning_v3/ppo_volatile_s{s}_v3a", range(5)),
         float(_seed_means(sel, sa, "volatile", "_v3a").mean()),
         float(_seed_means(c3a, a3a, "volatile", "_v3a").mean())),
        ("slower-learning", curve_final("runs_tuning_v3/ppo_volatile_s{s}_v3b", range(5)),
         float(_seed_means(sel, sa, "volatile", "_v3b").mean()), None),
        ("bigger-network", curve_final("runs_tuning_v3/ppo_volatile_s{s}_v1b", range(5)),
         float(_seed_means(sel, sa, "volatile", "_v1b").mean()),
         float(_seed_means(c1b, a1b, "volatile", "_v1b").mean())),
    ]
    fig, ax = plt.subplots(figsize=(6.4, 3.9))
    for label, m, d, sld in rows:
        xs, ys = [0, 1], [m, d]
        if sld is not None:
            xs.append(2); ys.append(sld)
        ax.plot(xs, ys, "-o", ms=5, label=label, alpha=0.9)
    _benchmark_hline(ax, x_frac=1.02, ha="left", va="center")
    ax.set_xticks([0, 1, 2])
    ax.set_xticklabels(["monitor block\n(never used for decisions)",
                        "development block\n(used for ALL decisions)",
                        "sealed blocks\n(one-shot each)"], fontsize=8.5)
    ax.set_ylabel(BPS)
    ax.set_title("Every configuration flips sign across evaluation blocks\n"
                 "(the development-block 'edge' was block luck, not skill)")
    fig.legend(loc="upper center", bbox_to_anchor=(0.5, -0.02), ncol=2,
               frameon=False, fontsize=8.5)
    save(fig, "fig4_three_block_signflip")


# ---------------------------------------------------------------- figure 5
def fig_learning_curves():
    """Training progress (curve.json): base PPO + selected config, volatile."""
    fig, ax = plt.subplots(figsize=(6.4, 3.8))
    groups = [("runs_primary_v3/ppo_volatile_s{s}", range(5), "base PPO", BLUE),
              ("runs_tuning_v3/ppo_volatile_s{s}_v3a", range(5),
               "faster-learning (selected)", RED)]
    for pat, seeds, label, color in groups:
        curves = []
        for s in seeds:
            c = json.loads((S / pat.format(s=s) / "curve.json").read_text())
            curves.append([p["mean_diff_bps"] for p in c])
        steps = [p["steps"] for p in json.loads(
            (S / pat.format(s=0) / "curve.json").read_text())]
        vals = np.array(curves)
        ax.plot(steps, vals.mean(0), color=color, lw=1.6)
        ax.fill_between(steps, vals.mean(0) - vals.std(0), vals.mean(0) + vals.std(0),
                        color=color, alpha=0.15)
    _benchmark_hline(ax, x_frac=1.02, ha="left", va="center")
    ax.set_xlabel("training steps")
    ax.set_ylabel("cost vs adaptive-TWAP (bps)\non the monitor block (n=200 episodes)")
    ax.set_title("Training curves, volatile regime")
    handles = [
        Line2D([], [], color=BLUE, lw=1.6, label="base PPO — mean over 5 seeds"),
        Line2D([], [], color=RED, lw=1.6, label="faster-learning (selected) — mean over 5 seeds"),
        Patch(facecolor="grey", alpha=0.25, label="shaded band = ± 1 standard deviation across the 5 seeds"),
    ]
    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, -0.02),
               ncol=2, frameon=False, fontsize=8.5)
    save(fig, "fig5_learning_curves_volatile")





# ---------------------------------------------------------------- figure 6
def fig_drift_confound():
    """The drift confound: identical 20-run design with and without the directional drift."""
    v2, a2 = _load("SUPERSEDED_step5_v2")
    v3, a3 = _load("step5_v3")
    groups = [("calm regime", _seed_means(v2, a2, "calm", ""), _seed_means(v3, a3, "calm", "")),
              ("volatile regime", _seed_means(v2, a2, "volatile", ""), _seed_means(v3, a3, "volatile", ""))]
    fig, ax = plt.subplots(figsize=(6.4, 3.8))
    centers = []
    for i, (name, drifty, fixed) in enumerate(groups):
        base = i * 2.0
        for k, (vals, color) in enumerate([(drifty, "#7f7f7f"), (fixed, BLUE)]):
            xc = base + (-0.35 if k == 0 else 0.35)
            ax.scatter(np.full(len(vals), xc - 0.14) + _seed_offsets(len(vals)),
                       vals, s=22, facecolors="none", edgecolors=color,
                       linewidths=1.1, zorder=2)
            ax.errorbar(xc + 0.10, vals.mean(), yerr=_ci95(vals), fmt="D", ms=7,
                        color=color, capsize=5, elinewidth=1.6, zorder=3)
        centers.append(base)
    _benchmark_hline(ax, x_frac=0.99, ha="right")
    ax.set_xticks(centers)
    ax.set_xticklabels([g[0] for g in groups])
    ax.set_xlim(-1.1, centers[-1] + 1.75)
    ax.set_ylabel(BPS + "\n(base PPO, development block)")
    ax.set_title("Cost with and without a directional drift in the background\n"
                 "price process, by regime (identical 20-run design)")
    handles = [
        Line2D([], [], marker="D", ls="", color="#7f7f7f", ms=7,
               label="with drift: mean ± 95% CI"),
        Line2D([], [], marker="D", ls="", color=BLUE, ms=7,
               label="drift-neutralised: mean ± 95% CI"),
        Line2D([], [], marker="o", ls="", markerfacecolor="none",
               markeredgecolor="grey", ms=6, label="individual seeds"),
    ]
    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, -0.02),
               ncol=2, frameon=False, fontsize=8.5)
    save(fig, "fig6_drift_confound")


# ---------------------------------------------------------------- figure 7
def fig_dqn_collapse():
    """DQN's degenerate-policy collapse vs PPO health (behaviour audit, base campaign).
    v3 (2026-07-14, user review): the zoom is its own SIDE PANEL (the old overlaid inset
    covered the dqn point at (50, 29)); filled (invalid) markers get a thin WHITE edge so
    the two near-coincident points at (87.1, 86.5)/(87.1, 89.5) read as two distinct
    squares (true positions kept; nothing nudged)."""
    audit = json.load(open(S / "step5_v3" / "behaviour_audit.json"))

    def plot_points(target):
        for e in audit:
            dn = e["action_shares"][0] * 100          # ACTIONS[0] = 0.0x = "trade nothing"
            rf = e["deadline_residual_frac"] * 100
            color = RED if e["algo"] == "dqn" else BLUE
            marker = "s" if e["algo"] == "dqn" else "o"
            if e["valid"]:
                target.scatter(dn, rf, s=44, marker=marker, facecolors="none",
                               edgecolors=color, linewidths=1.2, zorder=3)
            else:
                # white edge -> a visible seam where two filled markers touch
                target.scatter(dn, rf, s=48, marker=marker, facecolors=color,
                               edgecolors="white", linewidths=1.3, zorder=3)

    ZX, ZY = (5, 30), (-1.5, 12)   # zoom window: the 13-run healthy cluster
    fig, (ax, axz) = plt.subplots(1, 2, figsize=(9.8, 4.4),
                                  gridspec_kw={"width_ratios": [1.45, 1]})
    # ---- left: full view (nothing overlaid on the data area)
    ax.axhspan(10, 104, color=RED, alpha=0.06, zorder=0)
    ax.axhline(10, color=RED, lw=1.0, ls="--")
    # threshold label in the verified-empty top-left region (do-nothing<40, residual>40)
    ax.text(2, 100, "audit threshold: the forced-deadline buy finishes\n"
            ">10% of episodes.  Runs in the shaded zone are\n"
            "INVALID (degenerate 'do-nothing-then-dump' policy).",
            fontsize=8.5, color="#222222", ha="left", va="top", linespacing=1.3)
    plot_points(ax)
    # grey box marking the region the right panel magnifies
    from matplotlib.patches import Rectangle
    ax.add_patch(Rectangle((ZX[0], ZY[0]), ZX[1] - ZX[0], ZY[1] - ZY[0],
                           fill=False, edgecolor="grey", lw=1.0, zorder=4))
    ax.annotate("magnified in the\nright panel", xy=(ZX[1], 5), xytext=(44, 4),
                fontsize=8.5, color="#222222", va="center",
                arrowprops=dict(arrowstyle="->", color="#222222", lw=0.9))
    ax.set_xlabel('share of decisions that were "do nothing" (%)')
    ax.set_ylabel("episodes finished by the forced-deadline buy (%)")
    ax.set_title("all 20 base-campaign runs", fontsize=10)
    ax.set_xlim(-3, 100)
    ax.set_ylim(-4, 106)

    # ---- right: the zoom as its own panel (covers no data)
    axz.axhspan(10, ZY[1], color=RED, alpha=0.06, zorder=0)
    axz.axhline(10, color=RED, lw=1.0, ls="--")
    plot_points(axz)
    axz.set_xlim(*ZX)
    axz.set_ylim(*ZY)
    axz.set_xlabel('"do nothing" decisions (%)')
    axz.set_title("zoom: the 13 healthy runs (grey box, left)", fontsize=10)
    fig.suptitle("DQN collapses into do-nothing-then-dump; PPO trades properly\n"
                 "(behavioural audit of the 20 base-campaign runs)", y=1.04)

    handles = [
        Line2D([], [], marker="s", ls="", markerfacecolor=RED, markeredgecolor="white",
               ms=7, label="DQN, INVALID (filled)"),
        Line2D([], [], marker="s", ls="", markerfacecolor="none", markeredgecolor=RED,
               ms=7, label="DQN, valid (open)"),
        Line2D([], [], marker="o", ls="", markerfacecolor="none", markeredgecolor=BLUE,
               ms=7, label="PPO, valid (open)"),
    ]
    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, -0.02),
               ncol=3, frameon=False, fontsize=8.5)
    save(fig, "fig7_dqn_collapse")


# ---------------------------------------------------------------- figure 8
def fig_regime_comparison():
    """RQ2: calm vs volatile, base PPO + selected agent (development block)."""
    v3, a3 = _load("step5_v3")
    sel, sa = _load("step5_selection_v3")
    groups = [("base PPO", _seed_means(v3, a3, "calm", ""), _seed_means(v3, a3, "volatile", "")),
              ("faster-learning\n(selected)", _seed_means(sel, sa, "calm", "_v3a"),
               _seed_means(sel, sa, "volatile", "_v3a"))]
    fig, ax = plt.subplots(figsize=(6.4, 3.8))
    centers = []
    for i, (name, calm, vol) in enumerate(groups):
        base = i * 2.0
        for k, (vals, color, mk) in enumerate([(calm, BLUE, "s"), (vol, RED, "o")]):
            xc = base + (-0.35 if k == 0 else 0.35)
            ax.scatter(np.full(len(vals), xc - 0.14) + _seed_offsets(len(vals)),
                       vals, s=22, facecolors="none", edgecolors=color,
                       linewidths=1.1, zorder=2)
            ax.errorbar(xc + 0.10, vals.mean(), yerr=_ci95(vals), fmt="D", ms=7,
                        color=color, capsize=5, elinewidth=1.6, zorder=3)
        centers.append(base)
    _benchmark_hline(ax, x_frac=0.99, ha="right")
    ax.set_xticks(centers)
    ax.set_xticklabels([g[0] for g in groups])
    ax.set_xlim(-1.1, centers[-1] + 1.75)
    ax.set_ylabel(BPS + "\n(development block)")
    # DESCRIPTIVE, and no chronology. The previous title argued ("The apparent advantage was
    # volatile-regime-only; calm shows nothing") and dated itself ("the volatile signal LATER
    # failed the sealed test"). Arguments belong in prose, not on the exhibit, and
    # house style bans narrating a sequence. It also over-claimed: two CALM cells
    # in the reacting grid triggered follow-up, one at p = 0.0003, so "calm shows nothing" is
    # not true of the campaign. The title now names the two configurations plotted and the
    # block they were measured on, which is all this figure shows.
    ax.set_title("Cost against adaptive TWAP by regime, two configurations\n"
                 "(development block; per-seed values with 95% confidence intervals)")
    handles = [
        Line2D([], [], marker="D", ls="", color=BLUE, ms=7, label="calm regime: mean ± 95% CI"),
        Line2D([], [], marker="D", ls="", color=RED, ms=7, label="volatile regime: mean ± 95% CI"),
        Line2D([], [], marker="o", ls="", markerfacecolor="none",
               markeredgecolor="grey", ms=6, label="individual seeds"),
    ]
    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, -0.02),
               ncol=2, frameon=False, fontsize=8.5)
    save(fig, "fig8_regime_comparison")


# ---------------------------------------------------------------- figure 9
# The 16 grid cells: 11 judged 2026-07-15 (step5_grid_*) + the 5 already-tested cells
# (5-min column from the sweeps + selection centre; 25 BTC/10-min from the h600 sweep).
GRID_SOURCES = [
    # (size_btc, horizon_min, dirname, tag)
    (5.0, 2.5, "step5_grid_b5h150", "_gS5H150"),
    (12.5, 2.5, "step5_grid_b12h150", "_gS12H150"),
    (25.0, 2.5, "step5_grid_b25h150", "_gS25H150"),
    (50.0, 2.5, "step5_grid_b50h150", "_gS50H150"),
    (5.0, 5.0, "step5_sweep_b5", "_v3aB5"),
    (12.5, 5.0, "step5_sweep_b12", "_v3aB12"),
    (25.0, 5.0, "step5_selection_v3", "_v3a"),
    (50.0, 5.0, "step5_sweep_b50", "_v3aB50"),
    (5.0, 10.0, "step5_grid_b5h600", "_gS5H600"),
    (12.5, 10.0, "step5_grid_b12h600", "_gS12H600"),
    (25.0, 10.0, "step5_sweep_h600", "_v3aH600"),
    (50.0, 10.0, "step5_grid_b50h600", "_gS50H600"),
    (5.0, 20.0, "step5_grid_b5h1200", "_gS5H1200"),
    (12.5, 20.0, "step5_grid_b12h1200", "_gS12H1200"),
    (25.0, 20.0, "step5_grid_b25h1200", "_gS25H1200"),
    (50.0, 20.0, "step5_grid_b50h1200", "_gS50H1200"),
]


def grid_cell_stats(regime: str):
    """Uniform recomputation for all 16 cells: valid-seed means vs adaptive, pooled,
    one-sided across-seed t p, n_valid/n_total, §7.5 trigger (pooled<=-0.02, p<0.05,
    fully valid). Single code path for figure 9 + table T4."""
    from scipy.stats import ttest_1samp
    out = {}
    for size, hz, dirname, tag in GRID_SOURCES:
        j, a = _load(dirname)
        vals, n_total = [], 0
        for r in j["per_run"]:
            if r["algo"] != "ppo" or r["regime"] != regime:
                continue
            if _tag_of(r["run"], r["seed"]) != tag:
                continue
            n_total += 1
            if a[r["run"]]["valid"]:
                vals.append(r["mean_vs_adaptive_bps"])
        vals = np.array(vals)
        pooled = float(vals.mean()) if len(vals) else np.nan
        p = float(ttest_1samp(vals, 0.0, alternative="less").pvalue) if len(vals) >= 2 else np.nan
        n_cheap = int((vals < 0).sum())
        trigger = (len(vals) == n_total and pooled <= -0.02
                   and not np.isnan(p) and p < 0.05)
        out[(size, hz)] = dict(pooled=pooled, p=p, n_valid=len(vals),
                               n_total=n_total, n_cheaper=n_cheap, trigger=trigger)
    return out


def fig_grid_heatmap():
    """F9: 4 sizes x 4 horizons pooled cost vs adaptive-TWAP, per regime; §7.5
    triggers outlined. Dev-block evidence; caption carries the interpretation cap."""
    from matplotlib.patches import Rectangle
    sizes = [5.0, 12.5, 25.0, 50.0]
    horizons = [2.5, 5.0, 10.0, 20.0]
    fig, axes = plt.subplots(1, 2, figsize=(10.2, 4.6))
    vmax = 0.17
    for ax, regime in zip(axes, ["calm", "volatile"]):
        stats = grid_cell_stats(regime)
        M = np.array([[stats[(s, h)]["pooled"] for s in sizes] for h in horizons])
        im = ax.imshow(M, cmap="coolwarm", vmin=-vmax, vmax=vmax, aspect="auto")
        for i, h in enumerate(horizons):
            for j, s in enumerate(sizes):
                st = stats[(s, h)]
                # the centre cell's dev-block signal was already chased through the §6
                # confirmation family and FAILED both sealed tests — resolved, not a trigger
                resolved = (s, h) == (25.0, 5.0) and regime == "volatile"
                lines = [f"{st['pooled']:+.3f}", f"p={st['p']:.3f}"]
                if st["n_valid"] != st["n_total"]:
                    lines.append(f"{st['n_valid']}/{st['n_total']} valid")
                elif st["n_total"] != 3:
                    lines.append(f"n={st['n_total']}")
                if resolved:
                    lines.append("sealed: FAIL ×2")
                ax.text(j, i, "\n".join(lines), ha="center", va="center", fontsize=7.6)
                if resolved:
                    ax.add_patch(Rectangle((j - 0.5, i - 0.5), 1, 1, fill=False,
                                           edgecolor="dimgrey", lw=2.0, ls="--", zorder=4))
                elif st["trigger"]:
                    ax.add_patch(Rectangle((j - 0.5, i - 0.5), 1, 1, fill=False,
                                           edgecolor="black", lw=2.4, zorder=4))
        ax.set_xticks(range(4)); ax.set_xticklabels([f"{s:g}" for s in sizes])
        ax.set_yticks(range(4)); ax.set_yticklabels([f"{h:g}" for h in horizons])
        ax.set_xlabel("order size (BTC)")
        ax.set_title(f"{regime} regime", fontsize=10)
        ax.spines[:].set_visible(False)
    axes[0].set_ylabel("deadline (minutes)")
    cb = fig.colorbar(im, ax=axes, fraction=0.032, pad=0.02)
    cb.set_label("pooled cost vs adaptive-TWAP (bps)\nnegative = agent cheaper", fontsize=8.5)
    fig.suptitle("Robustness grid, development-block evidence: two calm cells (solid outline) meet the\n"
                 "pre-registered follow-up trigger; the centre volatile signal (dashed) already failed two confirmations",
                 y=1.06)
    save(fig, "fig9_grid_heatmap")


# ---------------------------------------------------------------- figure 14
def fig_forest_calm():
    """F14: the calm-regime companion to fig3 (same variants, same layout)."""
    sel, sa = _load("step5_selection_v3")
    NAMES = [("_v3a", "faster learning (lr 1e-3)"), ("_w3a", "combo: net 128 + reward x100"),
             ("_v3b", "slower learning (lr 1e-4)"), ("_v1b", "bigger network 128"),
             ("_v6", "reward x100 @ 10M steps"), ("_v5", "longer rollout"),
             ("_v4a", "no exploration bonus"), ("_v2", "reward x100"),
             ("_v6b", "slower-lr @ 10M steps"), ("_w3b", "combo: slow-lr + reward"),
             ("_v4b", "more exploration"), ("_v1a", "bigger network 64")]
    from matplotlib.transforms import blended_transform_factory
    fig, ax = plt.subplots(figsize=(6.6, 4.8))
    ys, ticklabels = [], []
    for i, (tag, label) in enumerate(NAMES):
        vals = _seed_means(sel, sa, "calm", tag)
        if len(vals) == 0:
            continue
        y = len(NAMES) - i
        ax.plot([vals.min(), vals.max()], [y, y], color="grey", lw=1.2, alpha=0.7, zorder=1)
        ax.scatter(vals, np.full(len(vals), y), s=20, facecolors="white",
                   edgecolors="grey", linewidths=1.0, zorder=2)
        ax.plot(vals.mean(), y, "D", ms=6.5, color=BLUE, zorder=3)
        ys.append(y); ticklabels.append(f"{label}  (n={len(vals)})")
    ax.axvline(0, color="k", lw=0.9)
    ax.axvline(-0.05, color="grey", lw=1.0, ls="--")
    tr = blended_transform_factory(ax.transData, ax.transAxes)
    ax.text(-0.05, 1.015, "materiality threshold (-0.05)", transform=tr,
            ha="center", va="bottom", fontsize=8, color="grey")
    ax.text(0.0, 1.015, "adaptive-TWAP benchmark", transform=tr,
            ha="center", va="bottom", fontsize=8, color="k")
    ax.set_yticks(ys); ax.set_yticklabels(ticklabels, fontsize=8.5)
    ax.set_ylim(min(ys) - 0.7, max(ys) + 0.7)
    ax.set_xlabel(BPS + "  —  development block")
    ax.set_title("Tuning screen, CALM regime: no variant shows a consistent advantage\n"
                 "(companion to the volatile-regime panel)", pad=18)
    handles = [
        Line2D([], [], marker="D", ls="", color=BLUE, ms=6.5, label="mean over seeds"),
        Line2D([], [], marker="o", ls="-", color="grey", markerfacecolor="white",
               markeredgecolor="grey", ms=6, lw=1.2,
               label="individual seeds (whisker = seed range)"),
    ]
    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, -0.02),
               ncol=2, frameon=False, fontsize=8.5)
    save(fig, "fig14_forest_tuning_calm")


# ---------------------------------------------------------------- figure 19
def fig_ladder_collapse():
    """F19: the §7.5a ladder outcome — each escalated agent's development-block result
    vs its result on the first-use reserve block. The p=0.0001 group sign-flips."""
    groups = [("b50h600", "50 BTC / 10-min", BLUE),
              ("b25h1200", "25 BTC / 20-min", RED)]
    fig, ax = plt.subplots(figsize=(6.6, 4.2))
    for cell, label, color in groups:
        dev = {r["run"]: r["mean_vs_adaptive_bps"]
               for r in json.load(open(S / f"step5_esc_{cell}" / "judgement.json"))["per_run"]
               if r["regime"] == "calm"}
        res = {r["run"]: r["mean_vs_adaptive_bps"]
               for r in json.load(open(S / f"step5_xblock_{cell}" / "judgement.json"))["per_run"]
               if r["regime"] == "calm"}
        for run in dev:
            ax.plot([0, 1], [dev[run], res[run]], "-o", color=color, ms=4,
                    lw=1.0, alpha=0.55, zorder=2)
        dm = np.mean(list(dev.values())); rm = np.mean(list(res.values()))
        ax.plot([0, 1], [dm, rm], "-D", color=color, ms=8, lw=2.6, zorder=3, label=label)
    _benchmark_hline(ax, x_frac=1.02, ha="left", va="center")
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["development block\n(where the triggers fired;\n5-seed escalation survived)",
                        "reserve block\n(first ever use;\nsame 10 agents, no retraining)"])
    ax.set_xlim(-0.25, 1.25)
    ax.set_ylabel(BPS)
    ax.set_title("Cross-block replication kills both triggered groups\n"
                 "(thin lines = individual seeds; thick = group mean)")
    fig.legend(loc="upper center", bbox_to_anchor=(0.5, -0.02), ncol=2,
               frameon=False, fontsize=9)
    save(fig, "fig19_ladder_collapse")


# ---------------------------------------------------------------- figure 20
def fig_dqn_collapse_by_setting():
    """F20: DQN behaviour-audit pass rate across every tested setting, base config vs
    the library-default update rhythm (d3)."""
    def rate(audit_path, algo="dqn"):
        a = [e for e in json.load(open(audit_path)) if e["algo"] == algo]
        return sum(e["valid"] for e in a), len(a)

    settings = ["5 BTC\n2.5-min", "25 BTC\n2.5-min", "25 BTC\n5-min\n(primary)", "25 BTC\n20-min"]
    base = [rate(S / "step5_dqnprobe_b5h150" / "behaviour_audit.json"),
            rate(S / "step5_dqnprobe_b25h150" / "behaviour_audit.json"),
            rate(S / "step5_v3" / "behaviour_audit.json"),
            rate(S / "step5_dqnprobe_b25h1200" / "behaviour_audit.json")]
    d3 = [rate(S / "step5_d3_b5h150" / "behaviour_audit.json"),
          rate(S / "step5_d3_b25h150" / "behaviour_audit.json"),
          rate(S / "step5_d3_b25h300" / "behaviour_audit.json"),
          None]
    fig, ax = plt.subplots(figsize=(6.8, 3.9))
    x = np.arange(len(settings))
    w = 0.36
    for vals, off, color, label in [
            (base, -w / 2, RED, "base configuration"),
            (d3, w / 2, "#7f7f7f", "library-default update rhythm (d3)")]:
        for j, v in enumerate(vals):
            if v is None:
                ax.text(x[j] + off, 0.03, "not\nrun", ha="center", fontsize=7.5, color="grey")
                continue
            k, n = v
            ax.bar(x[j] + off, k / n, width=w, color=color, alpha=0.85,
                   label=label if j == 0 else None)
            ax.text(x[j] + off, k / n + 0.02, f"{k}/{n}", ha="center", fontsize=8.5)
    ax.set_xticks(x); ax.set_xticklabels(settings, fontsize=9)
    ax.set_xlim(-0.6, 3.75)
    ax.set_ylim(0, 1.12)
    ax.set_ylabel("share of seeds passing the behavioural audit")
    ax.set_title("DQN collapse across settings: size-driven, and not cured by the\n"
                 "library-default update rhythm at the primary setting")
    fig.legend(loc="upper center", bbox_to_anchor=(0.5, -0.02), ncol=2,
               frameon=False, fontsize=9)
    save(fig, "fig20_dqn_collapse_by_setting")


# ---------------------------------------------------------------- figure 21
def fig_q_tilt():
    """F21: per-agent share of states where the learned Q-values rank 'trade nothing'
    first, collapsed vs valid agents (diagnostic artifact of record)."""
    data = json.load(open(Path(__file__).resolve().parents[2]
                          / "diagnostics" / "dqn_q_flatness.json"))
    rows = sorted(data["rows"], key=lambda r: r["argmax0_share"])
    fig, ax = plt.subplots(figsize=(6.6, 4.4))
    ypos = np.arange(len(rows))
    for y, r in zip(ypos, rows):
        color = BLUE if r["valid"] else RED
        ax.barh(y, 100 * r["argmax0_share"], color=color, alpha=0.85)
    def label_of(run):
        parts = run.split("_")           # dqn, calm, s0, dqS5H150 / dqS25H150
        btc = "5 BTC" if "S5H" in parts[3] else "25 BTC"
        return f"{btc}  {parts[1]}  {parts[2]}"
    ax.set_yticks(ypos)
    ax.set_yticklabels([label_of(r["run"]) for r in rows], fontsize=8)
    coll = [r["argmax0_share"] for r in rows if not r["valid"]]
    val = [r["argmax0_share"] for r in rows if r["valid"]]
    ax.axvline(100 * np.mean(coll), color=RED, ls="--", lw=1.2)
    ax.axvline(100 * np.mean(val), color=BLUE, ls="--", lw=1.2)
    from matplotlib.transforms import blended_transform_factory
    tr = blended_transform_factory(ax.transData, ax.transAxes)
    ax.text(100 * np.mean(coll), 1.01, f"collapsed mean {100*np.mean(coll):.0f}%",
            transform=tr, color=RED, fontsize=8, ha="center", va="bottom")
    ax.text(100 * np.mean(val), 1.01, f"valid mean {100*np.mean(val):.0f}%",
            transform=tr, color=BLUE, fontsize=8, ha="center", va="bottom")
    ax.set_xlabel('share of states where the network ranks "do nothing" first (%)')
    ax.set_title("Preference for inaction, by behavioural-audit outcome\n"
                 "(both 2.5-min probe cells, 5 evaluation episodes per agent)",
                 pad=24)
    handles = [Patch(facecolor=RED, alpha=0.85, label="collapsed (failed behavioural audit)"),
               Patch(facecolor=BLUE, alpha=0.85, label="valid")]
    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, -0.02),
               ncol=2, frameon=False, fontsize=9)
    save(fig, "fig21_q_tilt")


# ---------------------------------------------------------------- figure 22
def fig_pipeline_schematic():
    """F22: the evaluation architecture — episode-seed blocks and the
    screen -> replicate -> confirm ladder, with what actually happened at each layer."""
    from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
    fig, ax = plt.subplots(figsize=(9.6, 5.2))
    ax.set_xlim(0, 10); ax.set_ylim(0, 10); ax.axis("off")

    def box(x, y, w, h, text, fc, fontsize=8.4):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.12",
                                    facecolor=fc, edgecolor="k", lw=0.9))
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center",
                fontsize=fontsize, linespacing=1.35)

    def arrow(x1, y1, x2, y2, text="", color="k"):
        ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>",
                                     mutation_scale=14, color=color, lw=1.3))
        if text:
            ax.text((x1 + x2) / 2 + 0.12, (y1 + y2) / 2, text, fontsize=7.8,
                    ha="left", va="center", color=color)

    ax.text(2.1, 9.6, "EPISODE-SEED BLOCKS (disjoint by construction)",
            fontsize=9.5, weight="bold", ha="center")
    box(0.3, 8.3, 3.6, 0.85, "TRAINING episodes\nper-seed private ranges (seed x 10M+)", "#f0f0f0")
    box(0.3, 7.15, 3.6, 0.85, "MONITOR block @ 1.0M (n=200)\nlearning curves only, never decisions", "#f0f0f0")
    box(0.3, 6.0, 3.6, 0.85, "DEVELOPMENT block @ 5.0M (n=2000)\nALL screening, tuning, selection", "#dbe9f6")
    box(0.3, 4.85, 3.6, 0.85, "RESERVE block @ 6.0M (n=2000)\nreplication; first use = grid ladder", "#dff0d8")
    box(0.3, 3.7, 3.6, 0.85, "SEALED blocks @ 9.0M / 13.0M (n=2000)\none-shot confirmations, both SPENT", "#f6dbdb")

    ax.text(7.0, 9.6, "THE THREE-LAYER LADDER (pre-registered)",
            fontsize=9.5, weight="bold", ha="center")
    box(5.3, 7.9, 3.4, 1.15, "LAYER 1: SCREEN (development block)\n98 tuning runs, 22 grid groups,\nfrozen rules + across-seed check", "#dbe9f6")
    box(5.3, 5.8, 3.4, 1.15, "LAYER 2: REPLICATE\nescalate to 5 seeds, then re-test on the\nnever-used reserve block", "#dff0d8")
    box(5.3, 3.7, 3.4, 1.15, "LAYER 3: CONFIRM (sealed, one-shot)\nat most one test per candidate family,\nregistered before running", "#f6dbdb")
    arrow(7.0, 7.9, 7.0, 6.95, " candidates only")
    arrow(7.0, 5.8, 7.0, 4.85, " survivors only")

    box(0.3, 0.9, 9.3, 2.1,
        "WHAT HAPPENED: 2 tuning champions passed layer 1 and went to sealed tests:\n"
        "BOTH FAILED (pooled -0.002, p=0.38 / p=0.39). 2 grid triggers passed layer 1\n"
        "(p=0.014, p=0.0003), survived 5-seed escalation, then BOTH DIED on the reserve\n"
        "block (one sign-flipped to +0.018); no third sealed test was spent.\n"
        "Headline: the agent does not beat TWAP out-of-sample.", "#fffbe6", fontsize=8.4)
    ax.set_title("Evaluation architecture: disjoint episode blocks and the screen / replicate / confirm ladder",
                 fontsize=11)
    save(fig, "fig22_pipeline_schematic")


# ---------------------------------------------------------------- figures 10a/10b
def _per_episode(regime: str):
    """Load the per-episode re-eval arrays + pool agents by algo over AUDIT-VALID seeds."""
    d = np.load(S / "per_episode_v3" / f"{regime}.npz")
    audit = {e["run"]: e["valid"] for e in
             json.load(open(S / "step5_v3" / "behaviour_audit.json"))}
    pooled = {"fixed TWAP": d["fixed"], "adaptive TWAP": d["adaptive"]}
    for algo in ("ppo", "dqn"):
        arrs = [d[k] for k in d.files if k.startswith(f"{algo}_") and audit[k]]
        pooled[f"{algo.upper()} (valid seeds, n={len(arrs)})"] = np.concatenate(arrs)
    return d, pooled


def fig_cost_distributions():
    """F10a: absolute per-episode execution costs per policy and regime (violin)."""
    fig, axes = plt.subplots(1, 2, figsize=(10.6, 4.2), sharey=True)
    for ax, regime in zip(axes, ("calm", "volatile")):
        _, pooled = _per_episode(regime)
        labels, data = list(pooled.keys()), list(pooled.values())
        parts = ax.violinplot(data, showmeans=True, showextrema=False, widths=0.8)
        for pc, c in zip(parts["bodies"], ["#7f7f7f", "k", BLUE, RED]):
            pc.set_facecolor(c); pc.set_alpha(0.45)
        parts["cmeans"].set_color("k")
        ax.axhline(0, color="k", lw=0.6, alpha=0.5)
        ax.set_xticks(range(1, len(labels) + 1))
        ax.set_xticklabels(labels, fontsize=8, rotation=12)
        ax.set_title(f"{regime} regime", fontsize=10)
    axes[0].set_ylabel("implementation shortfall per episode (bps)")
    fig.suptitle("Absolute per-episode execution costs (25 BTC / 5-min, development block,\n"
                 "2,000 episodes per policy; agents pooled over audit-valid seeds)", y=1.06)
    save(fig, "fig10a_cost_distributions")


def fig_paired_distributions():
    """F10b: distribution of PAIRED per-episode differences (agent minus adaptive TWAP)."""
    fig, axes = plt.subplots(1, 2, figsize=(10.6, 4.0), sharey=False)
    for ax, regime in zip(axes, ("calm", "volatile")):
        d, _ = _per_episode(regime)
        audit = {e["run"]: e["valid"] for e in
                 json.load(open(S / "step5_v3" / "behaviour_audit.json"))}
        diffs, labels = [], []
        for algo, color in (("ppo", BLUE), ("dqn", RED)):
            arrs = [d[k] - d["adaptive"] for k in d.files
                    if k.startswith(f"{algo}_") and audit[k]]
            diffs.append(np.concatenate(arrs))
            labels.append(f"{algo.upper()} - adaptive TWAP")
        parts = ax.violinplot(diffs, showmeans=True, showextrema=False, widths=0.8)
        for pc, c in zip(parts["bodies"], [BLUE, RED]):
            pc.set_facecolor(c); pc.set_alpha(0.45)
        parts["cmeans"].set_color("k")
        ax.axhline(0, color="k", lw=0.9)
        span = max(a.max() for a in diffs) - min(a.min() for a in diffs)
        for i, arr in enumerate(diffs):
            ax.text(i + 1, arr.mean() - 0.05 * span, f"mean {arr.mean():+.3f}",
                    fontsize=8, va="top", ha="center")
        ax.set_xticks(range(1, len(labels) + 1))
        ax.set_xticklabels(labels, fontsize=8.5)
        ax.set_title(f"{regime} regime", fontsize=10)
    axes[0].set_ylabel("paired per-episode difference (bps)\nnegative = agent cheaper")
    fig.suptitle("Paired per-episode cost differences vs adaptive TWAP (development block;\n"
                 "the paired mean is tiny relative to the episode-level spread)", y=1.06)
    save(fig, "fig10b_paired_distributions")


if __name__ == "__main__":
    fig_dev_vs_sealed()
    fig_size_response()
    fig_forest_selection()
    fig_three_block()
    fig_learning_curves()
    fig_drift_confound()
    fig_dqn_collapse()
    fig_regime_comparison()
    fig_grid_heatmap()
    fig_forest_calm()
    fig_ladder_collapse()
    fig_dqn_collapse_by_setting()
    fig_q_tilt()
    fig_pipeline_schematic()
    print("ALL FIGURES DONE ->", OUT)
