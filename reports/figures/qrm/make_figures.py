"""Results-section figures for the LaTeX report.

Generates PDF (vector, for \\includegraphics) + PNG (preview) into reports/figures/
from the EXISTING sealed result files only (no new evaluation). Every number is read
from the source-of-record JSONs listed in qrm_step5_remediation.md.

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
               label="sealed block: mean ± 95% CI (one-shot, fresh seeds)"),
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
    ax.text(-0.05, 1.015, "materiality floor (-0.05)", transform=tr,
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
    """The drift confound: identical 20-run design before/after the drift fix."""
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
    ax.set_title("A hidden background price drift manufactured a spurious calm-regime edge;\n"
                 "neutralising the drift dissolved it (identical 20-run design before/after)")
    handles = [
        Line2D([], [], marker="D", ls="", color="#7f7f7f", ms=7,
               label="WITH drift (before the fix): mean ± 95% CI"),
        Line2D([], [], marker="D", ls="", color=BLUE, ms=7,
               label="drift-neutralised (after the fix): mean ± 95% CI"),
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
            fontsize=7.5, color=RED, ha="left", va="top", linespacing=1.3)
    plot_points(ax)
    # grey box marking the region the right panel magnifies
    from matplotlib.patches import Rectangle
    ax.add_patch(Rectangle((ZX[0], ZY[0]), ZX[1] - ZX[0], ZY[1] - ZY[0],
                           fill=False, edgecolor="grey", lw=1.0, zorder=4))
    ax.annotate("magnified in the\nright panel", xy=(ZX[1], 5), xytext=(44, 4),
                fontsize=7.5, color="grey", va="center",
                arrowprops=dict(arrowstyle="->", color="grey", lw=0.9))
    ax.set_xlabel('share of decisions that were "trade nothing" (%)')
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
    axz.set_xlabel('"trade nothing" decisions (%)')
    axz.set_title("zoom: the 13 healthy runs (grey box, left)", fontsize=10)
    fig.suptitle("DQN collapses into do-nothing-then-dump; PPO trades properly\n"
                 "(behaviour audit of the 20 base-campaign runs)", y=1.04)

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
    ax.set_title("The apparent advantage was volatile-regime-only; calm shows nothing\n"
                 "(development-block evidence; the volatile signal later failed the sealed test)")
    handles = [
        Line2D([], [], marker="D", ls="", color=BLUE, ms=7, label="calm regime: mean ± 95% CI"),
        Line2D([], [], marker="D", ls="", color=RED, ms=7, label="volatile regime: mean ± 95% CI"),
        Line2D([], [], marker="o", ls="", markerfacecolor="none",
               markeredgecolor="grey", ms=6, label="individual seeds"),
    ]
    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, -0.02),
               ncol=2, frameon=False, fontsize=8.5)
    save(fig, "fig8_regime_comparison")


if __name__ == "__main__":
    fig_dev_vs_sealed()
    fig_size_response()
    fig_forest_selection()
    fig_three_block()
    fig_learning_curves()
    fig_drift_confound()
    fig_dqn_collapse()
    fig_regime_comparison()
    print("ALL FIGURES DONE ->", OUT)
