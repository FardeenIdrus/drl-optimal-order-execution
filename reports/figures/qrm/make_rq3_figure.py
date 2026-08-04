"""RQ3 figure: what drives the agents' decisions, and what that attention is worth.

RQ3: which order-book features drive the agents' execution decisions, and does this differ by
market regime?

The figure has to carry four claims, and they are deliberately in this order because the later
ones are only interesting once the earlier ones are established:

  (a) THE AGENTS ARE NOT INATTENTIVE. In the injected environment the observation contains the
      measured venue signal that a one-line rule monetised; in the frozen-replay observation the
      counterpart is queue imbalance. Panel A plots each agent's attribution share on that
      feature against the equal-attribution reference 1/obs_dim -- the share a single dimension
      would receive if attribution were spread evenly. Above the line means the agent weights
      the paid feature MORE than chance.

  (b) THEIR POLICIES ARE NOT CONSTANTS -- except on frozen replay. Panel B plots |r| between
      each agent's per-episode premium over TWAP and that of a FIXED front-loading rule. r near
      1 means one constant reproduces the whole pattern; r near 0.35 means it does not.

  (c) NONE OF IT IS WORTH ANYTHING. Panel C plots alpha, the part of the premium a constant
      front-loading dose does NOT explain, against zero and against the pre-registered 0.05 bps
      materiality band.

  (d) REGIME REALLOCATES ATTENTION WITHOUT CHANGING THE ANSWER. Panel D plots the within-agent
      calm->volatile shift in each feature's share on the frozen track, where the same agent is
      attributed under both regimes so the comparison is paired.

DESIGN RULES OBSERVED HERE
  - One meaning per visual channel. Colour encodes ALGORITHM everywhere and nothing else;
    environment is encoded by position; regime, where it appears, by marker. An earlier figure
    in this project had to be rebuilt because colour meant two different things in two panels.
  - Reference lines are drawn from the data's own structure (1/obs_dim, zero, the registered
    0.05 bps floor), never chosen to flatter a reading.
  - Every agent is a visible dot. Cell means are drawn on top, not instead.

EXCLUSIONS, BOTH SHOWN ON THE FIGURE RATHER THAN BURIED IN THE CAPTION
  - Panel A: eight frozen DQN agents emit a SINGLE action in every sampled state. The map from
    observation to action is constant, so every SHAP value is exactly zero and the share is
    undefined, not zero. They are excluded from the shares and counted on the panel.
  - Panel C: alpha is a cost quantity, so agents that failed the behaviour audit (deadline
    residual > 10%, i.e. the parent order was not finished) are excluded -- their cost is not
    comparable with TWAP's. The count dropped is printed on the panel. This matters: including
    them produced an apparently significant alpha for injected DQN that vanishes once the
    agents that never finished the order are removed.

Sources (absolute):
  /Users/fardeenidrus/Desktop/MSc Dissertation/scratch_hyperliquid/oxford_l4/rq3_attribution/*.json
  /Users/fardeenidrus/Desktop/MSc Dissertation/scratch_hyperliquid/l2_test_results/l2_inversion_stage3.json
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

# The paired regime test in panel D reuses rq3_analysis's Wilcoxon and Holm helpers rather
# than reimplementing them: a second implementation is a second thing that can disagree with
# the numbers in the text.
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "diagnostics"))

S = Path("/Users/fardeenidrus/Desktop/MSc Dissertation/scratch_hyperliquid")
RQ3 = S / "oxford_l4" / "rq3_attribution"
OUT = Path(__file__).resolve().parent

PPO_C, DQN_C = "#1f77b4", "#d62728"
INK, MUTED, GRID = "#222222", "#666666", "#cccccc"
ACOL = {"ppo": PPO_C, "dqn": DQN_C}
BASE_RE = re.compile(r"^(ppo|dqn)_(calm|volatile)_s\d+$")
MATERIALITY = 0.05          # bps, pre-registered
AUDIT = {"runs_primary_v3": "step5_v3", "runs_signal_phaseD": "step5_signal_dev"}
FROZEN_SETS = ["runs", "runs_10s", "runs_10s_10min"]

plt.rcParams.update({"font.size": 9.5, "axes.titlesize": 10, "axes.labelsize": 9.5,
                     "figure.dpi": 150, "savefig.bbox": "tight",
                     "axes.spines.top": False, "axes.spines.right": False})


def jload(p: Path):
    return json.loads(p.read_text()) if p.exists() else []


def audit_valid(runs: str) -> dict[str, bool]:
    p = S / "oxford_l4" / AUDIT[runs] / "behaviour_audit.json"
    return {r["run"]: bool(r.get("valid", True)) for r in jload(p)}


def jitter(n: int, w: float = 0.16) -> np.ndarray:
    """Deterministic spread so overlapping dots stay countable. Fixed seed: the figure must
    be byte-reproducible across rebuilds."""
    return np.linspace(-w, w, n) if n > 1 else np.zeros(1)


# --------------------------------------------------------------------------------------
def panel_a(ax):
    """Attribution share on the feature that was demonstrably worth money."""
    cells = []
    for algo in ("ppo", "dqn"):
        rows = [r for r in jload(RQ3 / f"rq3_attribution_runs_signal_phaseD_{algo}.json")
                if BASE_RE.match(r["run"])]
        vals = [r["group_share"]["injected signal"] for r in rows]
        # One dot per agent: an injected/reacting agent is trained in ONE regime.
        cells.append({"label": "injected\nsignal", "env": "injected signal", "algo": algo,
                      "vals": vals, "ref": 1.0 / rows[0]["obs_dim"], "n_deg": 0,
                      "nlab": f"n={len(vals)}"})
    for algo in ("ppo", "dqn"):
        vals, n_deg, n_ag, obs_dim = [], 0, 0, 7
        for ds in FROZEN_SETS:
            for r in jload(RQ3 / f"rq3_attribution_frozen_{ds}_{algo}.json"):
                n_ag += 1
                for v in r["by_regime"].values():
                    if sum(v["share"].values()) < 1e-9:
                        n_deg += 1
                    else:
                        vals.append(v["share"]["queue imbalance"])
        # A frozen agent trains on mixed data and is attributed under BOTH regimes, so it
        # contributes two dots. Said in the tick label, because otherwise this cell's n looks
        # inconsistent with the injected cells beside it.
        cells.append({"label": "queue\nimbalance", "env": "recorded books", "algo": algo,
                      "vals": vals, "ref": 1.0 / obs_dim, "n_deg": n_deg,
                      "nlab": f"{n_ag} × 2 regimes"})

    for i, c in enumerate(cells):
        v = np.asarray(c["vals"], float) * 100
        x = i + jitter(len(v))
        ax.scatter(x, v, s=16, color=ACOL[c["algo"]], alpha=0.55,
                   edgecolors="none", zorder=3)
        ax.hlines(v.mean(), i - 0.3, i + 0.3, color=ACOL[c["algo"]], lw=2.4, zorder=4)
        ax.hlines(c["ref"] * 100, i - 0.36, i + 0.36, color=INK, lw=1.3, ls="--", zorder=5)
        # Place the mean label AWAY from the reference line, whichever side that is, so the
        # two never overprint. A first version pinned it above the mean unconditionally and
        # the frozen-DQN label landed on top of its own dashed reference.
        above = v.mean() > c["ref"] * 100
        ax.annotate(f"{v.mean():.1f}%", xy=(i, v.mean()), xytext=(26, 6 if above else -12),
                    textcoords="offset points", ha="left", fontsize=8.5,
                    color=ACOL[c["algo"]], fontweight="bold")
    # Headroom FIRST, then the note inside it: the note has to sit above every dot, and the
    # tallest dot is not known until the data are drawn.
    top = max(max(np.asarray(c["vals"])) for c in cells) * 100
    ax.set_ylim(0, top * 1.30)
    n_deg_total = sum(c["n_deg"] for c in cells)
    if n_deg_total:
        which = ", ".join(f"{c['env']} {c['algo'].upper()}" for c in cells if c["n_deg"])
        ax.annotate(f"{n_deg_total} agent-regime rows excluded ({which}): the policy emits a\n"
                    f"single action in every state, so attribution is undefined, not zero",
                    xy=(0.985, 0.985), xycoords="axes fraction", ha="right", va="top",
                    fontsize=7.0, color=MUTED)
    ax.set_xticks(range(len(cells)))
    ax.set_xticklabels([f"{c['env']}\n{c['algo'].upper()}\n{c['nlab']}" for c in cells],
                       fontsize=8.0)
    ax.set_ylabel("share of attribution on the paid feature (%)")
    ax.set_title("A. The agents are not ignoring the feature that was worth money\n"
                 "dashed line = share a single input would get if attention were spread evenly",
                 fontsize=9.5, loc="left")
    ax.grid(axis="y", color=GRID, lw=0.5, zorder=0)
    ax.set_axisbelow(True)
    ax.set_ylim(bottom=0)
    return cells


def panel_b(ax):
    """|r| against a fixed front-loading rule: is the policy just one constant?"""
    cells = []
    for env, runs in (("reacting market", "runs_primary_v3"), ("injected signal", "runs_signal_phaseD")):
        for algo in ("ppo", "dqn"):
            rows = [r for r in jload(RQ3 / f"rq3_reduction_{runs}_{algo}.json")
                    if BASE_RE.match(r["run"])]
            cells.append({"env": env, "algo": algo,
                          "vals": [abs(r["corr_with_probe"]) for r in rows]})
    # Frozen replay: ALL THREE datasets, from rq3_reduction_frozen.py. The inversion study's
    # stage 3 measured the same quantity but only on runs_10s; using it here would have made
    # panel B's frozen sample a subset of panel A's without saying so. The two implementations
    # agree exactly on their overlap (max |delta r| = 0), which is checked by that script.
    frozen = jload(RQ3 / "rq3_reduction_frozen_all.json")
    if not frozen:
        raise SystemExit("rq3_reduction_frozen_all.json not found -- run "
                         "reports/diagnostics/rq3_reduction_frozen.py first")
    for algo in ("ppo", "dqn"):
        rows = [r for r in frozen if r["algo"] == algo]
        cells.append({"env": "recorded books", "algo": algo,
                      "vals": [abs(r["corr_with_probe"]) for r in rows]})

    for i, c in enumerate(cells):
        v = np.asarray(c["vals"], float)
        ax.scatter(i + jitter(len(v)), v, s=16, color=ACOL[c["algo"]], alpha=0.55,
                   edgecolors="none", zorder=3)
        ax.hlines(v.mean(), i - 0.3, i + 0.3, color=ACOL[c["algo"]], lw=2.4, zorder=4)
        # Offset sideways, not upward: the frozen means sit at ~0.95 and an upward label
        # collided with the r = 1 reference line.
        ax.annotate(f"{v.mean():.2f}", xy=(i, v.mean()), xytext=(24, -3),
                    textcoords="offset points", ha="left", fontsize=8.5,
                    color=ACOL[c["algo"]], fontweight="bold")
    ax.axhline(1.0, color=INK, lw=1.1, ls="--")
    ax.annotate("r = 1: the whole policy IS one constant front-loading dose",
                xy=(0.02, 1.0), xycoords=("axes fraction", "data"), xytext=(0, 6),
                textcoords="offset points", ha="left", fontsize=7.2, color=INK)
    ax.set_xticks(range(len(cells)))
    # Six columns here versus four elsewhere, so the environment name is wrapped rather than
    # shrunk further -- adjacent labels were touching.
    ax.set_xticklabels([f"{c['env'].replace(' ', chr(10))}\n{c['algo'].upper()}\n"
                        f"n={len(c['vals'])}" for c in cells], fontsize=8.0)
    ax.set_ylabel("|r| with a fixed front-loading rule")
    ax.set_ylim(0, 1.15)
    ax.set_title("B. In the simulated markets the policies are genuinely state-dependent;"
                 "\non recorded books they collapse to a single constant",
                 fontsize=9.5, loc="left")
    ax.grid(axis="y", color=GRID, lw=0.5, zorder=0)
    ax.set_axisbelow(True)
    return cells


def panel_c(ax):
    """alpha: what the state-dependence achieves beyond a constant dose. Audit-valid only."""
    cells = []
    for env, runs in (("reacting market", "runs_primary_v3"), ("injected signal", "runs_signal_phaseD")):
        valid = audit_valid(runs)
        for algo in ("ppo", "dqn"):
            allrows = [r for r in jload(RQ3 / f"rq3_reduction_{runs}_{algo}.json")
                       if BASE_RE.match(r["run"])]
            rows = [r for r in allrows if valid.get(r["run"], True)]
            cells.append({"env": env, "algo": algo,
                          "vals": [r["alpha_bps"] for r in rows],
                          "n_drop": len(allrows) - len(rows)})

    MIN_TESTABLE = 3            # below this a cell supports no claim either way
    ax.axhspan(-MATERIALITY, MATERIALITY, color="#999999", alpha=0.13, zorder=0)
    ax.axhline(0.0, color=INK, lw=1.1)
    for i, c in enumerate(cells):
        v = np.asarray(c["vals"], float)
        testable = v.size >= MIN_TESTABLE
        if v.size:
            ax.scatter(i + jitter(len(v)), v, s=18, color=ACOL[c["algo"]],
                       alpha=0.6 if testable else 0.28, edgecolors="none", zorder=3)
            if testable:
                ax.hlines(v.mean(), i - 0.3, i + 0.3, color=ACOL[c["algo"]], lw=2.4, zorder=4)
        # A cell with one or two survivors gets NO mean bar and is marked untestable. Drawing
        # a cell mean over n=2 would invite exactly the reading the sample cannot support.
        # The caption belongs beside the cell's own dots, not parked at the axis floor where
        # the reader has to hunt for which column it refers to.
        if not testable and v.size:
            ax.annotate("n too small\nto test", xy=(i + 0.28, v.mean()),
                        ha="left", va="center", fontsize=7.0, color=MUTED, style="italic")
    ax.annotate("±0.05 bps: the pre-registered materiality floor",
                xy=(0.02, MATERIALITY), xycoords=("axes fraction", "data"), xytext=(0, 4),
                textcoords="offset points", ha="left", fontsize=7.2, color=MUTED)
    ax.set_xticks(range(len(cells)))
    # n and the audit exclusion go IN the tick label. Annotating them near the axis floor
    # overprinted the tick labels themselves in a first version.
    ax.set_xticklabels([
        f"{c['env']}\n{c['algo'].upper()}\nn={len(c['vals'])}"
        + (f"  ({c['n_drop']} unfinished)" if c["n_drop"] else "")
        for c in cells], fontsize=8.0)
    ax.set_ylabel("α: premium a constant dose does not explain (bps)\n← cheaper than TWAP")
    ax.set_title("C. What the state-dependence buys: in every cell large enough to test, "
                 "nothing\ndistinguishable from zero — audit-valid agents only, since an "
                 "unfinished order has no comparable cost", fontsize=9.5, loc="left")
    ax.grid(axis="y", color=GRID, lw=0.5, zorder=0)
    ax.set_axisbelow(True)
    return cells


def panel_d(ax, algo="ppo"):
    """Within-agent calm -> volatile reallocation of attention, frozen track (paired)."""
    from rq3_analysis import wilcoxon_signed_rank, holm            # reuse, do not re-implement
    by: dict[str, dict] = {}
    for ds in FROZEN_SETS:
        for r in jload(RQ3 / f"rq3_attribution_frozen_{ds}_{algo}.json"):
            for regime, v in r["by_regime"].items():
                if sum(v["share"].values()) < 1e-9:
                    continue
                by.setdefault(f"{ds}/{r['run']}", {})[regime] = v["share"]
    pairs = [v for v in by.values() if "calm" in v and "volatile" in v]
    feats = sorted({f for v in pairs for f in v["calm"]})
    deltas, praw = {}, {}
    for f in feats:
        a = np.array([v["calm"][f] for v in pairs])
        b = np.array([v["volatile"][f] for v in pairs])
        deltas[f] = (b - a) * 100
        _, praw[f] = wilcoxon_signed_rank(a, b)
    padj = holm(praw)
    order = sorted(feats, key=lambda f: deltas[f].mean())

    y = np.arange(len(order))
    means = np.array([deltas[f].mean() for f in order])
    ses = np.array([deltas[f].std(ddof=1) / np.sqrt(len(deltas[f])) for f in order])
    # ONE colour. Direction is already encoded by which side of zero the bar is on; colouring
    # by sign would make blue mean "positive here" and "PPO" in the other three panels.
    ax.barh(y, means, color=ACOL[algo], alpha=0.72, height=0.6, zorder=3)
    ax.errorbar(means, y, xerr=1.96 * ses, fmt="none", ecolor=INK, elinewidth=1.1,
                capsize=2.5, zorder=4)
    # Significance marks sit outside the whisker, not on top of it.
    for i, f in enumerate(order):
        mark = "***" if padj[f] < 0.001 else "**" if padj[f] < 0.01 else \
               "*" if padj[f] < 0.05 else "n.s."
        tip = means[i] + np.sign(means[i] or 1) * (1.96 * ses[i] + 0.35)
        ax.annotate(mark, xy=(tip, i), ha="left" if means[i] > 0 else "right",
                    va="center", fontsize=7.5, color=INK)
    ax.axvline(0, color=INK, lw=1.1)
    ax.set_yticks(y)
    ax.set_yticklabels(order)
    ax.set_xlabel("change in attribution share, calm → volatile (percentage points)")
    ax.set_title(f"D. The same {algo.upper()} agent reallocates attention by regime "
                 f"({len(pairs)} agents, paired)\n"
                 "toward spread and recent return, away from depth — and it changes no verdict",
                 fontsize=9.5, loc="left")
    ax.grid(axis="x", color=GRID, lw=0.5, zorder=0)
    ax.set_axisbelow(True)
    ax.margins(x=0.22)
    return {"n_pairs": len(pairs), "deltas": {f: float(deltas[f].mean()) for f in order},
            "p_holm": {f: float(padj[f]) for f in order}}


def main() -> None:
    fig, axes = plt.subplots(2, 2, figsize=(14.6, 10.4))
    a = panel_a(axes[0, 0])
    b = panel_b(axes[0, 1])
    c = panel_c(axes[1, 0])
    d = panel_d(axes[1, 1], algo="ppo")

    handles = [Line2D([], [], marker="o", ls="", color=PPO_C, label="PPO", markersize=6),
               Line2D([], [], marker="o", ls="", color=DQN_C, label="DQN", markersize=6),
               Line2D([], [], color=INK, lw=2.2, label="cell mean"),
               Line2D([], [], color=INK, lw=1.2, ls="--", label="reference line (see panel)"),
               Line2D([], [], color=INK, lw=1.1, label="95% CI (panel D)")]
    fig.legend(handles=handles, loc="lower center", ncol=5, frameon=False,
               bbox_to_anchor=(0.5, 0.003), fontsize=9)
    fig.suptitle("RQ3 — the agents attend to the order-book features that matter, "
                 "reallocate that attention by regime, and gain nothing from either",
                 y=0.995, fontsize=12.5)
    # Reserve the strip the legend occupies BEFORE laying the axes out; letting tight_layout
    # use the full height put the legend on top of panel C's tick labels.
    fig.tight_layout(rect=[0, 0.038, 1, 0.975])
    for ext in ("pdf", "png"):
        fig.savefig(OUT / f"fig24_rq3_attribution.{ext}")
    plt.close(fig)
    print(f"wrote {OUT}/fig24_rq3_attribution.pdf/.png")

    print("\n--- numbers on the figure, for the caption and for cross-checking the prose ---")
    for cell in a:
        v = np.asarray(cell["vals"]) * 100
        print(f"  A  {cell['env']:<14} {cell['algo'].upper():<4} n={len(v):<3} "
              f"mean {v.mean():5.2f}%  reference {cell['ref']*100:5.2f}%  "
              f"{'ABOVE' if v.mean() > cell['ref']*100 else 'BELOW'}"
              f"   excluded(constant)={cell['n_deg']}")
    for cell in b:
        v = np.asarray(cell["vals"])
        print(f"  B  {cell['env']:<14} {cell['algo'].upper():<4} n={len(v):<3} "
              f"|r| {v.mean():.3f}")
    for cell in c:
        v = np.asarray(cell["vals"])
        m = v.mean() if v.size else float("nan")
        print(f"  C  {cell['env']:<14} {cell['algo'].upper():<4} n={v.size:<3} "
              f"alpha {m:+.4f} bps   dropped(audit)={cell['n_drop']}")
    print(f"  D  frozen PPO, {d['n_pairs']} paired agents")
    for f, dv in d["deltas"].items():
        print(f"       {f:<22} {dv:+5.2f} pp   Holm p={d['p_holm'][f]:.4g}")


if __name__ == "__main__":
    main()
