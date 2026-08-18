"""Measured-signal extension figures (Phase F).

Sibling suite to reports/figures/qrm/ and reports/figures/l2/. Generates PDF (vector,
for \\includegraphics) + PNG (preview) from the FROZEN result JSONs only -- no new
evaluation, no re-runs. Every number traces to a source-of-record file listed in
reports/qrm_step5_remediation.md addenda (G)/(H)/(I)/(J) and criteria section 8.

House style inherited from reports/figures/qrm/make_figures.py:
  - legends OUTSIDE the axes, never overlapping data;
  - per-seed points as small open circles at deterministic offsets (no jitter),
    clearly separated from the filled mean +- 95% CI marker;
  - top/right spines off; plain numbers on axes; explicit legend entry per band.

Palette (colour-blindness validated, OKLab dE under Machado-2009 CVD simulation):
  - two-category figures keep the house pair blue #1f77b4 / red #d62728
    (normal dE 31.7, worst-CVD dE 21.1 -> PASS);
  - figures needing 3-4 categories use the Okabe-Ito subset (worst-CVD dE 8.6 ->
    PASS). matplotlib's default cycle is deliberately NOT extended: its red/green
    (CVD dE 3.9) and green/orange (CVD dE 0.7) pairs are indistinguishable to
    colour-blind readers.

TIERING (see methodology_defensibility.md section 7 / the report structure plan):
  MAIN      s1_injection_fidelity, s3_dev_campaign_forest, s4_exploiter_vs_agents
  SUPPORT   s5_base_vs_injected, s6_training_curves, s7_policy_sensitivity
  APPENDIX  s8_kernel_structure
  (s2_three_environment is BLOCKED on the L2 sealed exam -- see build_s2_when_ready.)

Run:  PYTHONPATH=src .venv/bin/python reports/figures/sigext/make_sigext_figures.py
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
DEV = S / "step5_signal_dev"
DIAG = DEV / "diagnostics_postnull"
RUNS = S / "runs_signal_phaseD"

plt.rcParams.update({
    "font.size": 10, "axes.titlesize": 11, "axes.labelsize": 10,
    "figure.dpi": 150, "savefig.bbox": "tight",
    "axes.spines.top": False, "axes.spines.right": False,
})

BLUE, RED = "#1f77b4", "#d62728"          # house pair (validated)
OI_BLUE, OI_VERM = "#0072B2", "#D55E00"   # Okabe-Ito (validated 4-set)
OI_GREEN, OI_ORANGE = "#009E73", "#E69F00"
INK, MUTED = "#222222", "#666666"
BPS = "cost vs adaptive TWAP (bps)"
REGIMES = ("calm", "volatile")
RCOL = {"calm": BLUE, "volatile": RED}


def save(fig, name: str, tier: str = "main_body") -> None:
    """tier -> subfolder. main_body/ = the figures that carry the argument in the
    Results chapter; appendix/ = supporting evidence and technical detail."""
    d = OUT / tier
    d.mkdir(parents=True, exist_ok=True)
    fig.savefig(d / f"{name}.pdf")
    fig.savefig(d / f"{name}.png")
    plt.close(fig)
    print(f"  wrote {tier}/{name}.pdf/.png")


def _ci95(x: np.ndarray) -> float:
    """95% CI half-width of the mean (t-based; matches the qrm suite)."""
    x = np.asarray(x, float)
    n = len(x)
    if n < 2:
        return float("nan")
    from scipy.stats import t as tdist
    return float(tdist.ppf(0.975, n - 1) * x.std(ddof=1) / np.sqrt(n))


def _seed_offsets(n: int, width: float = 0.06) -> np.ndarray:
    if n <= 1:
        return np.zeros(n)
    return np.linspace(-width, width, n)


def _zero_line(ax, label: str = "adaptive TWAP (zero line)") -> Line2D:
    ax.axhline(0.0, color=MUTED, lw=1.0, ls="--", zorder=1)
    return Line2D([], [], color=MUTED, lw=1.0, ls="--", label=label)


# --------------------------------------------------------------- data loaders
def load_gates() -> dict:
    return json.loads((S / "signal" / "gates" / "sigext_gates_v4c_PASS.json").read_text())


def load_dev() -> tuple[dict, dict]:
    j = json.loads((DEV / "judgement.json").read_text())
    a = {e["run"]: e for e in json.loads((DEV / "behaviour_audit.json").read_text())}
    return j, a


def load_follower() -> dict:
    return json.loads((DIAG / "diag_signal_follower.json").read_text())


def load_base_env() -> dict:
    return json.loads((DIAG / "diag_base_env.json").read_text())


def load_kernel() -> dict:
    return json.loads((S / "signal" / "kernel_solution.json").read_text())


def _tag_of(run: str, seed: int) -> str:
    m = re.search(rf"_s{seed}(_.+)?$", run)
    return (m.group(1) or "") if m else ""


def _group_seeds(j: dict, audit: dict, algo: str, regime: str, tag: str,
                 valid_only: bool = True) -> list[float]:
    out = []
    for r in j["per_run"]:
        if r["algo"] != algo or r["regime"] != regime:
            continue
        if _tag_of(r["run"], r["seed"]) != tag:
            continue
        if valid_only and not audit[r["run"]]["valid"]:
            continue
        out.append(r["mean_vs_adaptive_bps"])
    return out


# ============================================================== MAIN BODY
def fig_s1_injection_fidelity() -> None:
    """MAIN. Does the injected signal reproduce the REAL venue predictability?
    Per regime: measured-in-simulator vs real-data predictive slope by horizon,
    with the registered +/-20% acceptance band on the gated horizons (1,2,5,10 s).
    Source: signal/gates/sigext_gates_v4c_PASS.json -> injection_matching."""
    g = load_gates()["injection_matching"]
    fig, axes = plt.subplots(1, 2, figsize=(9.2, 3.6), sharey=False)
    for ax, regime in zip(axes, REGIMES):
        rows = g["regimes"][regime]["horizons"]
        hs = sorted(rows, key=lambda h: float(h))
        x = np.arange(len(hs))
        real = np.array([rows[h]["real_calibrate_slope"] for h in hs])
        sim = np.array([rows[h]["sim_total_slope"] for h in hs])
        gated = np.array([rows[h]["gated"] for h in hs])
        ax.fill_between(x, real * 0.8, real * 1.2, color=MUTED, alpha=0.18, zorder=1)
        ax.plot(x, real, "-o", color=INK, lw=1.8, ms=5, zorder=3)
        ax.plot(x, sim, "--s", color=RCOL[regime], lw=1.8, ms=5, zorder=4)
        for i, gt in enumerate(gated):
            if gt:
                ax.axvspan(i - 0.5, i + 0.5, color=RCOL[regime], alpha=0.05, zorder=0)
        ax.set_xticks(x)
        ax.set_xticklabels(hs)
        ax.set_xlabel("forward horizon (s)")
        ax.set_title(f"{regime}  (gated horizons shaded)")
        if ax is axes[0]:
            ax.set_ylabel("predictive slope (bps per unit imbalance)")
    handles = [Line2D([], [], color=INK, marker="o", lw=1.8, label="measured on Hyperliquid"),
               Line2D([], [], color=BLUE, marker="s", ls="--", lw=1.8, label="injected simulator (calm)"),
               Line2D([], [], color=RED, marker="s", ls="--", lw=1.8, label="injected simulator (volatile)"),
               Patch(facecolor=MUTED, alpha=0.18, label="registered +/-20% acceptance band")]
    fig.legend(handles=handles, loc="lower center", ncol=2, frameon=False,
               bbox_to_anchor=(0.5, -0.19))
    # Title states what the figure shows, not the verdict. The verdict is a methods
    # certification and belongs in the caption, where it can carry its band. Author's
    # labelling rule, 2026-08-12: short, specific, no internal names, no outcomes.
    fig.suptitle("Injected signal against Hyperliquid's BTC perpetual",
                 y=1.02, fontsize=11)
    fig.tight_layout()
    save(fig, "s1_injection_fidelity", "main_body")


def fig_s3_dev_campaign_forest() -> None:
    """MAIN. The primary result: all 38 agents, per-group pooled mean +- 95% CI with
    per-seed points. Nothing crosses to a material advantage.
    Source: step5_signal_dev/judgement.json + behaviour_audit.json."""
    j, audit = load_dev()
    groups = [("ppo", "", "PPO base"), ("ppo", "_v1a", "PPO net[64,64]"),
              ("ppo", "_v1b", "PPO net[128,128]"), ("ppo", "_v2", "PPO reward x100"),
              ("dqn", "", "DQN base")]
    fig, axes = plt.subplots(1, 2, figsize=(10.4, 3.9), sharex=True)
    for ax, regime in zip(axes, REGIMES):
        ys = np.arange(len(groups))[::-1]
        for y, (algo, tag, label) in zip(ys, groups):
            seeds = _group_seeds(j, audit, algo, regime, tag)
            if not seeds:
                # axes-fraction x so the note never lands on the data or the band
                ax.text(0.03, y, "no audit-valid seeds (all collapsed)", va="center",
                        ha="left", fontsize=8, style="italic", color=MUTED,
                        transform=ax.get_yaxis_transform(), zorder=5,
                        bbox=dict(facecolor="white", edgecolor="none", pad=1.5))
                continue
            m = float(np.mean(seeds))
            ci = _ci95(np.array(seeds))
            ax.plot(seeds, y + _seed_offsets(len(seeds)), "o", mfc="none",
                    mec=RCOL[regime], ms=4.5, alpha=0.85, zorder=2)
            if np.isfinite(ci):
                ax.errorbar([m], [y], xerr=[[ci], [ci]], fmt="o", color=RCOL[regime],
                            ms=7, capsize=3, lw=1.8, zorder=3)
            else:
                ax.plot([m], [y], "o", color=RCOL[regime], ms=7, zorder=3)
        ax.axvline(0.0, color=MUTED, lw=1.0, ls="--", zorder=1)
        ax.axvspan(-0.05, 0.05, color=MUTED, alpha=0.12, zorder=0)
        ax.set_yticks(ys)
        ax.set_yticklabels([g[2] for g in groups])
        ax.set_xlabel("cost vs adaptive TWAP (bps; negative = cheaper)")
        ax.set_title(regime)
    handles = [Line2D([], [], color=INK, marker="o", ls="none", ms=7, label="group mean +- 95% CI"),
               Line2D([], [], color=INK, marker="o", ls="none", mfc="none", ms=4.5, label="individual seed"),
               Patch(facecolor=MUTED, alpha=0.12, label="+/-0.05 bps materiality band"),
               Line2D([], [], color=MUTED, lw=1.0, ls="--", label="adaptive TWAP")]
    fig.legend(handles=handles, loc="lower center", ncol=4, frameon=False,
               bbox_to_anchor=(0.5, -0.16))
    fig.suptitle("No agent configuration beats TWAP in the injected market (development block, n=2000/agent)",
                 y=1.02, fontsize=11)
    fig.tight_layout()
    save(fig, "s3_dev_campaign_forest", "main_body")


def fig_s4_exploiter_vs_agents() -> None:
    """MAIN -- the decisive figure. The injected signal WAS capturable: a fixed
    one-line rule reading the agents' own observation beats TWAP by 4.6x-9.8x the
    materiality bar, while every trained agent sits at ~0 or worse.
    Sources: diagnostics_postnull/diag_signal_follower.json (exploiters),
    judgement.json (agents). Same env, same seeds, same CRN pairing."""
    fol = load_follower()
    j, audit = load_dev()
    rows = [("follower", "signal-follower\n(registered benchmark)", "exploit"),
            ("bangbang", "threshold rule\n(diagnostic)", "exploit"),
            ("half", "half-strength rule\n(diagnostic)", "exploit"),
            ("__ppo", "PPO base\n(trained agent)", "agent"),
            ("__dqn", "DQN base\n(trained agent)", "agent")]
    fig, axes = plt.subplots(1, 2, figsize=(10.6, 4.0), sharex=True)
    for ax, regime in zip(axes, REGIMES):
        ys = np.arange(len(rows))[::-1]
        for y, (key, label, kind) in zip(ys, rows):
            if kind == "exploit":
                e = fol["regimes"][regime]["exploiters"][key]
                m, se = e["mean_diff_bps"], e["se"]
                col = OI_VERM
                ax.errorbar([m], [y], xerr=[[1.96 * se], [1.96 * se]], fmt="s",
                            color=col, ms=7, capsize=3, lw=1.8, zorder=3)
                ax.text(m, y + 0.28, f"{m:+.3f}", ha="center", va="bottom",
                        fontsize=8, color=INK)
            else:
                algo = "ppo" if key == "__ppo" else "dqn"
                seeds = _group_seeds(j, audit, algo, regime, "")
                col = OI_BLUE
                if not seeds:
                    ax.text(0.03, y, "no audit-valid seeds (all collapsed)", va="center",
                            ha="left", fontsize=8, style="italic", color=MUTED,
                            transform=ax.get_yaxis_transform(), zorder=5,
                            bbox=dict(facecolor="white", edgecolor="none", pad=1.5))
                    continue
                m = float(np.mean(seeds))
                ci = _ci95(np.array(seeds))
                ax.plot(seeds, y + _seed_offsets(len(seeds)), "o", mfc="none",
                        mec=col, ms=4.5, alpha=0.85, zorder=2)
                if np.isfinite(ci):
                    ax.errorbar([m], [y], xerr=[[ci], [ci]], fmt="o", color=col,
                                ms=7, capsize=3, lw=1.8, zorder=3)
                ax.text(m, y + 0.28, f"{m:+.3f}", ha="center", va="bottom",
                        fontsize=8, color=INK)
        ax.axvline(0.0, color=MUTED, lw=1.0, ls="--", zorder=1)
        ax.axvspan(-0.05, 0.05, color=MUTED, alpha=0.12, zorder=0)
        ax.set_yticks(ys)
        ax.set_yticklabels([r[1] for r in rows], fontsize=8.5)
        ax.set_xlabel("cost vs adaptive TWAP (bps; negative = cheaper)")
        ax.set_title(regime)
    handles = [Line2D([], [], color=OI_VERM, marker="s", ls="none", ms=7,
                      label="signal-reading rule (mean +- 95% CI)"),
               Line2D([], [], color=OI_BLUE, marker="o", ls="none", ms=7,
                      label="trained RL agent (mean +- 95% CI)"),
               Line2D([], [], color=OI_BLUE, marker="o", ls="none", mfc="none", ms=4.5,
                      label="individual agent seed"),
               Patch(facecolor=MUTED, alpha=0.12, label="+/-0.05 bps materiality band")]
    fig.legend(handles=handles, loc="lower center", ncol=2, frameon=False,
               bbox_to_anchor=(0.5, -0.20))
    fig.suptitle("The signal was capturable; the agents did not capture it",
                 y=1.02, fontsize=11)
    fig.tight_layout()
    save(fig, "s4_exploiter_vs_agents", "main_body")


# ============================================================== SUPPORTING
def fig_s5_base_vs_injected() -> None:
    """SUPPORT. What the injection actually added: it AMPLIFIED the capturable edge
    ~4-6x rather than creating a channel from nothing. Best hand-coded reader in the
    base (no-signal) env vs the injected env, per regime.
    Sources: diagnostics_postnull/diag_base_env.json + diag_signal_follower.json."""
    base, inj = load_base_env(), load_follower()
    fig, ax = plt.subplots(figsize=(7.4, 3.6))
    x = np.arange(len(REGIMES))
    w = 0.34
    b_best, i_best, b_lab = [], [], []
    for regime in REGIMES:
        bex = base["regimes"][regime]["exploiters"]
        k = min(bex, key=lambda n: bex[n]["mean_diff_bps"])
        b_best.append(bex[k]["mean_diff_bps"])
        b_lab.append(k)
        i_best.append(inj["regimes"][regime]["exploiters"]["follower"]["mean_diff_bps"])
    # Plotted as SAVING (positive = cheaper than TWAP) so a taller bar simply means
    # better; an inverted cost axis reads as a trick and was rejected.
    b_sav = [-v for v in b_best]
    i_sav = [-v for v in i_best]
    ax.bar(x - w / 2, b_sav, w, color=OI_BLUE, edgecolor="white", linewidth=1.0)
    ax.bar(x + w / 2, i_sav, w, color=OI_VERM, edgecolor="white", linewidth=1.0)
    for xi, v in zip(x - w / 2, b_sav):
        ax.text(xi, v + 0.012, f"{v:.3f}", ha="center", va="bottom", fontsize=8.5, color=INK)
    for xi, v in zip(x + w / 2, i_sav):
        ax.text(xi, v + 0.012, f"{v:.3f}", ha="center", va="bottom", fontsize=8.5, color=INK)
    top = max(i_sav) * 1.30
    for xi, b, i in zip(x, b_sav, i_sav):
        ax.annotate(f"x{i / b:.1f}", xy=(xi, max(b, i) * 1.12), ha="center",
                    fontsize=9.5, color=INK, fontweight="bold")
    ax.axhline(0.0, color=MUTED, lw=1.0, ls="--")
    ax.axhspan(-0.05, 0.05, color=MUTED, alpha=0.12, zorder=0)
    ax.set_ylim(-0.06, top)
    ax.set_xticks(x)
    ax.set_xticklabels(REGIMES)
    ax.set_ylabel("saving vs adaptive TWAP (bps)")
    ax.set_title("The injection amplified the capturable edge; it did not create it",
                 fontsize=11)
    handles = [Patch(facecolor=OI_BLUE, label="base environment (best hand-coded reader)"),
               Patch(facecolor=OI_VERM, label="injected environment (registered follower)"),
               Patch(facecolor=MUTED, alpha=0.12, label="+/-0.05 bps materiality band")]
    fig.legend(handles=handles, loc="lower center", ncol=1, frameon=False,
               bbox_to_anchor=(0.5, -0.22))
    fig.tight_layout()
    save(fig, "s5_base_vs_injected", "appendix")


def fig_s6_training_curves() -> None:
    """SUPPORT. No learning trend: paired evaluation vs adaptive TWAP every 100k
    steps across the whole 2M-step budget, all base runs, both regimes.
    Source: runs_signal_phaseD/<run>/curve.json."""
    fig, axes = plt.subplots(1, 2, figsize=(9.4, 3.5), sharey=True)
    for ax, regime in zip(axes, REGIMES):
        for algo, col, ls in (("ppo", RCOL[regime], "-"), ("dqn", MUTED, ":")):
            for seed in range(5):
                p = RUNS / f"{algo}_{regime}_s{seed}" / "curve.json"
                if not p.exists():
                    continue
                c = json.loads(p.read_text())
                xs = [e["steps"] / 1e6 for e in c]
                ys = [e["mean_diff_bps"] for e in c]
                ax.plot(xs, ys, ls, color=col, lw=1.1, alpha=0.75)
        ax.axhline(0.0, color=INK, lw=1.0, ls="--", zorder=1)
        ax.set_xlabel("training steps (millions)")
        ax.set_title(regime)
        if ax is axes[0]:
            ax.set_ylabel(BPS)
    handles = [Line2D([], [], color=BLUE, lw=1.4, label="PPO seeds (calm)"),
               Line2D([], [], color=RED, lw=1.4, label="PPO seeds (volatile)"),
               Line2D([], [], color=MUTED, lw=1.4, ls=":", label="DQN seeds"),
               Line2D([], [], color=INK, lw=1.0, ls="--", label="adaptive TWAP")]
    fig.legend(handles=handles, loc="lower center", ncol=4, frameon=False,
               bbox_to_anchor=(0.5, -0.14))
    fig.suptitle("No agent trends toward an edge over the full training budget",
                 y=1.02, fontsize=11)
    fig.tight_layout()
    save(fig, "s6_training_curves", "appendix")


# ============================================================== APPENDIX
def fig_s8_kernel_structure() -> None:
    """APPENDIX. The calibrated injection kernel: gain per basis timescale, and the
    driver persistence it reproduces. Explains WHY the injected signal is capturable
    (8 s-dominated, i.e. persistent) where the base env's endogenous imbalance is not.
    Source: signal/kernel_solution.json."""
    k = load_kernel()
    fig, axes = plt.subplots(1, 2, figsize=(9.2, 3.4))
    ax = axes[0]
    hl = k["regimes"]["calm"]["kernel"]["halflives_s"]
    x = np.arange(len(hl))
    w = 0.36
    for i, regime in enumerate(REGIMES):
        g = k["regimes"][regime]["kernel"]["gains_bps"]
        ax.bar(x + (i - 0.5) * w, g, w, color=RCOL[regime], label=regime,
               edgecolor="white", linewidth=1.0)
    ax.axhline(0.0, color=MUTED, lw=1.0)
    ax.set_xticks(x)
    ax.set_xticklabels([f"{h:g}" for h in hl])
    ax.set_xlabel("basis half-life (s)")
    ax.set_ylabel("gain (bps per unit demeaned imbalance)")
    ax.set_title("Calibrated kernel gains")
    ax = axes[1]
    for regime in REGIMES:
        rho = k["regimes"][regime]["rho_first10"]
        ax.plot(np.arange(len(rho)) * 0.5, rho, "-o", color=RCOL[regime], ms=4,
                lw=1.6, label=regime)
    ax.axhline(0.0, color=MUTED, lw=1.0)
    ax.set_xlabel("lag (s)")
    ax.set_ylabel("autocorrelation of the measured signal")
    ax.set_title("Measured signal persistence (real data)")
    handles = [Line2D([], [], color=BLUE, lw=2, label="calm"),
               Line2D([], [], color=RED, lw=2, label="volatile")]
    fig.legend(handles=handles, loc="lower center", ncol=2, frameon=False,
               bbox_to_anchor=(0.5, -0.10))
    fig.tight_layout()
    save(fig, "s8_kernel_structure", "appendix")


L2_TEST = Path("/Users/fardeenidrus/Desktop/MSc Dissertation/scratch_hyperliquid/l2_test_results")
L2_LEVER = {                      # folder -> the design cell it represents
    "runs":            "1-min data, 30-min deadline",
    "runs_10s":        "10-s data, 30-min deadline",
    "runs_10s_10min":  "10-s data, 10-min deadline",
}


def load_l2_sealed_arms() -> list[dict]:
    """Every pre-registered L2 arm scored on the SEALED test split, pooled over
    BEHAVIOUR-VALID seeds only. The stored `arm_summary.pooled_mean_paired_diff_bps`
    pools ALL seeds including agents the deadline audit flags invalid, which
    contradicts the project-wide audit-before-cost rule; it is recomputed here.
    No PASS verdict moves (all four nominally-passing arms have zero invalid seeds),
    but four DQN control arms shift, all in the direction that strengthens the
    control. See addendum in reports/l2_test_protocol.md."""
    arms = []
    for f in sorted(L2_TEST.glob("test_*.json")):
        d = json.loads(f.read_text())
        for a in d["arm_summary"]:
            rs = [r for r in d["runs_flat"]
                  if r["algo"] == a["algo"] and r["size_btc"] == a["size_btc"]]
            ok = [r for r in rs if not r["dl_flag"]]
            if not ok:
                continue
            arms.append({
                "lever": L2_LEVER[a["runs_dir"]],
                "algo": a["algo"],
                "size": a["size_btc"],
                "bps": float(np.mean([r["mean_paired_diff_bps"] for r in ok])),
                "n_valid": len(ok),
                "n_total": len(rs),
                "n_cheaper": int(sum(r["mean_paired_diff_bps"] < 0 for r in ok)),
            })
    return arms


def load_reactive_sealed_arms() -> list[dict]:
    """The two sealed out-of-sample confirmations of the no-signal reactive track
    (R8b selected config v3a on block 9e6; remedial literal-rule pick v1b on 13e6).
    Both PASS=False -> the boundary null."""
    out = []
    for folder, label in (("step5_confirm_v3a", "selected config (lr 1e-3), volatile"),
                          ("step5_confirm_v1b", "remedial pick (bigger net), volatile")):
        v = json.loads((S / folder / "judgement.json").read_text())["verdicts"]["ppo_volatile"]
        out.append({"lever": label, "algo": "ppo", "size": None,
                    "bps": float(v["pooled_vs_adaptive_bps"]),
                    "n_valid": v["n_valid_seeds"], "n_total": v["n_valid_seeds"],
                    "n_cheaper": v["n_cheaper_of_valid"]})
    return out


def load_injected_sealed_arms() -> list[dict]:
    """The sealed exhibit for the injected environment (registered amendment, one shot,
    block 17e6): both regimes, PPO, 5 seeds each. Both PASS=False."""
    v = json.loads((S / "step5_signal_sealed" / "judgement.json").read_text())["verdicts"]
    return [{"lever": f"{reg} regime", "algo": "ppo", "size": None,
             "bps": float(v[f"ppo_{reg}"]["pooled_vs_adaptive_bps"]),
             "n_valid": v[f"ppo_{reg}"]["n_valid_seeds"],
             "n_total": v[f"ppo_{reg}"]["n_valid_seeds"],
             "n_cheaper": v[f"ppo_{reg}"]["n_cheaper_of_valid"]} for reg in REGIMES]


def fig_s2_three_environment() -> None:
    """MAIN -- the headline figure. Three independent environments of increasing
    favourability to the agent, each judged on data sealed until the verdict was
    fixed; no stable saving against TWAP anywhere. Panel B answers the obvious
    objection to a null (`maybe there was nothing to find'): in the third
    environment a registered rule with zero fitted parameters banks a large,
    strongly significant saving, and the agents capture none of it.

    CAPTION CAVEAT that must travel with this figure: the three environments differ
    on more than one axis (frozen replay of real book data with 1-min/10-s decisions
    vs a reactive simulator with 1-s decisions). The CONTROLLED contrast is
    environments 2 -> 3, where only the injected signal changes. Environment 1 is a
    qualitative anchor, not a matched comparison; and its result is reported as `no
    stable edge' rather than a clean null, because four of its fourteen arms do clear
    the bar nominally while a diagnosed-broken learner clears it equally (see the L2
    inversion figure) -- i.e. the apparent edge is a property of the test period.
    That last clause is no longer an inference: as of 2026-07-30 the mechanism is
    MEASURED (l2_test_protocol.md mechanism addendum; figure l2_inversion_mechanism).
    It is a pacing exposure -- within-episode drift reverses sign between the periods,
    a fixed rule incapable of learning reproduces the whole reversal, and 95% of the
    shift decomposes onto each agent's front-loading dose. Captions for environment 1
    may now say `attributable to a pacing exposure' rather than `unexplained'.

    Blocks are NOT shared across panels: the agents' sealed numbers come from block
    17e6, the ceiling confirmation from the minted block 21e6. Both are held out; the
    ceiling is a property of the environment, not of a particular block, and its
    calm/volatile values on the dev block (0.230/0.490 bps) bracket the same picture.

    Sources: l2_test_results/test_*.json; step5_confirm_v3a, step5_confirm_v1b,
    step5_signal_sealed/judgement.json; step5_signal_ceiling21e6/ceiling_confirmation.json.
    """
    envs = [
        ("1. Frozen replay of real order-book data", load_l2_sealed_arms(),
         "no stable edge — 4 of 14 arms clear the bar, and a diagnosed-broken learner clears it too"),
        ("2. Reactive simulator, no injected signal", load_reactive_sealed_arms(),
         "null — both sealed confirmations fail"),
        ("3. Reactive simulator + measured signal", load_injected_sealed_arms(),
         "null — both regimes fail"),
    ]
    # Horizontal, one sub-panel per environment, shared cost axis: 18 arms cannot carry
    # readable vertical tick labels, and stacking keeps each environment's verdict beside
    # its own arms instead of in a shared margin.
    flat = [a for _, arms, _ in envs for a in arms]
    lo = min(a["bps"] for a in flat) - 0.10
    hi = max(a["bps"] for a in flat) + 0.10
    fig = plt.figure(figsize=(13.4, 6.4))
    outer = fig.add_gridspec(1, 2, width_ratios=[2.05, 1.0], wspace=0.02)
    left = outer[0, 0].subgridspec(3, 1, height_ratios=[len(a) + 1.4 for _, a, _ in envs],
                                   hspace=0.42)
    axes_a = []
    for row, (name, arms, verdict) in enumerate(envs):
        ax = fig.add_subplot(left[row, 0], sharex=axes_a[0] if axes_a else None)
        axes_a.append(ax)
        ys = np.arange(len(arms))[::-1]
        ax.axvspan(-0.05, 0.05, color=MUTED, alpha=0.13, zorder=0)
        ax.axvline(0.0, color=INK, lw=1.2, ls="--", zorder=1)
        for y, a in zip(ys, arms):
            col = BLUE if a["algo"] == "ppo" else RED
            mk = "o" if a["algo"] == "ppo" else "s"
            ax.plot([a["bps"]], [y], mk, color=col, ms=8, zorder=3,
                    markeredgecolor="white", markeredgewidth=1.0)
            ax.annotate(f"{a['n_cheaper']}/{a['n_valid']} seeds cheaper",
                        xy=(a["bps"], y), xytext=(9, 0), textcoords="offset points",
                        va="center", fontsize=6.8, color=MUTED)
        ax.set_yticks(ys)
        ax.set_yticklabels(
            [f"{a['lever']}" + (f"  ·  {a['algo'].upper()} {a['size']:.0f} BTC"
                                if a["size"] else f"  ·  {a['algo'].upper()}")
             for a in arms], fontsize=7.2)
        ax.set_ylim(-0.85, len(arms) - 0.15)
        ax.set_xlim(lo, hi)
        ax.tick_params(axis="y", length=0)
        ax.set_title(name, fontsize=9.8, loc="left", color=INK, fontweight="bold", pad=13)
        ax.annotate(verdict, xy=(0.0, 1.0), xycoords="axes fraction", xytext=(0, 3),
                    textcoords="offset points", fontsize=7.8, color=MUTED, style="italic")
        if row < 2:
            ax.tick_params(axis="x", labelbottom=False)
    axes_a[-1].set_xlabel("pooled cost vs TWAP (bps)   [left of the line = cheaper than TWAP]")
    # the one matched comparison in the figure, marked as such
    con = fig.add_subplot(left[1:, 0], frameon=False)
    con.set_xticks([]); con.set_yticks([])
    con.patch.set_alpha(0.0)
    con.annotate("", xy=(1.012, 0.04), xytext=(1.012, 0.90), xycoords="axes fraction",
                 arrowprops=dict(arrowstyle="<->", color=INK, lw=1.2))
    con.annotate("controlled contrast: only the signal differs", xy=(1.028, 0.47),
                 xycoords="axes fraction", va="center", ha="center", rotation=90,
                 fontsize=7.6, color=INK)

    # ---- panel B: the edge was there to take, and the agents did not take it ----
    ax2 = fig.add_subplot(outer[0, 1])
    ax2.set_position(ax2.get_position().translated(0.075, 0.11).shrunk(0.76, 0.64))
    ceil = json.loads((S / "step5_signal_ceiling21e6" /
                       "ceiling_confirmation.json").read_text())["regimes"]
    inj = {a["lever"].split()[0]: a for a in load_injected_sealed_arms()}
    x = np.arange(len(REGIMES))
    w = 0.36
    c_sav = [-ceil[r]["vs_adaptive"]["mean_diff_bps"] for r in REGIMES]
    a_sav = [-inj[r]["bps"] for r in REGIMES]
    ax2.bar(x - w / 2, c_sav, w, color=OI_GREEN, edgecolor="white", linewidth=1.0)
    ax2.bar(x + w / 2, a_sav, w, color=OI_ORANGE, edgecolor="white", linewidth=1.0)
    for xi, v in zip(x - w / 2, c_sav):
        ax2.text(xi, v + 0.014, f"{v:.3f}", ha="center", va="bottom", fontsize=8.5, color=INK)
    for xi, v in zip(x + w / 2, a_sav):
        ax2.text(xi, min(v, 0) - 0.014, f"{v:+.3f}", ha="center", va="top",
                 fontsize=8.5, color=INK)
    for xi, c, a in zip(x, c_sav, a_sav):
        ax2.annotate(f"{100 * a / c:.0f}% of the\navailable edge\ncaptured",
                     xy=(xi + w / 2, min(a, 0) - 0.055), ha="center", va="top",
                     fontsize=8, color=INK, fontweight="bold")
    ax2.axhline(0.0, color=INK, lw=1.2, ls="--")
    ax2.axhspan(-0.05, 0.05, color=MUTED, alpha=0.13, zorder=0)
    ax2.set_xticks(x)
    ax2.set_xticklabels(REGIMES)
    ax2.set_ylim(min(min(a_sav) - 0.20, -0.20), max(c_sav) * 1.15)
    ax2.set_ylabel("saving vs TWAP (bps)")
    ax2.set_title("Environment 3: the edge was real\nand the agents left all of it",
                  fontsize=10.5)

    handles = [
        Line2D([], [], color=BLUE, marker="o", ls="", ms=8, label="PPO arm (sealed)"),
        Line2D([], [], color=RED, marker="s", ls="", ms=8, label="DQN arm (sealed)"),
        Line2D([], [], color=INK, lw=1.2, ls="--", label="TWAP"),
        Patch(facecolor=MUTED, alpha=0.13, label="+/-0.05 bps materiality band"),
        Patch(facecolor=OI_GREEN, label="registered rule, zero fitted parameters"),
        Patch(facecolor=OI_ORANGE, label="trained agents (sealed)"),
    ]
    fig.legend(handles=handles, loc="lower center", ncol=3, frameon=False,
               bbox_to_anchor=(0.5, -0.09))
    fig.suptitle("Three environments, three sealed verdicts: no stable improvement on TWAP "
                 "anywhere -- including where a simple rule proves the edge exists",
                 y=1.03, fontsize=11.5)
    save(fig, "s2_three_environment", "main_body")
    print(f"    fractions of ceiling captured: "
          f"{[f'{100*a/c:.1f}%' for a, c in zip(a_sav, c_sav)]}")


def main() -> None:
    print("building measured-signal extension figures ->", OUT)
    print(" MAIN:")
    fig_s1_injection_fidelity()
    fig_s3_dev_campaign_forest()
    fig_s4_exploiter_vs_agents()
    fig_s2_three_environment()
    print(" SUPPORTING:")
    fig_s5_base_vs_injected()
    fig_s6_training_curves()
    fig_s10_frontier()
    fig_s11_a4_observation()
    print(" APPENDIX:")
    fig_s8_kernel_structure()
    print("done.")




# ============================================================== A3 COMPARATORS
def fig_s10_frontier() -> None:
    """SUPPORT (promotable). Risk-return frontier, injected env, dev seeds: every feasible
    policy as (cost std, mean cost), with the efficient set joined and the 5 agents per
    regime plotted individually. Registered question (criteria A3): do the agents lie
    strictly inside the frontier? Calm 5/5 dominated; volatile 3/5 (caveats in addendum O).
    Sources: step5_comparators/{injected_dev.json, agents_dev_per_episode.npz,
    frontier_summary.json}."""
    comp = json.loads((S / "step5_comparators" / "injected_dev.json").read_text())
    fr = json.loads((S / "step5_comparators" / "frontier_summary.json").read_text())
    ag = np.load(S / "step5_comparators" / "agents_dev_per_episode.npz")
    label = {"adaptive": "adaptive TWAP", "fixed": "fixed TWAP", "ac_kT0": "AC $\\kappa T$=0",
             "ac_kT1": "AC $\\kappa T$=1", "ac_kT2": "AC $\\kappa T$=2",
             "ac_kT4": "AC $\\kappa T$=4", "vwap_oracle": "oracle VWAP"}
    fig, axes = plt.subplots(1, 2, figsize=(10.2, 4.2))
    for ax, regime in zip(axes, REGIMES):
        rows = comp["regimes"][regime]
        front = {f["policy"] for f in fr[regime]["frontier"]}
        fx = [f["std"] for f in fr[regime]["frontier"]]
        fy = [f["mean"] for f in fr[regime]["frontier"]]
        order = np.argsort(fx)
        ax.plot(np.array(fx)[order], np.array(fy)[order], "-", color=MUTED, lw=1.4,
                zorder=1, alpha=0.9)
        # group coincident benchmarks so their labels do not overprint. The TWAP/AC-kT0
        # cluster is coincident BECAUSE of the lambda=0 identity, so one merged label
        # states that result instead of stacking three names on one marker.
        drawn: list = []
        for k, v in rows.items():
            s, m = v["std_cost_bps"], v["mean_cost_bps"]
            if k == "vwap_oracle":
                ax.plot(s, m, "x", color=OI_ORANGE, ms=8, mew=2, zorder=4)
                ax.annotate("oracle VWAP", (s, m), fontsize=7.5, color=INK,
                            xytext=(4, 4), textcoords="offset points")
                continue
            hit = next((g for g in drawn
                        if abs(g["s"] - s) < 0.02 and abs(g["m"] - m) < 0.005), None)
            if hit is not None:
                hit["keys"].append(k)
                hit["on"] = hit["on"] or (k in front)
                continue
            drawn.append({"s": s, "m": m, "keys": [k], "on": k in front})
        for g in drawn:
            on = g["on"]
            ax.plot(g["s"], g["m"], "s", color=OI_VERM if on else MUTED,
                    mfc=OI_VERM if on else "white", mec=OI_VERM if on else MUTED,
                    ms=7, zorder=3)
            ks = g["keys"]
            if {"adaptive", "fixed", "ac_kT0"} <= set(ks):
                txt = "TWAP = AC $\\kappa T$=0"          # the identity, shown
            elif len(ks) > 1:
                txt = " = ".join(label.get(k, k) for k in ks)
            else:
                txt = label.get(ks[0], ks[0])
            ax.annotate(txt, (g["s"], g["m"]), fontsize=7.5, color=INK,
                        xytext=(5, 4), textcoords="offset points")
        for sd in range(5):
            c = ag[f"ppo_{regime}_s{sd}"]
            ax.plot(c.std(ddof=1), c.mean(), "o", mfc="none", mec=OI_BLUE, ms=7,
                    mew=1.6, zorder=5)
        ax.set_xlabel("cost standard deviation (bps)   [risk]")
        if ax is axes[0]:
            ax.set_ylabel("mean cost (bps)   [lower = cheaper]")
        n_dom = fr[regime]["n_dominated"]
        ax.set_title(f"{regime}  ({n_dom}/5 agents strictly dominated)")
    handles = [Line2D([], [], color=OI_VERM, marker="s", ls="none", ms=7,
                      label="benchmark on the efficient frontier"),
               Line2D([], [], color=MUTED, marker="s", ls="none", mfc="white", ms=7,
                      label="benchmark off the frontier"),
               Line2D([], [], color=MUTED, lw=1.4, label="efficient frontier"),
               Line2D([], [], color=OI_BLUE, marker="o", ls="none", mfc="none", ms=7,
                      label="trained RL agent (one per seed)"),
               Line2D([], [], color=OI_ORANGE, marker="x", ls="none", ms=8, mew=2,
                      label="oracle VWAP (infeasible; excluded from the frontier)")]
    fig.legend(handles=handles, loc="lower center", ncol=2, frameon=False,
               bbox_to_anchor=(0.5, -0.24))
    fig.suptitle("Agents pay for their deviation from uniform pacing without buying risk reduction",
                 y=1.02, fontsize=11)
    fig.tight_layout()
    save(fig, "s10_risk_return_frontier", "appendix")




def fig_s11_a4_observation() -> None:
    """MAIN. Amendment A4: the observation variant. Left = the mechanism (critic explained
    variance per run, before vs after). Right = the verdict (performance unchanged).
    Sources: diagnostics_postnull/diag_learning{,_a4}.json, step5_signal_obsfix/."""
    orig = {r["run"]: r for r in json.loads((DIAG / "diag_learning.json").read_text())}
    a4 = {r["run"]: r for r in json.loads((DIAG / "diag_learning_a4.json").read_text())}
    j4 = json.loads((S / "step5_signal_obsfix" / "judgement.json").read_text())
    au4 = {x["run"]: x for x in json.loads(
        (S / "step5_signal_obsfix" / "behaviour_audit.json").read_text())}
    jd, aud = load_dev()
    fig, axes = plt.subplots(1, 2, figsize=(10.2, 4.2))

    # ---- left: the mechanism ----
    ax = axes[0]
    runs = sorted(orig)
    y = np.arange(len(runs))
    for i, r in enumerate(runs):
        o, n = orig[r]["critic"]["explained_variance"], a4[r]["ev"]
        ax.plot([o, n], [i, i], "-", color=MUTED, lw=1.0, zorder=1)
        ax.plot(o, i, "o", color=MUTED, mfc="white", ms=6, zorder=3)
        ax.plot(n, i, "o", color=OI_VERM, ms=6, zorder=3)
    ax.axvline(0.0, color=INK, lw=1.0, ls="--", zorder=0)
    ax.set_yticks(y)
    ax.set_yticklabels([r.replace("ppo_", "").replace("_", " ") for r in runs], fontsize=8)
    ax.set_xlabel("how much of the outcome the value function predicts   [higher is better]")
    ax.set_title("How well the value function predicts the outcome")

    # ---- right: the verdict ----
    ax = axes[1]
    labels, orig_m, new_m, orig_n, new_n = [], [], [], [], []
    for reg in REGIMES:
        o = [r["mean_vs_adaptive_bps"] for r in jd["per_run"]
             if r["algo"] == "ppo" and r["regime"] == reg
             and r["run"].endswith(f"s{r['seed']}") and aud[r["run"]]["valid"]]
        n = [r["mean_vs_adaptive_bps"] for r in j4["per_run"]
             if r["regime"] == reg and au4[r["run"]]["valid"]]
        labels.append(reg); orig_m.append(np.mean(o)); new_m.append(np.mean(n))
        orig_n.append(len(o)); new_n.append(len(n))
    x = np.arange(len(labels)); w = 0.34
    b1 = ax.bar(x - w / 2, orig_m, w, color=MUTED, label="without the arrival price",   # NOT rendered: the legend is built from Line2D handles below. Kept in step with them so the two can never disagree.
                edgecolor="white", linewidth=1.0)
    b2 = ax.bar(x + w / 2, new_m, w, color=OI_VERM, label="with the arrival price",
                edgecolor="white", linewidth=1.0)
    for b, m, k in list(zip(b1, orig_m, orig_n)) + list(zip(b2, new_m, new_n)):
        ax.text(b.get_x() + b.get_width() / 2, m + 0.0015, f"{m:+.4f}\n({k}/5 valid)",
                ha="center", va="bottom", fontsize=7.5, color=INK)
    ax.axhline(0.0, color=MUTED, lw=1.0, ls="--")
    ax.axhspan(-0.05, 0.05, color=MUTED, alpha=0.12, zorder=0)
    ax.set_xticks(x); ax.set_xticklabels(labels)
    ax.set_ylabel("cost vs adaptive TWAP (bps)")
    ax.set_ylim(-0.055, max(max(orig_m), max(new_m)) * 1.9 + 0.02)
    ax.set_title("Execution cost against TWAP")
    handles = [Line2D([], [], color=MUTED, marker="o", mfc="white", ls="none", ms=6,
                      label="without the arrival price (28 inputs)"),
               Line2D([], [], color=OI_VERM, marker="o", ls="none", ms=6,
                      label="with the arrival price (29 inputs)"),
               Patch(facecolor=MUTED, alpha=0.12, label="+/-0.05 bps materiality band")]
    fig.legend(handles=handles, loc="lower center", ncol=3, frameon=False,
               bbox_to_anchor=(0.5, -0.13))
    # TITLE RULE: name what was ADDED, never "repairing"/"making learnable" -- those concede a
    # defect where the accurate statement is a specification test. See F23/F26/F27 for the
    # matching wording; the four titles must stay consistent.
    fig.suptitle("Adding the arrival price to the agent's inputs improves the value function "
                 "but not execution cost",
                 y=1.02, fontsize=11)
    fig.tight_layout()
    save(fig, "s11_a4_observation", "main_body")


if __name__ == "__main__":
    main()
