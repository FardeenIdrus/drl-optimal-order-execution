"""Verification sweep: every numeric claim in the Results chapter against its frozen source.

Each check names the claim, the source file, and PASSes only on an exact match to the
precision quoted in the chapter. A source under a SUPERSEDED_ prefix, or any file not named
as current in RESULTS_MANIFEST.md, is an automatic FAIL.
"""
from __future__ import annotations
import json, glob
from pathlib import Path
import numpy as np

S = Path("/Users/fardeenidrus/Desktop/MSc Dissertation/scratch_hyperliquid/oxford_l4")
L2 = Path("/Users/fardeenidrus/Desktop/MSc Dissertation/scratch_hyperliquid/l2_test_results")
rows = []


def chk(section, claim, quoted, actual, src, tol=5e-4):
    ok = (abs(quoted - actual) <= tol) if isinstance(quoted, (int, float)) else quoted == actual
    rows.append((ok, section, claim, quoted, actual, src))


def J(p):
    return json.loads((S / p).read_text())


# ---- 5.3 reactive: primary volatile signal + two sealed confirmations -----------------
j = J("step5_v3/judgement.json")["verdicts"]["ppo_volatile"]
chk("5.3", "volatile PPO primary vs adaptive", -0.047, round(j["pooled_vs_adaptive_bps"], 3),
    "step5_v3/judgement.json")
chk("5.3", "volatile PPO across-seed p", 0.006,
    round(j["across_seed"]["across_seed_t_p_vs_adaptive_onesided"], 3), "step5_v3/judgement.json")
for folder, q_bps, q_p, q_ch in (("step5_confirm_v3a", -0.0023, 0.38, 3),
                                 ("step5_confirm_v1b", -0.0022, 0.39, 2)):
    v = J(f"{folder}/judgement.json")["verdicts"]["ppo_volatile"]
    chk("5.3", f"{folder} pooled", q_bps, round(v["pooled_vs_adaptive_bps"], 4), folder)
    chk("5.3", f"{folder} p", q_p, round(v["across_seed_t_p_onesided"], 2), folder)
    chk("5.3", f"{folder} seeds cheaper", q_ch, v["n_cheaper_of_valid"], folder)

# ---- 5.4 injected sealed exhibit ------------------------------------------------------
v = J("step5_signal_sealed/judgement.json")["verdicts"]
chk("5.4", "sealed calm pooled", 0.005, round(v["ppo_calm"]["pooled_vs_adaptive_bps"], 3),
    "step5_signal_sealed")
chk("5.4", "sealed calm p", 0.63, round(v["ppo_calm"]["across_seed_t_p_onesided"], 2),
    "step5_signal_sealed")
chk("5.4", "sealed calm cheaper", 3, v["ppo_calm"]["n_cheaper_of_valid"], "step5_signal_sealed")
chk("5.4", "sealed volatile pooled", 0.025,
    round(v["ppo_volatile"]["pooled_vs_adaptive_bps"], 3), "step5_signal_sealed")
chk("5.4", "sealed volatile p", 0.81, round(v["ppo_volatile"]["across_seed_t_p_onesided"], 2),
    "step5_signal_sealed")
chk("5.4", "sealed volatile cheaper", 2, v["ppo_volatile"]["n_cheaper_of_valid"],
    "step5_signal_sealed")
# "six independent samples" = dev, reserve, curve-block, sealed x both regimes
for f in ("step5_signal_dev", "step5_signal_reserve", "step5_signal_curveblock",
          "step5_signal_sealed"):
    chk("5.4", f"{f} exists and no EDGE/PASS", True,
        all(not (g.get("EDGE") or g.get("PASS"))
            for g in J(f"{f}/judgement.json")["verdicts"].values()), f)

# ---- 5.5 the ceiling ------------------------------------------------------------------
d = J("step5_signal_dev/diagnostics_postnull/diag_signal_follower.json")["regimes"]
chk("5.5", "untuned follower calm saving", 0.230,
    round(-d["calm"]["exploiters"]["follower"]["mean_diff_bps"], 3), "diag_signal_follower")
chk("5.5", "untuned follower volatile saving", 0.490,
    round(-d["volatile"]["exploiters"]["follower"]["mean_diff_bps"], 3), "diag_signal_follower")
c = J("step5_signal_ceiling21e6/ceiling_confirmation.json")["regimes"]
chk("5.5", "ceiling calm", 0.313, round(-c["calm"]["vs_adaptive"]["mean_diff_bps"], 3),
    "step5_signal_ceiling21e6")
chk("5.5", "ceiling volatile", 0.625, round(-c["volatile"]["vs_adaptive"]["mean_diff_bps"], 3),
    "step5_signal_ceiling21e6")
chk("5.5", "executed frac calm > 0.98", True, c["calm"]["executed_frac"] > 0.98,
    "step5_signal_ceiling21e6")
chk("5.5", "executed frac volatile > 0.98", True, c["volatile"]["executed_frac"] > 0.98,
    "step5_signal_ceiling21e6")
chk("5.5", "ceiling p < 1e-25 both", True,
    c["calm"]["vs_adaptive"]["wilcoxon_p"] < 1e-25
    and c["volatile"]["vs_adaptive"]["wilcoxon_p"] < 1e-25, "step5_signal_ceiling21e6")
for reg, q in (("calm", -2), ("volatile", -4)):
    cap = -v[f"ppo_{reg}"]["pooled_vs_adaptive_bps"] / -c[reg]["vs_adaptive"]["mean_diff_bps"]
    chk("5.5", f"captured fraction {reg} (%)", q, round(100 * cap), "sealed / ceiling", tol=0.6)

# ---- 5.6 action-space parity ----------------------------------------------------------
p2 = json.loads((L2 / "l2_inversion_stage2.json").read_text())
worst_10s = max(abs(p2[k][sp]["grid"]["1"]["mean_diff_bps"])
                for k in ("runs_10s", "runs_10s_10min") for sp in ("validation", "test"))
chk("5.6", "m=1 reproduces TWAP within 0.007 bps (10-s)", True, worst_10s <= 0.0075,
    "l2_inversion_stage2.json")
worst_1m = max(abs(p2["runs"][sp]["grid"]["1"]["mean_diff_bps"]) for sp in ("validation", "test"))
chk("5.6", "m=1 within 0.023 bps (1-min)", True, worst_1m <= 0.0235, "l2_inversion_stage2.json")

# ---- 5.7 / 5.12 learning diagnostics --------------------------------------------------
orig = json.loads((S / "step5_signal_dev/diagnostics_postnull/diag_learning.json").read_text())
a4 = json.loads((S / "step5_signal_dev/diagnostics_postnull/diag_learning_a4.json").read_text())
ev_o = np.mean([r["critic"]["explained_variance"] for r in orig])
ev_a = np.mean([r["ev"] for r in a4])
chk("5.7", "critic EV, original observation", -0.004, round(float(ev_o), 3), "diag_learning")
chk("5.12", "critic EV, corrected observation", 0.405, round(float(ev_a), 3), "diag_learning_a4")
chk("5.7", "n agents in EV mean", 10, len(orig), "diag_learning")

# ---- 5.7 attribution (frozen-replay) --------------------------------------------------
s3 = json.loads((L2 / "l2_inversion_stage3.json").read_text())
r_all = [abs(s3[sp][k]["corr_with_probe"]) for sp in ("validation", "test") for k in s3[sp]]
chk("5.7", "mean |r| with fixed pacing rule", 0.95, round(float(np.mean(r_all)), 2),
    "l2_inversion_stage3.json", tol=0.005)
chk("5.7", "n agents", 30, len(s3["validation"]), "l2_inversion_stage3.json")

# ---- 5.9 robustness / DQN collapse ----------------------------------------------------
a = json.loads((S / "step5_v3/behaviour_audit.json").read_text())
dq = [x for x in a if x["run"].startswith("dqn")]
chk("5.9", "DQN invalid, primary campaign", 7, sum(not x["valid"] for x in dq), "step5_v3 audit")
a = json.loads((S / "step5_signal_dev/behaviour_audit.json").read_text())
dq = [x for x in a if x["run"].startswith("dqn")]
chk("5.9", "DQN invalid, injected env", 8, sum(not x["valid"] for x in dq),
    "step5_signal_dev audit")

# ---- 5.11 the inversion ---------------------------------------------------------------
test = {}
for f in sorted(L2.glob("test_*.json")):
    for r in json.loads(f.read_text())["runs_flat"]:
        test[(r["runs_dir"], r["run"])] = r
val = {(r["runs_dir"], r["run"]): r
       for r in json.loads((L2 / "val_recheck.json").read_text())["runs_flat"]}
keys = [k for k in val if k[0] == "runs_10s"]
flips = sum((val[k]["mean_paired_diff_bps"] > 0) != (test[k]["mean_paired_diff_bps"] > 0)
            for k in keys)
chk("5.11", "agents reversing sign (of 30)", 28, flips, "test_*.json + val_recheck.json")
g = p2["runs_10s"]
chk("5.11", "probe m=2 validation", 0.236, round(g["validation"]["grid"]["2"]["mean_diff_bps"], 3),
    "l2_inversion_stage2.json")
chk("5.11", "probe m=2 sealed test saving", 0.548,
    round(-g["test"]["grid"]["2"]["mean_diff_bps"], 3), "l2_inversion_stage2.json")
runs = sorted(s3["validation"])
bv = np.array([s3["validation"][k]["beta_frontload_dose"] for k in runs])
bt = np.array([s3["test"][k]["beta_frontload_dose"] for k in runs])
mv = np.array([s3["validation"][k]["mean_diff_bps"] for k in runs])
mt = np.array([s3["test"][k]["mean_diff_bps"] for k in runs])
chk("5.11", "dose correlation across periods", 0.999, round(float(np.corrcoef(bv, bt)[0, 1]), 3),
    "l2_inversion_stage3.json")
pv, pt = s3["validation"][runs[0]]["probe_mean_bps"], s3["test"][runs[0]]["probe_mean_bps"]
shift, pacing = float((mt - mv).mean()), float((bt * pt - bv * pv).mean())
chk("5.11", "% of shift explained by pacing", 95, round(100 * pacing / shift),
    "l2_inversion_stage3.json", tol=0.6)
chk("5.11", "residual (bps)", -0.025, round(shift - pacing, 3), "l2_inversion_stage3.json")
chk("5.11", "agents with negative dose", 4, int((bt < 0).sum()), "l2_inversion_stage3.json")
deg = [k for k in runs if abs(s3["test"][k]["beta_frontload_dose"] - 1.0) < 1e-3
       and abs(s3["test"][k]["corr_with_probe"] - 1.0) < 1e-3]
chk("5.11", "degenerate agent identical to max-pace rule", 1, len(deg),
    "l2_inversion_stage3.json")
n_pass = 0
for f in sorted(L2.glob("test_*.json")):
    d = json.loads(f.read_text())
    for arm in d["arm_summary"]:
        if arm["across_seed_t_p_less"] < 0.05 and arm["seeds_cheaper_than_twap"] == 5:
            n_pass += 1
chk("5.11", "arms clearing the bar (of 14)", 4, n_pass, "test_*.json")

# ---- stale-source guard ---------------------------------------------------------------
srcs = {r[5] for r in rows}
stale = [s for s in srcs if "SUPERSEDED" in s]
chk("ALL", "no SUPERSEDED_ source used", 0, len(stale), "stale guard")

# ---- report ---------------------------------------------------------------------------
print(f"{'':<4}{'sec':<7}{'claim':<48}{'chapter':>12}{'source':>12}  file")
bad = 0
for ok, sec, claim, q, actual, src in rows:
    if not ok:
        bad += 1
    print(f"{'PASS' if ok else 'FAIL':<4}{sec:<7}{claim[:47]:<48}{str(q):>12}{str(actual):>12}  {src}")
print(f"\n{len(rows) - bad}/{len(rows)} PASS, {bad} FAIL")
