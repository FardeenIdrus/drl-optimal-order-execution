"""Logged re-runs: capture average reward per episode for the agents that never had it.

WHY. The diagnostic question: does average episode reward trend upward during
training? fig23 answers it for the ORIGINAL Phase D agents (28 inputs, broken critic) because
those ten were retrained with logging in `runs_signal_logged`. It cannot answer it for:

  * the A4 agents   (injected track, 29 inputs, critic WORKS)   -- the obvious follow-up
  * the A4.3 agents (primary track, 28 inputs, critic WORKS)    -- the track carrying the claim
  * the primary campaign of record (27 inputs, critic broken)   -- the primary-track baseline

None of those three campaigns trained with a logger, so no reward series exists for any of
them. This retrains all thirty with `--log-learning` and NOTHING else changed, writing to NEW
directories so the agents of record are never touched.

THE GATE, and it is the whole point. Training is deterministic given the seed, and
`--log-learning` only attaches an SB3 logger, so each re-run must reproduce its original's
final curve value EXACTLY. A re-run that does not reproduce is not the same agent and its
reward series would be a look-alike, not evidence. Any mismatch is reported, not tolerated.
This is the same procedure that produced `runs_signal_logged` (all 20 passed).

Output dirs:
  runs_signal_obsfix_logged      <- runs_signal_obsfix       (inject + price-vs-arrival)
  runs_primary_v3_obsfix_logged  <- runs_primary_v3_obsfix   (price-vs-arrival)
  runs_primary_v3_logged         <- runs_primary_v3          (base, PPO only)
"""
import json
import os
import subprocess
import time
from pathlib import Path

REPO = Path("/Users/fardeenidrus/Desktop/MSc Dissertation/code/drl-optimal-order-execution")
S = Path("/Users/fardeenidrus/Desktop/MSc Dissertation/scratch_hyperliquid/oxford_l4")
PY = str(REPO / ".venv" / "bin" / "python")
CONC = 5

# (source campaign, logged output dir, extra training flags)
ARMS = [
    ("runs_signal_obsfix", "runs_signal_obsfix_logged",
     ["--inject", "--obs-price-vs-arrival"]),
    ("runs_primary_v3_obsfix", "runs_primary_v3_obsfix_logged",
     ["--obs-price-vs-arrival"]),
    ("runs_primary_v3", "runs_primary_v3_logged", []),
]


def specs():
    out = []
    for src, dst, flags in ARMS:
        for regime in ("calm", "volatile"):
            for seed in range(5):
                out.append({"src": src, "dst": dst, "flags": flags,
                            "regime": regime, "seed": seed,
                            "run": f"ppo_{regime}_s{seed}"})
    return out


def cmd(r):
    # The injected arm's per-regime signal offset is not passed here: --inject makes
    # train_reactive re-derive it from the frozen kernel_solution.json, which is exactly
    # how the source campaign obtained it. The gate below is what proves that held.
    return ([PY, "-m", "execution.qrm.train_reactive", "--scratch", str(S),
             "--algo", "ppo", "--regime", r["regime"], "--seed", str(r["seed"]),
             "--log-learning", "--out", str(S / r["dst"])] + r["flags"])


def rundir(r):
    return S / r["dst"] / r["run"]


def final_curve(path: Path):
    c = json.loads(path.read_text())
    return c[-1]["mean_diff_bps"], c[-1]["steps"]


def main():
    pending = [r for r in specs() if not (rundir(r) / "model.zip").exists()]
    print(f"logged re-runs: {len(specs())} total, {len(pending)} to run "
          f"(CONC={CONC})", flush=True)
    env = dict(os.environ, PYTHONPATH="src")
    active, idx, t0 = [], 0, time.time()
    while idx < len(pending) or active:
        while len(active) < CONC and idx < len(pending):
            r = pending[idx]; idx += 1
            d = rundir(r); d.mkdir(parents=True, exist_ok=True)
            lf = open(d / "train.log", "w")
            p = subprocess.Popen(cmd(r), cwd=str(REPO), env=env, stdout=lf,
                                 stderr=subprocess.STDOUT)
            active.append((p, r, lf))
            print(f"[{time.strftime('%H:%M:%S')}] START {r['dst']}/{r['run']}", flush=True)
        still = []
        for p, r, lf in active:
            if p.poll() is None:
                still.append((p, r, lf)); continue
            lf.close()
            ok = (rundir(r) / "model.zip").exists()
            print(f"[{time.strftime('%H:%M:%S')}] {'DONE' if ok else 'FAILED'} "
                  f"{r['dst']}/{r['run']}", flush=True)
        active = still
        time.sleep(5)

    # ---------------------------------------------------------------- the gate
    print(f"\n=== REPRODUCTION GATE ({(time.time()-t0)/3600:.2f} h elapsed) ===", flush=True)
    report, n_pass, n_fail = [], 0, 0
    for r in specs():
        src_c = S / r["src"] / r["run"] / "curve.json"
        new_c = rundir(r) / "curve.json"
        if not new_c.exists():
            print(f"MISSING  {r['dst']}/{r['run']}", flush=True)
            n_fail += 1
            report.append({**{k: r[k] for k in ("src", "dst", "run")},
                           "status": "missing"})
            continue
        o, os_ = final_curve(src_c)
        n, ns = final_curve(new_c)
        exact = (o == n) and (os_ == ns)
        n_pass += exact
        n_fail += (not exact)
        print(f"{'PASS' if exact else 'MISMATCH'}  {r['dst']}/{r['run']:16s} "
              f"orig {o:+.10f} @{os_}  new {n:+.10f} @{ns}  delta {n-o:+.2e}", flush=True)
        report.append({**{k: r[k] for k in ("src", "dst", "run", "regime", "seed")},
                       "orig_final_bps": o, "new_final_bps": n,
                       "delta": n - o, "steps": ns, "exact": bool(exact)})
    (S / "logged_rerun_gate.json").write_text(json.dumps(report, indent=1))
    print(f"\nGATE: {n_pass} exact, {n_fail} not exact, of {len(specs())}", flush=True)
    print(f"wrote {S / 'logged_rerun_gate.json'}", flush=True)


if __name__ == "__main__":
    main()
