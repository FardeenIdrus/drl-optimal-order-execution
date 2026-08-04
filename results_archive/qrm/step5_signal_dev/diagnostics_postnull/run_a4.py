"""Amendment A4: the fair-chance observation test. 10 runs = base PPO x 2 regimes x 5 seeds,
injected env, EVERYTHING identical to Phase D/E except obs gains price-vs-arrival (obs_dim 29).
5-way parallel; resumable (skips runs whose model.zip exists)."""
import subprocess, os, time
from pathlib import Path

REPO = Path("/Users/fardeenidrus/Desktop/MSc Dissertation/code/drl-optimal-order-execution")
SCRATCH = Path("/Users/fardeenidrus/Desktop/MSc Dissertation/scratch_hyperliquid/oxford_l4")
PY = str(REPO / ".venv" / "bin" / "python")
OUT = SCRATCH / "runs_signal_obsfix"
CONC = 5


def specs():
    return [{"regime": r, "seed": s} for r in ("calm", "volatile") for s in range(5)]


def cmd(r):
    return [PY, "-m", "execution.qrm.train_reactive", "--scratch", str(SCRATCH),
            "--algo", "ppo", "--regime", r["regime"], "--seed", str(r["seed"]),
            "--inject", "--obs-price-vs-arrival", "--out", str(OUT)]


def rundir(r):
    return OUT / f"ppo_{r['regime']}_s{r['seed']}"


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    pending = [r for r in specs() if not (rundir(r) / "model.zip").exists()]
    print(f"A4: {len(specs())} total, {len(pending)} to run", flush=True)
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
            print(f"[{time.strftime('%H:%M:%S')}] START {d.name}", flush=True)
        still = []
        for p, r, lf in active:
            if p.poll() is None:
                still.append((p, r, lf)); continue
            lf.close()
            ok = (rundir(r) / "model.zip").exists()
            print(f"[{time.strftime('%H:%M:%S')}] {'DONE' if ok else 'FAILED'} "
                  f"{rundir(r).name}", flush=True)
        active = still
        time.sleep(5)
    n = sum((rundir(r) / "model.zip").exists() for r in specs())
    print(f"A4 COMPLETE: {n}/10 ({(time.time()-t0)/3600:.1f} h)", flush=True)


if __name__ == "__main__":
    main()
