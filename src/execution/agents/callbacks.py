"""Training-curve callback: the evidence that reward scaling produced *stable*
learning, not just that a scale was chosen.

Every ``eval_freq`` steps the current (deterministic) policy is evaluated on a
fixed validation subset through the SAME loop/fill engine as the benchmarks, and
the paired mean vs TWAP is logged. A healthy run shows ``val_vs_twap_mean``
falling below zero (agent beats TWAP) then flattening -- the rising-then-flat
convergence curve. The series is written to CSV incrementally so a long background
run can be watched live, and returned in-memory for plotting/asserting.
"""
from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
from stable_baselines3.common.callbacks import BaseCallback

from execution.agents.policy import SB3Policy
from execution.eval.runner import run_episodes


class ValidationCurveCallback(BaseCallback):
    """Periodically evaluate on a fixed validation subset; log the paired curve."""

    def __init__(self, val_env, val_indices, twap_is, *, eval_freq: int,
                 csv_path: str | Path | None = None, verbose: int = 0) -> None:
        super().__init__(verbose)
        self.val_env = val_env
        self.val_indices = list(val_indices)
        self.twap_is = np.asarray(twap_is, dtype=float)   # aligned to val_indices order
        self.eval_freq = int(eval_freq)
        self.csv_path = Path(csv_path) if csv_path else None
        self.rows: list[dict] = []
        self._last_eval = -1

    def _on_training_start(self) -> None:
        if self.csv_path and self.csv_path.exists():
            self.csv_path.unlink()
        self._evaluate()   # baseline point (near-random policy)

    def _on_step(self) -> bool:
        if self.num_timesteps - self._last_eval >= self.eval_freq:
            self._evaluate()
        return True

    def _on_training_end(self) -> None:
        if self.num_timesteps != self._last_eval:
            self._evaluate()   # final point

    def _evaluate(self) -> None:
        self._last_eval = self.num_timesteps
        df = run_episodes(self.val_env, SB3Policy(self.model), self.val_indices)
        diff = df["is_bps"].to_numpy() - self.twap_is        # paired, agent - TWAP
        buf = self.model.ep_info_buffer
        ep_r = float(np.mean([e["r"] for e in buf])) if buf else float("nan")
        row = {
            "timesteps": int(self.num_timesteps),
            "train_ep_reward": ep_r,
            "val_is_mean": float(df["is_bps"].mean()),
            "val_vs_twap_mean": float(np.mean(diff)),       # < 0 == beating TWAP
            "val_full_exec_frac": float((df["inventory_left"].abs() < 1e-6).mean()),
            "val_residual_freq": float((df["residual_btc"] > 1e-9).mean()),
        }
        self.rows.append(row)
        if self.csv_path is not None:
            write_header = not self.csv_path.exists()
            self.csv_path.parent.mkdir(parents=True, exist_ok=True)
            with open(self.csv_path, "a", newline="") as fh:
                writer = csv.DictWriter(fh, fieldnames=list(row))
                if write_header:
                    writer.writeheader()
                writer.writerow(row)
        if self.verbose:
            print(f"[curve] t={row['timesteps']:>9}  val_vs_twap={row['val_vs_twap_mean']:+.3f} bps"
                  f"  val_is={row['val_is_mean']:+.3f}  resid={row['val_residual_freq']*100:.2f}%")
