"""Unit tests for the exogenous reference-move extension (Stage 3 Option A).

Run with:  PYTHONPATH=src pytest tests/test_qrm_exo_ref.py
"""
import numpy as np
import pandas as pd

from execution.qrm.assemble import assemble
from execution.qrm.exo_ref_sim import (
    MAX_ABS_MOVE,
    MoveProcess,
    measure_move_process,
    run_exo_qrm,
)


def test_measure_move_process_counts_tick_moves(tmp_path):
    p = tmp_path / "day.parquet"
    pd.DataFrame({"mid": [100.0, 101.0, 101.0, 99.0]}).to_parquet(p)  # deltas: +1, 0, -2
    mp = measure_move_process([str(p)], tick=1.0)
    z = MAX_ABS_MOVE
    assert mp.probs[z + 1] == 1 / 3   # +1
    assert mp.probs[z] == 1 / 3       # 0
    assert mp.probs[z - 2] == 1 / 3   # -2
    assert abs(mp.probs.sum() - 1.0) < 1e-12


def test_move_process_std_per_s():
    moves = np.array([-1, 0, 1])
    probs = np.array([0.5, 0.0, 0.5])   # E[m^2] = 1 per 0.5s -> var 2/s -> std sqrt(2)
    mp = MoveProcess(interval_s=0.5, moves=moves, probs=probs)
    assert abs(mp.std_per_s - np.sqrt(2.0)) < 1e-12


def _tiny_bundle():
    K, Q = 3, 6
    counts = np.zeros((K, Q + 1, 2, 3))
    time_in = np.ones((K, Q + 1, 2))
    counts[:, :, :, 0] = 4.0   # limit
    counts[:, :, :, 1] = 3.0   # cancel
    counts[:, :, :, 2] = 2.0   # market  (net drift negative -> lively queues)
    return assemble(counts, time_in, aes=[1.0, 1.0, 1.0], tick=1.0)


def test_run_exo_qrm_multitick_jumps_reform_book_without_crash():
    # +3-tick jump every 0.5s: must re-form the book (Model-III style), never crash,
    # and the price must track the jumps (~6 ticks/s)
    b = _tiny_bundle()
    jumps = MoveProcess(interval_s=0.5, moves=np.array([0, 3]), probs=np.array([0.0, 1.0]))
    out = run_exo_qrm(b, jumps, horizon_s=5.0, initial_price=100.0, seed=2)
    mids = out["mid_path_05s"]
    assert mids[-1] > mids[0] + 20.0          # ~10 jumps x 3 ticks, minus book noise
    assert out["vol_1s"] > 0.0


def test_run_exo_qrm_price_moves_and_tracks_process():
    b = _tiny_bundle()
    always_up = MoveProcess(
        interval_s=0.5,
        moves=np.array([0, 1]),
        probs=np.array([0.0, 1.0]),      # +1 tick every 0.5s, deterministically
    )
    out = run_exo_qrm(b, always_up, horizon_s=5.0, initial_price=100.0, seed=1)
    mids = out["mid_path_05s"]
    # ten boundaries of +1 tick: the price must END ~10 ticks higher and NOT be frozen
    assert mids[-1] > mids[0] + 8.0
    assert out["vol_1s"] > 0.0
    assert out["n_events"] > 0
    assert 1.0 <= out["spread_ticks_mean"] <= b.K * 2  # book stays sane
