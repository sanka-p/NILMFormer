"""Where does the test MSE actually go, and does F1 care?

MSE charges for distance in watts; F1 charges for which side of the threshold a sample
falls on. These are different functions, so some of the MSE is spent on errors that
leave the classification unchanged. This script measures that share directly by
partitioning the test samples into TP/TN/FP/FN and attributing the squared error to
each cell.

  TP + TN  -> classification already correct; MSE spent here buys no F1 at all.
  FP + FN  -> classification wrong; this is the only MSE that F1 would reward fixing.

CAVEAT: the stored `test_metrics_yhat` is CLIPPED -- the trainer zeroes predictions
below the appliance threshold before saving (trainer.py:278). On a true-OFF sample a
raw 30 W prediction is stored as 0, contributing 0 instead of 900. So this measurement
UNDERSTATES the OFF/TN share and OVERSTATES the ON/TP share. Any conclusion that the
OFF side dominates therefore holds a fortiori.

Usage:
  PYTHONPATH=. ./.venv/bin/python -m scripts.decompose_mse --arm TCN_KL_scratch
"""
from __future__ import annotations

import argparse
import os

import numpy as np
import torch

from scripts.calibrate_threshold import DEFAULT_GRID, build_config
from src.helpers.preprocessing import REDD_DataBuilder
from scripts.run_one_expe import _cached_nilm_dataset

SEEDS = (0, 1, 2)


def truth_power_and_state(cfg):
    """(true power W, true state) for the test split, in trainer order."""
    builder = REDD_DataBuilder(
        data_path=f"{cfg.data_path}/REDD/redd.h5",
        mask_app=cfg.app,
        sampling_rate=cfg.sampling_rate,
        window_size=cfg.window_size,
        synth_aggregate_apps=cfg.synth_aggregate_apps,
    )
    data, _ = _cached_nilm_dataset(builder, cfg, list(cfg.ind_house_test), True)
    power = np.asarray(data[:, 1, 0, :]).flatten().astype(np.float64)
    state = np.asarray(data[:, 1, 1, :]).flatten().astype(np.int64)
    return power, state, float(builder.appliance_param[cfg.app]["min_threshold"])


def _find(root, app, ws, arm, seed):
    for key in ({app, "WasherDryer"} if app == "WashingMachine" else {app}):
        f = os.path.join(root, f"REDD_{key}_10s", str(ws), f"{arm}_{seed}.pt")
        if os.path.exists(f + ".done"):
            return f
    return None


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--root", default="result-redd-fix")
    p.add_argument("--arm", default="TCN_KL_scratch")
    p.add_argument("--apps", nargs="+", default=list(DEFAULT_GRID))
    a = p.parse_args()

    for app in a.apps:
        ws = DEFAULT_GRID[app]
        cfg = build_config("REDD", app, ws)
        y_pow, y_st, floor = truth_power_and_state(cfg)
        print(f"\n=== {app} / ws{ws} ===  {100*y_st.mean():.2f}% ON, floor {floor:.0f} W, "
              f"mean ON power {y_pow[y_st == 1].mean():.0f} W")
        for seed in SEEDS:
            f = _find(a.root, app, ws, a.arm, seed)
            if f is None:
                continue
            log = torch.load(f, map_location="cpu", weights_only=False)
            yh = np.asarray(log["test_metrics_yhat"]).ravel().astype(np.float64)
            if yh.size != y_pow.size:
                print(f"  seed{seed}: size mismatch {yh.size} vs {y_pow.size}, skipping")
                continue
            se = (yh - y_pow) ** 2
            pred_on = (yh > floor).astype(np.int64)
            cells = {"TP": (y_st == 1) & (pred_on == 1),
                     "FN": (y_st == 1) & (pred_on == 0),
                     "FP": (y_st == 0) & (pred_on == 1),
                     "TN": (y_st == 0) & (pred_on == 0)}
            tot = se.sum()
            print(f"  seed{seed}: test MSE {se.mean():.1f} W^2   "
                  f"(RMSE {np.sqrt(se.mean()):.1f} W)")
            for k, m in cells.items():
                n = int(m.sum())
                share = 100.0 * se[m].sum() / tot if tot > 0 else 0.0
                rmse = np.sqrt(se[m].mean()) if n else 0.0
                print(f"    {k}  n={n:>8} ({100.0*n/se.size:>5.2f}%)  "
                      f"MSE share {share:>5.1f}%   RMSE {rmse:>7.1f} W")
            right = 100.0 * (se[cells["TP"]].sum() + se[cells["TN"]].sum()) / tot
            print(f"    -> {right:.1f}% of the MSE sits on samples ALREADY classified "
                  f"correctly (F1 gains nothing from fixing it)")


if __name__ == "__main__":
    main()
