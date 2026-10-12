#################################################################################################################
#
# @description : Amplitude calibration at inference time, with no retraining.
#
# The UK-DALE models respond to absolute watts, not to activation shape: NILMscaler divides
# every input by one global constant, so a 1.4 kW kettle and a 3 kW kettle are different
# inputs rather than one shape at two scales. A locally collected kettle therefore never
# crosses the amplitude the model learned to respond to, and its raw output stays under
# 200 W no matter where the detection threshold is put.
#
# The workaround: multiply the aggregate by `g` before scaling, run inference, divide the
# predictions back by `g`. `g` is NOT tuned -- it is the ratio of UK-DALE's median ON-power
# for that appliance to the local one, so it is a stated per-deployment constant.
#
# Both arms use the SAME detection threshold, applied after dividing back, so the only
# difference between them is the gain. Predictions are taken with threshold_small_values=0
# so nothing is suppressed inside evaluate() and the threshold can be applied once, here.
#
# Usage:
#     PYTHONPATH=. .venv/bin/python -m scripts.run_deee_calibrated
#
#################################################################################################################

import argparse
import csv
import glob
import logging
import sys
from pathlib import Path

import numpy as np

from scripts.build_deee_aggregate import aggregate_path
from scripts.run_deee_inference import (
    APP_KEY, SCORED_APPLIANCES, infer, load_model, load_scaler_stats, pinned_scaler,
)
from src.helpers.deee_aggregate import build_regimes, load_aggregate, windows_from_streams
from src.helpers.metrics import NILMmetrics
from src.helpers.preprocessing import UKDALE_DataBuilder

OUT = Path("results/deee_calibration.csv")
FIELDS = ["Appliance", "WindowSize", "Seed", "Gain", "Arm", "Threshold",
          "ACCURACY", "BALANCED_ACCURACY", "F1_SCORE", "PRECISION", "RECALL", "MAE"]


def derive_gain(appliance, agg, regime="adapted"):
    """UK-DALE median ON-power / local median ON-power, both measured not assumed."""
    key = APP_KEY[appliance]
    files = sorted(glob.glob(f"results/osw_gt_cache/UKDALE_{appliance}_10s_w256_*.npz"))
    if not files:
        return None
    z = np.load(files[0], allow_pickle=True)
    up, us = z["power"].ravel(), z["status"].ravel().astype(bool)
    dp = agg.appliance_power[key]
    ds = agg.status[key][regime].astype(bool)
    if not us.any() or not ds.any():
        return None
    return float(np.median(up[us])) / float(np.median(dp[ds]))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--appliances", nargs="+", default=SCORED_APPLIANCES)
    p.add_argument("--window-sizes", type=int, nargs="+", default=[128, 256, 512])
    p.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    p.add_argument("--mode", default="session")
    p.add_argument("--variant", default="pure")
    p.add_argument("--scope", default="ukdale")
    p.add_argument("--regime", default="adapted")
    p.add_argument("--device", default="cuda")
    p.add_argument("--out", type=Path, default=OUT)
    a = p.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    stats = load_scaler_stats()
    builder = UKDALE_DataBuilder(
        data_path="data/UKDALE/", mask_app=["kettle"], sampling_rate="10s", window_size=128
    )
    regimes = build_regimes(builder.appliance_param)
    agg = load_aggregate(aggregate_path(a.mode, a.variant, 0, scope=a.scope))
    metrics = NILMmetrics()

    rows = []
    for appliance in a.appliances:
        key = APP_KEY[appliance]
        g = derive_gain(appliance, agg, a.regime)
        if g is None:
            logging.warning("no gain derivable for %s, skipping", appliance)
            continue
        threshold = float(regimes[a.regime][key]["min_threshold"])
        y_true = agg.appliance_power[key]
        y_state = agg.status[key][a.regime].astype(int)
        logging.info("%s: gain %.3f, threshold %.0f W", appliance, g, threshold)

        for ws in a.window_sizes:
            if (appliance, ws) not in stats:
                continue
            scaler = pinned_scaler(stats[(appliance, ws)]["PowerStat2"])
            pinned = float(stats[(appliance, ws)]["PowerStat2"])
            for seed in a.seeds:
                try:
                    model, _, _ = load_model(appliance, ws, seed)
                except FileNotFoundError:
                    continue
                for gain, arm in ((1.0, "as-is"), (g, "rescaled")):
                    data = windows_from_streams(agg.aggregate * gain, y_true, y_state, ws)
                    # threshold 0: take the raw output, so the same threshold can be
                    # applied to both arms after dividing back.
                    _, y_hat, _ = infer(model, data, scaler, 0.0, a.device, pinned=pinned)
                    pred = np.asarray(y_hat, dtype=np.float64) / gain
                    pred[pred < threshold] = 0.0
                    n = min(pred.size, y_true.size)
                    m = metrics(
                        y_true[:n].astype(np.float64), pred[:n], y_state[:n],
                        (pred[:n] > threshold).astype(int),
                    )
                    rows.append({
                        "Appliance": appliance, "WindowSize": ws, "Seed": seed,
                        "Gain": round(gain, 4), "Arm": arm, "Threshold": threshold,
                        **{k: m.get(k) for k in FIELDS[6:]},
                    })
                    logging.info(
                        "  %-15s ws=%3d seed=%d %-9s gain=%.2f  F1=%6.2f BA=%6.2f "
                        "recall=%6.2f prec=%6.2f",
                        appliance, ws, seed, arm, gain, 100 * m.get("F1_SCORE", float("nan")),
                        100 * m.get("BALANCED_ACCURACY", float("nan")),
                        100 * m.get("RECALL", float("nan")),
                        100 * m.get("PRECISION", float("nan")),
                    )

    a.out.parent.mkdir(parents=True, exist_ok=True)
    with open(a.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        w.writeheader()
        w.writerows(rows)
    logging.info("wrote %s (%d rows)", a.out, len(rows))
    return 0


if __name__ == "__main__":
    sys.exit(main())
