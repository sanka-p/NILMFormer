#################################################################################################################
#
# @description : Cross-domain inference of UK-DALE-trained TCN_KL_aug_curated on DEEE_SmartHome.
#
# Loads the already-trained UK-DALE checkpoints and runs them, without any training or
# fine-tuning, over a synthetic aggregate built from the DEEE appliance traces. The question
# is whether a model trained on UK-DALE still disaggregates Sri Lankan 230 V appliances.
#
# Two things make this more than "call evaluate on new data":
#
#   1. THE SCALER IS NOT IN THE CHECKPOINT. The .pt files are the trainer's log dict
#      (trainer.py:375); NILMscaler is re-fit on every run and never persisted. It must be
#      pinned from results/ukdale_scaler_stats.csv (see scripts/pin_ukdale_scaler.py).
#      Re-fitting on DEEE would divide by ~3 kW instead of ~6.9 kW and feed the model inputs
#      roughly twice too large -- silently, with plausible-looking output. Hence the
#      assertions around the transform.
#
#   2. TWO THRESHOLD REGIMES. UK-DALE's kettle threshold (2000 W) exceeds the DEEE kettles'
#      peak (1370 W), and UK-DALE's washing_machine min_on_duration (1800 s) exceeds the
#      Singer's longest above-threshold run (730 s), so both appliances have identically
#      zero ground-truth activations as-trained. That is a real finding, reported as such,
#      and the `adapted` regime corrects those two priors so the waveform question can also
#      be asked. See src/helpers/deee_aggregate.ADAPTED_OVERRIDES.
#
# Fridge and Dishwasher are run as a SPECIFICITY check: neither appliance exists in DEEE, so
# the ground truth is identically zero, classification metrics are degenerate, and what is
# reported instead is a false-positive / phantom-energy rate (--specificity).
#
# Usage:
#     PYTHONPATH=. .venv/bin/python -m scripts.run_deee_inference --verify-ukdale   # V4 gate
#     PYTHONPATH=. .venv/bin/python -m scripts.run_deee_inference
#
#################################################################################################################

import argparse
import csv
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.baselines.nilm.tcn import TCN_NILM
from src.helpers.dataset import NILMDataset, NILMscaler
from src.helpers.deee_aggregate import (
    BASELOAD_VARIANTS,
    SCOPES,
    build_regimes,
    load_aggregate,
    windows_from_streams,
)
from src.helpers.metrics import NILMmetrics
from src.helpers.preprocessing import UKDALE_DataBuilder
from src.helpers.trainer import SeqToSeqTrainer
from scripts.build_deee_aggregate import aggregate_path

SCALER_STATS = Path("results/ukdale_scaler_stats.csv")
CKPT_ROOT = Path("result-tcn-ablation")
MODEL_KEY = "TCN_KL_aug_curated"
KL_ORDER = 10

#: The three DEEE appliances with a UK-DALE counterpart, and the two that have none.
SCORED_APPLIANCES = ["Kettle", "Microwave", "WashingMachine"]
SPECIFICITY_APPLIANCES = ["Fridge", "Dishwasher"]
#: repo appliance key -> the builder's `app` string.
APP_KEY = {
    "Kettle": "kettle",
    "Microwave": "microwave",
    "WashingMachine": "washing_machine",
    "Fridge": "fridge",
    "Dishwasher": "dishwasher",
}

#: The synthetic timeline's origin. Arbitrary but fixed, chosen as the first real DEEE
#: session date so hour-of-day is plausible (the TCN uses no time features, so it only
#: affects the exported timestamps).
SYNTH_T0 = pd.Timestamp("2026-08-25 00:00:00", tz="UTC")
SAMPLING_RATE_S = 10

METRIC_COLUMNS = [
    "MAE", "MSE", "RMSE", "TECA", "NDE", "SAE", "MR",
    "ACCURACY", "BALANCED_ACCURACY", "PRECISION", "RECALL", "F1_SCORE",
]


# --------------------------------------------------------------------------------- #
# Scaler pinning
# --------------------------------------------------------------------------------- #


def load_scaler_stats(path=SCALER_STATS):
    if not path.is_file():
        sys.exit(
            f"{path} not found -- run scripts/pin_ukdale_scaler.py first. Without it the "
            "UK-DALE normalisation cannot be reproduced and every result would be wrong."
        )
    with open(path, newline="") as f:
        return {(r["Appliance"], int(r["WindowSize"])): r for r in csv.DictReader(f)}


def pinned_scaler(power_stat2):
    """A NILMscaler carrying the UK-DALE statistics, constructed rather than fit.

    Satisfies transform (dataset.py:143) and inverse_transform_appliance
    (dataset.py:243-246) for the MaxScaling / SameAsPower configuration, which learns
    exactly power_stat1 = 0 and power_stat2 = max(aggregate).
    """
    sc = NILMscaler(power_scaling_type="MaxScaling", appliance_scaling_type="SameAsPower")
    sc.n_appliance = 1
    sc.power_stat1 = 0.0
    sc.power_stat2 = float(power_stat2)
    sc.appliance_stat1 = [0.0]
    sc.appliance_stat2 = [float(power_stat2)]
    sc.is_fitted = True
    return sc


# --------------------------------------------------------------------------------- #
# Model rebuild
# --------------------------------------------------------------------------------- #


def load_model(appliance, window_size, seed, ckpt_root=CKPT_ROOT, dataset="UKDALE"):
    """Rebuild TCN_NILM from a stored log dict. Returns (model, which_state_dict)."""
    path = Path(ckpt_root) / f"{dataset}_{appliance}_10s" / str(window_size) / \
        f"{MODEL_KEY}_{seed}.pt"
    if not path.is_file():
        raise FileNotFoundError(path)

    log = torch.load(path, map_location="cpu", weights_only=False)
    which = "best_model_state_dict" if "best_model_state_dict" in log else "model_state_dict"
    state = log[which]

    # The placeholder basis exists only to let the constructor run: tcn.py:43 raises
    # without one, and num_inputs is sized as basis.shape[0] + 1. load_state_dict then
    # overwrites the registered buffer with the trained basis.
    model = TCN_NILM(
        window_size=window_size, c_in=1, use_kl=True,
        kl_basis=np.eye(KL_ORDER, dtype=np.float32),
    )
    model.load_state_dict(state, strict=True)

    basis = model.kl_filter.basis
    assert basis.shape == (KL_ORDER, KL_ORDER), f"unexpected basis shape {tuple(basis.shape)}"
    # If the checkpoint had not carried a basis, the placeholder identity would survive and
    # the model would run with the wrong front end while looking perfectly healthy.
    assert not torch.allclose(basis, torch.eye(KL_ORDER)), (
        f"{path}: kl_filter.basis is still the identity placeholder -- the checkpoint did "
        "not carry a trained KL basis"
    )
    return model, which, log


# --------------------------------------------------------------------------------- #
# Inference
# --------------------------------------------------------------------------------- #


def infer(model, data, scaler, threshold, device, path_checkpoint=None, pinned=None):
    """Scale, run evaluate, and return (log, y_hat watts, diagnostics). Input is watts."""
    agg_max_w = float(data[:, 0, 0, :].max())

    # The direct anti-refit test is on the scaler's own parameters, not on a side-effect of
    # the data: assert it still carries the pinned UK-DALE statistics. (An earlier version
    # tested "scaled max < 1.0" instead, which conflates a re-fit with an input that simply
    # exceeds UK-DALE's range -- the remix aggregate legitimately does.)
    assert scaler.is_fitted and scaler.power_stat1 == 0.0, "scaler is not the pinned one"
    if pinned is not None:
        assert abs(scaler.power_stat2 - pinned) < 1e-6, (
            f"scaler.power_stat2 is {scaler.power_stat2} but the pinned UK-DALE value is "
            f"{pinned}; the scaler was re-fit somewhere"
        )

    data = scaler.transform(data.copy())
    scaled_max = float(data[:, 0, 0, :].max())
    frac_over = float((data[:, 0, 0, :] > 1.0).mean())
    if scaled_max > 1.0:
        # Not an error: it means this aggregate is louder than anything UK-DALE contained,
        # so the model is extrapolating. Worth recording next to the metrics it produced.
        logging.warning(
            "input exceeds the UK-DALE range: scaled max %.4f (raw %.0f W vs stat2 %.0f W), "
            "%.4f%% of samples above 1.0",
            scaled_max, agg_max_w, scaler.power_stat2, 100 * frac_over,
        )

    loader = DataLoader(NILMDataset(data), batch_size=1, shuffle=False)
    trainer = SeqToSeqTrainer(
        model,
        train_loader=loader,
        valid_loader=loader,  # evaluate divides by len(self.valid_loader)
        criterion=nn.MSELoss(),
        f_metrics=NILMmetrics(),
        device=device,
        all_gpu=False,  # all_gpu=True would iterate train_loader for a dummy forward
        verbose=False,
        plotloss=False,
        save_checkpoint=path_checkpoint is not None,
        path_checkpoint=path_checkpoint,
    )
    trainer.evaluate(
        loader, scaler=scaler, threshold_small_values=threshold,
        save_outputs=True, mask="test_metrics",
    )
    y_hat = np.asarray(trainer.log["test_metrics_yhat"], dtype=np.float32).ravel()
    return trainer.log, y_hat, {"ScaledMax": round(scaled_max, 5),
                                "FracInputOverRange": round(frac_over, 6)}


# --------------------------------------------------------------------------------- #
# V4: the pipeline-identity gate
# --------------------------------------------------------------------------------- #


def verify_ukdale(appliance, window_size, seed, device, tol=0.02):
    """Re-run a checkpoint on its own UK-DALE test set and compare to its stored metrics.

    If this reproduces, then the model rebuild, the pinned scaler and the threshold are all
    correct, and the DEEE numbers downstream can be trusted. No DEEE result should be
    reported before this passes.
    """
    stats = load_scaler_stats()
    key = (appliance, window_size)
    if key not in stats:
        sys.exit(f"no pinned scaler for {key}; run scripts/pin_ukdale_scaler.py --only {appliance}")

    app = APP_KEY[appliance]
    datasets = __import__("omegaconf").OmegaConf.load("configs/datasets.yaml")
    synth = list(datasets["UKDALE"]["synth_aggregate_apps"])
    test_houses = list(datasets["UKDALE"][appliance]["ind_house_test"])

    builder = UKDALE_DataBuilder(
        data_path="data/UKDALE/", mask_app=app, sampling_rate="10s",
        window_size=window_size, synth_aggregate_apps=synth,
    )
    data, _ = builder.get_nilm_dataset(house_indicies=test_houses)
    logging.info("V4: UK-DALE %s house(s) %s -> %d windows", appliance, test_houses, len(data))

    model, which, log = load_model(appliance, window_size, seed)
    scaler = pinned_scaler(stats[key]["PowerStat2"])
    threshold = builder.appliance_param[app]["min_threshold"]

    new_log, _, _ = infer(model, data, scaler, threshold, device,
                          pinned=float(stats[key]["PowerStat2"]))
    got, want = new_log["test_metrics_timestamp"], log["test_metrics_timestamp"]

    logging.info("V4: reproduced from %s", which)
    ok = True
    for k in ("MAE", "ACCURACY", "BALANCED_ACCURACY", "F1_SCORE"):
        if k not in want:
            continue
        a, b = float(got[k]), float(want[k])
        rel = abs(a - b) / max(abs(b), 1e-9)
        flag = "OK " if rel <= tol else "MISMATCH"
        ok &= rel <= tol
        logging.info("  %-18s stored=%10.4f  reproduced=%10.4f  rel=%.4f  %s", k, b, a, rel, flag)
    return ok


# --------------------------------------------------------------------------------- #
# Driver
# --------------------------------------------------------------------------------- #


def specificity_row(y_hat, aggregate, threshold, param, compute_status):
    """False-positive measures for an appliance that does not exist in this dataset."""
    fired = y_hat > threshold
    status = compute_status(
        fired.astype(int), param["min_on_duration"], param["min_off_duration"],
        param["min_activation_time"],
    )
    n_events = int(np.count_nonzero(np.diff(np.concatenate([[0], status])) == 1))
    phantom_wh = float(y_hat.sum()) * SAMPLING_RATE_S / 3600.0
    agg_wh = float(aggregate.sum()) * SAMPLING_RATE_S / 3600.0
    return {
        "FP_RATE": float(fired.mean()),
        "MEAN_PRED_W": float(y_hat.mean()),
        "MAX_PRED_W": float(y_hat.max()),
        "MAE": float(np.abs(y_hat).mean()),
        "PHANTOM_WH": phantom_wh,
        "PHANTOM_FRAC": phantom_wh / agg_wh if agg_wh else float("nan"),
        "N_FALSE_ACTIVATIONS": n_events,
    }


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--appliances", nargs="+", default=SCORED_APPLIANCES + SPECIFICITY_APPLIANCES)
    p.add_argument("--window-sizes", type=int, nargs="+", default=[128, 256, 512])
    p.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    p.add_argument("--modes", nargs="+", default=["session", "remix"])
    p.add_argument("--variants", nargs="+", default=list(BASELOAD_VARIANTS))
    p.add_argument("--scopes", nargs="+", default=list(SCOPES),
                   help="'all' = 12 categories in the aggregate; 'ukdale' = only the 3 "
                        "the models were trained to see")
    p.add_argument("--regimes", nargs="+", default=["astrained", "adapted"])
    p.add_argument("--agg-seed", type=int, default=0)
    p.add_argument("--device", default="cuda")
    p.add_argument("--result-path", type=Path, default=Path("result-deee"))
    p.add_argument("--metrics-csv", type=Path, default=Path("results/deee_metrics.csv"))
    p.add_argument("--specificity-csv", type=Path, default=Path("results/deee_specificity.csv"))
    p.add_argument("--pred-dir", type=Path, default=Path("results/deee_predictions"))
    p.add_argument("--dump-predictions", action="store_true",
                   help="write per-timestamp prediction npz files (one per run)")
    p.add_argument("--verify-ukdale", action="store_true",
                   help="V4 gate only: reproduce a checkpoint's stored UK-DALE metrics")
    a = p.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

    if not torch.cuda.is_available() and a.device == "cuda":
        logging.warning("CUDA unavailable, falling back to CPU")
        a.device = "cpu"

    if a.verify_ukdale:
        ok = True
        for appliance in a.appliances:
            for ws in a.window_sizes:
                for seed in a.seeds:
                    ok &= verify_ukdale(appliance, ws, seed, a.device)
        logging.info("V4 %s", "PASSED" if ok else "FAILED")
        return 0 if ok else 1

    stats = load_scaler_stats()
    builder = UKDALE_DataBuilder(
        data_path="data/UKDALE/", mask_app=["kettle"], sampling_rate="10s", window_size=128
    )
    regimes = build_regimes(builder.appliance_param)

    # Load every aggregate up front (each is a few MB) so the outer loop can be the model.
    # A checkpoint is ~hundreds of MB on NFS, so loading it once per (appliance, ws, seed)
    # instead of once per (mode, variant, regime) cuts 288 loads to 45.
    aggregates = {}
    for scope in a.scopes:
        for mode in a.modes:
            for variant in a.variants:
                path = aggregate_path(mode, variant, a.agg_seed, scope=scope)
                if not path.is_file():
                    sys.exit(f"{path} not found -- run scripts/build_deee_aggregate.py first")
                agg = load_aggregate(path)
                aggregates[(mode, variant, scope)] = agg
                logging.info(
                    "%s/%s/%s: L=%d, max=%.0f W, mean=%.0f W, baseload=%.1f W, reuse=%.1fx",
                    mode, variant, scope, agg.length, agg.aggregate.max(),
                    agg.aggregate.mean(), agg.baseload_w, agg.reuse_factor,
                )

    any_agg = next(iter(aggregates.values()))
    timestamps = pd.date_range(SYNTH_T0, periods=any_agg.length, freq=f"{SAMPLING_RATE_S}s")

    metric_rows, spec_rows = [], []

    for appliance in a.appliances:
        app = APP_KEY[appliance]
        is_scored = appliance in SCORED_APPLIANCES

        for ws in a.window_sizes:
            key = (appliance, ws)
            if key not in stats:
                logging.warning("no pinned scaler for %s, skipping", key)
                continue
            scaler = pinned_scaler(stats[key]["PowerStat2"])

            # The window tensors depend only on (aggregate, target, ws), not on the seed,
            # so build them once per window size and reuse across the three seeds.
            windows = {}
            for (mode, variant, scope), agg in aggregates.items():
                for regime in a.regimes:
                    if not is_scored and regime != a.regimes[0]:
                        continue
                    if is_scored:
                        y_true = agg.appliance_power[app]
                        y_state = agg.status[app][regime]
                    else:
                        y_true = np.zeros(agg.length, dtype=np.float32)
                        y_state = np.zeros(agg.length, dtype=np.int64)
                    windows[(mode, variant, scope, regime)] = (
                        windows_from_streams(agg.aggregate, y_true, y_state, ws),
                        y_true, y_state,
                    )

            for seed in a.seeds:
                try:
                    model, which, _ = load_model(appliance, ws, seed)
                except FileNotFoundError as e:
                    logging.warning("missing checkpoint %s, skipping", e)
                    continue

                for (mode, variant, scope, regime), (data, y_true, y_state) in windows.items():
                    agg = aggregates[(mode, variant, scope)]
                    param = regimes[regime][app] if is_scored else dict(builder.appliance_param[app])
                    threshold = param["min_threshold"]

                    ck = a.result_path / f"DEEE_{appliance}_10s" / str(ws)
                    ck.mkdir(parents=True, exist_ok=True)
                    stem = str(ck / f"{MODEL_KEY}_{mode}_{variant}_{scope}_{regime}_{seed}")

                    log, y_hat, diag = infer(
                        model, data, scaler, threshold, a.device, path_checkpoint=stem,
                        pinned=float(stats[key]["PowerStat2"]),
                    )
                    assert y_hat.size == agg.length, (
                        f"prediction length {y_hat.size} != timeline {agg.length}"
                    )

                    base = {
                        "Dataset": "DEEE", "Appliance": appliance, "Model": MODEL_KEY,
                        "SamplingRate": "10s", "WindowSize": ws, "Seed": seed,
                        "Mode": mode, "Variant": variant, "Scope": scope, "Regime": regime,
                        "Threshold": threshold,
                        "BaseloadW": agg.baseload_w,
                        "NoiseSigmaW": agg.noise_sigma_w,
                        "ReuseFactor": round(agg.reuse_factor, 2),
                        "StateDict": which,
                        **diag,
                    }

                    if is_scored:
                        m = log["test_metrics_timestamp"]
                        row = dict(base)
                        row.update({k: m.get(k) for k in METRIC_COLUMNS})
                        row["GT_ON_SAMPLES"] = int(y_state.sum())
                        row["DEGENERATE_GT"] = int(y_state.sum() == 0)
                        metric_rows.append(row)
                        logging.info(
                            "  %-15s ws=%3d seed=%d %-8s %-8s %-6s %-10s F1=%6.2f BA=%6.2f "
                            "MAE=%7.2f%s",
                            appliance, ws, seed, mode, variant, scope, regime,
                            100 * m.get("F1_SCORE", float("nan")),
                            100 * m.get("BALANCED_ACCURACY", float("nan")),
                            m.get("MAE", float("nan")),
                            "  [DEGENERATE GT]" if y_state.sum() == 0 else "",
                        )
                    else:
                        row = dict(base)
                        row.update(specificity_row(
                            y_hat, agg.aggregate, threshold, param, builder._compute_status
                        ))
                        spec_rows.append(row)
                        logging.info(
                            "  %-15s ws=%3d seed=%d %-8s %-8s %-6s SPECIFICITY "
                            "fp=%.5f phantom=%.4f events=%d",
                            appliance, ws, seed, mode, variant, scope,
                            row["FP_RATE"], row["PHANTOM_FRAC"], row["N_FALSE_ACTIVATIONS"],
                        )

                    if a.dump_predictions:
                        a.pred_dir.mkdir(parents=True, exist_ok=True)
                        out = a.pred_dir / (
                            f"{appliance}_{mode}_{variant}_{scope}_{regime}_ws{ws}_seed{seed}.npz"
                        )
                        np.savez_compressed(
                            out,
                            timestamp=timestamps.values.astype("datetime64[s]"),
                            aggregate_w=agg.aggregate,
                            y_true_w=y_true,
                            y_true_state=y_state,
                            y_hat_w=y_hat,
                            y_hat_state=(y_hat > threshold).astype(np.int8),
                            src_category_id=agg.src_category_id,
                            src_session_id=agg.src_session_id,
                            session_names=np.array(agg.session_names, dtype=object),
                            category_names=np.array(agg.category_names, dtype=object),
                        )

    if metric_rows:
        a.metrics_csv.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(metric_rows).to_csv(a.metrics_csv, index=False)
        logging.info("wrote %s (%d rows)", a.metrics_csv, len(metric_rows))
    if spec_rows:
        a.specificity_csv.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(spec_rows).to_csv(a.specificity_csv, index=False)
        logging.info("wrote %s (%d rows)", a.specificity_csv, len(spec_rows))
    return 0


if __name__ == "__main__":
    sys.exit(main())
