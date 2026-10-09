#################################################################################################################
#
# @description : Recover the UK-DALE NILMscaler statistics that the trained checkpoints imply.
#
# The `.pt` files under result-tcn-ablation/ are the trainer's log dict (trainer.py:375),
# not a checkpoint object: they carry the weights but NOT the scaler. `NILMscaler` is re-fit
# from scratch on every run (run_one_expe.py:340-344) and never persisted, so there is no
# record anywhere of the normalisation a trained model expects.
#
# Running inference on a new dataset therefore has to RECONSTRUCT that normalisation. It
# cannot be guessed: under the configured MaxScaling/SameAsPower the scaler learns
# `power_stat2 = max(aggregate)` over the all-houses array, and the aggregate is the
# synthetic one -- a sum of five channels each clipped at cutoff=6000
# (preprocessing.py:779) -- so it exceeds 6000 rather than saturating at it. It also varies
# with the run, because the window grid depends on which appliance is in `mask_app` (the
# _check_anynan drop at preprocessing.py:527) and on `window_size`. Hence up to
# 5 appliances x 3 window sizes = 15 distinct values.
#
# This script replays run_one_expe.py:244-252 and :340-344 verbatim and writes the values to
# results/ukdale_scaler_stats.csv, which scripts/run_deee_inference.py then pins. Letting the
# scaler re-fit on the new dataset instead would divide by its max (~1.6 kW rather than
# ~6 kW), feeding the model inputs several times too large -- silently, and with
# plausible-looking output.
#
# Usage:
#     PYTHONPATH=. .venv/bin/python -m scripts.pin_ukdale_scaler
#     PYTHONPATH=. .venv/bin/python -m scripts.pin_ukdale_scaler --only Microwave --window-sizes 256
#
#################################################################################################################

import argparse
import csv
import logging
import sys
from pathlib import Path

import numpy as np
from omegaconf import OmegaConf

from src.helpers.dataset import NILMscaler
from src.helpers.preprocessing import UKDALE_DataBuilder

DEFAULT_OUT = Path("results/ukdale_scaler_stats.csv")
ALL_HOUSES = [1, 2, 3, 4, 5]
DEFAULT_WINDOW_SIZES = [128, 256, 512]

#: repo appliance key -> the `app` string the builder wants, from configs/datasets.yaml.
#: Read from the config rather than retyped so a config change cannot desynchronise them.
DATASETS_CONFIG = Path("configs/datasets.yaml")
EXPES_CONFIG = Path("configs/expes.yaml")

FIELDS = [
    "Dataset", "Appliance", "App", "SamplingRate", "WindowSize",
    "PowerScalingType", "AppliancePowerScalingType",
    "PowerStat1", "PowerStat2", "ApplianceStat2", "NWindows", "AggMax", "Threshold",
]


def load_configs():
    """(appliance -> app string, synth_aggregate_apps, power/appliance scaling types)."""
    datasets = OmegaConf.load(DATASETS_CONFIG)
    expes = OmegaConf.load(EXPES_CONFIG)
    ukdale = datasets["UKDALE"]

    synth = list(ukdale["synth_aggregate_apps"])
    apps = {
        key: str(val["app"])
        for key, val in ukdale.items()
        if key != "synth_aggregate_apps"
    }
    return apps, synth, str(expes.power_scaling_type), str(expes.appliance_scaling_type)


def pin_one(app_key, app, synth, power_type, appliance_type, window_size,
            sampling_rate, data_path):
    """Replay the builder + scaler fit for one (appliance, window_size) cell."""
    builder = UKDALE_DataBuilder(
        data_path=f"{data_path}/UKDALE/",
        mask_app=app,
        sampling_rate=sampling_rate,
        window_size=window_size,
        synth_aggregate_apps=synth,
    )
    data, _ = builder.get_nilm_dataset(house_indicies=ALL_HOUSES)

    scaler = NILMscaler(
        power_scaling_type=power_type,
        appliance_scaling_type=appliance_type,
    )
    scaler.fit(data)

    agg_max = float(data[:, 0, 0, :].max())
    # V1: the aggregate must be the five-way synthetic sum, not the raw mains. A value of
    # exactly 6000.0 means the synth_aggregate column was missing and _get_stems fell back
    # to the single clipped `aggregate` channel (preprocessing.py:583-586) -- in which case
    # this number does not describe what the model was trained on.
    if abs(agg_max - 6000.0) < 1e-6:
        logging.warning(
            "%s/ws=%s: aggregate max is exactly 6000.0 -- the synthetic aggregate may not "
            "have been used. Verify 'synth_aggregate' is a column of the builder dataframe "
            "before trusting this value.",
            app_key, window_size,
        )

    return {
        "Dataset": "UKDALE",
        "Appliance": app_key,
        "App": app,
        "SamplingRate": sampling_rate,
        "WindowSize": builder.window_size,
        "PowerScalingType": power_type,
        "AppliancePowerScalingType": appliance_type,
        "PowerStat1": f"{float(scaler.power_stat1):.10g}",
        "PowerStat2": f"{float(scaler.power_stat2):.10g}",
        "ApplianceStat2": f"{float(scaler.appliance_stat2[0]):.10g}",
        "NWindows": int(data.shape[0]),
        "AggMax": f"{agg_max:.10g}",
        "Threshold": builder.appliance_param[app]["min_threshold"],
    }


def load_done(path):
    """Already-computed cells, so the script is resumable across the 15 builder passes."""
    if not path.is_file():
        return {}
    with open(path, newline="") as f:
        return {(r["Appliance"], r["WindowSize"]): r for r in csv.DictReader(f)}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out", type=Path, default=DEFAULT_OUT)
    p.add_argument("--data-path", default="data/")
    p.add_argument("--sampling-rate", default="10s")
    p.add_argument("--window-sizes", type=int, nargs="+", default=DEFAULT_WINDOW_SIZES)
    p.add_argument("--only", nargs="+", help="restrict to these appliance keys")
    p.add_argument("--force", action="store_true", help="recompute cells already in the CSV")
    a = p.parse_args()

    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
    )

    apps, synth, power_type, appliance_type = load_configs()
    if a.only:
        unknown = [k for k in a.only if k not in apps]
        if unknown:
            sys.exit(f"unknown appliance(s): {unknown}; known: {sorted(apps)}")
        apps = {k: apps[k] for k in a.only}

    logging.info("synth_aggregate_apps = %s", synth)
    logging.info("scaling = %s / %s", power_type, appliance_type)

    done = {} if a.force else load_done(a.out)
    rows = list(done.values())

    for app_key, app in sorted(apps.items()):
        for ws in a.window_sizes:
            if (app_key, str(ws)) in done:
                logging.info("skip %s/ws=%d (cached)", app_key, ws)
                continue
            logging.info("fitting %s (app=%s) / ws=%d ...", app_key, app, ws)
            row = pin_one(app_key, app, synth, power_type, appliance_type, ws,
                          a.sampling_rate, a.data_path)
            logging.info(
                "  -> power_stat2=%s  agg_max=%s  n_windows=%d",
                row["PowerStat2"], row["AggMax"], row["NWindows"],
            )
            rows.append(row)
            # Write after every cell: each pass is expensive and the box is shared.
            a.out.parent.mkdir(parents=True, exist_ok=True)
            with open(a.out, "w", newline="") as f:
                w = csv.DictWriter(f, fieldnames=FIELDS)
                w.writeheader()
                w.writerows(sorted(rows, key=lambda r: (r["Appliance"], int(r["WindowSize"]))))

    logging.info("wrote %d rows to %s", len(rows), a.out)

    stat2 = {r["Appliance"]: set() for r in rows}
    for r in rows:
        stat2[r["Appliance"]].add(r["PowerStat2"])
    for app_key, vals in sorted(stat2.items()):
        logging.info(
            "%s: power_stat2 %s across window sizes",
            app_key,
            "constant at " + next(iter(vals)) if len(vals) == 1 else f"VARIES {sorted(vals)}",
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
