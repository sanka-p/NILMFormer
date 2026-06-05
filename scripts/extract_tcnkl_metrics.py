import csv
import re
from collections import defaultdict
from pathlib import Path

import torch

RESULT_DIR = Path("result")
OUTPUT_CSV = Path("results/tcnkl_metrics.csv")
TARGET_MODEL = "TCN_KL"

COLUMNS = [
    "Model", "Dataset", "Appliance", "SamplingRate", "WindowSize",
    "MAE", "F1_SCORE", "SCA",
]

# {(Model, Dataset, Appliance, SamplingRate, WindowSize): [metrics_dict, ...]}
rows = defaultdict(list)

pt_files = sorted(RESULT_DIR.rglob("TCN_KL_*.pt"))
total = len(pt_files)

if total == 0:
    print(f"No TCN_KL .pt files found under {RESULT_DIR}")
    exit(0)

for i, pt_file in enumerate(pt_files, 1):
    print(f"[{i}/{total}] {pt_file}", flush=True)
    # result/{Dataset}_{Appliance}_{SamplingRate}/{WindowSize}/{Model}_{fold}.pt
    parts = pt_file.parts
    dataset_appliance_sr = parts[-3]
    window_size = parts[-2]
    stem = pt_file.stem  # e.g. TCN_KL_0

    match = re.match(r"^(.+)_(\d+)$", stem)
    if not match:
        continue
    model = match.group(1)
    if model != TARGET_MODEL:
        continue

    sr_match = re.match(r"^(.+)_(\d+\w+)$", dataset_appliance_sr)
    if not sr_match:
        continue
    dataset_appliance = sr_match.group(1)
    sampling_rate = sr_match.group(2)

    da_parts = dataset_appliance.split("_", 1)
    if len(da_parts) != 2:
        continue
    dataset, appliance = da_parts

    log = torch.load(pt_file, map_location="cpu", weights_only=False)
    m = log.get("test_metrics_timestamp", {})

    key = (model, dataset, appliance, sampling_rate, window_size)
    rows[key].append({
        "MAE": float(m.get("MAE", float("nan"))),
        "F1_SCORE": float(m.get("F1_SCORE", float("nan"))),
        "SCA": float(m.get("ACCURACY", float("nan"))),
    })

OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)

with open(OUTPUT_CSV, "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=COLUMNS)
    writer.writeheader()

    for (model, dataset, appliance, sr, ws), fold_metrics in sorted(rows.items()):
        n = len(fold_metrics)
        avg = {
            k: round(sum(fm[k] for fm in fold_metrics) / n, 3)
            for k in ("MAE", "F1_SCORE", "SCA")
        }
        writer.writerow({
            "Model": model,
            "Dataset": dataset,
            "Appliance": appliance,
            "SamplingRate": sr,
            "WindowSize": ws,
            **avg,
        })

print(f"Saved {len(rows)} rows to {OUTPUT_CSV}")
