"""Score trained models with the OSW appliance-combination metrics of Welikala et al. (2019).

Computes Aci, Afm and Apd/Apa per run and writes them to CSV, with no model retraining: the
per-appliance predictions are already stored in the `.pt` logs written by
`scripts/run_one_expe.py` (`log["test_metrics_yhat"]`, flat float32, in watts, already
inverse-transformed and already clipped at the appliance min_threshold).

The problem this script exists to solve
---------------------------------------
The three metrics score an appliance COMBINATION, but this repo trains one model per
appliance, so no single run can produce them. A synthetic multi-appliance predictor is
assembled from the per-appliance `.pt` files of the same (model, dataset, window size, seed),
and that is what gets scored.

Assembling it is not a matter of stacking arrays. Only the target appliance is merged
`how="inner"` in the builders, and its NaN-containing windows are dropped, so every appliance
ends up with a different window count AND a different time origin (UK-DALE/10s/ws=256: Kettle
7914 windows, Fridge 4802, Dishwasher 4803, Microwave 4797, WashingMachine 4796). Window
indices and even window boundaries therefore do not line up. The join must be at the
per-sample TIMESTAMP level, which is what this script does -- per house, so that no OSW ever
straddles a house boundary or a time gap.

Ground truth is not stored in the `.pt` logs, so it is rebuilt from `data/` with the same
builders the training run used and cached as `.npz`. The test split is seed-independent for
UK-DALE and REDD (fixed test houses, `shuffle=False`), so one rebuild per
(dataset, appliance, sampling rate, window size) serves every model and every seed.

Usage:
    uv run -m scripts.score_osw
    uv run -m scripts.score_osw --window-sizes 256 --seeds 0 --models NILMFormer -v
    uv run -m scripts.score_osw --build-gt-only          # warm the ground-truth cache
"""

import argparse
import csv
import hashlib
import json
import re
import subprocess
import sys
import warnings
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml

from src.helpers.osw import (
    DEFAULT_ON_THRESHOLDS_W,
    DEFAULT_OSW_LENGTH,
    build_osw,
    concat_osw,
)
from src.helpers.osw_metrics import per_combination, score, summary_lines

# Same default pair as scripts/make_table.py: baselines in one dir, the pretrained TCN_KL
# ("Proposed") in the other. The two agree on window counts, which is what makes them
# comparable -- 2207-results/result does not.
DEFAULT_RESULT_DIRS = [
    "result-synagg",
    "2207-results/result-synagg-overfit",
    "result-tcn-ablation",
]
DEFAULT_MODELS = [
    "TCN_KL", "TCN_KL_perapp", "TCN_KL_perapp_nilmformerlike", "TCN_KL_aug_curated", "TCN_KL_aug", "TCN_KL_scratch", "TCN",
    "NILMFormer", "BERT4NILM", "BiLSTM", "BiGRU", "CNN1D", "DAResNet",
]
DEFAULT_OUT = Path("results/osw_runs.csv")
DEFAULT_COMBOS_OUT = Path("results/osw_combinations.csv")
DEFAULT_GT_CACHE = Path("results/osw_gt_cache")
DEFAULT_PRED_CACHE = Path("results/osw_pred_cache")

# Fixed order => fixed bitmask bit order, so combination labels are comparable across runs.
APPLIANCE_ORDER = [
    "Dishwasher", "Fridge", "Kettle", "Microwave", "WashingMachine", "WasherDryer",
]

#: Bumped whenever a change here or in the builders would invalidate cached ground truth.
GT_CACHE_VERSION = 1

RUN_KEY_FIELDS = ("Model", "Dataset", "SamplingRate", "WindowSize", "Seed")

RUN_COLUMNS = [
    "SourceDirs", "Model", "Dataset", "SamplingRate", "WindowSize", "Seed",
    "Appliances", "NumAppliances", "SynthAggKey",
    "OswSize", "SamplingIntervalS", "OswDurationS", "OnSource", "NanPolicy", "PartialPolicy",
    "NumHouses", "NumWindowsMin", "NumSamplesJoined", "NumContiguousRuns",
    "NumOsw", "NumOswActive", "NumOswAllOff",
    "NumOswDroppedPartialSamples", "NumOswDroppedNan",
    "NumCombGT", "NumCombPred", "NumCombUnion",
    "AppCoverage", "AggMeanW",
    "ACI", "ACI_ACT", "AFM", "AFM_ACT", "APD", "APA_POOLED",
]

COMBO_COLUMNS = [
    "Model", "Dataset", "SamplingRate", "WindowSize", "Seed",
    "CombinationMask", "Combination", "NumAppliancesOn",
    "SupportGT", "SupportPred", "TP", "FP", "FN", "FM", "APA_CJ",
]


def warn(msg):
    print(f"WARN {msg}", file=sys.stderr)


class SkipRun(Exception):
    """A run cannot be scored; reported and skipped unless --strict."""


# --------------------------------------------------------------------------------------
# Stage 0: discover result files
# --------------------------------------------------------------------------------------


def parse_pt_path(pt_file):
    """Pull (model, dataset, appliance, sampling_rate, window_size, seed) out of a path.

    Copied from scripts/make_table.py rather than imported (that module has no public
    surface). The stem regex is what makes `TCN_KL_0` parse: the model name itself contains
    an underscore, so the seed must be split off greedily from the right.
    """
    parts = pt_file.parts
    if len(parts) < 3:
        return None
    dataset_appliance_sr = parts[-3]  # e.g. UKDALE_Dishwasher_10s
    window_size = parts[-2]  # e.g. 128

    stem_match = re.match(r"^(.+)_(\d+)$", pt_file.stem)
    if not stem_match:
        return None
    model, seed = stem_match.group(1), stem_match.group(2)

    sr_match = re.match(r"^(.+)_(\d+\w+)$", dataset_appliance_sr)
    if not sr_match:
        return None
    dataset_appliance, sampling_rate = sr_match.group(1), sr_match.group(2)

    da_parts = dataset_appliance.split("_", 1)
    if len(da_parts) != 2:
        return None
    dataset, appliance = da_parts

    return {
        "Model": model,
        "Dataset": dataset,
        "Appliance": appliance,
        "SamplingRate": sampling_rate,
        "WindowSize": window_size,
        "Seed": seed,
    }


def discover(result_dirs, filters):
    """Map run key -> {appliance: (source_dir, path)}, earlier --result-dir winning."""
    priority = {d: i for i, d in enumerate(result_dirs)}
    found = defaultdict(dict)

    for result_dir in result_dirs:
        root = Path(result_dir)
        if not root.is_dir():
            warn(f"result dir not found, skipping: {result_dir}")
            continue
        for pt_file in sorted(root.rglob("*.pt")):
            meta = parse_pt_path(pt_file)
            if meta is None:
                warn(f"unparseable path, skipping: {pt_file}")
                continue
            if filters["models"] and meta["Model"] not in filters["models"]:
                continue
            if filters["datasets"] and meta["Dataset"] not in filters["datasets"]:
                continue
            if filters["sampling_rate"] and meta["SamplingRate"] != filters["sampling_rate"]:
                continue
            if filters["window_sizes"] and meta["WindowSize"] not in filters["window_sizes"]:
                continue
            if filters["seeds"] and meta["Seed"] not in filters["seeds"]:
                continue
            if filters["appliances"] and meta["Appliance"] not in filters["appliances"]:
                continue

            run_key = tuple(meta[f] for f in RUN_KEY_FIELDS)
            appliance = meta["Appliance"]
            current = found[run_key].get(appliance)
            if current is None or priority[result_dir] < priority[current[0]]:
                found[run_key][appliance] = (result_dir, pt_file)
    return found


# --------------------------------------------------------------------------------------
# Stage 1: resolve the config exactly as the training run did
# --------------------------------------------------------------------------------------


def load_configs():
    with open("configs/expes.yaml") as f:
        expes = yaml.safe_load(f)
    with open("configs/datasets.yaml") as f:
        datasets = yaml.safe_load(f)
    with open("configs/models.yaml") as f:
        models = yaml.safe_load(f)
    return expes, datasets, models


def resolve_config(configs, dataset, model):
    """Replicate scripts/run_one_expe.py's config precedence.

    That script applies the model config (run_one_expe.py:215) and then overwrites it with
    the dataset-level non-appliance keys (run_one_expe.py:226). `synth_aggregate_apps` is
    defined at the dataset level in configs/datasets.yaml, so the dataset list WINS over the
    per-model list -- which means TCN_KL's `washer_dryer` entry in configs/models.yaml never
    took effect and every model on a given dataset trained against the same synthetic
    aggregate. That is why the two default result dirs agree on window counts, and why the
    ground-truth cache needs only one variant per dataset in practice.
    """
    expes, datasets, models = configs
    if dataset not in datasets:
        raise SkipRun(f"dataset {dataset} not in configs/datasets.yaml")
    if model not in models:
        raise SkipRun(f"model {model} not in configs/models.yaml")

    cfg = dict(expes)
    cfg.update(models[model])
    dataset_cfg = datasets[dataset]
    cfg.update({k: v for k, v in dataset_cfg.items() if not isinstance(v, dict)})
    return cfg, dataset_cfg


def appliance_keys_for(dataset_cfg, appliances_present):
    """Order appliances into bit order, dropping duplicate builder channels.

    REDD maps both `WashingMachine` and `WasherDryer` onto the builder app `WasherDryer`; if
    both were kept the combination space would contain the same physical appliance twice.
    """
    keys = [a for a in APPLIANCE_ORDER if a in appliances_present]
    keys += sorted(a for a in appliances_present if a not in APPLIANCE_ORDER)

    seen, kept, dropped = {}, [], []
    for key in keys:
        app = dataset_cfg.get(key, {}).get("app", key)
        if app in seen:
            dropped.append((key, seen[app]))
            continue
        seen[app] = key
        kept.append(key)
    return kept, dropped


def synth_key(synth_apps):
    """Short stable tag for the synthetic-aggregate variant, in config order."""
    joined = "|".join(synth_apps or [])
    return "sa" + hashlib.sha1(joined.encode()).hexdigest()[:8]


def git_sha():
    try:
        return subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True, text=True, check=True,
        ).stdout.strip()
    except Exception:  # pragma: no cover - forensic metadata only
        return "unknown"


# --------------------------------------------------------------------------------------
# Stage 2: ground-truth cache (the biggest performance lever)
# --------------------------------------------------------------------------------------


def gt_cache_path(cache_dir, dataset, appliance, sampling_rate, window_size, sa_key):
    return (
        Path(cache_dir)
        / f"{dataset}_{appliance}_{sampling_rate}_w{window_size}_{sa_key}.npz"
    )


def build_ground_truth(spec):
    """Rebuild one appliance's test set and write it to the cache. Runs in a worker process.

    `spec` is a plain dict so it pickles cleanly. Imports happen inside the function to keep
    the parent process from paying for torch in workers that only need the builders.
    """
    from src.helpers.preprocessing import (
        REDD_DataBuilder,
        REFIT_DataBuilder,
        UKDALE_DataBuilder,
    )

    dataset = spec["dataset"]
    data_path = spec["data_path"].rstrip("/")
    kwargs = dict(
        mask_app=spec["app"],
        sampling_rate=spec["sampling_rate"],
        window_size=int(spec["window_size"]),
        synth_aggregate_apps=spec["synth_aggregate_apps"],
    )
    if dataset == "UKDALE":
        builder = UKDALE_DataBuilder(data_path=f"{data_path}/UKDALE/", **kwargs)
    elif dataset == "REDD":
        builder = REDD_DataBuilder(data_path=f"{data_path}/REDD/redd.h5", **kwargs)
    elif dataset == "REFIT":
        builder = REFIT_DataBuilder(data_path=f"{data_path}/REFIT/RAW_DATA_CLEAN/", **kwargs)
    else:
        raise ValueError(f"unsupported dataset {dataset}")

    data, st_date = builder.get_nilm_dataset(house_indicies=list(spec["ind_house_test"]))
    if data.size == 0:
        raise RuntimeError(
            f"{dataset}/{spec['appliance']}/ws={spec['window_size']}: the rebuilt test set "
            f"is empty for houses {list(spec['ind_house_test'])}"
        )

    starts = pd.DatetimeIndex(st_date["start_date"]).to_numpy(dtype="datetime64[ns]")
    meta = {
        "dataset": dataset,
        "appliance": spec["appliance"],
        "app": spec["app"],
        "sampling_rate": spec["sampling_rate"],
        "window_size": int(spec["window_size"]),
        "synth_aggregate_apps": list(spec["synth_aggregate_apps"] or []),
        "threshold_min_w": float(
            builder.appliance_param[spec["app"]]["min_threshold"]
        ),
        "ind_house_test": list(spec["ind_house_test"]),
        "n_windows": int(data.shape[0]),
        "builder_version": GT_CACHE_VERSION,
        "git_sha": spec["git_sha"],
    }

    out = Path(spec["cache_path"])
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out,
        start_date=starts.astype(np.int64),
        house_id=np.asarray(st_date.index, dtype=np.int64),
        aggregate=data[:, 0, 0, :].astype(np.float32),
        power=data[:, 1, 0, :].astype(np.float32),
        status=data[:, 1, 1, :].astype(np.int8),
        meta=np.array(json.dumps(meta)),
    )
    return spec["cache_path"], meta["n_windows"]


def load_ground_truth(spec, refresh=False):
    """Load cached ground truth, rebuilding when absent or stale."""
    path = Path(spec["cache_path"])
    if path.exists() and not refresh:
        with np.load(path, allow_pickle=False) as z:
            meta = json.loads(str(z["meta"]))
            stale = (
                meta.get("builder_version") != GT_CACHE_VERSION
                or meta.get("window_size") != int(spec["window_size"])
                or meta.get("synth_aggregate_apps") != list(spec["synth_aggregate_apps"] or [])
                or meta.get("ind_house_test") != list(spec["ind_house_test"])
            )
            if not stale:
                return {
                    "start_date": z["start_date"],
                    "house_id": z["house_id"],
                    "aggregate": z["aggregate"],
                    "power": z["power"],
                    "status": z["status"],
                    "meta": meta,
                }
        warn(f"{path}: stale ground-truth cache; rebuilding")
    build_ground_truth(spec)
    return load_ground_truth(spec, refresh=False)


# --------------------------------------------------------------------------------------
# Stage 2b: prediction cache
# --------------------------------------------------------------------------------------


def _slug(text):
    return re.sub(r"[^A-Za-z0-9._-]+", "_", str(text)).strip("_")


def load_predictions(pt_path, source_dir, window_size, cache_dir, meta):
    """Return predictions as (n_windows, window_size) float32, watts.

    Cached as `.npy` next to a small json sidecar: the `.pt` files carry the model weights
    too (20-240 MB each), so re-reading them on every metric tweak dominates the runtime.
    """
    pt_path = Path(pt_path)
    stat = pt_path.stat()
    rel = (
        Path(_slug(source_dir))
        / f"{meta['Dataset']}_{meta['Appliance']}_{meta['SamplingRate']}"
        / str(window_size)
        / f"{meta['Model']}_{meta['Seed']}"
    )
    npy = Path(cache_dir) / rel.with_suffix(".npy")
    side = Path(cache_dir) / rel.with_suffix(".json")

    if npy.exists() and side.exists():
        try:
            info = json.loads(side.read_text())
            if info.get("st_mtime_ns") == stat.st_mtime_ns and info.get("st_size") == stat.st_size:
                return np.load(npy)
        except (OSError, ValueError, json.JSONDecodeError):
            pass  # fall through and re-read the .pt

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        # weights_only=False: the log mixes numpy arrays and metric dicts.
        log = torch.load(pt_path, map_location="cpu", weights_only=False)

    if "test_metrics_yhat" not in log:
        raise SkipRun(
            f"{pt_path}: no 'test_metrics_yhat' in the log (keys present: "
            f"{sorted(log)[:12]}); this run was saved without save_outputs=True"
        )
    flat = np.asarray(log["test_metrics_yhat"], dtype=np.float32).ravel()
    if flat.size % window_size:
        raise SkipRun(
            f"{pt_path}: test_metrics_yhat has {flat.size} values, not divisible by "
            f"window_size={window_size}"
        )
    yhat = flat.reshape(-1, window_size)

    npy.parent.mkdir(parents=True, exist_ok=True)
    np.save(npy, yhat)
    side.write_text(
        json.dumps(
            {
                "st_mtime_ns": stat.st_mtime_ns,
                "st_size": stat.st_size,
                "source": pt_path.as_posix(),
            }
        )
    )
    return yhat


# --------------------------------------------------------------------------------------
# Stage 3-4: per-sample frames and the cross-appliance timestamp join
# --------------------------------------------------------------------------------------


def per_sample_frames(appliance, gt, yhat, sampling_rate, context):
    """Expand windows to per-sample series, one frame per house.

    Windows are non-overlapping (`window_stride == window_size`) and each house's resampled
    index is a contiguous grid, so per-sample timestamps are the window start plus k*dt.
    """
    n_true = int(gt["power"].shape[0])
    if yhat.shape[0] != n_true:
        raise SkipRun(
            f"window count mismatch for {context['label']}: {context['pt_path']} holds "
            f"{yhat.shape[0]} windows ({yhat.shape[0] * yhat.shape[1]} samples / "
            f"{yhat.shape[1]}) but the rebuilt test set has {n_true} windows, so predictions "
            f"cannot be aligned to ground truth. Check that "
            f"synth_aggregate_apps={context['synth']} matches training (configs/models.yaml "
            f"is overridden by the dataset-level list in configs/datasets.yaml -- see "
            f"scripts/run_one_expe.py:215 then :226), that data/{context['dataset']} is "
            f"unchanged, and that ind_house_test={context['houses']}. GT cache: "
            f"{context['cache_path']} (delete it or pass --refresh-gt to rebuild)."
        )

    window_size = yhat.shape[1]
    step = pd.Timedelta(sampling_rate).value  # nanoseconds
    offsets = np.arange(window_size, dtype=np.int64) * step
    stamps = (gt["start_date"][:, None] + offsets[None, :]).ravel()

    frame = pd.DataFrame(
        {
            f"{appliance}_true": gt["power"].ravel().astype(np.float64),
            f"{appliance}_pred": yhat.ravel().astype(np.float64),
            f"{appliance}_status": gt["status"].ravel().astype(np.float64),
            f"{appliance}_agg": gt["aggregate"].ravel().astype(np.float64),
        },
        index=pd.DatetimeIndex(stamps.astype("datetime64[ns]")),
    )
    houses = np.repeat(gt["house_id"], window_size)

    out = {}
    for house in np.unique(houses):
        part = frame[houses == house]
        dup = int(part.index.duplicated().sum())
        if dup:
            raise SkipRun(
                f"{context['dataset']}/{appliance}/ws={window_size}: rebuilt test index has "
                f"{dup} duplicate timestamp(s) in house {house}; windows are assumed "
                f"non-overlapping (window_stride == window_size, "
                f"src/helpers/preprocessing.py)"
            )
        out[int(house)] = part.sort_index()
    return out


def join_appliances(per_appliance_frames, appliances):
    """Inner-join the appliances onto a common per-house timeline.

    Deterministic: fixed appliance order, inner join (commutative on the index), stable sort.
    """
    houses = set.intersection(*(set(f) for f in per_appliance_frames.values()))
    joined = {}
    for house in sorted(houses):
        frame = per_appliance_frames[appliances[0]][house]
        for appliance in appliances[1:]:
            frame = frame.join(per_appliance_frames[appliance][house], how="inner")
        joined[house] = frame.sort_index(kind="mergesort")
    return joined


# --------------------------------------------------------------------------------------
# Stage 5-6: score one run
# --------------------------------------------------------------------------------------


def score_run(run_key, appliance_files, args, configs, cache_meta, required):
    model, dataset, sampling_rate, window_size_s, seed = run_key
    window_size = int(window_size_s)
    cfg, dataset_cfg = resolve_config(configs, dataset, model)
    if dataset == "REFIT":
        raise SkipRun(
            "REFIT is not supported: its test split comes from "
            "split_train_test_pdl_nilmdataset with a seed-dependent house partition "
            "(scripts/run_one_expe.py:76-84), so the ground truth is not reconstructable "
            "from the result path alone"
        )

    appliances, dropped = appliance_keys_for(dataset_cfg, set(appliance_files))
    for key, kept in dropped:
        warn(
            f"{dataset}: appliance keys {key!r} and {kept!r} both map to the same builder "
            f"channel; keeping {kept!r} so the combination space has no duplicated appliance"
        )
    # Every scored run must cover the SAME appliance set, or the combination spaces differ
    # and the scores are not comparable across runs. Scoring a partial run with a smaller
    # appliance set silently would be worse than skipping it.
    missing = [a for a in required if a not in appliances]
    if missing:
        raise SkipRun(
            f"missing appliance(s) {missing}; a combination score needs every appliance, and "
            f"scoring the {len(appliances)} available one(s) {appliances} would not be "
            f"comparable with the full {len(required)}-appliance runs"
        )
    appliances = [a for a in appliances if a in required]
    if len(appliances) < 2:
        raise SkipRun(
            f"only {len(appliances)} appliance(s) available ({appliances}); a combination "
            "metric needs at least 2"
        )

    synth = list(cfg.get("synth_aggregate_apps") or [])
    sa_key = synth_key(synth)
    thresholds = {}
    for appliance in appliances:
        if args.threshold_source == "simple":
            if appliance not in DEFAULT_ON_THRESHOLDS_W:
                raise SkipRun(
                    f"no default ON threshold for {appliance!r}; add it to "
                    "src/helpers/osw.DEFAULT_ON_THRESHOLDS_W"
                )
            thresholds[appliance] = DEFAULT_ON_THRESHOLDS_W[appliance]

    # ---- ground truth + predictions per appliance ----
    frames, coverage, gt_metas = {}, {}, {}
    for appliance in appliances:
        source_dir, pt_path = appliance_files[appliance]
        app_cfg = dataset_cfg.get(appliance)
        if app_cfg is None:
            raise SkipRun(f"appliance {appliance} not in configs/datasets.yaml[{dataset}]")
        spec = {
            "dataset": dataset,
            "appliance": appliance,
            "app": app_cfg["app"],
            "sampling_rate": sampling_rate,
            "window_size": window_size,
            "synth_aggregate_apps": synth,
            "ind_house_test": list(app_cfg["ind_house_test"]),
            "data_path": args.data_path,
            "git_sha": cache_meta["git_sha"],
            "cache_path": gt_cache_path(
                args.gt_cache_dir, dataset, appliance, sampling_rate, window_size, sa_key
            ).as_posix(),
        }
        gt = load_ground_truth(spec, refresh=args.refresh_gt)
        gt_metas[appliance] = gt["meta"]
        if args.threshold_source == "kelly":
            thresholds[appliance] = gt["meta"]["threshold_min_w"]

        yhat = load_predictions(
            pt_path,
            source_dir,
            window_size,
            args.pred_cache_dir,
            {
                "Dataset": dataset, "Appliance": appliance, "SamplingRate": sampling_rate,
                "Model": model, "Seed": seed,
            },
        )
        context = {
            "label": f"{model}/{dataset}/{appliance}/{sampling_rate}/ws={window_size}/seed={seed}",
            "pt_path": pt_path,
            "synth": synth,
            "dataset": dataset,
            "houses": spec["ind_house_test"],
            "cache_path": spec["cache_path"],
        }
        frames[appliance] = per_sample_frames(
            appliance, gt, yhat, sampling_rate, context
        )
        coverage[appliance] = sum(len(f) for f in frames[appliance].values())

    # ---- cross-appliance timestamp join, per house ----
    joined = join_appliances(frames, appliances)
    n_joined = sum(len(f) for f in joined.values())
    if n_joined == 0:
        raise SkipRun(
            f"cross-appliance timestamp join is empty (per-appliance sample counts "
            f"{coverage}); the appliance test windows do not overlap in time"
        )

    # ---- OSW blocks, one build per house so no block straddles a house boundary ----
    true_cols = [f"{a}_true" for a in appliances]
    parts = []
    for house in sorted(joined):
        frame = joined[house]
        if len(frame) < args.osw_size:
            continue
        y_true = frame[true_cols].to_numpy(dtype=np.float64)
        parts.append(
            build_osw(
                # The aggregate defines the block grid. On the joined timeline the sum of the
                # measured appliance powers *is* the synthetic aggregate the models were
                # trained against, and unlike any single appliance's stored `_agg` column it
                # is model- and appliance-independent.
                y_true.sum(axis=1),
                y_true,
                frame[[f"{a}_pred" for a in appliances]].to_numpy(dtype=np.float64),
                appliances,
                thresholds=thresholds,
                osw_length=args.osw_size,
                index=frame.index,
                sampling_interval=sampling_rate,
                nan_policy=args.nan_policy,
                partial_policy=args.partial_policy,
                status_true=(
                    frame[[f"{a}_status" for a in appliances]].to_numpy(dtype=np.float64)
                    if args.gt_on_source == "status"
                    else None
                ),
            )
        )
    if not parts:
        raise SkipRun(
            f"no house has at least osw_size={args.osw_size} joined samples "
            f"(joined total {n_joined})"
        )
    blocks = concat_osw(parts)
    if blocks.n_osw == 0:
        raise SkipRun(
            f"0 OSW blocks survived from {n_joined} joined samples "
            f"(osw_size={args.osw_size}, drops={dict(blocks.drop_counts)})"
        )

    metrics = score(blocks)
    combos = per_combination(blocks)

    agg_ref = np.concatenate(
        [joined[h][f"{appliances[0]}_agg"].to_numpy(dtype=np.float64) for h in sorted(joined)]
    )
    row = {
        "SourceDirs": "|".join(
            sorted({appliance_files[a][0] for a in appliances})
        ),
        "Model": model, "Dataset": dataset, "SamplingRate": sampling_rate,
        "WindowSize": window_size_s, "Seed": seed,
        "Appliances": "|".join(appliances),
        "NumAppliances": len(appliances),
        "SynthAggKey": sa_key,
        "OswSize": args.osw_size,
        "SamplingIntervalS": metrics["SAMPLING_INTERVAL_S"],
        "OswDurationS": metrics["OSW_DURATION_S"],
        "OnSource": metrics["ON_SOURCE"],
        "NanPolicy": metrics["NAN_POLICY"],
        "PartialPolicy": metrics["PARTIAL_POLICY"],
        "NumHouses": len(joined),
        "NumWindowsMin": min(gt_metas[a]["n_windows"] for a in appliances),
        "NumSamplesJoined": n_joined,
        "NumContiguousRuns": metrics["N_CONTIGUOUS_RUNS"],
        "NumOsw": metrics["N_OSW"],
        "NumOswActive": metrics["N_OSW_ACT"],
        "NumOswAllOff": metrics["N_OSW_ALL_OFF"],
        "NumOswDroppedPartialSamples": metrics["N_DROP_PARTIAL_SAMPLES"],
        "NumOswDroppedNan": metrics["N_DROP_NAN"],
        "NumCombGT": metrics["N_COMB_TRUE"],
        "NumCombPred": metrics["N_COMB_PRED"],
        "NumCombUnion": metrics["N_COMB_UNION"],
        "AppCoverage": ";".join(
            f"{a}:{n_joined / coverage[a]:.3f}" if coverage[a] else f"{a}:0"
            for a in appliances
        ),
        "AggMeanW": round(float(agg_ref.mean()), 3),
        "ACI": metrics["ACI"], "ACI_ACT": metrics["ACI_ACT"],
        "AFM": metrics["AFM"], "AFM_ACT": metrics["AFM_ACT"],
        "APD": metrics["APD"], "APA_POOLED": metrics["APA_POOLED"],
    }

    combo_rows = []
    for record in combos.to_dict("records"):
        combo_rows.append(
            {
                "Model": model, "Dataset": dataset, "SamplingRate": sampling_rate,
                "WindowSize": window_size_s, "Seed": seed,
                **{k: record[k] for k in COMBO_COLUMNS[5:]},
            }
        )

    if args.verbose:
        label = f"{model}/{dataset}/ws={window_size}/seed={seed}"
        print(f"\n{label}", file=sys.stderr)
        print(
            f"  appliances: {', '.join(appliances)}  (bit 0 = {appliances[0]})",
            file=sys.stderr,
        )
        print(f"  thresholds: {thresholds}", file=sys.stderr)
        for appliance in appliances:
            own = coverage[appliance]
            print(
                f"  {appliance:<16} own {own:>9,} samples -> joined {n_joined:>9,} "
                f"({n_joined / own:.1%})" if own else f"  {appliance:<16} own 0 samples",
                file=sys.stderr,
            )
        for line in summary_lines(metrics):
            print(f"  {line}", file=sys.stderr)
        top = combos.head(5)
        print("  top combinations by GT support:", file=sys.stderr)
        for record in top.to_dict("records"):
            print(
                f"    {record['Combination']:<40} n={record['SupportGT']:>7} "
                f"Fm={record['FM']:.3f}",
                file=sys.stderr,
            )
    return row, combo_rows


# --------------------------------------------------------------------------------------
# CSV I/O -- both files are read-modify-write, keyed on the run
# --------------------------------------------------------------------------------------


def read_rows(path, columns):
    path = Path(path)
    if not path.exists():
        return []
    with open(path, newline="") as f:
        return [{c: row.get(c, "") for c in columns} for row in csv.DictReader(f)]


def write_rows(path, rows, columns):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=columns, restval="")
        writer.writeheader()
        writer.writerows(rows)


def run_key_of(row):
    return tuple(str(row[f]) for f in RUN_KEY_FIELDS)


# --------------------------------------------------------------------------------------
# Entry point
# --------------------------------------------------------------------------------------


def parse_args():
    parser = argparse.ArgumentParser(
        description="Score models with the OSW appliance-combination metrics "
        "(Aci / Afm / Apd) of Welikala et al., IEEE TSG 2019.",
    )
    parser.add_argument("--result-dir", dest="result_dirs", action="append", default=None,
                        help="result directory to scan; repeatable, earlier wins "
                             f"(default: {' '.join(DEFAULT_RESULT_DIRS)})")
    parser.add_argument("--models", nargs="+", default=DEFAULT_MODELS)
    parser.add_argument("--datasets", nargs="+", default=["UKDALE"])
    parser.add_argument("--appliances", nargs="+", default=None,
                        help="require exactly these appliance keys (default: all present)")
    parser.add_argument("--sampling-rate", default="10s", help="'' for all")
    parser.add_argument("--window-sizes", nargs="+", default=None)
    parser.add_argument("--seeds", nargs="+", default=None)
    parser.add_argument("--osw-size", type=int, default=DEFAULT_OSW_LENGTH,
                        help=f"samples per OSW (default: {DEFAULT_OSW_LENGTH})")
    parser.add_argument("--nan-policy", choices=["drop", "interpolate", "zero"],
                        default="drop")
    parser.add_argument("--partial-policy", choices=["drop", "pad", "keep"], default="drop")
    parser.add_argument("--gt-on-source", choices=["threshold", "status"],
                        default="threshold",
                        help="ground-truth ON/OFF from the block-mean threshold test "
                             "(default) or from the dataset's duration-filtered status "
                             "channel")
    parser.add_argument("--threshold-source", choices=["simple", "kelly"], default="simple",
                        help="'simple' uses the builders' plain appliance thresholds "
                             "(kettle 500, WM/DW 300, microwave 200, fridge 50 W); 'kelly' "
                             "uses the min_threshold the trainer clipped predictions at, "
                             "which is very low for WM/DW because it is meant to be paired "
                             "with duration filtering")
    parser.add_argument("--data-path", default="data/")
    parser.add_argument("--gt-cache-dir", default=DEFAULT_GT_CACHE)
    parser.add_argument("--pred-cache-dir", default=DEFAULT_PRED_CACHE)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--combos-out", type=Path, default=DEFAULT_COMBOS_OUT)
    parser.add_argument("--refresh", action="store_true",
                        help="rescore runs already present in --out")
    parser.add_argument("--refresh-gt", action="store_true",
                        help="rebuild the ground-truth cache")
    parser.add_argument("--build-gt-only", action="store_true",
                        help="populate the ground-truth cache and exit")
    parser.add_argument("--jobs", type=int, default=4)
    parser.add_argument("--strict", action="store_true",
                        help="turn per-run skip warnings into hard failures")
    parser.add_argument("-v", "--verbose", action="store_true")

    args = parser.parse_args()
    if not args.result_dirs:
        args.result_dirs = list(DEFAULT_RESULT_DIRS)
    if args.osw_size < 1:
        parser.error("--osw-size must be >= 1")
    return args


def warm_gt_cache(runs, args, configs, cache_meta):
    """Build every ground-truth cache entry the selected runs need, in parallel."""
    specs = {}
    for (model, dataset, sampling_rate, window_size_s, _), files in runs.items():
        try:
            cfg, dataset_cfg = resolve_config(configs, dataset, model)
        except SkipRun as exc:
            warn(str(exc))
            continue
        if dataset == "REFIT":
            continue
        appliances, _ = appliance_keys_for(dataset_cfg, set(files))
        synth = list(cfg.get("synth_aggregate_apps") or [])
        sa_key = synth_key(synth)
        for appliance in appliances:
            app_cfg = dataset_cfg.get(appliance)
            if app_cfg is None:
                continue
            path = gt_cache_path(
                args.gt_cache_dir, dataset, appliance, sampling_rate,
                int(window_size_s), sa_key,
            )
            if path.exists() and not args.refresh_gt:
                continue
            specs[path.as_posix()] = {
                "dataset": dataset,
                "appliance": appliance,
                "app": app_cfg["app"],
                "sampling_rate": sampling_rate,
                "window_size": int(window_size_s),
                "synth_aggregate_apps": synth,
                "ind_house_test": list(app_cfg["ind_house_test"]),
                "data_path": args.data_path,
                "git_sha": cache_meta["git_sha"],
                "cache_path": path.as_posix(),
            }
    if not specs:
        return
    todo = list(specs.values())
    print(
        f"Rebuilding {len(todo)} ground-truth cache entr(ies) with {args.jobs} worker(s); "
        "this re-reads the raw dataset and is the slow part -- it happens once per "
        "(dataset, appliance, sampling rate, window size).",
        file=sys.stderr,
    )
    with ProcessPoolExecutor(max_workers=args.jobs) as executor:
        for i, (path, n_windows) in enumerate(
            executor.map(build_ground_truth, todo), 1
        ):
            print(f"[{i}/{len(todo)}] {path}  ({n_windows} windows)", file=sys.stderr,
                  flush=True)


def main():
    args = parse_args()
    configs = load_configs()
    cache_meta = {"git_sha": git_sha()}

    filters = {
        "models": set(args.models) if args.models else None,
        "datasets": set(args.datasets) if args.datasets else None,
        "sampling_rate": args.sampling_rate,
        "window_sizes": set(args.window_sizes) if args.window_sizes else None,
        "seeds": set(args.seeds) if args.seeds else None,
        "appliances": set(args.appliances) if args.appliances else None,
    }
    runs = discover(args.result_dirs, filters)
    if not runs:
        warn("no result files matched the given filters")
        return 1

    # The appliance set a run must cover: everything present on disk for that dataset
    # (deduplicated), or exactly --appliances when given. Runs that lack any of it are
    # skipped rather than scored over a smaller, incomparable combination space.
    required_by_dataset = defaultdict(set)
    for (_, dataset, _, _, _), files in runs.items():
        required_by_dataset[dataset] |= set(files)
    for dataset, present in list(required_by_dataset.items()):
        try:
            _, dataset_cfg = resolve_config(configs, dataset, args.models[0])
        except SkipRun:
            dataset_cfg = {}
        keys, _ = appliance_keys_for(dataset_cfg, present)
        if args.appliances:
            keys = [a for a in keys if a in set(args.appliances)]
        required_by_dataset[dataset] = keys
        print(
            f"{dataset}: scoring combinations over {len(keys)} appliance(s): "
            f"{', '.join(keys)}",
            file=sys.stderr,
        )

    warm_gt_cache(runs, args, configs, cache_meta)
    if args.build_gt_only:
        print("Ground-truth cache populated; exiting (--build-gt-only).", file=sys.stderr)
        return 0

    existing_runs = read_rows(args.out, RUN_COLUMNS)
    existing_combos = read_rows(args.combos_out, COMBO_COLUMNS)
    done = {run_key_of(r) for r in existing_runs}

    todo = sorted(k for k in runs if args.refresh or k not in done)
    skipped_done = len(runs) - len(todo)
    if skipped_done:
        print(
            f"{skipped_done} run(s) already scored in {args.out}; pass --refresh to redo",
            file=sys.stderr,
        )
    if not todo:
        print("Nothing to do.", file=sys.stderr)
        return 0

    rescored = set()
    new_rows, new_combos = [], []
    failures = 0
    for i, run_key in enumerate(todo, 1):
        label = "/".join(run_key)
        print(f"[{i}/{len(todo)}] {label}", file=sys.stderr, flush=True)
        try:
            row, combo_rows = score_run(
                run_key, runs[run_key], args, configs, cache_meta,
                required_by_dataset[run_key[1]],
            )
        except SkipRun as exc:
            failures += 1
            if args.strict:
                raise
            warn(f"{label}: {exc}")
            continue
        new_rows.append(row)
        new_combos.extend(combo_rows)
        rescored.add(run_key)

        # Flush periodically: a full sweep takes minutes and an interrupt should not lose it.
        if i % 10 == 0 or i == len(todo):
            kept = [r for r in existing_runs if run_key_of(r) not in rescored]
            kept_combos = [r for r in existing_combos if run_key_of(r) not in rescored]
            write_rows(args.out, kept + new_rows, RUN_COLUMNS)
            write_rows(args.combos_out, kept_combos + new_combos, COMBO_COLUMNS)

    if not new_rows:
        warn("no run could be scored")
        return 1

    print(
        f"\nWrote {len(new_rows)} run(s) to {args.out} and "
        f"{len(new_combos)} combination row(s) to {args.combos_out}"
        + (f"; {failures} run(s) skipped" if failures else ""),
        file=sys.stderr,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
