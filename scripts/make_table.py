"""Aggregate NILM experiment results into a comparison table printed to the terminal.

Reads the `.pt` logs written by `scripts/run_one_expe.py`, which encode every grouping
key in the path:

    <result-dir>/{Dataset}_{Appliance}_{SamplingRate}/{WindowSize}/{Model}_{seed}.pt

Metrics are averaged over window sizes *and* seeds, then laid out like the paper's
comparison table: metric blocks (SCA / F1 / MAE) as row groups, models as rows, and
dataset -> appliance as grouped columns.

Scanning is expensive (each `.pt` carries the model weights and full prediction arrays,
19-130 MB), so extracted metrics are cached one row per run in `results/runs_cache.csv`
and only unseen files are re-read on subsequent invocations.

A second table reports the appliance-combination metrics of Welikala et al. (IEEE TSG 2019)
-- Aci, Afm and Apd. Those are read from `results/osw_runs.csv`, written by
`scripts/score_osw.py`, because they are per-RUN rather than per-appliance: a combination
score covers every appliance jointly and so cannot come from a single `.pt`. Pass `--no-osw`
to omit the block.

Usage:
    uv run -m scripts.make_table
    uv run -m scripts.make_table --result-dir 2207-results/result
    uv run -m scripts.make_table --window-sizes 256 --models TCN_KL NILMFormer
"""

import argparse
import csv
import os
import re
import sys
import warnings
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import torch

# Baselines live in one directory, the pretrained TCN_KL ("Proposed") in another; the two
# are disjoint in model, so both are read by default and unioned.
DEFAULT_RESULT_DIRS = [
    "result-synagg",
    "2207-results/result-synagg-overfit",
    "result-tcn-ablation",
    # Curated pool-size sweep (Microwave only, caps 20/50/100/200). Kept out of
    # DEFAULT_MODELS so it does not clutter the main table, but cached for
    # scripts/make_pool_sweep.py.
    "result-curated-scaling",
    # REDD baselines (and a duplicate copy of the UK-DALE ones). Listed LAST so the
    # UK-DALE duplicates lose the priority tie-break in aggregate() and the existing
    # numbers are untouched; only the REDD rows, which exist nowhere else, are new.
    "2207-results/result",
]
DEFAULT_CACHE = Path("results/runs_cache.csv")

# Metrics pulled out of log["test_metrics_timestamp"] and stored in the cache.
PT_METRIC_COLUMNS = [
    "MAE", "F1_SCORE", "ACCURACY", "BALANCED_ACCURACY", "PRECISION", "RECALL",
    "TECA", "NDE", "SAE",
]
# Appliance-combination metrics. These are per-RUN, not per-appliance: they come from
# results/osw_runs.csv (written by scripts/score_osw.py) rather than from any single .pt.
OSW_METRIC_COLUMNS = ["ACI", "ACI_ACT", "AFM", "AFM_ACT", "APD", "APA_POOLED"]
METRIC_COLUMNS = PT_METRIC_COLUMNS + OSW_METRIC_COLUMNS
CACHE_COLUMNS = [
    "SourceDir", "SourceFile", "Model", "Dataset", "Appliance",
    "SamplingRate", "WindowSize", "Seed",
] + METRIC_COLUMNS
DEFAULT_OSW_CSV = Path("results/osw_runs.csv")
OSW_RUN_KEY = ("Model", "Dataset", "SamplingRate", "WindowSize", "Seed")

DATASET_LABEL = {"UKDALE": "UK-DALE", "REDD": "REDD", "REFIT": "REFIT"}
DATASET_ORDER = ["UKDALE", "REDD", "REFIT"]

APPLIANCE_LABEL = {
    "WashingMachine": "WM",
    "Dishwasher": "DW",
    "Kettle": "Kettle",
    "Microwave": "Micro.",
    "Fridge": "Fridge",
    "WasherDryer": "WD",
}
APPLIANCE_ORDER = ["WashingMachine", "Dishwasher", "Kettle", "Microwave", "Fridge", "WasherDryer"]

# TCN_KL is the pretrained "Proposed" model (src/helpers/expes.py:325).
MODEL_LABEL = {
    "TCN_KL": "Proposed",
    "TCN_KL_perapp": "Proposed 1app",
    "TCN_KL_aug_curated": "TCN+KL+Cur+Aug",
    "TCN_KL_perapp_nilmformerlike": "Prop 1app NFlike",
    # Ablation ladder for the Proposed model: no curation / no augmentation / no KL,
    # then +KL, then +synthetic training data. See src/helpers/tcn_ablation.py.
    "TCN": "TCN",
    "TCN_KL_scratch": "TCN+KL",
    "TCN_KL_aug": "TCN+KL+aug",
    # Variant C of the 2x2: curated synthetic data WITHOUT the KL front end.
    "TCN_aug_curated": "TCN+Cur+Aug",
    # Curated pool-size sweep; N is the number of distinct curated activations.
    "TCN_KL_aug_curated_n20": "TCN+KL+Cur+Aug n20",
    "TCN_KL_aug_curated_n50": "TCN+KL+Cur+Aug n50",
    "TCN_KL_aug_curated_n100": "TCN+KL+Cur+Aug n100",
    "TCN_KL_aug_curated_n200": "TCN+KL+Cur+Aug n200",
    "NILMFormer": "NILMFormer",
    "BERT4NILM": "BERT4NILM",
    "BiLSTM": "BiLSTM",
    "BiGRU": "BiGRU",
    "CNN1D": "CNN1D",
    "DAResNet": "DAResNet",
}
PROPOSED_MODEL = "TCN_KL"
DEFAULT_MODELS = [
    "TCN_KL", "TCN_KL_perapp", "TCN_KL_perapp_nilmformerlike", "TCN_KL_aug_curated", "TCN_KL_aug", "TCN_KL_scratch", "TCN",
    "NILMFormer", "BERT4NILM", "BiLSTM", "BiGRU",
]

# Display label -> (cache column, formatter, direction of "better").
# ACCURACY and F1_SCORE are stored as fractions (e.g. 0.893), hence the x100.
METRICS = {
    # SCA is plain per-timestamp accuracy on a heavily OFF-dominated signal, so it is
    # nearly saturated for every method (an all-OFF predictor scores ~99% on kettle).
    # BA is balanced accuracy -- the mean of sensitivity and specificity -- which is
    # already computed in src/helpers/metrics.py:128 and separates the methods properly.
    "SCA %": ("ACCURACY", lambda v: f"{100 * v:.2f}", "max"),
    "BA %": ("BALANCED_ACCURACY", lambda v: f"{100 * v:.2f}", "max"),
    "F1 %": ("F1_SCORE", lambda v: f"{100 * v:.2f}", "max"),
    "MAE": ("MAE", lambda v: f"{v:.2f}", "min"),
}

# Appliance-combination metrics, rendered in their own table (they are per-run, not
# per-appliance). Note the scale asymmetry inherited from src/helpers/osw_metrics.py: ACI
# and APD are already percentages, AFM is a fraction like F1_SCORE and so needs the x100.
OSW_METRICS = {
    "Aci %": ("ACI", lambda v: f"{v:.2f}", "max"),
    "Aci % act": ("ACI_ACT", lambda v: f"{v:.2f}", "max"),
    "Afm %": ("AFM", lambda v: f"{100 * v:.2f}", "max"),
    "Afm % act": ("AFM_ACT", lambda v: f"{100 * v:.2f}", "max"),
    "Apd %": ("APD", lambda v: f"{v:.2f}", "max"),
}

MISSING = "--"
BOLD = "\033[1m"
RESET = "\033[0m"


def warn(msg):
    print(f"WARN {msg}", file=sys.stderr)


# --------------------------------------------------------------------------------------
# Stage 1: scan .pt files into a per-run CSV cache
# --------------------------------------------------------------------------------------


def parse_pt_path(pt_file):
    """Pull (model, dataset, appliance, sampling_rate, window_size, seed) out of a path.

    The stem regex is what makes `TCN_KL_0` parse correctly -- the model name itself
    contains an underscore, so the seed must be split off greedily from the right.
    """
    parts = pt_file.parts
    if len(parts) < 3:
        return None
    dataset_appliance_sr = parts[-3]  # e.g. UKDALE_Dishwasher_10s
    window_size = parts[-2]  # e.g. 128

    stem_match = re.match(r"^(.+)_(\d+)$", pt_file.stem)  # e.g. CNN1D_0, TCN_KL_0
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


def read_metrics(pt_file):
    """Load one result file and return its test metrics. Runs in a worker process."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        # weights_only=False is required: the metric dicts contain numpy scalars.
        log = torch.load(pt_file, map_location="cpu", weights_only=False)

    # TCN_KL logs lack every training key (training is skipped for the pretrained
    # model), so read defensively -- only the test metrics are guaranteed present.
    metrics = log.get("test_metrics_timestamp", {}) or {}
    out = {}
    # Only the per-.pt metrics: the OSW columns are per-run and are merged in later.
    for name in PT_METRIC_COLUMNS:
        value = metrics.get(name)
        out[name] = "" if value is None else float(value)
    return out


def load_cache(cache_path):
    if not cache_path.exists():
        return []
    with open(cache_path, newline="") as f:
        # Backfill any column the cache predates (e.g. the OSW metrics) so old rows do not
        # KeyError in aggregate(). This is what lets new metric columns be added without
        # forcing a --refresh over every result file.
        return [
            {**dict.fromkeys(CACHE_COLUMNS, ""), **row} for row in csv.DictReader(f)
        ]


def write_cache(cache_path, rows):
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    with open(cache_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CACHE_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)


def scan(result_dirs, cache_path, refresh, jobs):
    """Return every cached run row, extracting metrics for any file not already cached."""
    rows = [] if refresh else load_cache(cache_path)
    known = {row["SourceFile"] for row in rows}

    pending = []
    for result_dir in result_dirs:
        root = Path(result_dir)
        if not root.is_dir():
            warn(f"result dir not found, skipping: {result_dir}")
            continue
        for pt_file in sorted(root.rglob("*.pt")):
            key = pt_file.as_posix()
            if key in known:
                continue
            meta = parse_pt_path(pt_file)
            if meta is None:
                warn(f"unparseable path, skipping: {pt_file}")
                continue
            meta["SourceDir"] = result_dir
            meta["SourceFile"] = key
            pending.append((pt_file, meta))
            known.add(key)

    if not pending:
        return rows

    total = len(pending)
    print(f"Reading {total} new result file(s)...", file=sys.stderr)
    with ProcessPoolExecutor(max_workers=jobs) as executor:
        results = executor.map(read_metrics, [pt for pt, _ in pending])
        for i, (metrics, (pt_file, meta)) in enumerate(zip(results, pending), 1):
            print(f"[{i}/{total}] {pt_file}", file=sys.stderr, flush=True)
            rows.append({**meta, **metrics})
            # Flush periodically so an interrupted scan does not lose its work --
            # a full pass over the larger result dirs takes minutes.
            if i % 50 == 0:
                write_cache(cache_path, rows)

    write_cache(cache_path, rows)
    print(f"Cached {len(rows)} run(s) in {cache_path}", file=sys.stderr)
    return rows


def merge_osw(rows, osw_csv):
    """Stamp the per-run OSW metrics from `osw_csv` onto the scanned rows.

    The OSW metrics are per (model, dataset, sampling rate, window size, seed) -- one
    appliance-combination score for the whole run -- while `rows` has one entry per
    appliance. The same value is written to every appliance row of a run, which is harmless
    because aggregate() means over window sizes and seeds and the value is constant within
    a run.

    The values are always overwritten, never cached-and-kept: re-running
    scripts/score_osw.py is what refreshes them, so runs_cache.csv is not authoritative for
    these columns and no cache invalidation dance is needed.
    """
    osw_csv = Path(osw_csv)
    if not osw_csv.exists():
        warn(f"OSW score file not found, ACI/AFM/APD columns will be empty: {osw_csv} "
             f"(create it with: uv run -m scripts.score_osw)")
        return

    by_run = {}
    with open(osw_csv, newline="") as f:
        for record in csv.DictReader(f):
            by_run[tuple(record.get(k, "") for k in OSW_RUN_KEY)] = record

    matched_models = set()
    for row in rows:
        record = by_run.get(tuple(row[k] for k in OSW_RUN_KEY))
        if record is None:
            continue
        matched_models.add(row["Model"])
        for column in OSW_METRIC_COLUMNS:
            row[column] = record.get(column, "")

    for model in sorted({row["Model"] for row in rows} - matched_models):
        warn(f"no OSW scores for model {model} in {osw_csv}; run: "
             f"uv run -m scripts.score_osw --models {model}")


# --------------------------------------------------------------------------------------
# Stage 2: aggregate over window sizes and seeds
# --------------------------------------------------------------------------------------


def aggregate(rows, result_dirs, sampling_rate, window_sizes, seeds):
    """Group runs by (model, dataset, appliance) and mean each metric across them.

    Windows and seeds are pooled with equal weight. Rows from earlier `--result-dir`
    entries win when the same run appears in more than one directory.
    """
    priority = {d: i for i, d in enumerate(result_dirs)}

    best_by_run = {}
    for row in rows:
        if row["SourceDir"] not in priority:
            continue
        if sampling_rate and row["SamplingRate"] != sampling_rate:
            continue
        if window_sizes and row["WindowSize"] not in window_sizes:
            continue
        if seeds and row["Seed"] not in seeds:
            continue
        key = (row["Model"], row["Dataset"], row["Appliance"], row["WindowSize"], row["Seed"])
        current = best_by_run.get(key)
        if current is None or priority[row["SourceDir"]] < priority[current["SourceDir"]]:
            best_by_run[key] = row

    groups = defaultdict(list)
    for (model, dataset, appliance, _, _), row in best_by_run.items():
        groups[(model, dataset, appliance)].append(row)

    agg = {}
    for key, group_rows in groups.items():
        means = {}
        for column in METRIC_COLUMNS:
            # .get, not [...]: rows scanned fresh from .pt files in this same invocation
            # carry only PT_METRIC_COLUMNS. The OSW columns arrive later via merge_osw (for
            # runs that have OSW scores) or via load_cache's backfill (for cached rows), so
            # a freshly-read run with no OSW score has no such key at all.
            values = [
                float(r[column])
                for r in group_rows
                if r.get(column) not in ("", None)
            ]
            means[column] = sum(values) / len(values) if values else None
        means["n"] = len(group_rows)
        agg[key] = means
    return agg


def report_coverage(agg):
    """Warn about groups backed by fewer runs than the typical cell. Returns (modal_n, n_short)."""
    counts = [m["n"] for m in agg.values()]
    if not counts:
        return 0, 0
    expected = Counter(counts).most_common(1)[0][0]
    short = 0
    for (model, dataset, appliance), means in sorted(agg.items()):
        if means["n"] < expected:
            warn(f"{dataset}/{appliance}/{model}: {means['n']} runs (expected {expected})")
            short += 1
    return expected, short


def ordered(values, canonical):
    """Sort by a canonical order, appending anything unrecognised alphabetically."""
    known = [v for v in canonical if v in values]
    extra = sorted(v for v in values if v not in canonical)
    return known + extra


# --------------------------------------------------------------------------------------
# Stage 3: render
# --------------------------------------------------------------------------------------


def cell(text, width, bold, use_color):
    """Right-align `text` in `width`, applying bold *after* padding so widths stay true."""
    padded = text.rjust(width)
    if bold and use_color:
        return f"{BOLD}{padded}{RESET}"
    return padded


def span_width(widths):
    """Width of a group of columns once rendered as ` a │ b │ c `."""
    return sum(w + 2 for w in widths) + max(len(widths) - 1, 0)


def rule(left, mid, right, widths):
    return left + mid.join("─" * (w + 2) for w in widths) + right


def group_rule(left, mid, right, group_widths):
    return left + mid.join("─" * w for w in group_widths) + right


def render(agg, models, use_color):
    """Build the table as a list of lines. Column blocks follow the data present."""
    datasets = ordered({d for _, d, _ in agg}, DATASET_ORDER)
    blocks = []  # [(dataset, [appliance, ...]), ...]
    for dataset in datasets:
        appliances = ordered({a for _, d, a in agg if d == dataset}, APPLIANCE_ORDER)
        if appliances:
            blocks.append((dataset, appliances))

    metric_width = max(len("Metric"), *(len(label) for label in METRICS))
    # +2 leaves room for the ` *` marker on the Proposed row.
    method_width = max(len("Method"), max(len(MODEL_LABEL.get(m, m)) for m in models) + 2)

    # Data column widths: the header label, or the widest formatted value beneath it.
    data_widths = {}
    for dataset, appliances in blocks:
        for appliance in appliances:
            width = len(APPLIANCE_LABEL.get(appliance, appliance))
            for label, (column, fmt, _) in METRICS.items():
                for model in models:
                    means = agg.get((model, dataset, appliance))
                    value = means[column] if means else None
                    text = fmt(value) if value is not None else MISSING
                    width = max(width, len(text))
            data_widths[(dataset, appliance)] = width

    flat_widths = [metric_width, method_width]
    for dataset, appliances in blocks:
        flat_widths += [data_widths[(dataset, a)] for a in appliances]

    # Top border and dataset header span whole blocks (the LaTeX \multicolumn rows);
    # the appliance header below splits them (the \cline).
    group_widths = [metric_width + 2, method_width + 2]
    for dataset, appliances in blocks:
        group_widths.append(span_width([data_widths[(dataset, a)] for a in appliances]))

    lines = []
    lines.append(group_rule("┌", "┬", "┐", group_widths))

    header = ["Metric".ljust(metric_width), "Method".ljust(method_width)]
    for i, (dataset, appliances) in enumerate(blocks):
        label = DATASET_LABEL.get(dataset, dataset)
        header.append(label.center(group_widths[2 + i] - 2))
    lines.append("│ " + " │ ".join(header) + " │")

    # Split each dataset span into its appliance columns.
    parts = ["─" * (metric_width + 2), "─" * (method_width + 2)]
    for dataset, appliances in blocks:
        parts.append("┬".join("─" * (data_widths[(dataset, a)] + 2) for a in appliances))
    lines.append("├" + "┼".join(parts) + "┤")

    header = [" " * metric_width, " " * method_width]
    for dataset, appliances in blocks:
        for appliance in appliances:
            name = APPLIANCE_LABEL.get(appliance, appliance)
            header.append(name.rjust(data_widths[(dataset, appliance)]))
    lines.append("│ " + " │ ".join(header) + " │")

    for label, (column, fmt, direction) in METRICS.items():
        lines.append(rule("├", "┼", "┤", flat_widths))

        # Per column, find the winning value so it can be bolded (min for MAE).
        best = {}
        for dataset, appliances in blocks:
            for appliance in appliances:
                values = [
                    agg[(m, dataset, appliance)][column]
                    for m in models
                    if (m, dataset, appliance) in agg
                    and agg[(m, dataset, appliance)][column] is not None
                ]
                if values:
                    best[(dataset, appliance)] = max(values) if direction == "max" else min(values)

        for i, model in enumerate(models):
            name = MODEL_LABEL.get(model, model)
            is_proposed = model == PROPOSED_MODEL
            method = name.ljust(method_width - 2) + " *" if is_proposed else name.ljust(method_width)
            row = [
                (label if i == 0 else "").ljust(metric_width),
                f"{BOLD}{method}{RESET}" if is_proposed and use_color else method,
            ]
            for dataset, appliances in blocks:
                for appliance in appliances:
                    means = agg.get((model, dataset, appliance))
                    value = means[column] if means else None
                    text = fmt(value) if value is not None else MISSING
                    is_best = value is not None and value == best.get((dataset, appliance))
                    row.append(cell(text, data_widths[(dataset, appliance)], is_best, use_color))
            lines.append("│ " + " │ ".join(row) + " │")

    lines.append(rule("└", "┴", "┘", flat_widths))
    return lines


def render_osw(agg, models, use_color):
    """Build the appliance-combination table: metrics x models, dataset columns only.

    These metrics are per (model, dataset) -- a combination score spans every appliance at
    once -- so unlike the main table there is no appliance split; showing one would just
    repeat the same number five times.
    """
    datasets = ordered({d for _, d, _ in agg}, DATASET_ORDER)

    def value_for(model, dataset, column):
        """The run's value, taken from any appliance group; they must all agree."""
        values = [
            agg[(m, d, a)][column]
            for (m, d, a) in agg
            if m == model and d == dataset and agg[(m, d, a)][column] is not None
        ]
        if not values:
            return None
        if max(values) - min(values) > 1e-6:
            warn(f"{dataset}/{model}: OSW metric {column} differs across appliance groups "
                 f"({min(values):.4f}..{max(values):.4f}); the run is only partially scored")
        return values[0]

    present = [
        d for d in datasets
        if any(value_for(m, d, c) is not None for m in models for c in OSW_METRIC_COLUMNS)
    ]
    if not present:
        return []

    metric_width = max(len("Metric"), *(len(label) for label in OSW_METRICS))
    method_width = max(len("Method"), max(len(MODEL_LABEL.get(m, m)) for m in models) + 2)

    data_widths = {}
    for dataset in present:
        width = len(DATASET_LABEL.get(dataset, dataset))
        for label, (column, fmt, _) in OSW_METRICS.items():
            for model in models:
                value = value_for(model, dataset, column)
                width = max(width, len(fmt(value) if value is not None else MISSING))
        data_widths[dataset] = width

    flat_widths = [metric_width, method_width] + [data_widths[d] for d in present]

    lines = [rule("┌", "┬", "┐", flat_widths)]
    header = ["Metric".ljust(metric_width), "Method".ljust(method_width)]
    header += [DATASET_LABEL.get(d, d).rjust(data_widths[d]) for d in present]
    lines.append("│ " + " │ ".join(header) + " │")

    for label, (column, fmt, direction) in OSW_METRICS.items():
        lines.append(rule("├", "┼", "┤", flat_widths))

        best = {}
        for dataset in present:
            values = [
                v for v in (value_for(m, dataset, column) for m in models) if v is not None
            ]
            if values:
                best[dataset] = max(values) if direction == "max" else min(values)

        for i, model in enumerate(models):
            name = MODEL_LABEL.get(model, model)
            is_proposed = model == PROPOSED_MODEL
            method = (
                name.ljust(method_width - 2) + " *" if is_proposed
                else name.ljust(method_width)
            )
            row = [
                (label if i == 0 else "").ljust(metric_width),
                f"{BOLD}{method}{RESET}" if is_proposed and use_color else method,
            ]
            for dataset in present:
                value = value_for(model, dataset, column)
                text = fmt(value) if value is not None else MISSING
                is_best = value is not None and value == best.get(dataset)
                row.append(cell(text, data_widths[dataset], is_best, use_color))
            lines.append("│ " + " │ ".join(row) + " │")

    lines.append(rule("└", "┴", "┘", flat_widths))
    return lines


def write_csv(path, agg, models):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    columns = ["Model", "Dataset", "Appliance", "NumRuns"] + METRIC_COLUMNS
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        for (model, dataset, appliance), means in sorted(agg.items()):
            if model not in models:
                continue
            row = {"Model": model, "Dataset": dataset, "Appliance": appliance,
                   "NumRuns": means["n"]}
            for column in METRIC_COLUMNS:
                value = means[column]
                row[column] = "" if value is None else round(value, 6)
            writer.writerow(row)
    print(f"Wrote aggregated metrics to {path}", file=sys.stderr)


# --------------------------------------------------------------------------------------


def parse_args():
    parser = argparse.ArgumentParser(
        description="Print a NILM model comparison table averaged over windows and seeds.",
    )
    parser.add_argument("--result-dir", action="append", dest="result_dirs", metavar="DIR",
                        help=f"result directory; repeatable, earlier wins on duplicate runs "
                             f"(default: {' '.join(DEFAULT_RESULT_DIRS)})")
    parser.add_argument("--models", nargs="+", default=DEFAULT_MODELS,
                        help=f"model rows, in display order (default: {' '.join(DEFAULT_MODELS)})")
    parser.add_argument("--sampling-rate", default="10s",
                        help="sampling rate to report; empty string for all (default: 10s)")
    parser.add_argument("--window-sizes", nargs="+", help="window sizes to pool (default: all)")
    parser.add_argument("--seeds", nargs="+", help="seeds to pool (default: all)")
    parser.add_argument("--cache", type=Path, default=DEFAULT_CACHE,
                        help=f"per-run metrics cache (default: {DEFAULT_CACHE})")
    parser.add_argument("--refresh", action="store_true",
                        help="discard the cache and re-read every result file")
    parser.add_argument("--jobs", type=int, default=4,
                        help="parallel result-file readers (default: 4)")
    parser.add_argument("--csv", metavar="FILE", help="also write the aggregated numbers as CSV")
    parser.add_argument("--keep-empty-models", action="store_true",
                        help="keep model rows that have no runs instead of dropping them")
    parser.add_argument("--no-color", action="store_true", help="disable ANSI highlighting")
    parser.add_argument("--osw-csv", type=Path, default=DEFAULT_OSW_CSV,
                        help=f"per-run appliance-combination scores from "
                             f"scripts/score_osw.py (default: {DEFAULT_OSW_CSV})")
    parser.add_argument("--no-osw", action="store_true",
                        help="ignore the OSW score file and drop the Aci/Afm/Apd block")
    args = parser.parse_args()
    if not args.result_dirs:
        args.result_dirs = list(DEFAULT_RESULT_DIRS)
    return args


def main():
    args = parse_args()

    rows = scan(args.result_dirs, args.cache, args.refresh, args.jobs)
    if not rows:
        print("No result files found.", file=sys.stderr)
        return 1

    if not args.no_osw:
        merge_osw(rows, args.osw_csv)

    agg = aggregate(rows, args.result_dirs, args.sampling_rate, args.window_sizes, args.seeds)
    if not agg:
        print("No runs matched the requested filters.", file=sys.stderr)
        return 1

    models = args.models
    if not args.keep_empty_models:
        present = {m for m, _, _ in agg}
        missing = [m for m in models if m not in present]
        for model in missing:
            warn(f"no runs for model {model}, dropping its row")
        models = [m for m in models if m in present]
    if not models:
        print("None of the requested models have any runs.", file=sys.stderr)
        return 1

    expected, short = report_coverage(agg)

    use_color = not args.no_color and sys.stdout.isatty() and not os.environ.get("NO_COLOR")
    for line in render(agg, models, use_color):
        print(line)

    footer = f"Averaged over {expected} run(s) per cell"
    if args.window_sizes or args.seeds:
        details = []
        if args.window_sizes:
            details.append(f"windows {','.join(args.window_sizes)}")
        if args.seeds:
            details.append(f"seeds {','.join(args.seeds)}")
        footer += f" ({'; '.join(details)})"
    footer += f"; * marks the proposed method{'' if not short else f'; {short} cell(s) short -- see warnings above'}"
    print(footer)

    if not args.no_osw:
        osw_lines = render_osw(agg, models, use_color)
        if osw_lines:
            print()
            for line in osw_lines:
                print(line)
            print(
                "Appliance-combination metrics over all appliances jointly (Welikala et al., "
                "IEEE TSG 2019);\n'act' rows exclude OSWs whose ground truth is all-OFF. "
                f"Source: {args.osw_csv}"
            )

    if args.csv:
        write_csv(args.csv, agg, models)
    return 0


if __name__ == "__main__":
    sys.exit(main())
