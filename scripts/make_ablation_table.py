#################################################################################################################
#
# @description : Paper tables for the TCN ablation (UK-DALE), rendered from results/runs_cache.csv
#
# Two presets:
#   --preset ablation  the ablation ladder (default)
#   --preset sota      TCN + KL + Curated + Aug as "Proposed" against the SotA baselines
#
# The cache is populated by scripts/make_table.py; run that first when new results exist:
#     PYTHONPATH=. .venv/bin/python -m scripts.make_table --no-color
#
# Usage:
#     PYTHONPATH=. .venv/bin/python -m scripts.make_ablation_table
#     PYTHONPATH=. .venv/bin/python -m scripts.make_ablation_table --preset sota
#     PYTHONPATH=. .venv/bin/python -m scripts.make_ablation_table --csv results/ablation_metrics.csv
#
#################################################################################################################

import argparse
import csv
import statistics as st
import sys
from collections import defaultdict
from pathlib import Path

from scripts.make_table import DEFAULT_RESULT_DIRS

DEFAULT_CACHE = Path("results/runs_cache.csv")
#: Distinct curated activations each appliance's synthetic stream was built from. The
#: curated arm generates the SAME number of training windows as the real arms, so this --
#: not sample count -- is what limits it: on REDD, curation helps where the pool is large
#: (fridge ~1000) and hurts where it is tiny (11-44).
ACTIVATION_COUNTS = Path("results/curated_activation_counts.csv")

# (cache model key, printed label, training epochs)
PRESETS = {
    # Only the in-repo arms: every row here is trained on the same splits at the same
    # 3-epoch budget, so the ladder is like-for-like. The externally pretrained rows
    # (TCN_KL, TCN_KL_perapp, TCN_KL_perapp_nilmformerlike) ran at 95-10000 epochs and
    # are deliberately excluded, as is TCN_KL_aug (repo-extracted activations): it differs
    # from the curated arm in activation-pool size as well as source, so that pair does not
    # isolate curation. scripts/make_table.py still reports all of them.
    "ablation": [
        ("TCN_KL_aug_curated", "TCN + KL + Curated + Aug", "3"),
        ("TCN_aug_curated", "TCN + Curated + Aug", "3"),
        ("TCN_KL_scratch", "TCN + KL", "3"),
        ("TCN", "Vanilla TCN", "3"),
    ],
    # Everything here is trained in-repo at the same 3-epoch budget on the same splits,
    # so this comparison is like-for-like; the externally pretrained rows are excluded.
    "sota": [
        ("TCN_KL_aug_curated", "Proposed (TCN+KL+Cur+Aug)", "3"),
        ("NILMFormer", "NILMFormer", "3"),
        ("BERT4NILM", "BERT4NILM", "3"),
        ("BiLSTM", "BiLSTM", "3"),
        ("BiGRU", "BiGRU", "3"),
        ("CNN1D", "CNN1D", "3"),
        ("DAResNet", "DAResNet", "3"),
    ],
}

APPS_BY_DATASET = {
    "UKDALE": [
        ("WashingMachine", "WM"), ("Dishwasher", "DW"), ("Kettle", "Kettle"),
        ("Microwave", "Micro."), ("Fridge", "Fridge"),
    ],
    # REDD has no kettle, and its washer/dryer is a single combined meter.
    "REDD": [
        ("WasherDryer", "WD"), ("Dishwasher", "DW"),
        ("Microwave", "Micro."), ("Fridge", "Fridge"),
    ],
}
APPS = APPS_BY_DATASET["UKDALE"]
# (label, cache column, scale, direction of "better", format)
METRICS = [
    ("SCA", "ACCURACY", 100, "max", "{:.2f}%"),
    ("BA", "BALANCED_ACCURACY", 100, "max", "{:.2f}%"),
    ("F1", "F1_SCORE", 100, "max", "{:.2f}%"),
    ("MAE", "MAE", 1, "min", "{:.2f}"),
]

LABEL_W = 26


def load(cache, dataset, sampling_rate):
    """{(model, appliance, metric): [per-run values]} for one dataset.

    The same run can appear in more than one result directory -- the UK-DALE baselines
    live in both `result-synagg` and `2207-results/result` -- so runs are deduplicated by
    (model, appliance, window, seed) before averaging, with earlier entries of
    make_table.DEFAULT_RESULT_DIRS winning. Without this the duplicated cells would be
    counted twice and pull the means toward whichever copy is duplicated.
    """
    if not cache.is_file():
        sys.exit(f"{cache} not found -- run scripts.make_table first to populate it.")
    priority = {d: i for i, d in enumerate(DEFAULT_RESULT_DIRS)}

    best = {}
    for row in csv.DictReader(open(cache, newline="")):
        if row["Dataset"] != dataset or row["SamplingRate"] != sampling_rate:
            continue
        key = (row["Model"], row["Appliance"], row["WindowSize"], row["Seed"])
        rank = priority.get(row.get("SourceDir", ""), len(priority))
        if key not in best or rank < best[key][0]:
            best[key] = (rank, row)

    vals = defaultdict(list)
    for _, row in best.values():
        for _, column, _, _, _ in METRICS:
            raw = row.get(column, "")
            if raw not in ("", None):
                vals[(row["Model"], row["Appliance"], column)].append(float(raw))
    return vals


def render(vals, rows, dataset, sampling_rate, title):
    APPS = APPS_BY_DATASET[dataset]
    out = [
        f"{title} -- {dataset}, {sampling_rate}, "
        "mean of 9 runs/cell (3 window sizes x 3 seeds)",
        "",
    ]
    lone_cols = set()
    for name, column, scale, better, fmt in METRICS:
        out.append(name)
        out.append(f"  {'Method':<{LABEL_W}}{'Ep':>6}" + "".join(f"{lbl:>10}" for _, lbl in APPS))
        out.append("  " + "-" * (LABEL_W + 6 + 10 * len(APPS)))

        cells = {}
        for key, _, _ in rows:
            for app, _ in APPS:
                series = vals.get((key, app, column))
                cells[(key, app)] = st.mean(series) * scale if series else None

        best = {}
        lone = set()
        for app, _ in APPS:
            present = [(cells[(k, app)], k) for k, _, _ in rows if cells[(k, app)] is not None]
            if len(present) > 1:
                best[app] = (min(present) if better == "min" else max(present))[1]
            elif present:
                # A single populated cell is not a winner. On REDD the WasherDryer
                # baselines are absent entirely (they live under REDD_WashingMachine and
                # predate the rename), so marking the lone Proposed cell "best" would read
                # as beating baselines that were never in the column.
                lone.add(app)
        lone_cols |= lone

        for key, label, epochs in rows:
            line = f"  {label:<{LABEL_W}}{epochs:>6}"
            for app, _ in APPS:
                value = cells[(key, app)]
                text = "--" if value is None else fmt.format(value)
                if value is not None and best.get(app) == key:
                    text = "[" + text + "]"
                line += f"{text:>10}"
            out.append(line)
        out.append("")

    if lone_cols:
        out += [
            "  (!) " + ", ".join(sorted(lone_cols)) + ": only one method has runs in this "
            "column, so no",
            "      best-marker is shown. On REDD the WasherDryer baselines are absent -- "
            "they were",
            "      run under the REDD_WashingMachine key before the WashingMachine -> "
            "WasherDryer",
            "      config change, so they are not comparable and were not re-run.",
            "",
        ]
    out += [
        "  [ ] = best in column among the rows shown; a column with only one populated",
        "        row carries no marker.",
        "  SCA is per-timestamp accuracy; on appliances with a low duty cycle it saturates",
        "  (an all-OFF predictor scores ~99% on Kettle). BA is balanced accuracy -- the mean",
        "  of sensitivity and specificity -- which separates the methods properly.",
        "  Ep = training epochs.",
    ]
    counts = load_activation_counts(dataset)
    if counts:
        out += [
            "",
            "  Distinct curated activations behind the synthetic stream (per appliance):",
            "    " + "   ".join(f"{lbl} {counts[app]}" for app, lbl in APPS if app in counts),
            "  The curated arm generates as many training windows as the real arms, so this",
            "  is a diversity limit, not a sample-count one.",
        ]
    return "\n".join(out)


def load_activation_counts(dataset):
    """{appliance: usable activations} for the curated arm, if the file exists."""
    if not ACTIVATION_COUNTS.is_file():
        return {}
    return {
        r["appliance"]: int(r["usable_activations"])
        for r in csv.DictReader(open(ACTIVATION_COUNTS, newline=""))
        if r["dataset"] == dataset
    }


def write_csv(vals, rows, path, dataset, sampling_rate):
    """One row per (method, appliance) with mean and standard deviation over the 9 runs."""
    APPS = APPS_BY_DATASET[dataset]
    header = ["dataset", "sampling_rate", "model", "method", "epochs", "appliance", "n_runs"]
    for name, _, _, _, _ in METRICS:
        header += [f"{name}_mean", f"{name}_std"]
    header.append("curated_activations")
    counts = load_activation_counts(dataset)

    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(header)
        for key, label, epochs in rows:
            for app, short in APPS:
                n = 0
                cells = []
                for name, column, scale, _, _ in METRICS:
                    series = vals.get((key, app, column)) or []
                    n = max(n, len(series))
                    if not series:
                        cells += ["", ""]
                        continue
                    mean = st.mean(series) * scale
                    sd = (st.stdev(series) * scale) if len(series) > 1 else 0.0
                    cells += [f"{mean:.4f}", f"{sd:.4f}"]
                if n:
                    act = counts.get(app, "") if key == "TCN_KL_aug_curated" else ""
                    w.writerow([dataset, sampling_rate, key, label, epochs, short, n]
                               + cells + [act])
    return path


def main():
    ap = argparse.ArgumentParser(description="Render the TCN ablation / SotA tables.")
    ap.add_argument("--cache", type=Path, default=DEFAULT_CACHE)
    ap.add_argument("--dataset", default="UKDALE", choices=sorted(APPS_BY_DATASET))
    ap.add_argument("--sampling-rate", default="10s")
    ap.add_argument("--preset", choices=sorted(PRESETS), default="ablation")
    ap.add_argument("--out", type=Path, help="also write the rendered table to this file")
    ap.add_argument("--csv", type=Path, help="write per-(method, appliance) metrics as CSV")
    args = ap.parse_args()

    rows = PRESETS[args.preset]
    title = "ABLATION STUDY" if args.preset == "ablation" else "COMPARISON WITH STATE OF THE ART"
    vals = load(args.cache, args.dataset, args.sampling_rate)

    table = render(vals, rows, args.dataset, args.sampling_rate, title)
    print(table)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(table + "\n")
        print(f"\nwritten to {args.out}", file=sys.stderr)
    if args.csv:
        write_csv(vals, rows, args.csv, args.dataset, args.sampling_rate)
        print(f"csv written to {args.csv}", file=sys.stderr)


if __name__ == "__main__":
    main()
