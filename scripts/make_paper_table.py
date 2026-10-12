#################################################################################################################
#
# @description : The paper's main comparison table (Table: performance_comparison) as LaTeX.
#
# "Proposed" is TCN_KL_aug_curated -- the in-repo TCN + KL + Curated + Aug arm, trained per
# appliance on the same splits and the same 3-epoch budget as every baseline, so the whole
# table is like-for-like. (The earlier "Proposed" row was the externally pretrained TCN_KL,
# which ran a different recipe at a far larger epoch budget and is not comparable.)
#
# REDD's washer/dryer is keyed "WashingMachine" in runs made before 2026-06-22 and
# "WasherDryer" after. REDD_METER_MAP (preprocessing.py:1258-1296) maps BOTH names to the
# same meter list in every house, with identical appliance_param and identical splits, so
# they are aliased into one column here.
#
# Usage:
#     PYTHONPATH=. .venv/bin/python -m scripts.make_paper_table --out results/paper_table.tex
#
#################################################################################################################

import argparse
import csv
import statistics as st
import sys
from collections import defaultdict
from pathlib import Path

from scripts.make_table import DEFAULT_RESULT_DIRS

CACHE = Path("results/runs_cache.csv")
SAMPLING_RATE = "10s"

# (cache key, printed label with citation)
#: THE PROPOSED MODEL IS `arm3` (TCN_KL_aug): the in-repo, 3-epoch, one-model-per-
#: appliance arm with KL features and synthetic augmentation from repo-extracted
#: activations. Fixed by the user on 2026-10-12; do not substitute another arm without
#: being asked. The other entries exist only for the head-width and recipe comparisons.
PROPOSED_CHOICES = {
    # Arm 3 of the ablation ladder: KL + synthetic augmentation built from activations
    # extracted from this repo's own training houses -- NO curated store. Same 3-epoch
    # in-repo recipe and splits as every baseline.
    "arm3": ("TCN_KL_aug", "3"),
    # In-repo per-appliance arm: same splits, same scaler, same 3-epoch budget as every
    # baseline, so the table is like-for-like.
    "inrepo": ("TCN_KL_aug_curated", "3"),
    # Externally trained, ONE model with an 8-appliance head, ~10,000 epochs on curated
    # data with a different optimiser/loss. Stronger on several cells, but NOT trained
    # under the baselines' protocol -- the epoch budget alone differs by 3 orders of
    # magnitude, so a win here is not evidence of a better method.
    "external": ("TCN_KL", "10000"),
}

PRESETS = {
    # Proposed against published baselines.
    "sota": [
        ("TCN_KL_aug_curated", r"Proposed"),
        ("NILMFormer", r"NILMFormer\cite{petralia_nilmformer_2025}"),
        ("BERT4NILM", r"BERT4NILM\cite{yue_bert4nilm_2020}"),
        ("BiLSTM", r"BiLSTM\cite{10.1145/2821650.2821672}"),
        ("BiGRU", r"BiGRU\cite{Precioso2023}"),
    ],
    # The 2x2 ablation. Every row is trained in-repo on the same splits at the same
    # 3-epoch budget, so the ladder is strictly like-for-like and no epoch caveat applies.
    #   A = neither, B = KL only, C = curated+aug only, D = both.
    # Reading it: B-A is the KL front end, C-A is curated augmentation on its own, and
    # (D-C)-(B-A) is their interaction.
    # The two heads of the SAME externally trained model: identical curated data, recipe
    # and epoch budget, differing only in whether one network predicts all appliances or
    # one network is trained per appliance.
    "heads": [
        ("TCN_KL", r"One model, all appliances"),
        ("TCN_KL_perapp", r"One model per appliance"),
    ],
    # Ascending: neither -> KL -> aug -> both, so each row adds one ingredient.
    "ablation": [
        ("TCN", r"Vanilla TCN"),
        ("TCN_KL_scratch", r"TCN + KL"),
        ("TCN_aug_curated", r"TCN + Syn"),
        # Arm 3, so the bottom row of the ladder IS the Proposed row of the SotA table.
        # It was TCN_KL_aug_curated, which made the two tables disagree cell by cell
        # (UK-DALE WM F1 21.23 vs 23.12) for what the paper calls the same model.
        ("TCN_KL_aug", r"TCN + KL + Syn"),
    ],
}
CAPTIONS = {
    "sota": ("Performance Comparison on UK-DALE and REDD Datasets",
             "table:performance_comparison"),
    "ablation": ("Ablation Study on UK-DALE and REDD Datasets", "table:ablation_study"),
    "heads": ("Shared versus per-appliance output head, same trained model",
              "table:head_width"),
}
#: Per-dataset captions, used when --split emits one table per dataset.
SPLIT_CAPTIONS = {
    ("sota", "UKDALE"): ("Performance Comparison on the UK-DALE Dataset",
                         "table:performance_ukdale"),
    ("sota", "REDD"): ("Performance Comparison on the REDD Dataset",
                       "table:performance_redd"),
    ("ablation", "UKDALE"): ("Ablation Study on the UK-DALE Dataset",
                             "table:ablation_ukdale"),
    ("ablation", "REDD"): ("Ablation Study on the REDD Dataset", "table:ablation_redd"),
}
DATASET_TITLE = {"UKDALE": "UK-DALE", "REDD": "REDD"}
METHODS = PRESETS["sota"]
BLOCKS = [
    ("UKDALE", [("WashingMachine", "WM"), ("Dishwasher", "DW"), ("Kettle", "Kettle"),
                ("Microwave", "Micro."), ("Fridge", "Fridge")]),
    ("REDD", [("WasherDryer", "WM"), ("Dishwasher", "DW"),
              ("Microwave", "Micro."), ("Fridge", "Fridge")]),
]
ALIASES = {"REDD": {"WashingMachine": "WasherDryer"}}
# (block label, cache column, scale, direction, format)
METRICS = [
    ("SCA", "ACCURACY", 100, "max", "{:.2f}\\%"),
    ("F1", "F1_SCORE", 100, "max", "{:.2f}\\%"),
    ("MAE", "MAE", 1, "min", "{:.2f}"),
]


def load():
    """{(dataset, model, appliance, column): mean}, deduplicated across result dirs."""
    priority = {d: i for i, d in enumerate(DEFAULT_RESULT_DIRS)}
    best = {}
    with open(CACHE, newline="") as f:
        for r in csv.DictReader(f):
            if r["SamplingRate"] != SAMPLING_RATE:
                continue
            ds = r["Dataset"]
            app = ALIASES.get(ds, {}).get(r["Appliance"], r["Appliance"])
            key = (ds, r["Model"], app, r["WindowSize"], r["Seed"])
            rank = priority.get(r.get("SourceDir", ""), len(priority))
            if key not in best or rank < best[key][0]:
                best[key] = (rank, r)

    vals = defaultdict(list)
    for (ds, model, app, _, _), (_, r) in best.items():
        for _, col, _, _, _ in METRICS:
            v = r.get(col)
            if v not in (None, ""):
                vals[(ds, model, app, col)].append(float(v))
    return ({k: st.mean(v) for k, v in vals.items()},
            {k: len(v) for k, v in vals.items()})


def render(blocks, means, caption, label, a, highlight):
    """One table* environment covering `blocks` (one or both datasets)."""
    ncol = sum(len(apps) for _, apps in blocks)
    L = []
    L.append(r"\begin{table*}[!t]")
    L.append(r"\centering")
    L.append(r"\caption{%s}" % caption)
    L.append(r"\label{%s}" % label)
    # A full-width table* fits both datasets only when scaled; a single dataset fits at
    # normal size, and \resizebox would then MAGNIFY it, so only scale the wide variant.
    scale = a.delta and len(blocks) > 1
    if scale:
        L.append(r"\resizebox{\textwidth}{!}{%")
    elif a.delta:
        L.append(r"\setlength{\tabcolsep}{4pt}\small")
    # Keep the vertical rule between dataset groups: "ccccc|cccc", not "ccccccccc".
    spec = "|".join("c" * len(apps) for _, apps in blocks)
    L.append(r"\begin{tabular}{c|c|%s}" % spec)
    L.append(r"\hline")

    if len(blocks) > 1:
        spans = []
        for i, (ds, apps) in enumerate(blocks):
            fmt_c = "c|" if i < len(blocks) - 1 else "c"
            spans.append(r"\multicolumn{%d}{%s}{%s}" % (len(apps), fmt_c, DATASET_TITLE[ds]))
        L.append(r"\multirow{2}{*}{Metrics} & \multirow{2}{*}{Methods} & "
                 + " & ".join(spans) + r" \\ \cline{3-%d} " % (ncol + 2))
        L.append(" & & " + " & ".join(lbl for _, apps in blocks for _, lbl in apps)
                 + r" \\ \hline")
    else:
        L.append(r"Metrics & Methods & "
                 + " & ".join(lbl for _, apps in blocks for _, lbl in apps) + r" \\ \hline")

    missing = []
    for mlabel, column, scale_m, better, fmt in METRICS:
        L.append(r"\multirow{%d}{*}{%s} " % (len(METHODS), mlabel))

        cells = {}
        for model, _ in METHODS:
            for ds, apps in blocks:
                for app, _ in apps:
                    v = means.get((ds, model, app, column))
                    cells[(model, ds, app)] = None if v is None else scale_m * v
                    if v is None:
                        missing.append(f"{mlabel} {model} {ds}/{app}")

        best = {}
        for ds, apps in blocks:
            for app, _ in apps:
                present = [(cells[(m, ds, app)], m) for m, _ in METHODS
                           if cells[(m, ds, app)] is not None]
                if len(present) > 1:
                    best[(ds, app)] = (min(present) if better == "min" else max(present))[1]

        base_model = METHODS[0][0]
        for model, mlab in METHODS:
            row = []
            for ds, apps in blocks:
                for app, _ in apps:
                    v = cells[(model, ds, app)]
                    t = "--" if v is None else fmt.format(v)
                    if v is not None and best.get((ds, app)) == model:
                        t = r"\textbf{%s}" % t
                    if a.delta and model != base_model and v is not None:
                        b = cells[(base_model, ds, app)]
                        if b is not None:
                            # Change vs the baseline row, in the metric's own units.
                            d = v - b
                            txt = f"({d:+.2f})"
                            if a.color and abs(d) >= 0.005:
                                # Colour by IMPROVEMENT, not sign: for MAE lower is better,
                                # so a negative delta is green.
                                good = d > 0 if better == "max" else d < 0
                                txt = r"\%s{%s}" % ("gain" if good else "loss", txt)
                            row.append(f"{t} {txt}")
                            continue
                    row.append(t)
            shade = r" \rowcolor{gray!15}" if model == highlight else ""
            L.append(f"{shade} & {mlab} & " + " & ".join(row) + r" \\")
        L.append(r" \hline")

    L.append(r"\end{tabular}")
    if scale:
        L.append(r"}")
    L.append(r"\end{table*}")
    return L, missing


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out", type=Path)
    p.add_argument("--preset", choices=sorted(PRESETS), default="sota")
    p.add_argument("--proposed", choices=sorted(PROPOSED_CHOICES), default="arm3",
                   help="which model fills the Proposed row (default: arm3 = the in-repo "
                        "3-epoch TCN + KL + Syn arm, TCN_KL_aug)")
    p.add_argument("--delta", action="store_true",
                   help="show each cell's change against the first row in brackets")
    p.add_argument("--color", action="store_true",
                   help="colour the bracketed deltas green for better, red for worse "
                        "(direction-aware: for MAE a NEGATIVE delta is the improvement)")
    p.add_argument("--datasets", nargs="+", choices=["UKDALE", "REDD"],
                   help="restrict to these datasets (default: both)")
    p.add_argument("--split", action="store_true",
                   help="emit one table per dataset instead of a single combined table")
    p.add_argument("--accuracy", choices=["SCA", "BA"], default="SCA",
                   help="SCA = plain per-timestamp accuracy; BA = balanced accuracy")
    a = p.parse_args()

    global METHODS, METRICS
    METHODS = list(PRESETS[a.preset])
    if a.preset == "sota":
        key, _ = PROPOSED_CHOICES[a.proposed]
        METHODS = [(key, r"Proposed")] + METHODS[1:]
    if a.accuracy == "BA":
        METRICS = [("BA", "BALANCED_ACCURACY", 100, "max", "{:.2f}\\%")] + METRICS[1:]

    global BLOCKS
    if a.datasets:
        BLOCKS = [b for b in BLOCKS if b[0] in a.datasets]

    means, counts = load()
    highlight = "TCN_KL_aug_curated" if a.preset == "ablation" else METHODS[0][0]

    out_lines, missing = [], []
    if a.color:
        out_lines += [
            r"% requires \usepackage{xcolor}; \rowcolor also needs \usepackage{colortbl}",
            r"\providecommand{\gain}[1]{\textcolor{green!55!black}{#1}}",
            r"\providecommand{\loss}[1]{\textcolor{red!75!black}{#1}}",
        ]
    groups = [[b] for b in BLOCKS] if a.split else [BLOCKS]
    for i, blocks in enumerate(groups):
        if a.split:
            caption, label = SPLIT_CAPTIONS[(a.preset, blocks[0][0])]
        else:
            caption, label = CAPTIONS[a.preset]
        lines, miss = render(blocks, means, caption, label, a, highlight)
        if i:
            out_lines.append("")
        out_lines += lines
        missing += miss

    text = "\n".join(out_lines)
    print(text)
    if missing:
        print("\n% MISSING CELLS: " + "; ".join(sorted(set(missing))), file=sys.stderr)
    ns = {counts.get((ds, m, app, "F1_SCORE"), 0)
          for m, _ in METHODS for ds, apps in BLOCKS for app, _ in apps}
    print(f"\n% runs per cell: {sorted(ns)}", file=sys.stderr)
    if a.out:
        a.out.parent.mkdir(parents=True, exist_ok=True)
        a.out.write_text(text + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
