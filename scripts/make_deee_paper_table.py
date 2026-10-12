#################################################################################################################
#
# @description : Paper tables for the DEEE_SmartHome cross-domain evaluation.
#
# Condenses results/deee_metrics.csv and deee_specificity.csv into two LaTeX tables:
#   1. accuracy on the three appliances DEEE shares with UK-DALE, across the four
#      aggregate conditions (which loads are in the mains x whether a base load is added)
#   2. specificity -- what the Fridge and Dishwasher models invent on a house that has
#      neither
#
# Only the `session` aggregate mode is tabulated. `remix` is the robustness check and is
# reported in text: it uses the same crop-and-remix generator that produced this model's
# TRAINING data, so it structurally resembles what the model saw and flatters it.
#
# Only the `adapted` threshold regime appears for Kettle and WM: under the as-trained
# UK-DALE thresholds both have identically zero ground truth (kettle peaks 1370 W vs a
# 2000 W threshold; the Singer's longest ON run is 730 s vs an 1800 s minimum), so those
# cells are undefined rather than poor.
#
# Usage:
#     PYTHONPATH=. .venv/bin/python -m scripts.make_deee_paper_table --out results/deee_paper_tables.tex
#
#################################################################################################################

import argparse
import csv
import statistics as st
import sys
from collections import defaultdict
from pathlib import Path

METRICS_CSV = Path("results/deee_metrics.csv")
SPEC_CSV = Path("results/deee_specificity.csv")
CACHE = Path("results/runs_cache.csv")

APPS = [("Kettle", "Kettle"), ("Microwave", "Micro."), ("WashingMachine", "WM"),
        ("Fridge", "Fridge")]
#: (scope, variant) -> printed row label, in the order a reader should walk them.
CONDITIONS = [
    ("all", "pure", r"All 12 loads"),
    ("all", "baseload", r"All 12 loads + base load"),
    ("ukdale", "pure", r"UK-DALE loads only"),
    ("ukdale", "baseload", r"UK-DALE loads only + base load"),
]
METRICS = [("BA", "BALANCED_ACCURACY", 100, "{:.2f}\\%"),
           ("F1", "F1_SCORE", 100, "{:.2f}\\%"),
           ("MAE", "MAE", 1, "{:.2f}")]
SPEC_ROWS = [("FP rate", "FP_RATE", 100, "{:.2f}\\%"),
             ("Phantom energy share", "PHANTOM_FRAC", 100, "{:.2f}\\%"),
             ("Mean invented power", "MEAN_PRED_W", 1, "{:.2f} W")]


def load(path):
    return list(csv.DictReader(open(path, newline=""))) if path.is_file() else []


def ukdale_ref():
    """TCN_KL_aug_curated on UK-DALE -- the in-domain number each cell is measured against."""
    v = defaultdict(list)
    seen = set()
    for r in csv.DictReader(open(CACHE, newline="")):
        if (r.get("Dataset"), r.get("Model"), r.get("SamplingRate")) != (
            "UKDALE", "TCN_KL_aug_curated", "10s"):
            continue
        k = (r["Appliance"], r["WindowSize"], r["Seed"])
        if k in seen:
            continue
        seen.add(k)
        for _, col, _, _ in METRICS:
            if r.get(col) not in (None, ""):
                v[(r["Appliance"], col)].append(float(r[col]))
    return {k: st.mean(x) for k, x in v.items()}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out", type=Path)
    p.add_argument("--mode", default="session")
    p.add_argument("--regime", default="adapted")
    a = p.parse_args()

    rows, spec = load(METRICS_CSV), load(SPEC_CSV)
    if not rows:
        sys.exit(f"{METRICS_CSV} not found -- run scripts/run_deee_inference.py first")
    ref = ukdale_ref()

    vals = defaultdict(list)
    for r in rows:
        if r["Mode"] != a.mode or r["Regime"] != a.regime:
            continue
        for _, col, _, _ in METRICS:
            if r.get(col) not in (None, ""):
                vals[(r["Scope"], r["Variant"], r["Appliance"], col)].append(float(r[col]))

    L = ["% Table 1: cross-domain accuracy",
         r"\begin{table*}[!t]", r"\centering",
         r"\caption{Cross-domain inference on DEEE\_SmartHome: UK-DALE-trained "
         r"TCN + KL + Syn applied without retraining. Mean $\pm$ sd over 9 runs "
         r"(3 window sizes $\times$ 3 seeds).}",
         r"\label{table:deee_crossdomain}",
         r"\setlength{\tabcolsep}{5pt}\small",
         r"\begin{tabular}{c|l|cccc}", r"\hline",
         r"Metric & Aggregate composition & Kettle & Micro. & WM & Fridge \\ \hline"]

    for mlabel, col, scale, fmt in METRICS:
        L.append(r"\multirow{%d}{*}{%s} " % (len(CONDITIONS) + 1, mlabel))
        for scope, variant, label in CONDITIONS:
            cells = []
            for app, _ in APPS:
                v = vals.get((scope, variant, app, col), [])
                if not v:
                    cells.append("--")
                    continue
                sd = st.stdev(v) if len(v) > 1 else 0.0
                cells.append(f"{fmt.format(scale * st.mean(v))}$\\pm${scale * sd:.2f}")
            L.append(f" & {label} & " + " & ".join(cells) + r" \\")
        refc = [fmt.format(scale * ref[(app, col)]) if (app, col) in ref else "--"
                for app, _ in APPS]
        L.append(r" & \textit{UK-DALE (in-domain)} & " +
                 " & ".join(r"\textit{%s}" % c for c in refc) + r" \\")
        L.append(r" \hline")
    L += [r"\end{tabular}", r"\end{table*}", ""]

    # ---- specificity ----
    if spec:
        sv = defaultdict(list)
        for r in spec:
            if r["Mode"] != a.mode:
                continue
            for _, col, _, _ in SPEC_ROWS:
                if r.get(col) not in (None, ""):
                    sv[(r["Scope"], r["Variant"], r["Appliance"], col)].append(float(r[col]))
        L += ["% Table 2: specificity",
              r"\begin{table}[!t]", r"\centering",
              r"\caption{Specificity: what the Fridge and Dishwasher models report on a "
              r"house containing neither. Phantom energy is the share of total measured "
              r"energy assigned to an absent appliance.}",
              r"\label{table:deee_specificity}",
              r"\setlength{\tabcolsep}{5pt}\small",
              r"\begin{tabular}{l|l|c}", r"\hline",
              r"Measure & Aggregate composition & Dishwasher \\ \hline"]
        for slabel, col, scale, fmt in SPEC_ROWS:
            L.append(r"\multirow{%d}{*}{%s} " % (len(CONDITIONS), slabel))
            for scope, variant, label in CONDITIONS:
                cells = []
                for app in ("Dishwasher",):
                    v = sv.get((scope, variant, app, col), [])
                    cells.append(fmt.format(scale * st.mean(v)) if v else "--")
                L.append(f" & {label} & " + " & ".join(cells) + r" \\")
            L.append(r" \hline")
        L += [r"\end{tabular}", r"\end{table}"]

    text = "\n".join(L)
    print(text)
    if a.out:
        a.out.parent.mkdir(parents=True, exist_ok=True)
        a.out.write_text(text + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
