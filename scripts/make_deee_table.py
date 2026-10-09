#################################################################################################################
#
# @description : Tables for the DEEE_SmartHome cross-domain inference.
#
# Renders results/deee_metrics.csv and results/deee_specificity.csv in the same style as
# scripts/make_ablation_table.py, with a side-by-side UK-DALE reference column pulled from
# results/runs_cache.csv so the generalisation gap is visible in one place.
#
# A caveats footer is printed INTO the output file rather than left to a README: several of
# these numbers cannot be read correctly without it (two appliances have identically zero
# ground truth under the as-trained thresholds, and the aggregate is synthetic).
#
# Usage:
#     PYTHONPATH=. .venv/bin/python -m scripts.make_deee_table
#     PYTHONPATH=. .venv/bin/python -m scripts.make_deee_table --out results/deee_table.txt
#
#################################################################################################################

import argparse
import csv
import statistics as st
import sys
from collections import defaultdict
from pathlib import Path

DEFAULT_METRICS = Path("results/deee_metrics.csv")
DEFAULT_SPECIFICITY = Path("results/deee_specificity.csv")
DEFAULT_CACHE = Path("results/runs_cache.csv")
UKDALE_REF_MODEL = "TCN_KL_aug_curated"

APPS = [("Kettle", "Kettle"), ("Microwave", "Micro."), ("WashingMachine", "WM")]
# (label, column, scale, direction, format)
METRICS = [
    ("SCA", "ACCURACY", 100, "max", "{:.2f}%"),
    ("BA", "BALANCED_ACCURACY", 100, "max", "{:.2f}%"),
    ("F1", "F1_SCORE", 100, "max", "{:.2f}%"),
    ("MAE", "MAE", 1, "min", "{:.2f}"),
]
SPEC_METRICS = [
    ("FP rate", "FP_RATE", 100, "{:.3f}%"),
    ("Phantom energy", "PHANTOM_FRAC", 100, "{:.2f}%"),
    ("Mean pred", "MEAN_PRED_W", 1, "{:.2f} W"),
    ("False activations", "N_FALSE_ACTIVATIONS", 1, "{:.0f}"),
]
LABEL_W = 30
CELL_W = 18


def read_rows(path):
    if not path.is_file():
        return []
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def ukdale_reference(cache_path):
    """{appliance: {column: mean}} for TCN_KL_aug_curated on UK-DALE, as already reported."""
    if not cache_path.is_file():
        return {}
    vals = defaultdict(lambda: defaultdict(list))
    seen = set()
    with open(cache_path, newline="") as f:
        for r in csv.DictReader(f):
            if r.get("Dataset") != "UKDALE" or r.get("Model") != UKDALE_REF_MODEL:
                continue
            if r.get("SamplingRate") != "10s":
                continue
            key = (r["Appliance"], r["WindowSize"], r["Seed"])
            if key in seen:
                continue
            seen.add(key)
            for _, col, _, _, _ in METRICS:
                v = r.get(col)
                if v not in (None, ""):
                    vals[r["Appliance"]][col].append(float(v))
    return {app: {c: st.mean(v) for c, v in per.items()} for app, per in vals.items()}


def agg(rows, keyfn):
    """{(key, appliance, column): [values]}"""
    out = defaultdict(list)
    for r in rows:
        for _, col, _, _, _ in METRICS:
            v = r.get(col)
            if v not in (None, ""):
                out[(keyfn(r), r["Appliance"], col)].append(float(v))
    return out


def fmt_cell(values, scale, fmt):
    """"mean ±sd" over the runs in a cell, or "--" when the cell has none."""
    if not values:
        return "--"
    mean = scale * st.mean(values)
    sd = scale * (st.stdev(values) if len(values) > 1 else 0.0)
    return f"{fmt.format(mean)}±{sd:.2f}"


def render_block(rows, title, keyfn, key_order, ref, degenerate):
    out = [title, ""]
    for label, col, scale, direction, fmt in METRICS:
        head = f"  {label:<{LABEL_W}}" + "".join(f"{s:>{CELL_W}}" for _, s in APPS)
        out += [head, "  " + "-" * (len(head) - 2)]

        for key in key_order:
            line = f"  {key:<{LABEL_W}}"
            for app, _ in APPS:
                vals = rows.get((key, app, col), [])
                cell = fmt_cell(vals, scale, fmt)
                if (app, key) in degenerate:
                    cell = "n/a"
                line += f"{cell:>{CELL_W}}"
            out.append(line)

        if ref:
            line = f"  {'UK-DALE reference':<{LABEL_W}}"
            for app, _ in APPS:
                v = ref.get(app, {}).get(col)
                line += f"{(fmt.format(scale * v) if v is not None else '--'):>{CELL_W}}"
            out.append(line)
        out.append("")
    return out


def render_specificity(rows):
    """False-positive behaviour, broken out by aggregate scope.

    This is the comparison the scope experiment exists for: `all` puts nine appliance types
    the UK-DALE models have never seen into the mains, `ukdale` puts in only the three they
    were trained on. Any drop from one to the other is hallucination caused by
    out-of-vocabulary load rather than by the target appliances.
    """
    if not rows:
        return []
    apps = sorted({r["Appliance"] for r in rows})
    scopes = sorted({r.get("Scope", "all") for r in rows})

    out = [
        "SPECIFICITY -- appliances that do not exist in DEEE_SmartHome",
        "",
        "  Neither a fridge nor a dishwasher was recorded, so ground truth is identically",
        "  zero and classification metrics are undefined. What is measured instead is how",
        "  much of a house the model invents. Phantom energy is the share of TOTAL measured",
        "  energy the model assigns to an appliance that is not there.",
        "",
        "  scope 'all'    = all 12 DEEE categories in the aggregate (9 of them are loads the",
        "                   UK-DALE models were never trained on)",
        "  scope 'ukdale' = only the 3 categories the models know (kettle, microwave,",
        "                   washing machine)",
        "",
    ]
    variants = sorted({r.get("Variant", "pure") for r in rows})

    def mean_of(col, a, sc, v):
        vals = [float(r[col]) for r in rows
                if r["Appliance"] == a and r.get("Scope", "all") == sc
                and r.get("Variant", "pure") == v and r.get(col) not in (None, "")]
        return st.mean(vals) if vals else None

    # Broken out by variant as well as scope: averaging over the base-load variants hides
    # the dominant effect, which is that an 80 W constant floor is itself read as a fridge.
    for v in variants:
        out += [f"  base-load variant: {v}", ""]
        head = f"    {'measure':<{LABEL_W}}" + "".join(
            f"{a + ' [' + sc + ']':>{CELL_W}}" for a in apps for sc in scopes
        )
        out += [head, "    " + "-" * (len(head) - 4)]
        for label, col, scale, fmt in SPEC_METRICS:
            line = f"    {label:<{LABEL_W}}"
            for a in apps:
                for sc in scopes:
                    m = mean_of(col, a, sc, v)
                    line += f"{(fmt.format(scale * m) if m is not None else '--'):>{CELL_W}}"
            out.append(line)
        out.append("")

    if len(scopes) > 1:
        out += ["  Effect of removing the 9 out-of-vocabulary categories:", ""]
        for v in variants:
            for a in apps:
                p_all, p_uk = mean_of("MEAN_PRED_W", a, "all", v), mean_of("MEAN_PRED_W", a, "ukdale", v)
                f_all, f_uk = mean_of("FP_RATE", a, "all", v), mean_of("FP_RATE", a, "ukdale", v)
                if None in (p_all, p_uk, f_all, f_uk):
                    continue
                delta = (p_uk - p_all) / p_all * 100 if p_all else float("nan")
                out.append(
                    f"    {v:<9s} {a:<12s} phantom power {p_all:7.2f} W -> {p_uk:6.2f} W "
                    f"({delta:+6.1f}%)   FP {100*f_all:6.2f}% -> {100*f_uk:6.2f}%"
                )
        out += [
            "",
            "  Read phantom-energy PERCENTAGES with care: the ukdale-scope mains averages",
            "  10.6 W (pure) against 162.3 W for all 12 categories, so a share of total",
            "  energy rises even when the absolute invented power falls. Mean predicted",
            "  watts is the figure that is comparable across scopes.",
        ]
    return out + [""]


def caveats(rows, spec_rows):
    """Printed into the file: several numbers above are unreadable without these."""
    baseload = sorted({(r["Variant"], r["BaseloadW"], r["NoiseSigmaW"])
                       for r in rows + spec_rows if "Variant" in r})
    reuse = sorted({r["ReuseFactor"] for r in rows + spec_rows if "ReuseFactor" in r})
    out = [
        "CAVEATS (these numbers cannot be read correctly without them)",
        "",
        "  1. THE AGGREGATE IS SYNTHETIC. DEEE_SmartHome has no mains channel -- each trace",
        "     is one appliance recorded alone, by design, so that an aggregate could be",
        "     built later. Loads were placed on a synthetic timeline; the three scored",
        "     appliances at their MEASURED UK-DALE house-2 duty cycles (kettle 0.590%,",
        "     microwave 0.442%, washing machine 1.119%), the nine distractor categories at",
        "     ASSUMED household duty cycles (see deee_aggregate.DISTRACTOR_DUTY). The",
        "     `ukdale` scope drops those nine entirely, leaving a much quieter mains",
        "     (mean 10.6 W vs 162.3 W) in which the three known appliances are the only",
        "     load -- an easier but far less realistic test.",
        "",
        "  2. TWO APPLIANCES HAVE NO GROUND TRUTH AS-TRAINED, shown as n/a:",
        "     * Kettle: UK-DALE's threshold is 2000 W for a ~3 kW UK element. The DEEE",
        "       kettles are 230 V units peaking at 1370 and 1341 W, so every sample labels",
        "       OFF. The model cannot detect them by construction, not by failure.",
        "     * WashingMachine: UK-DALE requires 1800 s of continuous ON. The Singer",
        "       top-loader's longest above-threshold run is 730 s (7 runs, gaps up to 940 s),",
        "       so no activation survives. This is a duration prior, not an amplitude one.",
        "     The `adapted` regime corrects exactly these two priors (kettle threshold 500 W,",
        "     the repo's own non-Kelly constant; washing machine min_on_duration 180 s) so",
        "     the waveform question can still be asked. Adapted and as-trained numbers are",
        "     NOT comparable to each other -- only as-trained is comparable to UK-DALE.",
        "",
        "  3. GROUND TRUTH AND PREDICTION ARE LABELLED ASYMMETRICALLY. GT passes through",
        "     _compute_status (duration filtering); predicted state is a bare y_hat >",
        "     threshold (trainer.py:341). This is the repo's existing behaviour, applied to",
        "     UK-DALE too, but it is what drives F1 to 0 when GT is empty.",
        "",
        "  4. THE WASHING MACHINE IS NOT THE SAME KIND OF APPLIANCE. The Singer is a",
        "     cold-fill top-loader: 677 W peak, 103 W mean. UK-DALE washing machines carry a",
        "     ~2 kW heating element and the model keys on it. A low score is substantially an",
        "     appliance-population difference, not a model failure.",
        "",
        "  5. SOURCE DATA IS 6.11 HOURS IN 35 DISJOINT SESSIONS. Effective diversity is the",
        "     number of distinct activations -- washing machine 1, kettle 2, microwave 5 --",
        "     not the number of timesteps. Stream length is not independent evidence.",
    ]
    if reuse:
        out.append(f"     Reuse factor (synthetic length / real samples): {', '.join(reuse)}x.")
    out += [
        "",
        "  6. BASE-LOAD VARIANTS. A noiseless sum of clean traces has no standby draw and no",
        "     meter noise, both of which make real disaggregation harder, so `pure` is an",
        "     optimistic bound. Injected values:",
    ]
    for variant, bl, ns in baseload:
        out.append(f"     * {variant:<9s} base load {float(bl):.1f} W, Gaussian noise sigma {float(ns):.1f} W")
    out += [
        "",
        "  7. MODE. `session` places whole recorded sessions (primary). `remix` uses the",
        "     crop-and-remix generator that produced this model's TRAINING data, so it",
        "     structurally resembles what the model was trained on and flatters it; any",
        "     remix-over-session gap is that augmentation-matching bonus, not generalisation.",
        "",
        "  8. ws=512 NEVER CONTAINS A COMPLETE APPLIANCE CYCLE (longest DEEE session is 368",
        "     samples at 10 s), so those models are disadvantaged for an unrelated reason.",
    ]
    return out


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--metrics", type=Path, default=DEFAULT_METRICS)
    p.add_argument("--specificity", type=Path, default=DEFAULT_SPECIFICITY)
    p.add_argument("--cache", type=Path, default=DEFAULT_CACHE)
    p.add_argument("--out", type=Path)
    a = p.parse_args()

    rows = read_rows(a.metrics)
    spec_rows = read_rows(a.specificity)
    if not rows and not spec_rows:
        sys.exit(f"no results in {a.metrics} -- run scripts/run_deee_inference.py first")

    ref = ukdale_reference(a.cache)
    degenerate = {
        (r["Appliance"], f"{r['Mode']}/{r['Scope']}/{r['Variant']}/{r['Regime']}")
        for r in rows if r.get("DEGENERATE_GT") == "1"
    }

    keyfn = lambda r: f"{r['Mode']}/{r['Scope']}/{r['Variant']}/{r['Regime']}"
    key_order = sorted({keyfn(r) for r in rows})
    grouped = agg(rows, keyfn)

    n_runs = len({(r["Appliance"], r["WindowSize"], r["Seed"]) for r in rows})
    lines = [
        "CROSS-DOMAIN INFERENCE -- UK-DALE-trained TCN_KL_aug_curated on DEEE_SmartHome",
        f"10s, mean ± sd over 9 runs/cell (3 window sizes x 3 seeds); {n_runs} runs per arm",
        "",
    ]
    lines += render_block(grouped, "ACCURACY BY AGGREGATE MODE / BASE LOAD / THRESHOLD REGIME",
                          keyfn, key_order, ref, degenerate)
    lines += render_specificity(spec_rows)
    lines += caveats(rows, spec_rows)

    text = "\n".join(lines)
    print(text)
    if a.out:
        a.out.parent.mkdir(parents=True, exist_ok=True)
        a.out.write_text(text + "\n")
        print(f"\nwritten to {a.out}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
