#################################################################################################################
#
# @description : Curated activation-pool size sweep (UK-DALE, Microwave).
#
# The 9 appliance/dataset cells of the main ablation split cleanly: every curated pool of
# >=163 activations beat the real-data arm, every pool of <=44 lost to it. That brackets a
# threshold but does not locate it. This traces the curve directly by subsampling the
# UK-DALE Microwave curated pool to 20/50/100/200 activations and re-running the full
# 3x3 grid at each point. The uncapped arm (3000 activations, MAX_SEGMENTS_PER_APPLIANCE)
# is the right-hand end; "TCN + KL" -- same model, real training data -- is the flat
# reference line the curve has to cross.
#
# The caps are exact: the per-appliance cap is applied after the length filter, so a cap
# of N yields exactly N usable samplers (verified in the run logs).
#
# Reads results/runs_cache.csv; run scripts/make_table.py first when new results exist.
#
# Usage:
#     PYTHONPATH=. .venv/bin/python -m scripts.make_pool_sweep
#     PYTHONPATH=. .venv/bin/python -m scripts.make_pool_sweep --csv results/pool_sweep.csv
#
#################################################################################################################

import argparse
import csv
import statistics as st
import sys
from collections import defaultdict
from pathlib import Path

DEFAULT_CACHE = Path("results/runs_cache.csv")
DATASET = "UKDALE"
APPLIANCE = "Microwave"
SAMPLING_RATE = "10s"

#: (cache model key, pool size, printed label). None = not a curated arm; the reference row.
POINTS = [
    ("TCN_KL_scratch", None, "TCN + KL (real data)"),
    ("TCN_KL_aug_curated_n20", 20, "curated, 20 activations"),
    ("TCN_KL_aug_curated_n50", 50, "curated, 50 activations"),
    ("TCN_KL_aug_curated_n100", 100, "curated, 100 activations"),
    ("TCN_KL_aug_curated_n200", 200, "curated, 200 activations"),
    ("TCN_KL_aug_curated", 3000, "curated, 3000 (uncapped)"),
]

# (label, cache column, scale, format)
METRICS = [
    ("F1", "F1_SCORE", 100, "{:6.2f}"),
    ("BA", "BALANCED_ACCURACY", 100, "{:6.2f}"),
    ("SCA", "ACCURACY", 100, "{:6.2f}"),
    ("MAE", "MAE", 1, "{:6.2f}"),
]


def load(cache_path):
    """{model: {metric column: [per-run values]}} for the swept cell."""
    if not cache_path.is_file():
        sys.exit(f"cache not found: {cache_path} -- run scripts/make_table.py first")
    vals = defaultdict(lambda: defaultdict(list))
    with open(cache_path, newline="") as f:
        for r in csv.DictReader(f):
            if (r.get("Dataset"), r.get("Appliance"), r.get("SamplingRate")) != (
                DATASET, APPLIANCE, SAMPLING_RATE
            ):
                continue
            for _, column, _, _ in METRICS:
                raw = r.get(column)
                if raw not in (None, ""):
                    vals[r["Model"]][column].append(float(raw))
    return vals


def render(vals):
    label_w = max(len(lbl) for _, _, lbl in POINTS)
    head = f"{'arm':<{label_w}}  {'n':>5}  {'runs':>4}"
    for name, _, _, _ in METRICS:
        head += f"  {name + ' (mean+-sd)':>20}"
    out = [
        f"Curated pool-size sweep -- {DATASET} {APPLIANCE} @ {SAMPLING_RATE}",
        "3 window sizes x 3 seeds per point; +-sd is across those 9 runs.",
        "",
        head,
        "-" * len(head),
    ]
    ref = {}
    for model, n, label in POINTS:
        per = vals.get(model, {})
        runs = max((len(v) for v in per.values()), default=0)
        line = f"{label:<{label_w}}  {('--' if n is None else n):>5}  {runs:>4}"
        for _, column, scale, fmt in METRICS:
            v = per.get(column, [])
            if not v:
                line += f"  {'--':>20}"
                continue
            mean = scale * st.mean(v)
            sd = scale * (st.stdev(v) if len(v) > 1 else 0.0)
            if n is None:
                ref[column] = mean
            cell = fmt.format(mean) + " +-" + f"{sd:.2f}"
            line += f"  {cell:>20}"
        out.append(line)

    # Where the curve crosses the real-data reference is the answer the sweep exists for.
    if "F1_SCORE" in ref:
        out += ["", f"  Reference (real data) F1 = {ref['F1_SCORE']:.2f}%.  Delta vs reference:"]
        crossed = None
        for model, n, _ in POINTS:
            if n is None:
                continue
            v = vals.get(model, {}).get("F1_SCORE", [])
            if not v:
                continue
            d = 100 * st.mean(v) - ref["F1_SCORE"]
            out.append(f"    n={n:<5} {d:+6.2f}")
            if d > 0 and crossed is None:
                crossed = n
        out.append(
            f"    Curation first overtakes real data at n={crossed}."
            if crossed else
            "    Curation does not overtake real data anywhere in the swept range."
        )
    return "\n".join(out)


def write_csv(vals, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    header = ["dataset", "appliance", "sampling_rate", "model", "arm", "pool_size", "n_runs"]
    for name, _, _, _ in METRICS:
        header += [f"{name}_mean", f"{name}_std"]
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(header)
        for model, n, label in POINTS:
            per = vals.get(model, {})
            runs = max((len(v) for v in per.values()), default=0)
            if not runs:
                continue
            row = [DATASET, APPLIANCE, SAMPLING_RATE, model, label, "" if n is None else n, runs]
            for _, column, scale, _ in METRICS:
                v = per.get(column, [])
                if v:
                    row += [f"{scale * st.mean(v):.4f}",
                            f"{scale * (st.stdev(v) if len(v) > 1 else 0.0):.4f}"]
                else:
                    row += ["", ""]
            w.writerow(row)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--cache", type=Path, default=DEFAULT_CACHE)
    p.add_argument("--csv", type=Path, help="also write the sweep as a CSV")
    p.add_argument("--out", type=Path, help="also write the rendered table to this file")
    a = p.parse_args()

    vals = load(a.cache)
    text = render(vals)
    print(text)
    if a.out:
        a.out.parent.mkdir(parents=True, exist_ok=True)
        a.out.write_text(text + "\n")
    if a.csv:
        write_csv(vals, a.csv)
    return 0


if __name__ == "__main__":
    sys.exit(main())
