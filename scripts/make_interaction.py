#################################################################################################################
#
# @description : KL x Curation interaction term, from the completed 2x2 of the TCN ablation.
#
# The ladder (A -> B -> D) reports each ingredient conditional on the ones below it. With
# variant C -- curated synthetic data WITHOUT the KL front end -- the design closes into a
# full 2x2 and the interaction becomes computable:
#
#       interaction = (D - C) - (B - A)
#
# i.e. how much more KL is worth on curated synthetic data than on real data. A term near
# zero means the two ingredients are additive and the ladder's A->B->D decomposition is
# unconfounded; a large term either way means it is not.
#
#   A = TCN                   (neither)       C = TCN_aug_curated        (curated only)
#   B = TCN_KL_scratch        (KL only)       D = TCN_KL_aug_curated     (both)
#
# Reads results/runs_cache.csv; run scripts/make_table.py first when new results exist.
#
# Usage:
#     PYTHONPATH=. .venv/bin/python -m scripts.make_interaction --dataset UKDALE
#
#################################################################################################################

import argparse
import csv
import statistics as st
import sys
from collections import defaultdict
from pathlib import Path

DEFAULT_CACHE = Path("results/runs_cache.csv")

CELLS = [("A", "TCN"), ("B", "TCN_KL_scratch"),
         ("C", "TCN_aug_curated"), ("D", "TCN_KL_aug_curated")]

APPS_BY_DATASET = {
    "UKDALE": [("WashingMachine", "WM"), ("Dishwasher", "DW"), ("Kettle", "Kettle"),
               ("Microwave", "Micro."), ("Fridge", "Fridge")],
    "REDD": [("WasherDryer", "WD"), ("Dishwasher", "DW"),
             ("Microwave", "Micro."), ("Fridge", "Fridge")],
}
# (label, cache column, scale)
METRICS = [("F1", "F1_SCORE", 100), ("BA", "BALANCED_ACCURACY", 100)]


def load(cache_path, dataset, sampling_rate):
    """{(model, appliance): mean over the 9 runs} per metric column."""
    if not cache_path.is_file():
        sys.exit(f"cache not found: {cache_path} -- run scripts/make_table.py first")
    raw = defaultdict(list)
    with open(cache_path, newline="") as f:
        for r in csv.DictReader(f):
            if r.get("Dataset") != dataset or r.get("SamplingRate") != sampling_rate:
                continue
            for _, column, _ in METRICS:
                v = r.get(column)
                if v not in (None, ""):
                    raw[(r["Model"], r["Appliance"], column)].append(float(v))
    return {k: st.mean(v) for k, v in raw.items() if v}


def render(means, dataset, sampling_rate):
    apps = APPS_BY_DATASET[dataset]
    out = [f"KL x CURATION INTERACTION -- {dataset}, {sampling_rate}, mean of 9 runs/cell",
           "",
           "  A = TCN   B = TCN+KL   C = TCN+Cur+Aug   D = TCN+KL+Cur+Aug",
           "  KL on real data    = B - A        KL on curated data = D - C",
           "  interaction        = (D - C) - (B - A)",
           ""]
    for label, column, scale in METRICS:
        w = 24
        head = f"  {label:<{w}}" + "".join(f"{s:>10}" for _, s in apps) + f"{'mean':>10}"
        out += [head, "  " + "-" * (len(head) - 2)]

        vals = {}
        missing = False
        for key, model in CELLS:
            row = [means.get((model, app, column)) for app, _ in apps]
            if any(v is None for v in row):
                missing = True
            vals[key] = row

        def line(name, row):
            cells = "".join(f"{v:>10.2f}" if v is not None else f"{'--':>10}" for v in row)
            present = [v for v in row if v is not None]
            mean = f"{st.mean(present):>10.2f}" if present else f"{'--':>10}"
            return f"  {name:<{w}}{cells}{mean}"

        for key, model in CELLS:
            out.append(line(f"{key}  {model}", [scale * v if v is not None else None
                                                for v in vals[key]]))
        out.append("")

        def diff(x, y):
            return [scale * (a - b) if a is not None and b is not None else None
                    for a, b in zip(vals[x], vals[y])]

        kl_real, kl_cur = diff("B", "A"), diff("D", "C")
        inter = [a - b if a is not None and b is not None else None
                 for a, b in zip(kl_cur, kl_real)]
        out.append(line("KL on real   (B-A)", kl_real))
        out.append(line("KL on curated (D-C)", kl_cur))
        out.append(line("interaction", inter))
        if missing:
            out.append("  (some cells missing -- means are over the available appliances only)")
        out.append("")
    out += [
        "  A positive interaction means KL and curated synthetic data are complementary --",
        "  KL is worth more on curated data than on real data -- and a negative one means",
        "  they are partly redundant. A term near zero means the two are additive, so the",
        "  A->B->D ladder reports each ingredient's effect without confounding.",
    ]
    return "\n".join(out)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--cache", type=Path, default=DEFAULT_CACHE)
    p.add_argument("--dataset", default="UKDALE", choices=sorted(APPS_BY_DATASET))
    p.add_argument("--sampling-rate", default="10s")
    p.add_argument("--out", type=Path)
    a = p.parse_args()

    text = render(load(a.cache, a.dataset, a.sampling_rate), a.dataset, a.sampling_rate)
    print(text)
    if a.out:
        a.out.parent.mkdir(parents=True, exist_ok=True)
        a.out.write_text(text + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
