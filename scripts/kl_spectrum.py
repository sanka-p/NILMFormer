"""Variance spectrum of the fitted KL (Karhunen-Loeve) basis, per dataset.

The TCN's KL front end replaces the single aggregate channel with `kl_order` projections
of a sliding `kl_order`-wide window (src/baselines/nilm/tcn_kl.py:KLFilter). Those
projections are the eigenvectors of the ensemble autocorrelation matrix R fitted by
tcn_ablation.fit_kl_basis. The fit returns eigenvalues alongside the basis, but
run_one_expe.py discards them -- so how much signal variance each of the 10 channels
actually carries has never been written down. This script recovers it.

Each segment is standardized before accumulation, so diag(R) ~= 1 and trace(R) ~= order.
The share of variance on channel k is lambda_k / sum(lambda), readable directly as "what
fraction of an activation window this channel sees".

Reads the cached activation segments under results/tcn_aug_cache/ -- the same pickles
run_one_expe.py fits from -- so no .h5 is touched and the numbers are exactly the ones
the trained arms used.
"""

import argparse
import pickle
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.helpers.tcn_ablation import fit_kl_basis  # noqa: E402

CACHE = Path("results/tcn_aug_cache")

#: The cell each reported appliance is scored at. REDD splits the training houses PER
#: APPLIANCE (configs/datasets.yaml:60-86), so each appliance fits its basis from a
#: different segment pool; UK-DALE uses one split (h1,3,4,5) for every appliance, so its
#: spectrum varies only with window size.
CANONICAL = {
    "REDD": [
        ("Fridge",         "REDD_h3-5-6_10s_w256"),
        ("Microwave",      "REDD_h3-5_10s_w128"),
        ("Dishwasher",     "REDD_h3-4-5-6_10s_w512"),
        ("WashingMachine", "REDD_h3-5-6_10s_w768"),
    ],
    "UKDALE": [
        ("ws 128", "UKDALE_h1-3-4-5_10s_w128"),
        ("ws 256", "UKDALE_h1-3-4-5_10s_w256"),
        ("ws 512", "UKDALE_h1-3-4-5_10s_w512"),
    ],
}


def load_segments(path):
    with open(path, "rb") as f:
        cached = pickle.load(f)
    return cached[0]


def spectrum(segments, order):
    pooled = [seg for segs in segments.values() for seg in segs]
    vals, basis = fit_kl_basis(pooled, order=order)
    vals = np.asarray(vals, dtype=np.float64)
    total = vals.sum()
    return vals, vals / total, np.cumsum(vals) / total, basis, len(pooled)


def _eff_channels(share):
    p = share[share > 0]
    return float(np.exp(-(p * np.log(p)).sum()))


def canonical_tables(order, seed):
    """Markdown tables of per-channel variance share, one per dataset."""
    for dataset, cells in CANONICAL.items():
        cols, meta = [], []
        for label, key in cells:
            path = CACHE / f"{key}_s{seed}.pkl"
            if not path.is_file():
                print("  !! missing {}, skipping {}".format(path, label))
                continue
            vals, share, cum, _, n_seg = spectrum(load_segments(path), order)
            cols.append((label, share, cum))
            meta.append((label, key, n_seg, _eff_channels(share)))
        if not cols:
            continue

        print("")
        print("### {}".format(dataset))
        print("")
        print("| channel | " + " | ".join(c[0] for c in cols) + " |")
        print("|---" * (len(cols) + 1) + "|")
        for k in range(order):
            print("| {} | ".format(k)
                  + " | ".join("{:.2f}%".format(100 * c[1][k]) for c in cols) + " |")
        print("| **cum. ch 0-1** | "
              + " | ".join("**{:.2f}%**".format(100 * c[2][1]) for c in cols) + " |")
        print("| **cum. ch 0-2** | "
              + " | ".join("**{:.2f}%**".format(100 * c[2][2]) for c in cols) + " |")
        print("| eff. channels | "
              + " | ".join("{:.2f}".format(m[3]) for m in meta) + " |")
        print("| segments | " + " | ".join(str(m[2]) for m in meta) + " |")
        print("")
        for m in meta:
            print("  {}: {}".format(m[0], m[1]))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--order", type=int, default=10)
    ap.add_argument("--seed", type=int, default=0,
                    help="which cached seed to read (REDD segments are seed-independent; "
                         "UK-DALE applies a seeded 80/20 split)")
    ap.add_argument("--canonical", action="store_true",
                    help="emit the two markdown tables (REDD per appliance, UK-DALE per "
                         "window) instead of the full per-cache dump")
    ap.add_argument("--out", default="results/kl_spectrum.csv")
    args = ap.parse_args()

    if args.canonical:
        canonical_tables(args.order, args.seed)
        return

    keys = sorted({p.name.rsplit("_s", 1)[0] for p in CACHE.glob("*.pkl")})
    rows = []
    for key in keys:
        path = CACHE / "{}_s{}.pkl".format(key, args.seed)
        if not path.is_file():
            continue
        segs = load_segments(path)
        vals, share, cum, basis, n_seg = spectrum(segs, args.order)

        dataset = key.split("_")[0]
        print("")
        print("=== {} ===  {} activation segments, {} appliances, trace(R) = {:.4f}"
              .format(key, n_seg, len(segs), vals.sum()))
        print("  ch   eigenvalue   share    cumulative")
        for k in range(args.order):
            print("  {:<3d}  {:10.4f}  {:6.2f}%  {:8.2f}%"
                  .format(k, vals[k], 100 * share[k], 100 * cum[k]))
            rows.append((dataset, key, n_seg, k, vals[k], share[k], cum[k]))
        print("  participation ratio (effective # channels) = {:.2f}"
              .format(_eff_channels(share)))

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        f.write("dataset,cache_key,n_segments,channel,eigenvalue,share,cumulative\n")
        for r in rows:
            f.write("{},{},{},{},{:.6f},{:.6f},{:.6f}\n".format(*r))
    print("")
    print("wrote {} ({} rows)".format(out, len(rows)))


if __name__ == "__main__":
    main()
