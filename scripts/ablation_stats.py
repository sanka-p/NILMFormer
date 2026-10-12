#################################################################################################################
#
# @description : Significance tests behind the paper's "Component Contributions" subsection.
#
# The ablation is a 2x2 over the KLT front end and the synthetic training data:
#
#                    real training data        synthetic training data
#   no KLT           A = TCN                   C = TCN_aug_curated
#   KLT              B = TCN_KL_scratch        D = TCN_KL_aug_curated  (--d-arm curated)
#                                              D = TCN_KL_aug          (--d-arm arm3)
#
# The synthetic arms REPLACE the real training windows with an equal number of generated
# ones (run_one_expe.py:250, aug_mode=replace, aug_ratio=1.0), so A->C and B->D are
# train-on-synthetic vs train-on-real comparisons, not "with and without extra data".
#
# Two D arms exist and they are NOT interchangeable: TCN_KL_aug_curated draws activations
# from the curated store, TCN_KL_aug from this repo's own training houses. Both are
# reported, because the draft's quoted numbers match one and the repo's "Proposed"
# convention names the other.
#
# Pairing unit is (Appliance, WindowSize, Seed): each arm has exactly one run per cell, so
# every contrast is a paired comparison over 45 cells (UK-DALE) or 36 (REDD).
#
# Everything is reported in PERCENTAGE POINTS for BA and F1 (the cache stores fractions)
# and in WATTS for MAE. SIGN CONVENTION: every difference is `second arm - first arm`, so
# for BA and F1 a POSITIVE difference is an improvement, and for MAE a NEGATIVE one is.
#
# Usage:
#     PYTHONPATH=. ./env/bin/python -m scripts.ablation_stats
#     PYTHONPATH=. ./env/bin/python -m scripts.ablation_stats --dataset UKDALE --margin 5
#
#################################################################################################################

import argparse
import csv
import itertools
import math
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.make_table import DEFAULT_RESULT_DIRS  # noqa: F401  (documents the source dirs)

CACHE = Path("results/runs_cache.csv")
SAMPLING_RATE = "10s"
WINDOW_SIZES = ("128", "256", "512")
SEEDS = ("0", "1", "2")

#: Cache column -> (printed name, unit, scale applied to the stored value)
METRICS = {
    "BALANCED_ACCURACY": ("BA", "pp", 100.0),
    "F1_SCORE": ("F1", "pp", 100.0),
    "MAE": ("MAE", "W", 1.0),
}
#: Lower is better, so the sign of an improvement flips.
LOWER_IS_BETTER = {"MAE"}

ARMS = {"A": "TCN", "B": "TCN_KL_scratch", "C": "TCN_aug_curated"}
D_ARMS = {"curated": "TCN_KL_aug_curated", "arm3": "TCN_KL_aug"}

CONTRASTS = [
    ("A->B", "A", "B", "KLT, real data"),
    ("C->D", "C", "D", "KLT, synthetic data"),
    ("A->C", "A", "C", "synthetic replaces real, no KLT"),
    ("B->D", "B", "D", "synthetic replaces real, with KLT"),
]
#: Only the "synthetic replaces real" contrasts get an equivalence test.
EQUIV_CONTRASTS = ("A->C", "B->D")
EQUIV_METRICS = ("BALANCED_ACCURACY", "F1_SCORE")

APPLIANCES = {
    "UKDALE": ["WashingMachine", "Dishwasher", "Kettle", "Microwave", "Fridge"],
    "REDD": ["WasherDryer", "Dishwasher", "Microwave", "Fridge"],
}
#: REDD exposes the washing machine on the combined washer/dryer meter, and it is the WM
#: column of Tables IV and VI. It is one of the four REDD appliances, always included.
#: The alias keeps a differently-keyed run joining to the same cell.
REDD_ALIAS = {"WashingMachine": "WasherDryer"}


def load_runs(cache):
    """{(dataset, model, appliance, window, seed): row} from the make_table cache."""
    if not cache.is_file():
        raise SystemExit(
            f"{cache} not found. Rebuild it with:\n"
            f"    PYTHONPATH=. ./env/bin/python -m scripts.make_table"
        )
    out = {}
    with open(cache) as f:
        for r in csv.DictReader(f):
            if r["SamplingRate"] != SAMPLING_RATE:
                continue
            app = r["Appliance"]
            if r["Dataset"] == "REDD":
                app = REDD_ALIAS.get(app, app)
            key = (r["Dataset"], r["Model"], app, r["WindowSize"], r["Seed"])
            if key in out:
                continue  # first hit wins, matching make_table's dedup
            out[key] = r
    return out


def paired_vectors(runs, dataset, m1, m2, metric, appliances):
    """(x, y, missing) aligned on (appliance, window, seed)."""
    x, y, missing = [], [], []
    for app, ws, seed in itertools.product(appliances, WINDOW_SIZES, SEEDS):
        a = runs.get((dataset, m1, app, ws, seed))
        b = runs.get((dataset, m2, app, ws, seed))
        if a is None or b is None or not a[metric] or not b[metric]:
            missing.append((app, ws, seed, "missing " + (m1 if a is None else m2)))
            continue
        scale = METRICS[metric][2]
        x.append(float(a[metric]) * scale)
        y.append(float(b[metric]) * scale)
    return np.array(x), np.array(y), missing


def tost_paired(d, margin):
    """Paired TOST, implemented with scipy (statsmodels is not installed here).

    Two one-sided t-tests on the differences: H0_lower says the true difference is at or
    below -margin, H0_upper that it is at or above +margin. Equivalence is declared when
    BOTH are rejected, i.e. max(p) < alpha, which is the same thing as the 90% CI lying
    inside the margin at alpha = 0.05.
    """
    n = len(d)
    mean = float(np.mean(d))
    se = float(np.std(d, ddof=1)) / math.sqrt(n)
    if se == 0:
        return mean, 0.0, 0.0, 0.0
    t_lower = (mean + margin) / se
    t_upper = (margin - mean) / se
    p_lower = float(stats.t.sf(t_lower, n - 1))   # H0: diff <= -margin
    p_upper = float(stats.t.sf(t_upper, n - 1))   # H0: diff >= +margin
    return mean, p_lower, p_upper, max(p_lower, p_upper)


def holm(pvals):
    """Holm-Bonferroni adjusted p-values, order preserved."""
    idx = np.argsort(pvals)
    adj = np.empty(len(pvals))
    running = 0.0
    for rank, i in enumerate(idx):
        val = (len(pvals) - rank) * pvals[i]
        running = max(running, val)
        adj[i] = min(1.0, running)
    return adj


def analyse(runs, dataset, arms, appliances, margin):
    """All contrasts x metrics for one dataset, with Holm applied across the family."""
    rows = []
    for cname, lhs, rhs, desc in CONTRASTS:
        m1, m2 = arms[lhs], arms[rhs]
        for metric, (mname, unit, _) in METRICS.items():
            x, y, missing = paired_vectors(runs, dataset, m1, m2, metric, appliances)
            n = len(x)
            if n < 2:
                rows.append({"dataset": dataset, "contrast": cname,
                             "description": desc, "arm_from": m1, "arm_to": m2,
                             "metric": mname, "unit": unit, "n": n,
                             "note": "insufficient paired runs"})
                continue
            d = y - x
            mean = float(np.mean(d))
            sd = float(np.std(d, ddof=1))
            se = sd / math.sqrt(n)
            t, p = stats.ttest_rel(y, x)
            try:
                w_p = float(stats.wilcoxon(d).pvalue)
            except ValueError:                      # all-zero differences
                w_p = 1.0
            ci90 = stats.t.ppf(0.95, n - 1) * se
            ci95 = stats.t.ppf(0.975, n - 1) * se
            better = "lower" if metric in LOWER_IS_BETTER else "higher"
            improves = (mean < 0) if metric in LOWER_IS_BETTER else (mean > 0)
            row = {
                "dataset": dataset, "contrast": cname,
                "description": desc, "arm_from": m1, "arm_to": m2,
                "metric": mname, "unit": unit, "n": n,
                "mean_diff": round(mean, 4), "sd": round(sd, 4),
                "t": round(float(t), 4), "p_two_sided": float(p),
                "ci90_lo": round(mean - ci90, 4), "ci90_hi": round(mean + ci90, 4),
                "ci95_lo": round(mean - ci95, 4), "ci95_hi": round(mean + ci95, 4),
                "cohen_dz": round(mean / sd, 4) if sd > 0 else float("inf"),
                "wilcoxon_p": w_p,
                "better_is": better,
                "direction": "improves" if improves else "degrades",
                "n_missing": len(missing),
            }
            if cname in EQUIV_CONTRASTS and metric in EQUIV_METRICS:
                _, p_lo, p_hi, p_tost = tost_paired(d, margin)
                row.update({
                    "tost_margin": margin,
                    "tost_p_lower": p_lo, "tost_p_upper": p_hi, "tost_p": p_tost,
                    "equivalent": bool(p_tost < 0.05),
                })
            rows.append(row)

    live = [r for r in rows if "p_two_sided" in r]
    adj = holm([r["p_two_sided"] for r in live])
    for r, a in zip(live, adj):
        r["p_holm"] = float(a)
    return rows


def arm_means(runs, dataset, arms, appliances):
    """Per-appliance mean of each arm, for the Table VI cross-check."""
    out = defaultdict(dict)
    for key, model in arms.items():
        for app in appliances:
            for metric in METRICS:
                vals = [
                    float(runs[(dataset, model, app, ws, s)][metric]) * METRICS[metric][2]
                    for ws, s in itertools.product(WINDOW_SIZES, SEEDS)
                    if (dataset, model, app, ws, s) in runs
                    and runs[(dataset, model, app, ws, s)][metric]
                ]
                if vals:
                    out[(key, model, app)][METRICS[metric][0]] = round(float(np.mean(vals)), 2)
    return out


def fmt_p(p):
    return "<0.001" if p < 0.001 else f"{p:.3f}"


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dataset", choices=["UKDALE", "REDD", "all"], default="all")
    ap.add_argument("--margin", type=float, default=5.0,
                    help="equivalence margin in percentage points for BA and F1")
    ap.add_argument("--d-arm", choices=["curated", "arm3", "both"], default="both",
                    help="which arm fills D; 'both' reports each separately")
    ap.add_argument("--cache", type=Path, default=CACHE)
    ap.add_argument("--out-csv", type=Path, default=Path("results/ablation_stats.csv"))
    ap.add_argument("--out-txt", type=Path, default=Path("results/ablation_stats.txt"))
    a = ap.parse_args()

    runs = load_runs(a.cache)
    datasets = ["UKDALE", "REDD"] if a.dataset == "all" else [a.dataset]
    d_keys = ["curated", "arm3"] if a.d_arm == "both" else [a.d_arm]

    all_rows, lines = [], []
    lines.append("Ablation significance tests")
    lines.append("=" * 100)
    lines.append("Differences are `second arm - first arm`. BA and F1 in percentage "
                 "points, MAE in watts.")
    lines.append("For MAE lower is better, so a NEGATIVE difference is the improvement.")
    lines.append("TOST implemented with scipy (statsmodels is not installed here).")
    lines.append("")

    for d_key in d_keys:
        arms = dict(ARMS, D=D_ARMS[d_key])
        lines.append("#" * 100)
        lines.append(f"# D = {arms['D']}  ({d_key})")
        lines.append("#" * 100)
        for ds in datasets:
            apps = APPLIANCES[ds]
            expected = len(apps) * len(WINDOW_SIZES) * len(SEEDS)
            rows = analyse(runs, ds, arms, apps, a.margin)
            for r in rows:
                r["d_arm"] = d_key
            all_rows.extend(rows)

            lines.append("")
            lines.append(f"--- {ds}  [{len(apps)} appliances, expected n = {expected}] ---")
            short = [r for r in rows if r.get("n", 0) != expected]
            if short:
                lines.append(f"  !! n differs from {expected} in "
                             f"{len(short)} test(s); see n column")
            lines.append(
                f"  {'contrast':7s} {'metric':6s} {'n':>3s} {'mean':>9s} {'sd':>8s} "
                f"{'t':>7s} {'p':>8s} {'p_holm':>8s} {'wilcox':>8s} {'dz':>7s} "
                f"{'90% CI':>18s}  verdict"
            )
            for r in rows:
                if "p_two_sided" not in r:
                    lines.append(f"  {r['contrast']:7s} {r['metric']:6s} "
                                 f"{r['n']:>3d}  {r.get('note','')}")
                    continue
                sig = "sig" if r["p_holm"] < 0.05 else "ns"
                lines.append(
                    f"  {r['contrast']:7s} {r['metric']:6s} {r['n']:>3d} "
                    f"{r['mean_diff']:>9.3f} {r['sd']:>8.3f} {r['t']:>7.2f} "
                    f"{fmt_p(r['p_two_sided']):>8s} {fmt_p(r['p_holm']):>8s} "
                    f"{fmt_p(r['wilcoxon_p']):>8s} {r['cohen_dz']:>7.2f} "
                    f"[{r['ci90_lo']:>7.2f},{r['ci90_hi']:>7.2f}]  "
                    f"{sig}, {r['direction']}"
                )
            eq = [r for r in rows if "tost_p" in r]
            if eq:
                lines.append(f"  equivalence (TOST, margin +-{a.margin:g}):")
                for r in eq:
                    lines.append(
                        f"    {r['contrast']:7s} {r['metric']:6s} "
                        f"mean {r['mean_diff']:+7.3f}  90% CI "
                        f"[{r['ci90_lo']:+7.3f},{r['ci90_hi']:+7.3f}]  "
                        f"p_lower {r['tost_p_lower']:.4f}  p_upper {r['tost_p_upper']:.4f}"
                        f"  TOST p {r['tost_p']:.4f}  -> "
                        f"{'EQUIVALENT' if r['equivalent'] else 'not shown equivalent'}"
                    )

        # Per-appliance arm means, for the Table VI cross-check.
        lines.append("")
        lines.append(f"--- per-appliance arm means (D = {arms['D']}) ---")
        for ds in datasets:
            lines.append(f"  {ds}")
            means = arm_means(runs, ds, arms, APPLIANCES[ds])
            lines.append(f"    {'arm':24s} {'appliance':16s} {'BA':>8s} {'F1':>8s} {'MAE':>8s}")
            for (key, model, app), m in means.items():
                lines.append(f"    {key + ' = ' + model:24s} {app:16s} "
                             f"{m.get('BA', float('nan')):8.2f} {m.get('F1', float('nan')):8.2f} "
                             f"{m.get('MAE', float('nan')):8.2f}")
            lines.append(f"    {'-- macro means --':24s}")
            for key, model in arms.items():
                vals = {k: [] for k in ("BA", "F1", "MAE")}
                for app in APPLIANCES[ds]:
                    m = means.get((key, model, app), {})
                    for k in vals:
                        if k in m:
                            vals[k].append(m[k])
                lines.append(f"    {key + ' = ' + model:24s} {'macro':16s} "
                             f"{np.mean(vals['BA']):8.2f} {np.mean(vals['F1']):8.2f} "
                             f"{np.mean(vals['MAE']):8.2f}")
        lines.append("")

    a.out_csv.parent.mkdir(parents=True, exist_ok=True)
    fields = sorted({k for r in all_rows for k in r})
    order = ["dataset", "d_arm", "contrast", "description", "arm_from", "arm_to",
             "metric", "unit", "n", "mean_diff", "sd", "t", "p_two_sided", "p_holm",
             "wilcoxon_p", "cohen_dz", "ci90_lo", "ci90_hi", "ci95_lo", "ci95_hi",
             "better_is", "direction", "tost_margin", "tost_p_lower", "tost_p_upper",
             "tost_p", "equivalent", "n_missing", "note"]
    fields = [f for f in order if f in fields] + [f for f in fields if f not in order]
    with open(a.out_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        w.writerows(all_rows)

    text = "\n".join(lines) + "\n"
    a.out_txt.write_text(text)
    print(text)
    print(f"wrote {a.out_csv} and {a.out_txt}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
