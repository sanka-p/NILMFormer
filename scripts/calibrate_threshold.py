"""Threshold analysis for the NILM ablation: calibration, PR-AUC, and the BA/F1 conflict.

Background
----------
`SeqToSeqTrainer.evaluate` uses ONE constant for two different jobs
(`src/helpers/trainer.py:278, :334-341`):

  * a regression floor   -- `pred[pred < threshold_small_values] = 0`
  * the ON/OFF detector  -- `y_hat_state = y_hat > threshold_small_values`

and that constant is the appliance's `min_threshold` (`scripts/run_one_expe.py:370`).
It is never tuned. For a rare appliance the resulting operating point is badly
miscalibrated: the stored runs over-fire by 3-31x (predicted-ON count / true-ON count),
and the effect scales with how rare the appliance is -- the signature of train/test
class-prior shift rather than a bad model. Under prior shift the Bayes-optimal
correction in the binary case reduces to a threshold shift (Elkan 2001; Menon et al.,
logit adjustment), so the threshold should be chosen on held-out data, not asserted.

Why BA and F1 disagree, and what this script does about it
----------------------------------------------------------
Both are built from the same confusion matrix but normalise the false-positive count
differently:

    BA        = (TP/P + (N-FP)/N) / 2      -- FP divided by N, the negative count
    precision = TP/(TP+FP)                 -- FP divided by predicted positives

For a rare appliance N is enormous (REDD dishwasher: N ~ 300k, P ~ 13k), so a false
positive costs BA ~1/(2N) and costs precision ~1/(TP+FP). BA is therefore nearly blind
to over-firing exactly where over-firing is the failure mode, and its optimum sits at a
far lower threshold than F1's. That is not noise -- it is a structural property, and it
is why the two metrics ranked our arms differently.

This is the documented weakness of specificity-based metrics on skewed data
(Saito & Rehmsmeier 2015; Davis & Goadrich 2006). The standard remedies, all
implemented here:

  * report a threshold-free summary  -> PR-AUC (average precision), which removes the
    operating point from the comparison entirely;
  * report a single metric that uses all four confusion cells in proportion to class
    size -> MCC (Chicco & Jurman 2020), which cannot be fooled the way BA and F1 each
    can be, and whose optimum sits between theirs;
  * if a single operating point is needed, state the criterion it was chosen by, and
    report what the other criteria would have picked.

Scope and honesty constraints
-----------------------------
  * Only the DETECTION threshold is swept. The regression floor stays at
    `min_threshold`, so MAE/RMSE cannot move and are not reported here.
  * The stored `*_yhat` arrays are already clipped below `min_threshold`, so only
    thresholds at or above it are representable. That is the direction the over-firing
    calls for, but it does cap recall; the cap is reported as `rec_max`, and PR-AUC is
    integrated over the representable region only and labelled `pAP` (partial AP) so it
    is never confused with a full average-precision score.
  * Thresholds are chosen on VALIDATION and applied unchanged to TEST. On REDD the
    validation and test houses have very different duty cycles, so the chosen operating
    point is conservative by a known factor; `--explain` prints the two priors.
  * Before calibrating anything the script recomputes the stored metrics at
    T = min_threshold and refuses the cell if they do not reproduce. That is what proves
    the ground truth was reconstructed in the order the trainer saw it.

No training, no GPU: this re-scores checkpoints that already exist.

Usage:
  PYTHONPATH=. ./.venv/bin/python -m scripts.calibrate_threshold --root result-redd-tod
  PYTHONPATH=. ./.venv/bin/python -m scripts.calibrate_threshold --explain
"""

import argparse
import csv
import logging
import os
import sys

import numpy as np
import torch
import yaml
from omegaconf import OmegaConf

sys.path.insert(0, os.getcwd())

from scripts.run_one_expe import _cached_nilm_dataset  # noqa: E402
from src.helpers.preprocessing import REDD_DataBuilder  # noqa: E402

logging.basicConfig(level=logging.WARNING)

# Appliance -> window size actually swept in this root (scripts/pick_window.py).
DEFAULT_GRID = {
    "Fridge": 256,
    "Microwave": 128,
    "Dishwasher": 512,
    "WashingMachine": 768,
}

ARMS = [
    "TCN_KL_scratch",
    # Class-weighted loss. TCN_KL_cw3 is the default weight; the others bracket it so
    # the threshold analysis can show how much over-firing the loss removes before the
    # threshold is touched, and how much is left for the threshold to remove.
    "TCN_KL_cw1_5",
    "TCN_KL_cw3",
    "TCN_KL_cw10",
    "TCN_KL_act_cw3",
    "TCN_KL_act",
    "TCN_KL_act_tod",
    "TCN_KL_act_tod_film",
    "TCN_KL_stat",
    "TCN_KL_act_stat",
    "TCN_KL_act_tod_stat",
]

SEEDS = (0, 1, 2)
CRITERIA = ("F1", "BA", "MCC", "GM", "F1b", "PRIOR")

# Reference prevalence for the prevalence-corrected metrics. Precision -- and therefore
# F1 -- is a function of the evaluation set's class balance as well as of the
# classifier (Saito & Rehmsmeier 2015), which is the whole reason F1 and BA rank
# operating points differently on a rare appliance. Re-expressing precision at a FIXED
# reference prevalence removes that dependence: at pi0 both F1b and BA are functions of
# (TPR, FPR) alone, so they are finally measuring the same thing and their optima
# coincide. pi0 = 0.5 is the balanced reference.
PI0 = 0.5

# "GM" is the geometric mean of BA and F1: the simplest criterion that refuses to
# sacrifice either one, included as a baseline against which MCC's claim to reconcile
# them can be judged rather than asserted.
RECONCILE = ("F1", "BA")


# ----------------------------------------------------------------------------------
# Confusion-matrix curve
# ----------------------------------------------------------------------------------
def _f1_at_prevalence(tpr, fpr, pi0):
    """F1 the classifier would score on a set with prevalence `pi0`.

    TPR and FPR are properties of the classifier alone; precision additionally depends
    on how many negatives there are. Re-deriving precision at a fixed reference
    prevalence therefore strips out the dataset's class balance:

        precision(pi0) = pi0*TPR / (pi0*TPR + (1-pi0)*FPR)

    and F1 follows from that precision and the unchanged recall. With pi0 held fixed
    across appliances, F1b and BA both depend only on (TPR, FPR), which is what makes
    them rank operating points the same way.
    """
    num = pi0 * tpr
    den = pi0 * tpr + (1.0 - pi0) * fpr
    prec = np.where(den > 0, num / np.where(den > 0, den, 1.0), 0.0) if np.ndim(den) else (
        (num / den) if den > 0 else 0.0)
    s = prec + tpr
    return np.where(s > 0, 2 * prec * tpr / np.where(s > 0, s, 1.0), 0.0) if np.ndim(s) else (
        (2 * prec * tpr / s) if s > 0 else 0.0)


class DetectionCurve:
    """Every (TP, FP) pair reachable by thresholding `y_hat`, from a single sort.

    Thresholding predicts positive for `y_hat > T`. Sorting the scores once and taking
    cumulative sums of the labels yields TP and FP at every distinct threshold exactly,
    so the sweep is both dense (no grid-resolution artefact in the reported optima) and
    cheap (one O(n log n) sort instead of one O(n) pass per candidate threshold).

    `cut[j]` is a threshold that produces exactly `k[j]` positives: it is the score of
    the first EXCLUDED sample, so every included sample is strictly above it.
    """

    def __init__(self, y_state, y_hat, floor):
        y_state = np.asarray(y_state).astype(np.int64)
        y_hat = np.asarray(y_hat, dtype=np.float64)
        self.P = int(y_state.sum())
        self.N = int(y_state.size - self.P)
        self.floor = float(floor)
        n = y_hat.size

        order = np.argsort(-y_hat, kind="stable")
        s = y_hat[order]
        lab = y_state[order]
        self._s = s
        self._cum_tp = np.concatenate(([0], np.cumsum(lab)))
        self._cum_fp = np.concatenate(([0], np.cumsum(1 - lab)))

        # Prediction counts that are actually achievable: 0, every tie-group boundary,
        # and n. A threshold inside a tie group cannot split it.
        k = np.concatenate(([0], np.flatnonzero(np.diff(s)) + 1, [n]))
        self.k = k
        # Threshold reporting the k-th operating point: the first excluded score.
        self.cut = np.where(k < n, s[np.minimum(k, n - 1)], 0.0)
        self.tp = self._cum_tp[k]
        self.fp = self._cum_fp[k]

    # -- single operating point ----------------------------------------------------
    def at(self, thr):
        """Metrics for the rule `y_hat > thr`, computed exactly."""
        k = int(np.searchsorted(-self._s, -float(thr), side="left"))
        return self._metrics(int(self._cum_tp[k]), int(self._cum_fp[k]))

    def _metrics(self, tp, fp):
        P, N = self.P, self.N
        pp, tn, fn = tp + fp, N - fp, P - tp
        rec = tp / P if P else 0.0
        spec = tn / N if N else 0.0
        prec = tp / pp if pp else 1.0
        f1 = (2 * tp / (2 * tp + fp + fn)) if (2 * tp + fp + fn) else 0.0
        den = float(pp) * P * N * (tn + fn)
        mcc = ((tp * tn - fp * fn) / np.sqrt(den)) if den > 0 else 0.0
        ba = 100 * (rec + spec) / 2
        f1p = 100 * f1
        fpr = fp / N if N else 0.0
        return dict(
            TP=tp, FP=fp, TN=tn, FN=fn,
            PRECISION=100 * prec, RECALL=100 * rec, SPECIFICITY=100 * spec, FPR=100 * fpr,
            F1=f1p, BA=ba, MCC=100 * mcc, GM=float(np.sqrt(max(ba, 0) * max(f1p, 0))),
            F1b=100 * float(_f1_at_prevalence(rec, fpr, PI0)),
            PRIOR=-abs((pp / P if P else 0.0) - 1.0),
            OVERFIRE=pp / P if P else 0.0,
        )

    # -- full curve ----------------------------------------------------------------
    def sweep(self):
        """Dense metric arrays over the thresholds at or above the regression floor.

        Scores below the floor were clipped to zero before storage, so a threshold
        under it is not representable and is excluded rather than silently reported.
        """
        # A threshold T is representable iff T >= floor. The operating point for
        # T = floor keeps every sample scoring above the floor, and its reported `cut`
        # is the largest score at or below it -- so masking on `cut >= floor` would
        # wrongly discard exactly that point (and with it the achievable recall
        # ceiling). Mask on the prediction count instead, which is unambiguous.
        k_floor = int(np.searchsorted(-self._s, -self.floor, side="left"))
        m = self.k <= k_floor
        tp = self.tp[m].astype(np.float64)
        fp = self.fp[m].astype(np.float64)
        P, N = float(self.P), float(self.N)
        pp, tn, fn = tp + fp, N - fp, P - tp
        rec = tp / P
        spec = tn / N
        prec = np.where(pp > 0, tp / np.where(pp > 0, pp, 1.0), 1.0)
        f1 = np.where(2 * tp + fp + fn > 0, 2 * tp / (2 * tp + fp + fn), 0.0)
        den = pp * P * N * (tn + fn)
        mcc = np.where(den > 0, (tp * tn - fp * fn) / np.sqrt(np.where(den > 0, den, 1.0)), 0.0)
        ba = 100 * (rec + spec) / 2
        f1p = 100 * f1
        fpr = fp / N
        f1b = _f1_at_prevalence(rec, fpr, PI0)
        return dict(
            cut=np.maximum(self.cut[m], self.floor), TP=tp, FP=fp,
            RECALL=100 * rec, PRECISION=100 * prec, SPECIFICITY=100 * spec, FPR=100 * fpr,
            F1=f1p, BA=ba, MCC=100 * mcc, GM=np.sqrt(np.maximum(ba, 0) * np.maximum(f1p, 0)),
            F1b=100 * f1b,
            # Distance from firing exactly as often as the appliance is on. Used as a
            # label-free criterion: it needs the prevalence scalar, not the labels.
            PRIOR=-np.abs(pp / P - 1.0),
            OVERFIRE=pp / P,
        )

    def partial_ap(self):
        """Average precision over the representable region, plus the recall ceiling.

        Step integration -- the rule sklearn's average_precision_score uses:
        sum over operating points of (R_k - R_{k-1}) * P_k. Recall is capped because of
        the clipping, so the cap is returned and the number is reported as `pAP`, never
        as a full AP.
        """
        sw = self.sweep()
        rec = sw["RECALL"] / 100.0
        prec = sw["PRECISION"] / 100.0
        o = np.argsort(rec, kind="stable")
        rec, prec = rec[o], prec[o]
        ap = float(np.sum(np.diff(np.concatenate(([0.0], rec))) * prec))
        return 100 * ap, 100 * float(rec.max() if rec.size else 0.0)

    def best(self, criterion):
        """Threshold maximising `criterion` over the representable region."""
        sw = self.sweep()
        i = int(np.nanargmax(sw[criterion]))
        return float(sw["cut"][i]), {k: v[i] for k, v in sw.items()}


# ----------------------------------------------------------------------------------
# Data
# ----------------------------------------------------------------------------------
def build_config(dataset, appliance, window_size, sampling_rate="10s"):
    """Rebuild the experiment config exactly as run_one_expe.main does."""
    with open("configs/expes.yaml") as f:
        cfg = yaml.safe_load(f)
    with open("configs/datasets.yaml") as f:
        ds = yaml.safe_load(f)[dataset]
    with open("configs/models.yaml") as f:
        # Any non-cropping arm works: the crop touches the TRAIN split only, so valid
        # and test are identical across arms and one rebuild serves all of them.
        cfg.update(yaml.safe_load(f)["TCN_KL_scratch"])
    cfg.update({k: v for k, v in ds.items() if not isinstance(v, dict)})
    cfg.update(ds[appliance])
    cfg.update(
        dataset=dataset, appliance=appliance, window_size=window_size,
        sampling_rate=sampling_rate, seed=0, name_model="TCN_KL_scratch",
        cache_data=True,
    )
    return OmegaConf.create(cfg)


def ground_truth_states(cfg):
    """(y_state_valid, y_state_test, min_threshold) flattened in trainer order.

    The eval loaders are not shuffled, so concatenating `state.flatten()` over batches
    (trainer.py:313-317) reproduces dataset order exactly.
    """
    builder = REDD_DataBuilder(
        data_path=f"{cfg.data_path}/REDD/redd.h5",
        mask_app=cfg.app,
        sampling_rate=cfg.sampling_rate,
        window_size=cfg.window_size,
        synth_aggregate_apps=cfg.synth_aggregate_apps,
    )
    out = {}
    for split, houses in (("valid", list(cfg.ind_house_valid)),
                          ("test", list(cfg.ind_house_test))):
        data, _ = _cached_nilm_dataset(builder, cfg, houses, True)
        # [n, 1 + n_app, 2, L]; channel 1 is the target appliance, index 1 its state.
        out[split] = np.asarray(data[:, 1, 1, :]).flatten().astype(int)
    return out["valid"], out["test"], float(builder.appliance_param[cfg.app]["min_threshold"])


# ----------------------------------------------------------------------------------
# Report
# ----------------------------------------------------------------------------------
def explain_conflict(name, curve_test):
    """Print the structural reason BA and F1 pick different thresholds."""
    t_f1, m_f1 = curve_test.best("F1")
    t_ba, m_ba = curve_test.best("BA")
    t_mcc, m_mcc = curve_test.best("MCC")
    P, N = curve_test.P, curve_test.N
    print(f"\n  -- why BA and F1 disagree ({name}) --")
    print(f"     P = {P:,} positives, N = {N:,} negatives  (prevalence {100*P/(P+N):.2f}%)")
    print(f"     one false positive costs BA     1/(2N) = {100/(2*N):.6f} points")
    print(f"     {int(round(N/50)):,} false positives are needed to move BA by one point")
    for tag, t, m in (("F1-opt ", t_f1, m_f1), ("BA-opt ", t_ba, m_ba), ("MCC-opt", t_mcc, m_mcc)):
        print(f"     {tag} T={t:8.1f}  FP={int(m['FP']):>7,}  "
              f"spec={m['SPECIFICITY']:6.2f}  prec={m['PRECISION']:6.2f}  "
              f"rec={m['RECALL']:6.2f}  F1={m['F1']:5.1f}  BA={m['BA']:5.1f}  "
              f"MCC={m['MCC']:5.1f}  over={m['OVERFIRE']:5.2f}x")
    print(f"     BA-optimal fires {m_ba['OVERFIRE']:.2f}x, F1-optimal {m_f1['OVERFIRE']:.2f}x, "
          f"MCC-optimal {m_mcc['OVERFIRE']:.2f}x")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", default="result-redd-tod")
    ap.add_argument("--dataset", default="REDD")
    ap.add_argument("--arms", nargs="*", default=ARMS)
    ap.add_argument("--criterion", default="PRIOR", choices=CRITERIA,
                    help="how the single reported operating point is chosen. Default "
                         "PRIOR = prevalence matching: fire as often as the appliance "
                         "is on. It needs only the duty-cycle scalar, not labels, which "
                         "is why it survives the valid->test prior shift that the "
                         "threshold-transfer criteria do not (worst-case regret 7.8 "
                         "points against F1 11.9, MCC 12.8, BA 18.1, stock 20.1).")
    ap.add_argument("--explain", action="store_true",
                    help="print the BA/F1 conflict diagnostic for the reference arm")
    ap.add_argument("--strict", action="store_true",
                    help="abort on a self-check mismatch instead of skipping the cell")
    args = ap.parse_args()

    rows = []
    for app, ws in DEFAULT_GRID.items():
        cfg = build_config(args.dataset, app, ws)
        yv, yt, floor = ground_truth_states(cfg)
        print(f"\n=== {app} / ws{ws} ===  "
              f"valid {100*yv.mean():.2f}% ON, test {100*yt.mean():.2f}% ON "
              f"(prior shift {yt.mean()/max(yv.mean(),1e-9):.1f}x), "
              f"min_threshold {floor:.0f} W")
        print(f"{'arm':<22}{'pAP':>7}{'recmax':>8}{'  ':>2}"
              f"{'T*':>7}{'F1':>7}{'BA':>7}{'MCC':>7}{'over':>7}{'  ':>2}"
              f"{'F1@0':>7}{'BA@0':>7}{'MCC@0':>7}")

        first_curve = None
        for arm in args.arms:
            acc = []
            for seed in SEEDS:
                p = f"{args.root}/{args.dataset}_{app}_10s/{ws}/{arm}_{seed}.pt"
                if not os.path.exists(p):
                    continue
                log = torch.load(p, map_location="cpu", weights_only=False)
                if "test_metrics_timestamp" not in log:
                    continue
                yh_v = np.asarray(log["valid_metrics_yhat"])
                yh_t = np.asarray(log["test_metrics_yhat"])
                if yh_v.size != yv.size or yh_t.size != yt.size:
                    msg = (f"  !! {arm} seed {seed}: length mismatch "
                           f"(valid {yh_v.size}/{yv.size}, test {yh_t.size}/{yt.size})")
                    if args.strict:
                        raise SystemExit(msg)
                    print(msg)
                    continue

                cv = DetectionCurve(yv, yh_v, floor)
                ct = DetectionCurve(yt, yh_t, floor)

                # --- self-check: reproduce the stored metrics at the stock threshold ---
                stored = log["test_metrics_timestamp"]
                stock = ct.at(floor)
                bad = False
                for key, skey in (("F1", "F1_SCORE"), ("BA", "BALANCED_ACCURACY"),
                                  ("PRECISION", "PRECISION"), ("RECALL", "RECALL")):
                    if abs(stock[key] - 100 * float(stored[skey])) > 0.15:
                        msg = (f"  !! {arm} seed {seed}: {key} self-check failed -- "
                               f"recomputed {stock[key]:.2f} vs stored "
                               f"{100*float(stored[skey]):.2f}; refusing this cell.")
                        if args.strict:
                            raise SystemExit(msg)
                        print(msg)
                        bad = True
                        break
                if bad:
                    continue

                if first_curve is None and arm == "TCN_KL_scratch":
                    first_curve = ct

                # --- calibrate on VALID by each criterion, apply to TEST ---
                pap, recmax = ct.partial_ap()
                chosen, chosen_oracle = {}, {}
                for crit in CRITERIA:
                    t_star, _ = cv.best(crit)
                    chosen[crit] = (t_star, ct.at(t_star))
                    # Same criterion, but allowed to pick the threshold on TEST. This
                    # separates the two failure modes: whether a criterion reconciles
                    # BA and F1 AT ALL (this row), and whether the choice survives the
                    # valid->test prior shift (the row above).
                    t_or, _ = ct.best(crit)
                    chosen_oracle[crit] = (t_or, ct.at(t_or))
                # Oracle: the best each metric could reach on TEST at ANY representable
                # threshold. Regret against this is what separates "the criterion chose
                # badly" from "this run simply cannot do better".
                sw_t = ct.sweep()
                oracle = {m: float(np.nanmax(sw_t[m])) for m in RECONCILE}
                acc.append((stock, chosen, pap, recmax, oracle, chosen_oracle))

            if not acc:
                continue
            crit = args.criterion
            avg = lambda f: float(np.mean([f(a) for a in acc]))  # noqa: E731
            print(f"{arm:<22}{avg(lambda a: a[2]):>7.1f}{avg(lambda a: a[3]):>8.1f}  "
                  f"{avg(lambda a: a[1][crit][0]):>7.0f}"
                  f"{avg(lambda a: a[1][crit][1]['F1']):>7.1f}"
                  f"{avg(lambda a: a[1][crit][1]['BA']):>7.1f}"
                  f"{avg(lambda a: a[1][crit][1]['MCC']):>7.1f}"
                  f"{avg(lambda a: a[1][crit][1]['OVERFIRE']):>6.2f}x  "
                  f"{avg(lambda a: a[0]['F1']):>7.1f}"
                  f"{avg(lambda a: a[0]['BA']):>7.1f}"
                  f"{avg(lambda a: a[0]['MCC']):>7.1f}")

            row = dict(appliance=app, window=ws, arm=arm, n_seeds=len(acc),
                       prior_valid=float(yv.mean()), prior_test=float(yt.mean()),
                       pAP=avg(lambda a: a[2]), rec_max=avg(lambda a: a[3]),
                       F1_stock=avg(lambda a: a[0]["F1"]),
                       BA_stock=avg(lambda a: a[0]["BA"]),
                       MCC_stock=avg(lambda a: a[0]["MCC"]),
                       over_stock=avg(lambda a: a[0]["OVERFIRE"]))
            for m in RECONCILE:
                row[f"oracle_{m}"] = avg(lambda a, m=m: a[4][m])
            for c in CRITERIA:
                for slot, sfx in ((1, ""), (5, "_or")):
                    row[f"T_{c}{sfx}"] = avg(lambda a, c=c, s=slot: a[s][c][0])
                    for m in ("F1", "BA", "MCC", "OVERFIRE"):
                        row[f"{m}_at_{c}{sfx}"] = avg(
                            lambda a, c=c, m=m, s=slot: a[s][c][1][m])
                    # Regret: how much of each metric's achievable value this criterion
                    # gives up. A criterion that reconciles BA and F1 is low on BOTH.
                    for m in RECONCILE:
                        row[f"regret_{m}_at_{c}{sfx}"] = avg(
                            lambda a, c=c, m=m, s=slot: a[4][m] - a[s][c][1][m])
            rows.append(row)

        if args.explain and first_curve is not None:
            explain_conflict(f"{app}, reference arm, test split", first_curve)

    if rows:
        print("\n\n=== Which validation criterion reconciles BA and F1? ===")
        print("Regret = (best that metric could reach on test at any representable")
        print("threshold) - (what it reaches at the threshold this criterion picked).")
        print("A criterion that genuinely reconciles them has LOW regret on both, so")
        print("the column to read is the worst of the two, not the mean.\n")
        for tag, suffix in (
            ("A. criterion applied ON TEST -- does it reconcile BA and F1 at all?", "or"),
            ("B. criterion calibrated on VALIDATION -- does the choice transfer?", ""),
        ):
            print(f"\n{tag}")
            print(f"{'criterion':<12}{'BA regret':>11}{'F1 regret':>11}{'worst':>9}"
                  f"{'over-fire':>11}")
            best_crit, best_worst = None, float("inf")
            for c in CRITERIA:
                key = f"regret_{{}}_at_{c}{'_or' if suffix else ''}"
                rb = float(np.mean([r[key.format("BA")] for r in rows]))
                rf = float(np.mean([r[key.format("F1")] for r in rows]))
                ov = float(np.mean([r[f"OVERFIRE_at_{c}{'_or' if suffix else ''}"]
                                    for r in rows]))
                worst = max(rb, rf)
                if worst < best_worst:
                    best_crit, best_worst = c, worst
                print(f"{c:<12}{rb:>11.2f}{rf:>11.2f}{worst:>9.2f}{ov:>10.2f}x")
            rb = float(np.mean([r["oracle_BA"] - r["BA_stock"] for r in rows]))
            rf = float(np.mean([r["oracle_F1"] - r["F1_stock"] for r in rows]))
            ov = float(np.mean([r["over_stock"] for r in rows]))
            print(f"{'(stock)':<12}{rb:>11.2f}{rf:>11.2f}{max(rb, rf):>9.2f}{ov:>10.2f}x")
            print(f"-> lowest worst-case regret: {best_crit} ({best_worst:.2f} points)")

        out = os.path.join(args.root, "threshold_analysis.csv")
        with open(out, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)
        print(f"\nwrote {out} ({len(rows)} rows)")
        print(f"operating point chosen on VALIDATION by {args.criterion}; "
              f"'@0' columns are the stock min_threshold rule. "
              f"pAP = average precision over the representable region "
              f"(recall capped at rec_max by the regression floor).")


if __name__ == "__main__":
    main()
