"""Estimate the TARGET house's ON prevalence from unlabelled predictions.

WHY THIS EXISTS
---------------
Prevalence-matched thresholding is the single largest scoring gain measured in this
project (WashingMachine F1 45.6 -> 65.7), but it needs one number the deployment site
does not hand over: the target appliance's duty cycle. Every table produced with
`--prior test` therefore carries an assumption, not a result. `--prior valid` is honest
but pays for it, because REDD's validation and test houses disagree on ON rate by
1.3x-7.3x (see `redd-split-duty-cycle-mismatch` and scripts/score_postprocess.py).

The label-shift literature says that scalar is recoverable from UNLABELLED target data:

  * Saerens, Latinne & Decaestecker (2002) -- EM on calibrated source posteriors
    (a.k.a. MLLS). Maximises the target likelihood under the label-shift assumption
    p(x|y) invariant, p(y) free.
  * Lipton, Wang & Smola (2018) -- BBSE: invert the source confusion matrix against
    the target's PREDICTED label distribution. In the binary hard-label case this
    reduces exactly to the Rogan & Gladen (1978) prevalence correction

        pi_hat = (q_t - FPR) / (TPR - FPR)

    where TPR/FPR are measured on the labelled validation split and q_t is the
    predicted-positive rate on the unlabelled target.

Both use target SCORES only, never target labels, so a threshold tuned to `pi_hat` is
label-free in a way `--prior test` is not.

WHAT IS ACTUALLY BEING TESTED
-----------------------------
Both estimators assume label shift: p(x | y) is the same in both houses, only p(y)
moves. Across REDD houses that is FALSE in detail -- a different physical washing
machine has a different power signature -- so TPR/FPR do not transfer exactly and the
estimates will be biased. The size of that bias is the measurement. The script reports
it directly (pi_hat vs the true target rate) AND the only thing that matters
downstream: the F1 obtained when the estimate is used in place of the oracle.

The anchor threshold for BBSE maximises Youden's J = TPR - FPR on validation, because
the estimator divides by J: its variance scales as 1/J^2 and the inversion is
ill-conditioned as J -> 0. J is reported so a reader can see how well-posed each
inversion was.

Usage:
  PYTHONPATH=. ./.venv/bin/python -m scripts.estimate_prior --root result-redd-tod
"""
from __future__ import annotations

import argparse
import csv
import os
import sys

import numpy as np
import torch

from src.helpers.postprocess import durations_for, postprocess_state
from scripts.calibrate_threshold import (DEFAULT_GRID, build_config,
                                         ground_truth_states)
from scripts.score_postprocess import (SEEDS, _pick_threshold, _timestamp_metrics)

#: Bins for the score -> P(ON) calibration map that EM consumes. Binning rather than a
#: full isotonic fit over ~3e5 points keeps this O(n log n) with a tiny constant; 200
#: quantile bins is far finer than the posterior's actual resolution.
N_BINS = 200
EM_ITERS = 2000
EM_TOL = 1e-10
#: Below this Youden's J the confusion-matrix inversion is refused. The estimator
#: divides by J, so an error in the predicted positive rate is amplified by 1/J.
#: MEASURED: WashingMachine reaches only J = 0.17 on REDD's validation house and its
#: estimate came out 19x too high (38.9% against a true 2.07%); every appliance whose
#: estimate was merely BIASED rather than absurd had J >= 0.70.
#:
#: This guard catches VARIANCE, not bias, and must not be read as a validity test:
#: Dishwasher had a well-conditioned J = 0.76 and still over-estimated by 2.5x,
#: because the failure there is the label-shift ASSUMPTION (p(x|y) invariant across
#: houses), which no function of the source confusion matrix can detect.
J_MIN = 0.50


# ----------------------------------------------------------------------------------
# Confusion-matrix inversion (BBSE-hard / Rogan-Gladen)
# ----------------------------------------------------------------------------------
def _roc(y, yh, floor):
    """(thresholds, TPR, FPR) over every representable threshold >= `floor`.

    Mirrors scripts/calibrate_threshold.DetectionCurve (same single-sort construction
    and the same `k <= k_floor` representability mask, which is correct where a
    `cut >= floor` mask would wrongly drop the recall ceiling); re-stated here because
    the threshold array itself is needed, not just the metric optima.
    """
    order = np.argsort(-yh, kind="stable")
    s = yh[order]
    lab = y[order].astype(np.int64)
    n = s.size
    ctp = np.concatenate(([0], np.cumsum(lab)))
    cfp = np.concatenate(([0], np.cumsum(1 - lab)))
    k = np.concatenate(([0], np.flatnonzero(np.diff(s)) + 1, [n]))
    cut = np.where(k < n, s[np.minimum(k, n - 1)], 0.0)
    k_floor = int(np.searchsorted(-s, -float(floor), side="left"))
    m = k <= k_floor
    P = max(int(lab.sum()), 1)
    N = max(int(n - lab.sum()), 1)
    return np.maximum(cut[m], floor), ctp[k][m] / P, cfp[k][m] / N


def _anchor(y_valid, yh_v, floor):
    """Validation threshold maximising Youden's J, with its TPR/FPR."""
    thr, tpr, fpr = _roc(y_valid, yh_v, floor)
    i = int(np.argmax(tpr - fpr))
    return float(thr[i]), float(tpr[i]), float(fpr[i])


def _bbse(q_t, tpr, fpr):
    """Rogan-Gladen / BBSE-hard prior estimate, clipped to a valid prevalence."""
    j = tpr - fpr
    if j < J_MIN:
        return float("nan"), j
    return float(np.clip((q_t - fpr) / j, 0.0, 1.0)), j


# ----------------------------------------------------------------------------------
# EM on calibrated posteriors (Saerens-Latinne-Decaestecker 2002 / MLLS)
# ----------------------------------------------------------------------------------
def _pava(p, w):
    """Non-decreasing weighted least-squares fit of `p` (PAVA), one value per input."""
    val, wt, cnt = [], [], []          # pooled mean, pooled weight, inputs in the block
    for pi, wi in zip(p, w):
        val.append(float(pi)); wt.append(float(wi)); cnt.append(1)
        while len(val) > 1 and val[-2] > val[-1]:
            tot = wt[-2] + wt[-1]
            val[-2] = (val[-2] * wt[-2] + val[-1] * wt[-1]) / tot
            wt[-2] = tot
            cnt[-2] += cnt[-1]
            val.pop(); wt.pop(); cnt.pop()
    return np.repeat(np.asarray(val), np.asarray(cnt))


def _calibrator(y_valid, yh_v):
    """Fit score -> P(ON) on validation; return a function mapping scores to posteriors.

    Quantile-binned empirical ON rate, made monotone by PAVA. Binning rather than a
    full isotonic fit over ~3e5 points keeps the cost negligible, and 200 bins is far
    finer than the posterior's real resolution. Monotonicity is imposed because it is a
    property we are willing to assume of the regressor (a higher predicted wattage
    should not mean a lower chance of being ON) and because it stops sparse high-score
    bins from producing non-monotone noise that EM would then amplify.

    EM needs posteriors calibrated on the SOURCE domain, so the map is fitted on
    validation and merely applied to the target scores.
    """
    edges = np.unique(np.quantile(yh_v, np.linspace(0.0, 1.0, N_BINS + 1)))
    if edges.size < 2:                      # degenerate: every score identical
        base = float(y_valid.mean())
        return lambda s: np.full(np.shape(s), base)
    nb = edges.size - 1
    idx = np.clip(np.searchsorted(edges, yh_v, side="right") - 1, 0, nb - 1)
    cnt = np.bincount(idx, minlength=nb).astype(float)
    pos = np.bincount(idx, weights=y_valid.astype(float), minlength=nb)
    keep = np.flatnonzero(cnt > 0)
    rate = pos[keep] / cnt[keep]
    # Empty bins are filled by interpolating between the occupied ones; PAVA's output
    # is non-decreasing, so the interpolation is too.
    fitted = np.interp(np.arange(nb), keep, _pava(rate, cnt[keep]))

    def f(s):
        i = np.clip(np.searchsorted(edges, s, side="right") - 1, 0, nb - 1)
        return fitted[i]
    return f


def _em(post_t, pi_src):
    """Saerens et al. (2002) EM for the target prior from source-calibrated posteriors."""
    eps = 1e-9
    pi_src = float(np.clip(pi_src, eps, 1 - eps))
    p = np.clip(np.asarray(post_t, dtype=np.float64), eps, 1 - eps)
    pi = pi_src
    for _ in range(EM_ITERS):
        a = (pi / pi_src) * p
        b = ((1 - pi) / (1 - pi_src)) * (1 - p)
        new = float(np.mean(a / (a + b)))
        if abs(new - pi) < EM_TOL:
            pi = new
            break
        pi = new
    return float(np.clip(pi, 0.0, 1.0))


# ----------------------------------------------------------------------------------
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--root", default="result-redd-tod")
    p.add_argument("--arms", nargs="+",
                   default=["TCN_KL_scratch", "TCN_KL_cw3", "TCN_KL_act"])
    p.add_argument("--postprocess", action="store_true",
                   help="also score with Kelly duration rules, threshold re-tuned "
                        "under the same smoothing")
    a = p.parse_args()

    rows = []
    for app, ws in DEFAULT_GRID.items():
        cfg = build_config("REDD", app, ws)
        y_valid, y_test, floor = ground_truth_states(cfg)
        dur = durations_for(app) if a.postprocess else None
        pi_v, pi_true = float(y_valid.mean()), float(y_test.mean())
        print(f"\n=== {app} / ws{ws} ===  valid {100*pi_v:.2f}% / test {100*pi_true:.2f}% ON"
              f"  (shift {pi_true/pi_v:.2f}x), floor {floor:.0f} W")
        print(f"{'arm':<18}{'piValid':>8}{'piBBSE':>8}{'piEM':>8}{'piTrue':>8}{'J':>6}"
              f"{'|errBBSE':>9}{'errEM':>7}{'| F1val':>8}{'F1BBSE':>8}{'F1EM':>8}{'F1orac':>8}")
        for arm in a.arms:
            acc = {}
            for seed in SEEDS:
                for key in ({app, "WasherDryer"} if app == "WashingMachine" else {app}):
                    f = os.path.join(a.root, f"REDD_{key}_10s", str(ws),
                                     f"{arm}_{seed}.pt")
                    if os.path.exists(f + ".done"):
                        break
                else:
                    continue
                log = torch.load(f, map_location="cpu", weights_only=False)
                yh_v = np.asarray(log["valid_metrics_yhat"], dtype=np.float64).ravel()
                yh_t = np.asarray(log["test_metrics_yhat"], dtype=np.float64).ravel()
                if yh_v.size != y_valid.size or yh_t.size != y_test.size:
                    print(f"  {arm} seed {seed}: size mismatch, skipping", file=sys.stderr)
                    continue

                # --- estimate the target prior, using target SCORES only -------------
                t_anchor, tpr, fpr = _anchor(y_valid, yh_v, floor)
                pred_v = (yh_v > t_anchor).astype(np.int8)
                pred_t = (yh_t > t_anchor).astype(np.int8)
                if dur is not None:
                    # Post-processing is part of the classifier, so TPR/FPR and q_t are
                    # all measured after it; pi_t itself is unchanged by our smoothing.
                    pred_v = postprocess_state(pred_v, **dur)
                    pred_t = postprocess_state(pred_t, **dur)
                    tpr = float(np.mean(pred_v[y_valid == 1]))
                    fpr = float(np.mean(pred_v[y_valid == 0]))
                pi_bbse, j = _bbse(float(pred_t.mean()), tpr, fpr)
                pi_em = _em(_calibrator(y_valid, yh_v)(yh_t), pi_v)

                # --- what each prior buys downstream --------------------------------
                # The threshold is read off the TARGET scores for every estimated
                # prior (transductive but label-free); the "valid" baseline reads it
                # off the validation scores, which is the only honest option when no
                # target prior is available at all.
                res = {}
                res["F1val"] = _timestamp_metrics(
                    y_test, _state(yh_t, _pick_threshold(yh_v, pi_v, floor, dur), dur))["F1"]
                for tag, pi in (("F1BBSE", pi_bbse), ("F1EM", pi_em), ("F1orac", pi_true)):
                    if not np.isfinite(pi):
                        res[tag] = float("nan")
                        continue
                    res[tag] = _timestamp_metrics(
                        y_test, _state(yh_t, _pick_threshold(yh_t, pi, floor, dur), dur))["F1"]
                res.update(piBBSE=100 * pi_bbse, piEM=100 * pi_em, J=j)
                for k, v in res.items():
                    acc.setdefault(k, []).append(v)
            if not acc:
                continue
            m = {k: float(np.nanmean(v)) for k, v in acc.items()}
            eb, ee = m["piBBSE"] - 100 * pi_true, m["piEM"] - 100 * pi_true
            print(f"{arm:<18}{100*pi_v:>8.2f}{m['piBBSE']:>8.2f}{m['piEM']:>8.2f}"
                  f"{100*pi_true:>8.2f}{m['J']:>6.2f}{eb:>+9.2f}{ee:>+7.2f}"
                  f"{m['F1val']:>8.1f}{m['F1BBSE']:>8.1f}{m['F1EM']:>8.1f}{m['F1orac']:>8.1f}")
            rows.append(dict(appliance=app, ws=ws, arm=arm, postprocess=bool(dur),
                             pi_valid=100 * pi_v, pi_true=100 * pi_true, **m))

    if rows:
        out = os.path.join(a.root, "prior_estimation%s.csv" % ("_pp" if a.postprocess else ""))
        with open(out, "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(rows[0]))
            w.writeheader(); w.writerows(rows)
        print(f"\nwrote {out} ({len(rows)} rows)")
        print("piBBSE = confusion-matrix inversion (Lipton 2018 / Rogan-Gladen 1978); "
              "piEM = Saerens et al. 2002 EM on calibrated posteriors.\n"
              "Both use UNLABELLED target scores only. F1orac is the --prior test "
              "oracle; F1val uses no target information at all. The gap F1orac - F1val "
              "is what a prior estimate has to close.")
    return 0


def _state(yh, thr, dur):
    s = (yh > thr).astype(np.int8)
    return postprocess_state(s, **dur) if dur is not None else s


if __name__ == "__main__":
    raise SystemExit(main())
