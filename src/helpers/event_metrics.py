"""Activation-level (event) scoring for NILM predictions.

Every metric in this repo is per-timestamp against a negative class that outnumbers the
positive one 25:1 (Fridge) to 85:1 (Microwave). That is why Balanced Accuracy is nearly
uninformative on the rare appliances: ~6,000 false positives are needed to move BA by a
single point, so a model can fire five times too often and still score 85 BA.

Event scoring asks the question a practitioner actually has -- "did it find the
activation?" -- by collapsing each maximal ON run to one object and matching predicted
objects against true ones.

MATCHING. Greedy one-to-one by overlap, highest first, subject to a minimum overlap
criterion. One-to-one matters: with many-to-one matching a single predicted ON run
spanning the whole test set "detects" every true activation and scores 100% recall at
100% precision, which is nonsense. Under one-to-one it matches exactly one true event
and the rest are misses.

OVERLAP CRITERION. Two are reported because they answer different questions:
  * `iou`      -- intersection over union of the two spans. Strict: penalises a
                  prediction that finds the activation but gets its extent wrong.
  * `coverage` -- intersection / length of the TRUE event. Lenient: credits any
                  prediction that covers enough of the real activation, regardless of
                  how much extra it predicts. This is closer to "was the appliance
                  detected" and is the usual NILM reading.
A threshold of 0 under `coverage` means "any overlap at all".

This mirrors the event-based metrics of Mesaros et al. (2016) for sound event
detection, which is the same problem shape: sparse labelled spans in a long stream.
"""
from __future__ import annotations

import numpy as np

from src.helpers.postprocess import runs

__all__ = ["match_events", "event_scores"]


def _overlap_matrix(true_ev, pred_ev, criterion):
    """(n_true, n_pred) overlap scores under `criterion`; 0 where spans are disjoint."""
    if len(true_ev) == 0 or len(pred_ev) == 0:
        return np.zeros((len(true_ev), len(pred_ev)), dtype=float)
    ts, te = true_ev[:, 0][:, None], true_ev[:, 1][:, None]
    ps, pe = pred_ev[:, 0][None, :], pred_ev[:, 1][None, :]
    inter = np.minimum(te, pe) - np.maximum(ts, ps)
    inter = np.clip(inter, 0, None).astype(float)
    if criterion == "iou":
        union = (te - ts) + (pe - ps) - inter
        with np.errstate(divide="ignore", invalid="ignore"):
            return np.where(union > 0, inter / union, 0.0)
    if criterion == "coverage":
        with np.errstate(divide="ignore", invalid="ignore"):
            return np.where((te - ts) > 0, inter / (te - ts), 0.0)
    raise ValueError(f"criterion must be 'iou' or 'coverage', got {criterion!r}")


def match_events(true_ev, pred_ev, threshold=0.0, criterion="coverage"):
    """Greedy one-to-one matching, best overlap first.

    Returns (pairs, n_true, n_pred) where `pairs` is a list of (true_idx, pred_idx).
    A pair is only formed when the overlap is strictly positive AND meets `threshold`
    (so threshold=0 means "any overlap", not "no overlap required").
    """
    true_ev = np.asarray(true_ev).reshape(-1, 2)
    pred_ev = np.asarray(pred_ev).reshape(-1, 2)
    sc = _overlap_matrix(true_ev, pred_ev, criterion)
    ok = (sc > 0) & (sc >= threshold)
    cand = np.argwhere(ok)
    if cand.size == 0:
        return [], len(true_ev), len(pred_ev)
    # Sort by score descending; ties broken by index so the result is deterministic.
    vals = sc[cand[:, 0], cand[:, 1]]
    order = np.lexsort((cand[:, 1], cand[:, 0], -vals))
    used_t, used_p, pairs = set(), set(), []
    for i in order:
        t, p = int(cand[i, 0]), int(cand[i, 1])
        if t in used_t or p in used_p:
            continue
        used_t.add(t); used_p.add(p); pairs.append((t, p))
    return pairs, len(true_ev), len(pred_ev)


def event_scores(y_true_state, y_pred_state, threshold=0.0, criterion="coverage"):
    """Event precision/recall/F1 plus the diagnostics needed to read them.

    `n_true`/`n_pred` are reported because event F1 alone hides the failure mode where
    a model emits a handful of enormous predictions: that shows up as n_pred << n_true
    with high precision, which is only legible if the counts are visible.
    """
    true_ev = runs(y_true_state, 1)
    pred_ev = runs(y_pred_state, 1)
    pairs, n_true, n_pred = match_events(true_ev, pred_ev, threshold, criterion)
    tp = len(pairs)
    prec = 100.0 * tp / n_pred if n_pred else 0.0
    rec = 100.0 * tp / n_true if n_true else 0.0
    f1 = (2 * prec * rec / (prec + rec)) if (prec + rec) > 0 else 0.0
    true_len = float(true_ev[:, 1].sum() - true_ev[:, 0].sum()) if n_true else 0.0
    pred_len = float(pred_ev[:, 1].sum() - pred_ev[:, 0].sum()) if n_pred else 0.0
    return dict(
        EV_F1=f1, EV_PREC=prec, EV_REC=rec,
        n_true=n_true, n_pred=n_pred, tp=tp,
        # Event count ratio: >1 means the model fragments or over-fires at the EVENT
        # level, which is a different pathology from per-sample over-firing.
        ev_ratio=(n_pred / n_true) if n_true else float("nan"),
        # Duration ratio: predicted ON time over true ON time.
        dur_ratio=(pred_len / true_len) if true_len else float("nan"),
    )
