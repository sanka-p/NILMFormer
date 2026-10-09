#################################################################################################################
#
# @description : NILMFormer - Appliance-combination metrics (OSW level)
#
# The three evaluation metrics of
#
#     Welikala et al., "Incorporating Appliance Usage Patterns for Non-Intrusive Load
#     Monitoring and Load Forecasting", IEEE Transactions on Smart Grid, 2019.
#
#   1. Aci  -- appliance combination identification accuracy
#   2. Fm / Afm -- per-combination F-measure and its unweighted average
#   3. Apa / Apd / Apf -- total power correctly assigned
#
# All three are scored per OSW (observation sliding window), never per raw sample: build the
# OSW blocks with src.helpers.osw first, then feed the block-level products here.
#
# Deliberately numpy + pandas only. The confusion counting is a bincount rather than an
# sklearn call: it is faster at this size and sidesteps the zero_division warning noise that
# src/helpers/metrics.py already suffers from.
#
#################################################################################################################

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

import numpy as np
import pandas as pd

from src.helpers.osw import (
    ALL_OFF,
    OSWBlocks,
    combination_cardinality,
    combination_name,
)

__all__ = [
    "ApdResult",
    "FmResult",
    "aci",
    "afm",
    "combination_identification_accuracy",
    "f_measure",
    "per_combination",
    "power_assigned_per_combination",
    "score",
    "total_power_assigned",
    "total_power_assigned_forecast",
]

#: Above this many distinct combinations the k*k confusion matrix is built label-by-label
#: instead of as one dense allocation. With 5 appliances k <= 32, so the fast path always
#: applies in practice.
_DENSE_CONFUSION_LIMIT = 4096


# ========================================= Helpers ========================================= #


def _check_labels(comb_true, comb_pred) -> tuple[np.ndarray, np.ndarray]:
    a = np.asarray(comb_true, dtype=np.int64).ravel()
    b = np.asarray(comb_pred, dtype=np.int64).ravel()
    if a.shape != b.shape:
        raise ValueError(
            f"combination label arrays must have equal length, got {a.shape[0]} and "
            f"{b.shape[0]}"
        )
    return a, b


def _check_power(y_true, y_pred) -> tuple[np.ndarray, np.ndarray]:
    # float64 throughout: OSW means are 1e2-1e3 W and there can be ~1e6 OSWs x 5 appliances,
    # so pooled sums reach 1e9-1e10. float32 carries ~7 significant digits, which would shift
    # Apa by whole percentage points. Predictions loaded from .pt logs are float32.
    a = np.asarray(y_true, dtype=np.float64)
    b = np.asarray(y_pred, dtype=np.float64)
    if a.ndim != 2 or b.ndim != 2:
        raise ValueError(
            f"y_true and y_pred must be 2-D (n_windows, n_appliances), got shapes "
            f"{a.shape} and {b.shape}"
        )
    if a.shape != b.shape:
        raise ValueError(f"y_true has shape {a.shape} but y_pred has {b.shape}")
    for name, arr in (("y_true", a), ("y_pred", b)):
        n_bad = int((~np.isfinite(arr)).sum())
        if n_bad:
            raise ValueError(
                f"{name} contains {n_bad} non-finite value(s); src/helpers/osw.py's "
                "nan_policy should have removed these. Silently zero-filling them (as "
                "NILMmetrics does) would inflate Apa, so this is an error."
            )
    return a, b


def _safe_div(num: float, den: float) -> float:
    return float(num) / float(den) if den else float("nan")


def _mean_defined(values: np.ndarray) -> float:
    """Unweighted mean over the defined (non-NaN) entries; NaN when none are defined."""
    arr = np.asarray(values, dtype=np.float64)
    ok = np.isfinite(arr)
    return float(arr[ok].mean()) if ok.any() else float("nan")


def _confusion(codes_true: np.ndarray, codes_pred: np.ndarray, k: int) -> np.ndarray:
    """Dense k x k confusion matrix, rows = truth, cols = prediction."""
    if k <= _DENSE_CONFUSION_LIMIT:
        flat = np.bincount(codes_true * k + codes_pred, minlength=k * k)
        return flat.reshape(k, k)
    cm = np.zeros((k, k), dtype=np.int64)  # pragma: no cover - not reachable at 5 appliances
    for i in range(k):
        sel = codes_true == i
        if sel.any():
            cm[i] = np.bincount(codes_pred[sel], minlength=k)
    return cm


# ========================================= 1. Aci ========================================= #


def combination_identification_accuracy(y_true, y_pred) -> float:
    """Appliance combination identification accuracy, Aci, as a percentage in [0, 100].

    The percentage of OSWs for which the predicted turned-ON appliance combination exactly
    matches the ground-truth combination::

        Aci = (number of OSWs correctly identified / total number of OSWs) * 100

    Parameters
    ----------
    y_true, y_pred : array-like, shape (n_windows,)
        Ground-truth and predicted combination label/ID per OSW. Integer bitmasks as
        produced by :func:`src.helpers.osw.states_to_bitmask` are the canonical encoding,
        but any hashable integer labelling works as long as both arrays share it.

    Returns
    -------
    float
        Accuracy as a percentage. NaN for empty input.
    """
    a, b = _check_labels(y_true, y_pred)
    if a.size == 0:
        return float("nan")
    return 100.0 * float((a == b).sum()) / float(a.size)


#: Short alias.
aci = combination_identification_accuracy


# ========================================= 2. Fm / Afm ========================================= #


@dataclass(frozen=True)
class FmResult:
    """Per-combination F-measure plus its unweighted average."""

    fm: dict[int, float]
    tp: dict[int, int]
    fp: dict[int, int]
    fn: dict[int, int]
    support_true: dict[int, int]
    support_pred: dict[int, int]
    afm: float
    n_combinations: int
    n_scored: int
    n_undefined: int


def afm(y_true, y_pred) -> FmResult:
    """Per-combination F-measure and its unweighted average, Afm (rich result).

    For each combination ``Cj`` appearing in ``y_true`` OR ``y_pred``::

        TP = #(true == Cj and pred == Cj)     # correctly predicted as Cj
        FN = #(true == Cj and pred != Cj)     # actual Cj, predicted something else
        FP = #(true != Cj and pred == Cj)     # predicted Cj, actual something else

        Fm,Cj = 2 * TP / (2 * TP + FN + FP)

    ``Afm`` is the UNWEIGHTED mean of ``Fm,Cj`` across all combinations, which is the paper's
    definition. Note what that implies: a combination observed in 6 OSWs counts exactly as
    much as one observed in 20 000. ``support_true`` / ``support_pred`` are returned so the
    per-combination breakdown can show it.

    Edge case ``2 * TP + FN + FP == 0``: ``Fm,Cj`` is defined as **NaN and excluded from the
    Afm average** (counted in ``n_undefined``). With the union label set this can only arise
    from a caller-side pre-filter, but it is guarded regardless.

    Parameters
    ----------
    y_true, y_pred : array-like, shape (n_windows,)
        Ground-truth and predicted combination ID per OSW.

    Returns
    -------
    FmResult
        ``.fm`` maps combination -> Fm,Cj and ``.afm`` is the average; see
        :func:`f_measure` for the plain ``(dict, float)`` form.
    """
    a, b = _check_labels(y_true, y_pred)
    if a.size == 0:
        return FmResult({}, {}, {}, {}, {}, {}, float("nan"), 0, 0, 0)

    labels = np.union1d(np.unique(a), np.unique(b))
    k = labels.size
    codes_true = np.searchsorted(labels, a)
    codes_pred = np.searchsorted(labels, b)

    cm = _confusion(codes_true, codes_pred, k)
    tp = np.diag(cm).astype(np.int64)
    sup_true = cm.sum(axis=1, dtype=np.int64)
    sup_pred = cm.sum(axis=0, dtype=np.int64)
    fn = sup_true - tp
    fp = sup_pred - tp

    den = 2 * tp + fn + fp
    with np.errstate(divide="ignore", invalid="ignore"):
        fm_vals = np.where(den > 0, 2.0 * tp / den, np.nan)

    keys = [int(v) for v in labels]
    n_scored = int(np.isfinite(fm_vals).sum())
    return FmResult(
        fm=dict(zip(keys, (float(v) for v in fm_vals))),
        tp=dict(zip(keys, (int(v) for v in tp))),
        fp=dict(zip(keys, (int(v) for v in fp))),
        fn=dict(zip(keys, (int(v) for v in fn))),
        support_true=dict(zip(keys, (int(v) for v in sup_true))),
        support_pred=dict(zip(keys, (int(v) for v in sup_pred))),
        afm=_mean_defined(fm_vals),
        n_combinations=k,
        n_scored=n_scored,
        n_undefined=k - n_scored,
    )


def f_measure(y_true, y_pred) -> tuple[dict[int, float], float]:
    """Per-combination F-measure and Afm.

    Thin wrapper over :func:`afm` returning exactly the two documented outputs:

    Returns
    -------
    (dict, float)
        ``(a)`` a dict mapping combination -> ``Fm,Cj`` (NaN where undefined), and
        ``(b)`` ``Afm``, the unweighted average of ``Fm,Cj`` across all combinations,
        with undefined entries excluded from the average.
    """
    result = afm(y_true, y_pred)
    return result.fm, result.afm


# ========================================= 3. Apa / Apd / Apf ========================================= #


def _active_rows(y_true: np.ndarray, comb_true, exclude_all_off: bool) -> np.ndarray:
    """Rows to score. All-OFF OSWs are excluded because Apa is undefined there."""
    if not exclude_all_off:
        return np.ones(y_true.shape[0], dtype=bool)
    if comb_true is not None:
        labels = np.asarray(comb_true, dtype=np.int64).ravel()
        if labels.shape[0] != y_true.shape[0]:
            raise ValueError(
                f"comb_true has length {labels.shape[0]} but y_true has "
                f"{y_true.shape[0]} rows"
            )
        return labels != ALL_OFF
    # No labels supplied: identify all-OFF from the measured power itself, so the
    # zero-denominator guarantee holds either way.
    return y_true.sum(axis=1, dtype=np.float64) > 0.0


def total_power_assigned(
    y_true, y_pred, *, comb_true=None, exclude_all_off: bool = True
) -> float:
    """Total power correctly assigned, Apa, as a percentage.

    ::

        Apa = (1 - [ sum_t sum_i |yhat_t(i) - y_t(i)| ] / [ 2 * sum_t ybar_t ]) * 100

        ybar_t = sum_i y_t(i)      # total measured power across appliances at OSW t

    where ``yhat_t(i)`` is the predicted and ``y_t(i)`` the measured mean power of appliance
    ``i`` over OSW ``t``. Both are OSW block-level MEANS, not raw samples -- build them with
    :func:`src.helpers.osw.build_osw`.

    Parameters
    ----------
    y_true, y_pred : array-like, shape (n_windows, n_appliances)
        Ground-truth and predicted power per appliance per OSW, in watts.
    comb_true : array-like, shape (n_windows,), optional
        Ground-truth combination labels, used only to identify all-OFF OSWs.
    exclude_all_off : bool, optional
        Default True. All-OFF OSWs are excluded because ``ybar_t == 0`` there, making the
        denominator zero and Apa UNDEFINED rather than zero. When ``comb_true`` is not
        supplied, all-OFF rows are identified from ``y_true`` itself.

    Returns
    -------
    float
        Apa as a percentage. NaN if no rows survive or the denominator is zero.

    Notes
    -----
    Apa is not clipped: it is unbounded below and goes negative when the model over-assigns
    power. ``NILMmetrics``' existing ``TECA`` (``src/helpers/metrics.py``) is the
    single-appliance special case of this formula.

    **The scale does not start at zero.** Because of the factor 2 in the denominator, a
    predictor that outputs zero everywhere scores ``Apa = 50%``::

        (1 - sum|0 - y| / (2 * sum y)) * 100 = (1 - 1/2) * 100 = 50

    So 50% is the do-nothing floor, not 0%, and a score below 50% means the model assigns
    more power than actually exists. Read reported Apa/Apd values against that baseline --
    the repo's ``TECA`` has the identical property.

    The average of ``Apa,Cj`` across combinations is the paper's ``Apd`` (disaggregation) --
    see :func:`power_assigned_per_combination`, which is a different quantity from this
    pooled value.
    """
    a, b = _check_power(y_true, y_pred)
    rows = _active_rows(a, comb_true, exclude_all_off)
    if not rows.any():
        return float("nan")
    a, b = a[rows], b[rows]
    abs_err = np.abs(b - a).sum(dtype=np.float64)
    den = 2.0 * a.sum(dtype=np.float64)
    ratio = _safe_div(abs_err, den)
    return float("nan") if not np.isfinite(ratio) else (1.0 - ratio) * 100.0


def total_power_assigned_forecast(
    y_true, y_pred, *, comb_true=None, exclude_all_off: bool = True
) -> float:
    """Total power correctly assigned for forecast output, Apf, as a percentage.

    Functionally identical to :func:`total_power_assigned`; kept as a separate name because
    the paper reports the disaggregation and forecasting cases as distinct quantities
    (``Apd`` versus ``Apf``), and scoring code reads more clearly when the intent is explicit.
    """
    return total_power_assigned(
        y_true, y_pred, comb_true=comb_true, exclude_all_off=exclude_all_off
    )


@dataclass(frozen=True)
class ApdResult:
    """Per-combination Apa plus Apd (their unweighted average) and the pooled value."""

    apa_per_combination: dict[int, float]
    support: dict[int, int]
    abs_err: dict[int, float]
    true_power: dict[int, float]
    apd: float
    apa_pooled: float
    n_combinations: int
    n_scored: int


def power_assigned_per_combination(
    y_true, y_pred, comb_true, *, exclude_all_off: bool = True
) -> ApdResult:
    """``Apa,Cj`` grouped by ground-truth combination, plus ``Apd``.

    ``Apd`` is the UNWEIGHTED MEAN of ``Apa,Cj`` across combinations -- the paper's actual
    definition -- and **not** the pooled global value, which is returned separately as
    ``apa_pooled``. The two differ whenever combination supports are unbalanced, which they
    always are in practice.

    Combinations whose ``2 * sum_t ybar_t == 0`` get ``Apa,Cj = NaN`` and are excluded from
    the ``Apd`` mean. The all-OFF combination is always excluded when ``exclude_all_off``.

    Parameters
    ----------
    y_true, y_pred : array-like, shape (n_windows, n_appliances)
    comb_true : array-like, shape (n_windows,)
        Ground-truth combination label per OSW; grouping is by ground truth, not prediction.
    """
    a, b = _check_power(y_true, y_pred)
    labels_all = np.asarray(comb_true, dtype=np.int64).ravel()
    if labels_all.shape[0] != a.shape[0]:
        raise ValueError(
            f"comb_true has length {labels_all.shape[0]} but y_true has {a.shape[0]} rows"
        )
    rows = _active_rows(a, labels_all, exclude_all_off)
    if not rows.any():
        return ApdResult({}, {}, {}, {}, float("nan"), float("nan"), 0, 0)

    a, b, labels_sel = a[rows], b[rows], labels_all[rows]
    labels, inv = np.unique(labels_sel, return_inverse=True)
    k = labels.size

    per_row_err = np.abs(b - a).sum(axis=1, dtype=np.float64)
    per_row_true = a.sum(axis=1, dtype=np.float64)
    num = np.bincount(inv, weights=per_row_err, minlength=k)
    den = 2.0 * np.bincount(inv, weights=per_row_true, minlength=k)
    support = np.bincount(inv, minlength=k)

    with np.errstate(divide="ignore", invalid="ignore"):
        apa_cj = np.where(den > 0, (1.0 - num / den) * 100.0, np.nan)

    keys = [int(v) for v in labels]
    n_scored = int(np.isfinite(apa_cj).sum())
    pooled_ratio = _safe_div(per_row_err.sum(dtype=np.float64), 2.0 * per_row_true.sum(dtype=np.float64))
    return ApdResult(
        apa_per_combination=dict(zip(keys, (float(v) for v in apa_cj))),
        support=dict(zip(keys, (int(v) for v in support))),
        abs_err=dict(zip(keys, (float(v) for v in num))),
        true_power=dict(zip(keys, (float(v) for v in den / 2.0))),
        apd=_mean_defined(apa_cj),
        apa_pooled=(
            float("nan") if not np.isfinite(pooled_ratio) else (1.0 - pooled_ratio) * 100.0
        ),
        n_combinations=k,
        n_scored=n_scored,
    )


# ========================================= Aggregators ========================================= #


def score(
    blocks: OSWBlocks,
    *,
    round_to: int = 3,
    emit_forecast_aliases: bool = False,
) -> dict:
    """Score OSW blocks and return one flat, CSV-friendly metric dict.

    Follows the conventions of ``src/helpers/metrics.py``: flat uppercase keys, rounding
    applied only at the very end, plain Python scalars (never numpy types) so the values can
    be written straight to CSV. Every key is always present, holding NaN where undefined.

    Scale asymmetry, deliberate and to be honoured by any renderer:

    - ``ACI``, ``ACI_ACT``, ``APD``, ``APA_POOLED`` are PERCENTAGES in [0, 100], per the paper.
    - ``AFM``, ``AFM_ACT`` are FRACTIONS in [0, 1], matching this repo's existing ``F1_SCORE``.

    ``_ACT`` variants are computed over only those OSWs whose GROUND-TRUTH combination is
    not all-OFF. The filter is on ground truth alone, so a model that predicts all-OFF on an
    active OSW still counts as wrong -- and that is the one realistic route to a
    defined-but-zero ``Fm,Cj`` rather than the NaN case.

    Apa/Apd always exclude all-OFF OSWs, whose zero denominator makes them undefined.
    """
    if not blocks.has_predictions:
        raise ValueError(
            "blocks carries no predictions; rebuild with build_osw(..., y_pred=...) before "
            "scoring"
        )

    ct, cp = blocks.label_true, blocks.label_pred
    active = blocks.active_mask()

    fm_all = afm(ct, cp)
    fm_act = afm(ct[active], cp[active])
    apd = power_assigned_per_combination(
        blocks.mean_true, blocks.mean_pred, ct, exclude_all_off=True
    )

    out: dict = {
        "N_OSW": int(ct.size),
        "N_OSW_ACT": int(active.sum()),
        "N_OSW_ALL_OFF": int(ct.size - active.sum()),
        "ACI": combination_identification_accuracy(ct, cp),
        "ACI_ACT": combination_identification_accuracy(ct[active], cp[active]),
        "AFM": fm_all.afm,
        "AFM_ACT": fm_act.afm,
        "APD": apd.apd,
        "APA_POOLED": apd.apa_pooled,
        "N_COMB_TRUE": int(np.unique(ct).size),
        "N_COMB_PRED": int(np.unique(cp).size),
        "N_COMB_UNION": fm_all.n_combinations,
        "N_COMB_FM_SCORED": fm_all.n_scored,
        "N_COMB_FM_UNDEFINED": fm_all.n_undefined,
        "N_COMB_UNION_ACT": fm_act.n_combinations,
        "N_COMB_FM_SCORED_ACT": fm_act.n_scored,
        "N_COMB_APD_SCORED": apd.n_scored,
    }
    if emit_forecast_aliases:
        # Same numbers under the paper's forecasting names, so a forecast run can be written
        # to an identical CSV schema.
        out["APF"] = out["APA_POOLED"]
        out["APF_D"] = out["APD"]

    rounded = {}
    for key, value in out.items():
        rounded[key] = (
            value if isinstance(value, int) else round(float(value), round_to)
        )
    rounded.update(blocks.metadata())
    return rounded


def per_combination(blocks: OSWBlocks, *, round_to: int = 6) -> pd.DataFrame:
    """One row per appliance combination -- the per-combination breakdown.

    A row is emitted for every combination with ``SupportGT > 0 or SupportPred > 0``.
    ``APA_CJ`` is NaN on the all-OFF row by construction (zero denominator). The frame always
    carries the full column set, even when empty, so concatenating many runs cannot produce a
    ragged CSV.
    """
    columns = [
        "CombinationMask", "Combination", "NumAppliancesOn", "SupportGT", "SupportPred",
        "TP", "FP", "FN", "FM", "APA_CJ", "AbsErrW", "TruePowerW", "IsAllOff",
    ]
    if not blocks.has_predictions:
        raise ValueError("blocks carries no predictions; nothing to break down")
    if blocks.n_osw == 0:
        return pd.DataFrame({c: pd.Series(dtype="object") for c in columns})

    ct, cp = blocks.label_true, blocks.label_pred
    fm_res = afm(ct, cp)
    apd_res = power_assigned_per_combination(
        blocks.mean_true, blocks.mean_pred, ct, exclude_all_off=True
    )
    names = blocks.appliance_names

    rows = []
    for mask in sorted(fm_res.fm):
        rows.append(
            {
                "CombinationMask": mask,
                "Combination": combination_name(mask, names),
                "NumAppliancesOn": int(combination_cardinality(mask)),
                "SupportGT": fm_res.support_true[mask],
                "SupportPred": fm_res.support_pred[mask],
                "TP": fm_res.tp[mask],
                "FP": fm_res.fp[mask],
                "FN": fm_res.fn[mask],
                "FM": round(fm_res.fm[mask], round_to),
                "APA_CJ": _round_or_nan(apd_res.apa_per_combination.get(mask), round_to),
                "AbsErrW": _round_or_nan(apd_res.abs_err.get(mask), round_to),
                "TruePowerW": _round_or_nan(apd_res.true_power.get(mask), round_to),
                "IsAllOff": mask == ALL_OFF,
            }
        )
    frame = pd.DataFrame(rows, columns=columns)
    return frame.sort_values(
        ["SupportGT", "CombinationMask"], ascending=[False, True]
    ).reset_index(drop=True)


def _round_or_nan(value, round_to: int) -> float:
    return float("nan") if value is None else round(float(value), round_to)


def summary_lines(metrics: Mapping[str, float]) -> list[str]:
    """Compact human-readable summary of a :func:`score` dict, for CLI output."""
    return [
        f"OSW: N={metrics['N_OSW']} (active {metrics['N_OSW_ACT']}, "
        f"all-OFF {metrics['N_OSW_ALL_OFF']}), duration {metrics['OSW_DURATION_S']:g}s",
        f"Aci  = {metrics['ACI']:.2f}%   (active-only {metrics['ACI_ACT']:.2f}%)",
        f"Afm  = {metrics['AFM']:.4f}    (active-only {metrics['AFM_ACT']:.4f})"
        f"  over {metrics['N_COMB_UNION']} combinations",
        f"Apd  = {metrics['APD']:.2f}%   (pooled Apa {metrics['APA_POOLED']:.2f}%)",
    ]
