# TODO

## Metrics warnings (`src/helpers/metrics.py`)

### Divide by zero in `NILMmetrics` (lines 95, 100, 104)

When the ground-truth appliance signal `y` is all zeros (appliance never on), three
NILM-specific metrics blow up:

| Metric | Line | Denominator |
|--------|------|-------------|
| TECA   | 95   | `2 * np.sum(np.abs(y))` |
| NDE    | 100  | `np.sum(y ** 2)` |
| SAE    | 104  | `np.sum(y)` |

**Fix:** Guard each division — return `float("nan")` (or a domain-appropriate
fallback) when the denominator is zero.

### sklearn classification warnings in `NILMmetrics` (~line 132–136)

- `UserWarning: y_pred contains classes not in y_true` — model predicts ON
  states when the appliance is always OFF in the test window.
- `UndefinedMetricWarning: Recall is ill-defined ... due to no true samples` —
  same root cause: `y_state` is all zeros.

**Fix:** Pass `zero_division=0` to `precision_score`, `recall_score`, and
`f1_score` calls in the `y_state` block.
