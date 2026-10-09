# Paper TODO — TCN ablation study

Tracking file for the ablation work. Updated 2026-10-05.
Numbers here are measured, not estimated; every claim names the file it came from.

---

## 1. Status

| Experiment | Scope | Status |
|---|---|---|
| UK-DALE 3-arm ladder (A, B, D) | 5 appliances x 3 windows x 3 seeds | **done** 45 cells/arm, 0 fail |
| REDD 3-arm ladder (A, B, D) | 4 appliances x 3 x 3 | **done** 108 cells, 0 fail |
| SotA comparison (UK-DALE) | Proposed vs 6 baselines | **done** |
| Pretrained reference rows | TCN_KL, 1app ~900ep, 1app ~95ep | **done** (excluded from the ablation table) |
| KLT inference latency | CUDA + CPU, batch 1 + 32 | **done** -> `results/kl_latency.json` |
| **Variant C** (aug, no KL) | UK-DALE 45 + REDD 36 | **done** 81 cells, 0 fail |
| **Curated pool-size sweep** | Microwave, caps 20/50/100/200 | **done** 36 cells, 0 fail |
| KL x Curation interaction (2x2) | both datasets | **done** -> `results/interaction_*.txt` |
| K-component sweep (K in 1..10) | optional | **not run** — see 5.3 |
| REDD SotA comparison | needs `--result-dir 2207-results/result` | **not run** — see 5.4 |

### Variant map (template letter -> model key)

| Variant | KL | Synth | Model key | Params |
|---|---|---|---|---|
| A | x | x | `TCN` | 29,409 |
| B | yes | x | `TCN_KL_scratch` | 31,009 |
| C | x | yes | `TCN_aug_curated` | 29,409 |
| D | yes | yes | `TCN_KL_aug_curated` | 31,009 |
| D' | yes | yes (repo acts, not curated) | `TCN_KL_aug` | 31,009 — UK-DALE only |

---

## 2. Headline findings

1. **KL features are the dominant component.** UK-DALE A->B = **+31.96 mean F1**, +18.75 BA.
   An order of magnitude larger than any other step. Replicates on REDD (+8.6 to +12.5 BA on
   3 of 4 appliances).
2. **Gains concentrate on SHORT-BURST loads, not multi-state ones.** Kettle +58.1, Fridge
   +53.3, Microwave +36.4 vs WM +1.1, DW +10.9. ⚠️ This **contradicts** the hypothesis in the
   draft ("expect the largest gains on multi-state loads WM, DW") — that sentence must be
   rewritten.
3. **Curated augmentation on its own does almost nothing — and can hurt.** With variant C
   (curated synthetic data, no KL) the design closes into a 2x2, so augmentation can be read
   *unconditionally* (A->C) as well as conditionally (B->D):

   | contrast | UK-DALE mean F1 | REDD mean F1 |
   |---|---|---|
   | A->C, augmentation alone | +3.02 | **-2.04** |
   | B->D, augmentation given KL | +5.20 | -1.53 |

   Variant C is catastrophic on Kettle (7.09 vs vanilla TCN's 17.36 F1) and below vanilla on
   every REDD appliance except WasherDryer. ⚠️ The earlier "+5.63 mean F1 for curation +
   augmentation" was the B->D figure only; it must not be quoted as augmentation's standalone
   effect. **Essentially the whole gain of the proposed model is the KL front end.**
4. **KL and curation are additive, not synergistic.** Interaction `(D-C) - (B-A)`:
   UK-DALE **+2.19 F1 / +1.20 BA**, REDD **+0.51 F1 / -8.04 BA** — small relative to the
   +31.96 main effect of KL. Per-appliance it swings both ways (UK-DALE Kettle +14.18,
   Fridge -5.94; REDD Fridge +22.44, WasherDryer -21.90), so the mean is near zero rather
   than consistently so. Practical consequence: the A->B->D ladder is **not** materially
   confounded, so its step-wise attribution stands. Tables in `results/interaction_*.txt`.
5. **Where curation does help (given KL), the benefit comes from activation *quality*,
   not pool size — and not sample count either.** (Scope: this is the B->D contrast, i.e.
   curation *on top of* KL; finding 3 covers curation on its own, which is a different and
   much weaker effect.) The augmented arm generates the *same* number of training windows as the
   real arms (verified in logs, e.g. `50938 real -> 50938`), so volume is not the variable.
   Across 9 appliance/dataset cells the outcome correlates with pool size:

   | pool | outcome |
   |---|---|
   | >= 163 activations (all 5 UK-DALE) | curation wins |
   | <= 44 activations (REDD micro 44, DW 11, WD 11) | curation loses |

   but the direct sweep shows that correlation is **not causal**. Subsampling the UK-DALE
   Microwave curated pool to 20/50/100/200 activations leaves the gain essentially intact
   at every point (9 runs per point):

   | pool | F1 | delta vs `TCN + KL` (56.58) |
   |---|---|---|
   | 20 | 73.61 +-3.86 | +17.03 |
   | 50 | 73.02 +-7.30 | +16.44 |
   | 100 | 72.01 +-7.00 | +15.43 |
   | 200 | 74.17 +-6.34 | +17.59 |
   | 3000 (uncapped) | 71.31 +-7.16 | +14.73 |

   Twenty curated activations are enough; the curve is flat across the whole range and the
   uncapped arm is, if anything, the weakest point. The cap is seeded per run, so each point averages
   three independent draws x three window sizes. **So the REDD losses are not a small-pool
   effect.** The likely cause is pool quality and cross-house heterogeneity -- REDD H5's
   microwave segments peak at 96 W and are dropped by the 200 W threshold, leaving H3's 46;
   REDD Dishwasher/WasherDryer draw their 11 activations from four and three houses
   respectively. Counts in `results/curated_activation_counts.csv`, curve in
   `results/pool_sweep.csv`.
6. **KLT energy compaction justifies K=10 without a sweep**: components 1-3 carry
   **90.5 %** of ensemble variance (77.9 / 9.3 / 3.3 %); components 4-10 ~1 % each. Two
   independent fits agree within 1 %.
7. **KLT inference overhead is negligible in absolute terms**: +0.116 ms per 1,000 samples
   (CUDA, batch 1), +22.5 % relative on a 0.5 ms model. +1,600 parameters (+5.4 %).

---

## 3. Corrections needed in the draft

- **Parameters "312K" is wrong** — the whole model is **31,009** params (29,409 without KL).
  That placeholder appears to come from a different architecture.
- **"LOHO splits"** is inaccurate for UK-DALE: validation is a seeded 80/20 split of training
  *windows*; only REDD holds out a validation *house*. Test is a fixed house in both.
- **"Variants without augmentation are trained on curated real sequences only"** — no: A and
  B train on **real windows** from the raw dataset. Only C and D touch curated activations.
- **KL hypothesis is inverted** (see finding 2).
- **"Is the gain larger on REDD (domain shift)?"** — no, it is *smaller* and partly negative.
- **Input channels K = 11** (10 KLT + 1 raw), not "K".

---

## 4. Caveats to state explicitly

- All arms run **3 epochs** with validation loss still falling. Fair to each other; a ceiling
  for all. Not a converged comparison.
- The augmented arms **replace** real windows rather than adding to them, so "synthetic"
  means "trained on generated instead of real data".
- **REDD WasherDryer is not comparable** to the existing REDD baselines — its `app` mapping
  changed (`WashingMachine` -> `WasherDryer`) after those baselines ran (commits 2026-06-22).
- **REDD cells are ~13x smaller** than UK-DALE's (Fridge 3,753 vs Kettle 50,938 training
  windows), so REDD numbers are inherently noisier.
- **REDD synthetic aggregate is inconsistent across houses**: appliances absent from a house
  are silently zero-filled (`preprocessing.py:1731`), so houses 4 and 6 contribute a weaker
  aggregate. Affects all arms and all baselines equally.
- **The REDD curated store has no external anchor.** `tcn_meta.json` describes a *different*
  store (`Data/redd_curated_formatted`); microwave mean on-power differs 882.5 vs 501.2 W.
  Validation was by independent re-derivation of the profile instead.
- **D vs D' differ in pool cap as well as source** (D uncapped, D' capped at 3000), so that
  specific pair does not cleanly isolate curation. The A/B/C/D 2x2 is unaffected.
- The pool-size sweep is **one appliance, one dataset**. It rules out pool size as the
  explanation for the UK-DALE/REDD split on Microwave, but does not establish that no
  appliance has a size floor.

---

## 5. Open items

### 5.1 Variant C — DONE
81 cells, 0 failures. Closed the 2x2; results folded into findings 3 and 7.

### 5.2 Pool-size sweep — DONE (result in 2.4)
Microwave, caps 20/50/100/200 plus the uncapped 3000 point. Reference line: `TCN + KL`
Microwave F1 56.58. Realised pools equal the nominal caps exactly -- the run logs show
`microwave -> 20/20`, `50/50`, `100/100`, `200/200` usable samplers, i.e. the cap is applied
*after* the `len > 10` filter. (An earlier note claiming ~13/37/63/126 realised was wrong.)
Rendered by `scripts/make_pool_sweep.py`; re-run it once the grid completes.
**Follow-up this opens:** the mechanism is now quality, not size, so an
alpha-Precision / beta-Recall / Authenticity analysis of the activation pools
(Alaa et al., ICML 2022) would measure it directly.

### 5.3 K-component sweep — judged NOT vital
K=10 is inherited from the reference implementation (`TCN_ukdale.py`), not tuned, and is not
a claimed contribution. The spectrum (finding 5) is the conventional justification. If a
reviewer pushes: reduced sweep = 1 appliance x 1 window x 3 seeds x K in {1,2,3,5,10} =
**45 runs, ~4 h** (not 225 runs / 19 h).

### 5.4 REDD SotA comparison — not run
REDD baselines live in `2207-results/result/`, which is not in `DEFAULT_RESULT_DIRS`, so they
are absent from the cache. One `make_table --result-dir 2207-results/result` would pull them
in, then `make_ablation_table --preset sota --dataset REDD`.
Comparable: Fridge, Dishwasher, Microwave. Not comparable: WasherDryer (see caveats).

### 5.5 Split-table mismatch
The REDD runs use the repo's **per-appliance** splits (Fridge [3,5,6], Microwave [3,5],
Dishwasher [3,4,5,6], WasherDryer [3,5,6], valid = held-out house, test = house 1) for
comparability with existing baselines — **not** the uniform "train 2,3,4,5,6 / test 1" in the
draft's Table 2. Either the caption changes or the runs do.

### 5.6 Convergence follow-up (optional)
All arms are under-trained at 3 epochs. A matched higher-budget grid would turn "A is worse"
into "A is worse at its own best". ~26 h scoped to one window size.

---

## 6. Artifacts

| File | Contents |
|---|---|
| `results/ablation_table.txt` / `ablation_metrics.csv` | UK-DALE ladder (mean + std) |
| `results/ablation_table_redd.txt` / `ablation_metrics_redd.csv` | REDD ladder |
| `results/sota_comparison.txt` / `sota_metrics.csv` | UK-DALE vs 6 baselines |
| `results/curated_activation_counts.csv` | activations per appliance, both datasets |
| `results/pool_sweep.txt` / `pool_sweep.csv` | curated pool-size sweep curve |
| `results/interaction_ukdale.txt` / `interaction_redd.txt` | KL x Curation 2x2 and interaction term |
| `results/kl_latency.json` | KLT inference overhead |
| `results/runs_cache.csv` | per-run table behind every number (691+ runs) |
| `result-tcn-ablation/` | UK-DALE + REDD ablation `.pt` files |
| `result-curated-scaling/` | pool-size sweep `.pt` files |

Regenerate the interaction: `PYTHONPATH=. .venv/bin/python -m scripts.make_interaction
--dataset {UKDALE,REDD} --out ...`

Regenerate the sweep: `PYTHONPATH=. .venv/bin/python -m scripts.make_pool_sweep
--out results/pool_sweep.txt --csv results/pool_sweep.csv`

Regenerate tables: `PYTHONPATH=. .venv/bin/python -m scripts.make_ablation_table
--dataset {UKDALE,REDD} --preset {ablation,sota} --out ... --csv ...`
(run `scripts.make_table` first if new `.pt` files exist, to refresh the cache).


---

## 7. Phase 5 (2026-10-09): cross-domain inference on DEEE_SmartHome

UK-DALE-trained `TCN_KL_aug_curated` run, inference-only, on a synthetic aggregate built
from the locally collected DEEE_SmartHome traces. 576 runs (3 scored + 2 specificity
appliances x 3 window sizes x 3 seeds x 2 aggregate modes x 2 base-load variants x 2
scopes x 2 threshold regimes), 0 failures.

**Gate.** The trained scaler is not stored in the checkpoints, so it was recovered by
replaying the builder (`scripts/pin_ukdale_scaler.py`): `power_stat2 = 6852` for all 5
appliances x 3 window sizes. V4 then reproduced each checkpoint's own stored UK-DALE
metrics EXACTLY (Dishwasher/128/s0: MAE 37.7570, F1 0.5080, rel = 0.0000), which validates
the model rebuild, the pinned scaler and the threshold path in one shot.

### 7.1 Two appliances cannot be scored at all under as-trained settings

- **Kettle**: UK-DALE threshold 2000 W vs DEEE peaks of 1370 / 1341 W (230 V ~1.4 kW
  elements). Every sample labels OFF; GT and prediction are both identically zero.
- **WashingMachine**: UK-DALE needs 1800 s of continuous ON; the Singer top-loader's
  above-threshold phases are 7 runs, longest 730 s, gaps up to 940 s. No activation
  survives. This is a **duration** prior, not an amplitude one -- threshold tuning cannot
  fix it.

Even with both priors corrected (`adapted` regime: kettle 500 W, WM min_on 180 s), the
**kettle model still detects nothing** -- BA 50.00%, F1 0.07% in the clean scope. The
amplitude domain shift is total: a 3 kW-trained detector does not fire on a 1.4 kW kettle.

### 7.2 Hallucination is driven by UNFAMILIAR LOAD -- except when a base load exists

Fridge and Dishwasher do not exist in DEEE, so anything they predict is invented. Removing
the 9 out-of-vocabulary categories (fan, iron, AC, laptop, bulb, ...) from the aggregate:

| variant | appliance | phantom power | FP rate |
|---|---|---|---|
| pure | Fridge | 14.92 W -> **1.65 W** (-88.9%) | 20.1% -> **1.7%** |
| pure | Dishwasher | 11.91 W -> 2.34 W (-80.3%) | 19.0% -> 13.0% |
| baseload | Fridge | 79.03 W -> **79.79 W (+1.0%)** | 89.3% -> **98.3%** |
| baseload | Dishwasher | 14.87 W -> 2.62 W (-82.4%) | 24.6% -> 13.3% |

1. **With no base load, the out-of-vocabulary loads are the whole cause.** Strip them and
   fridge hallucination drops ~89% to 1.65 W. The model is well behaved on a mains
   containing only appliances it knows.
2. **An 80 W constant floor is itself read as a fridge.** The fridge model reports ~79 W --
   essentially the entire injected floor -- regardless of scope, i.e. **86.7% of all energy
   in the quiet house**. This is arguably correct behaviour given training: UK-DALE fridge
   is a 50-300 W cycler at 37.5% duty, so a constant ~80 W draw is almost exactly its
   signature. It is still a failure mode for deployment.

⚠️ Phantom-energy *percentages* are not comparable across scopes: the ukdale-scope mains
averages 10.6 W against 162.3 W for all 12 categories, so the share rises even when the
invented watts fall. **Mean predicted watts is the comparable figure.**

### 7.3 Removing unfamiliar load also lifts the scored appliances

Microwave F1 2.60% -> 26-37%, WashingMachine F1 5.06% -> 27-42% (session/pure, adapted).
The distractors were masking the targets, not just inflating false positives.

### 7.4 Data defect found in the collection

Two of the three `Fan/Sisl_60W` leaves name the wrong CSV in `metadata.json`
(`Setting_Low` -> a non-existent file; `Setting_High` -> Setting_Low's file). The loader
falls back to the single CSV present and hard-fails if the trace count ever stops matching
the CSV count. **Worth fixing at source** -- the `samples_written` fields in those two
leaves may also be mislabelled.

### Artifacts

| File | Contents |
|---|---|
| `results/deee_table.txt` | full cross-domain table + specificity + caveats |
| `results/deee_metrics.csv` / `deee_specificity.csv` | 432 + 144 rows |
| `results/deee_trace_inventory.csv` | 35-trace census |
| `results/deee_aggregate_{mode}_{variant}[_ukdale]_seed0.npz` | the 8 synthetic aggregates |
| `results/deee_predictions/` | per-timestamp predictions with source provenance |
| `results/ukdale_scaler_stats.csv` | the recovered UK-DALE normalisation |
