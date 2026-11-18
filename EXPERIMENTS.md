# Experimental log — REDD held-out-house study

Exploratory work on the KLT + TCN model, REDD, held-out-house protocol. **Not the paper
run**: these experiments are free of the comparability constraints that bind the published
tables, which is why scoring rules are varied here that must stay fixed there.

Everything below is reproducible from the scripts named in each section. Where a result is
negative it is recorded as such, because several of the negatives are more useful than the
positives.

---

## 0. The headline, in one paragraph

The **decision rule has moved results far more than any model change**. Three separate
interventions on how a prediction becomes a detection — threshold criterion, prevalence
matching, and temporal post-processing — produced gains of 5-20 F1 points each, while the
model-side ablations (class weighting, activity-aware cropping, time-of-day conditioning,
basis geometry) produced a few points at best and several clear regressions. The practical
output is a **fully label-free scoring pipeline** that matches an oracle which knows the
target house's duty cycle:

```
scripts/make_sota_table_direct.py --score prevalence --prior valid --postprocess
```

Mean F1 over the four REDD appliances, reference arm: **label-free + post-processing 67.4**,
against **oracle-prior without post-processing 64.3**. The honest pipeline beats the
cheating one.

---

## 1. Class weighting — a calibration fix in disguise

Weighted MSE with `w_off > w_on` (the opposite of inverse-frequency weighting), arms
`cw1_5 [1.5,1.0]`, `cw3 [3.0,1.0]`, `cw10 [10.0,1.0]`.

At the stock threshold, over-firing (predicted-ON / true-ON = recall/precision) falls
**monotonically** with `w_off` on every appliance, MAE falls with it, and F1 peaks at
`w_off ≈ 3`.

* `cw10` **collapses** on Microwave: F1 4.6, recall 2.9%, over-firing 0.14x — a silent
  detector. I initially reported from WashingMachine alone that it did not collapse; the
  full grid refuted that.
* The cliff is **appliance-dependent and not ordered by prevalence**: Dishwasher (4.12% ON)
  degrades while WashingMachine (2.07% ON) does not. So rarity does not predict it.

**The finding that matters:** under prevalence-matched scoring the *unweighted* reference
beats `cw3` on WM, DW and Microwave. The loss function's apparent detection gain was mostly
**calibration**, and calibration is obtainable for free at scoring time. The MAE gain is
real and survives (WM 31.4 -> 23.3 W).

**Decision:** default stays unweighted; prevalence matching is the over-firing fix. Keep
`w_off = 3` only when MAE is the target.

---

## 2. Threshold criteria and prevalence matching

`scripts/calibrate_threshold.py`, 44 (appliance, arm, seed) rows.

Two structural facts drove this:

* **BA is F1 at 50% prevalence.** Precision depends on the evaluation set's class balance
  as well as the classifier (Saito & Rehmsmeier 2015), so BA and F1 rank operating points
  differently on a rare appliance. Re-expressing precision at a fixed reference prevalence
  makes both functions of (TPR, FPR) alone and their optima coincide. The BA-vs-F1 conflict
  is **one prevalence term**, not a deep disagreement.
* **REDD's validation and test houses differ 1.3x-7.3x in duty cycle**, so any threshold
  tuned on validation arrives at test mis-calibrated.

Criteria were therefore ranked by **measured transfer regret** (validation-calibrated minus
test-oracle), not by in-sample optimum. `PRIOR` — match the predicted-positive rate to the
appliance duty cycle — transfers nearly free (worst regret 8.04 oracle vs 8.07 calibrated),
while GM/F1/MCC lose 5-7 points in transfer. That is why PRIOR won, and it is also why
further threshold work is near its ceiling.

Prevalence matching lifts WashingMachine F1 from **27.5 to 65.6** with an oracle prior.
I first reported that number without noting the threshold was picked on `y_test`; the
label-free figure is **45.6**. Section 4 is about closing that gap.

---

## 3. Fragmentation, and temporal post-processing

`src/helpers/postprocess.py`, `src/helpers/event_metrics.py`, `scripts/score_postprocess.py`.

**The discovery.** Activation-level scoring with greedy one-to-one matching (so a single
blob prediction cannot score 100%) showed the models emit **44-72 predicted activations per
true one** on Dishwasher, 20-51 on WashingMachine. Event F1 read **3-6** while timestamp F1
read 45-66. The timestamp metrics were hiding a shattered output.

**The fix is label-free and principled.** Ground-truth states are built with Kelly's
duration rules (`preprocessing.py:406-440`): gaps shorter than `min_off_duration` bridged,
activations shorter than `min_on_duration` dropped. So a predicted ON run shorter than
`min_on_duration` **cannot match any true activation** — it is a guaranteed false positive,
removable without looking at labels. Applying the same rules to predictions makes the two
sequences members of the same family. Order follows Kelly: bridge first, then drop.

| | before | after |
|---|---|---|
| Dishwasher F1 | 46.3 | **60.3** |
| Dishwasher BA | 74.6 | **90.3** |
| Dishwasher event F1 | 3.3 | **57.2** |
| WashingMachine F1 (label-free) | 45.6 | **61.5** |

Gains track `min_on_duration`, **not rarity** — which is the signature of a fragmentation
fix rather than a class-imbalance fix.

**Side effects worth knowing.** Post-processing *rehabilitates* `cw3` (WM label-free
40.6 -> 68.5, above the reference's 61.5) and **reorders the leaderboard**: ActCrop is best
on WM without post-processing (66.5) and fifth with it (61.3). Any ranking claim must state
which scoring produced it.

---

## 4. Estimating the target prior from unlabelled data — negative

`scripts/estimate_prior.py`.

Prevalence matching needs the target house's duty cycle, which `--prior test` assumes is
known. The label-shift literature says it is recoverable from unlabelled target data:
**BBSE** (Lipton et al. 2018; in the binary hard-label case exactly the Rogan & Gladen 1978
correction `pi = (q_t - FPR)/(TPR - FPR)`) and **EM/MLLS** (Saerens et al. 2002).

**Verification first.** On simulated data satisfying label shift exactly, both estimators
recover the prior to **±0.1 points** across shifts from 0.17x to 5x. The `F1val` column also
reproduces `score_postprocess --prior valid` exactly. So the implementation is sound and
what follows is the assumption failing.

| appliance | shift | true | BBSE | EM | J |
|---|---|---|---|---|---|
| Fridge | 1.32x | 22.74% | 26.54% | 26.09% | 0.98 |
| Microwave | 7.15x | 1.17% | 2.55% | 0.16% | 0.98 |
| Dishwasher | 6.97x | 4.12% | 10.12% | 9.59% | 0.76 |
| WashingMachine | 2.69x | 2.07% | **38.90%** | 37.78% | **0.17** |

Both **over-estimate on every appliance**. A systematic one-sided bias is the signature of
`p(x|y)` changing, not `p(y)` — the test house contains a physically different appliance, so
the validation confusion matrix does not transfer. Label shift is precisely the assumption a
held-out-house protocol is guaranteed to violate. Using the estimates scores **worse than
using no target information at all** (mean F1 53.7 BBSE / 44.9 EM vs 59.5 validation).

**Conditioning diagnoses only half of it.** WM's Youden J = 0.17 correctly predicts garbage
(`J_MIN` is now 0.50, which refuses it), but Dishwasher had a healthy J = 0.76 and was still
2.5x too high. The guard catches variance; **no function of the source confusion matrix can
catch the bias**.

**What replaced it.** The gap was mostly fragmentation, not ignorance of the prior. Once
duration rules remove the spurious runs, a conservative validation-calibrated threshold is
nearly as good as the oracle:

| | WM | DW | Micro. | Fridge | mean |
|---|---|---|---|---|---|
| validation prior + pp (**label-free**) | 61.5 | 60.3 | 59.0 | 88.9 | **67.4** |
| test-prior oracle + pp | 66.0 | 61.5 | 58.9 | 88.6 | 68.7 |

Mean gap **1.3 points**, and on Microwave and Fridge the label-free threshold is the
*better* of the two. Full table: `result-redd-tod/table_tod_labelfree_pp.tex`, coverage 3/3
in every cell.

**Caveat, and it is not small.** This equivalence holds for the reference and
class-weighted arms. The **ActCrop family loses badly on WM** label-free (46.0 vs 61.3
oracle), so activity-aware cropping shifts the validation score distribution in a way that
breaks threshold transfer.

**Conclusion:** do not estimate the target prior. Report `--prior valid --postprocess`, and
keep `--prior test` only as a declared upper bound.

---

## 5. Tuning the post-processing — negative, with one adoption

`scripts/tune_postprocess.py`. Searches (min_on, min_off, median_k) per cell, fitting
threshold and post-processing **alternately** to convergence, and reports the identical
search fitted against test as an oracle.

Motivation: Kelly's durations are *label-generation* parameters, never validated as
prediction filters — the asymmetry argument licenses only `min_on` — and `median_filter` was
implemented, tested, and **never once applied** (`durations_for()` returns only
min_on/min_off).

| appliance | valid %ON | Kelly F1 | tuned F1 | oracle F1 | gain | regret |
|---|---|---|---|---|---|---|
| Fridge | 17.22 | 88.9 | **89.3** | 88.7 | +0.4 | -0.6 |
| Microwave | 0.16 | 59.0 | 55.6 | 59.5 | -3.4 | +3.9 |
| Dishwasher | 0.59 | 60.3 | 55.0 | 66.6 | -5.3 | +11.6 |
| WashingMachine | 0.77 | 61.5 | 36.8 | 68.8 | **-24.7** | **+32.1** |

**Regret tracks validation ON-mass almost perfectly.** Three hyperparameters cannot be
chosen from a single house with under 1% positive mass. This is **not** evidence Kelly is
optimal — the oracle shows **+6 to +7 F1 of real headroom** on DW and WM that validation
cannot locate. Closing it needs a better validation split (cross-validation over training
houses), not a better search.

**Two findings that survive:**

1. **Kelly's Fridge rules are wrong for filtering predictions.** `min_on = 6` samples (60 s)
   is far too permissive for a load whose cycles run ten minutes or more. Tuning selects
   `60/12/0` and lifts **event F1 from 67.2 to 92.1** with timestamp F1 also improving.
   Fridge is the one appliance whose validation split can support the choice.
2. **The median filter is vindicated, but only as an oracle.** The test-fitted search picks
   `median_k = 21` on DW and `5-21` on WM, often with `min_off = 0` — "do not bridge, just
   median filter" is a materially better operating point that cannot be selected label-free.
   The unused code path is not worthless; it is **unreachable under an honest protocol**.

---

## 6. Time-of-day conditioning — negative, and robustly so

Arms `TCN_KL_act` (activity-aware cropping alone), `TCN_KL_act_tod` (concatenated sin/cos
channels), `TCN_KL_act_tod_film` (FiLM modulation). The cropping-only arm **B** exists
precisely so that C-minus-A does not confound the data change with the prior.

**Every ToD arm loses to the reference on every appliance**, and the result is stable across
all three scorings (stock threshold, prevalence-matched, prevalence + post-processing).
This is a real negative result and worth reporting: NILMFormer's TimeRPE gives it wall-clock
information, and a plausible story said that was part of its margin. On a held-out-house
protocol the diurnal prior does not transfer.

**Incomplete:** the phase-shuffle control `TCN_KL_act_tod_shuf` (3 cells) has still never
been run. Without it we can say the prior does not help, but not whether the extra channels
hurt for a different reason. It is the cheapest outstanding experiment.

---

## 7. Basis geometry — the fitted KL is right, and DCT is the worst option

All 11 basis arms, full 3/3 coverage, scored label-free (`--prior valid`) with
post-processing. Mean F1 over the four appliances:

| rank | arm | mean F1pp | | rank | arm | mean F1pp |
|---|---|---|---|---|---|---|
| 1 | **TCN_KL_scratch** (fitted KL, order 10) | **67.4** | | 7 | TCN_KLwave64 | 63.5 |
| 2 | TCN_KLwhiten | 66.2 | | 8 | TCN_KL64_13 | 62.1 |
| 3 | TCN_KLperapp | 66.0 | | 9 | TCN_KLwave64_16 | 61.3 |
| 4 | TCN_KLlearn | 65.2 | | 10 | TCN_KL64_16 | 60.3 |
| 5 | TCN_KL4 | 64.9 | | 11 | **TCN_KLdct** | **59.2** |
| 6 | *TCN_KLrand (control)* | *64.7* | | | | |

**The current choice is the right one.** None of the ten variants improves on plain fitted
KL at order 10. Whitening does not help (66.2), a **learnable** basis does not help (65.2),
per-appliance fitting does not help (66.0, confirming the earlier wash), order 4 is worse
than order 10 (64.9), and Daubechies wavelets are worse (63.5).

**The random-basis control is the interesting result.** `TCN_KLrand` lands only **2.7
points** behind fitted KL overall, and is indistinguishable on Fridge (88.4 vs 88.9) and
Microwave (58.3 vs 59.0). The entire fitted advantage is concentrated on **WashingMachine
(61.5 vs 53.8, +7.7)** and Dishwasher (60.3 vs 58.5, +1.8). So KL structure earns its keep
only on **long, structured activation cycles**; for a quasi-periodic load (Fridge) or a
short impulsive one (Microwave), any random projection of the same rank does as well.
Given the wide per-seed spread on these arms, the 2.7-point aggregate gap is weak evidence
on its own — the WashingMachine-specific gap is the solid part.

**A random basis beats fixed DCT by 5.6 points**, and beats four of the "principled"
variants. DCT is last.

**I predicted this wrong, and the correction matters.** Under the older oracle-prior,
no-post-processing scoring I reported DCT within 3.8 F1 of KL and *equal or slightly better*
on WM/DW, and predicted the gap would persist concentrated on Microwave and Fridge. Under
the label-free post-processed scoring DCT is **8.2 behind overall and 19.6 behind on
WashingMachine** — the gap vanished on Fridge (87.8 vs 88.9), persisted on Microwave
(46.6 vs 59.0), and *exploded* on the two appliances where DCT had looked competitive.

The mechanism is visible in the over-firing columns: on WashingMachine DCT over-fires
**3.55x** against the reference's 2.33x, and post-processing fails to repair it (**2.86x**
vs 1.41x). DCT emits long spurious runs that satisfy the duration rules and therefore
survive them, whereas the reference's errors are short and get deleted. This is a concrete
case of a conclusion that was an artefact of the scoring rule, and it is why every ranking
in this log is stated together with the scoring that produced it.

**Consequence for a KL+DCT hybrid: do not build it.** The motivation was that DCT appeared
to win on WM/DW while KL won on Microwave/Fridge, suggesting complementary structure. That
premise is now refuted — DCT loses everywhere under honest scoring, and is beaten by a
random basis. Mixing in a basis that underperforms noise is not a promising direction.

---

## 8. Infrastructure findings

Not science, but each cost real time and would cost it again.

* **Completion was judged by `.pt` existence alone.** The trainer writes ~416 KB
  intermediate checkpoints mid-run, so a killed cell **permanently marked itself done** and
  silently vanished from the grid — this is what produced two mysterious n=2 rows. Fixed
  with a `.pt.done` sidecar written only after a clean exit (`scripts/sweep_parallel.sh`).
* **`pkill -f <pattern>` matches your own shell.** Killed my own session twice; the Bash
  tool's command line contains the pattern being searched for. Use exact-argument matching:
  `ps -o args= | awk '$2=="scripts/tod_supervisor.sh"'`.
* **An NFS stall looks exactly like CPU saturation.** Load average 80 on 64 cores, GPUs
  idle, jobs apparently running. In fact every trainer sat in uninterruptible `D` state on
  `rpc_wait_bit_killable` having used **4-7 seconds of CPU in 80 minutes**; `python -c
  "print('hello')"` took 75 s wall for 0.03 s CPU. **Linux load average counts D-state
  processes**, so the load *was* the stall. I initially misread this as CPU oversubscription
  and scaled workers up, which was the wrong direction; it is reverted.
* **The duration rules were the bottleneck of their own tuning.** `bridge_short_gaps` and
  `drop_short_runs` looped in Python over runs, and a realistic 300k-sample prediction has
  **31,164 ON runs**. Vectorised via a difference-array fill: **6-8x faster**, verified
  bit-identical against the original loops across 23 shapes x 9 widths.

---

## 9. Open threads

| thread | status |
|---|---|
| Basis geometry / DCT / `KLrand` | DONE — fitted KL order 10 wins; DCT last; hybrid ruled out (§7) |
| `TCN_KL_act_tod_shuf` control | 3 cells, configured, never run — cheapest open item |
| UK-DALE | stored predictions exist; blockers: locate `ukdale.h5`, and **valid ⊂ train** makes it a stress test, not a replication |
| Cross-validated validation split | newly motivated by §5 regret; would unlock the +6-7 post-processing headroom and a steadier threshold |
| Model selection on event F1 | selection, early stopping and LR schedule all watch validation **MSE** (`trainer.py:151-188`); nothing in training knows about detection |
| `scripts/benchmark_flops.py` | not re-run for the new arms |
