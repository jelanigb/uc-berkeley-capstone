# Tier-Split Models — Approach & Results

**Date:** 2026-05-30
**Status:** ❌ Negative result — approach rejected, not persisted as a model version.
**Basis:** model v6.2, data snapshot v3.5_real (19,589 train / 5,192 locked val rows).
**Code:** [`modeling/tier_routing.py`](../src/capstone/modeling/tier_routing.py),
[`utils/snapshot_experiment.py`](../src/capstone/utils/snapshot_experiment.py),
exploratory cells in [`notebooks/03_model_training_results.ipynb`](../src/capstone/notebooks/03_model_training_results.ipynb),
tests in [`tests/tier_routing_test.py`](../tests/tier_routing_test.py) and
[`tests/snapshot_experiment_test.py`](../tests/snapshot_experiment_test.py).
**Preservation:** §8 — saved to `experiments/tier_split/` in GCS (separate from the
`models/` lineage).

---

## 1. Motivation

The v6.2 segment audit flags **tier=S** (small channels) as the only consistent
blind spot. Every tree model drops 0.037–0.041 ROC-AUC below its global score on
S, while tier=M / tier=L stay strong and no *vertical* segment shows a blind spot:

| model | global AUC | tier=S AUC | drop |
|---|---|---|---|
| ensemble_stacking | 0.9201 | 0.8829 | 0.0372 |
| xgb | 0.9200 | 0.8821 | 0.0379 |
| lgb | 0.9188 | 0.8811 | 0.0377 |
| rf | 0.8861 | 0.8450 | 0.0411 |

**Hypothesis tested:** giving tier=S its own model — so it is no longer "diluted"
by M/L during training — would close the gap on small channels.

---

## 2. Approach

We trained one sub-model per tier for the three strongest families (XGB,
LightGBM, stacked ensemble) and routed each row to its tier's sub-model at
prediction time.

- **`TierRoutedClassifier`** — holds `{S: …, M: …, L: …}` sub-models and exposes
  a single `predict_proba` / `predict`. It routes each row to its tier's model by
  **DataFrame-index lookup** and reassembles the per-row probabilities in input
  order. Because it presents one `predict_proba`, it drops into the existing
  `Validator`, `SegmentAuditor`, and `ModelResult.from_sklearn` unchanged.
- Each sub-model is a `sklearn.clone()` of the trained global v6.2 model — i.e. it
  **inherits the global model's hyperparameters** — refit on only that tier's rows.

Two variants were compared head-to-head against the v6.2 globals:

- **`*_tier_full`** — separate S / M / L sub-models; each row routed to its own.
- **`*_tier_Sspec`** — a tier=S specialist only; M and L fall back to the v6.2 global.

### Why metrics are pooled, not averaged

Routing happens at the **prediction** level. Every row is scored by its tier's
sub-model, the per-row probabilities are reassembled into one full-length vector,
and each metric is computed **once over that pooled vector** — identical to how
the global models are scored, so the numbers are directly comparable. This is
**not** an average of per-tier metrics: ROC-AUC is non-decomposable
(`pooled AUC ≠ mean(AUC_S, AUC_M, AUC_L)`) because the pooled score also ranks
S-rows against M/L-rows.

### Training data per tier

| tier | train rows |
|---|---|
| L | 6,983 |
| M | 6,833 |
| S | 5,773 |
| **all (global)** | **19,589** |

Each S-specialist therefore trains on **5,773 rows vs 19,589** for the global — about 30%.

---

## 3. Results

### 3.1 Pooled validation metrics (global vs tiered)

| model | ROC-AUC | accuracy | f1_macro | f1_above | f1_below |
|---|---|---|---|---|---|
| xgb | **0.9200** | 0.8432 | 0.8407 | 0.8606 | 0.8209 |
| lgb | 0.9188 | 0.8442 | 0.8416 | 0.8619 | 0.8213 |
| ensemble_stacking | **0.9201** | 0.8428 | 0.8405 | 0.8599 | 0.8210 |
| xgb_tier_full | 0.9160 | 0.8415 | 0.8390 | 0.8590 | 0.8190 |
| xgb_tier_Sspec | 0.9166 | 0.8409 | 0.8383 | 0.8588 | 0.8179 |
| lgb_tier_full | 0.9130 | 0.8369 | 0.8341 | 0.8556 | 0.8125 |
| lgb_tier_Sspec | 0.9070 | 0.8411 | 0.8382 | 0.8598 | 0.8166 |
| ensemble_stacking_tier_full | 0.9140 | 0.8415 | 0.8390 | 0.8590 | 0.8191 |
| ensemble_stacking_tier_Sspec | 0.9172 | 0.8396 | 0.8369 | 0.8577 | 0.8162 |

Every tiered variant has **lower global AUC** than its global counterpart.

### 3.2 Per-tier ROC-AUC (segment audit)

| tier | xgb | xgb_full | xgb_Sspec | lgb | lgb_full | lgb_Sspec | stack | stack_full | stack_Sspec |
|---|---|---|---|---|---|---|---|---|---|
| L | 0.9481 | 0.9446 | 0.9481 | 0.9463 | 0.9418 | 0.9463 | 0.9483 | 0.9424 | 0.9483 |
| M | 0.9187 | 0.9183 | 0.9187 | 0.9179 | 0.9156 | 0.9179 | 0.9182 | 0.9166 | 0.9182 |
| **S** | **0.8821** | 0.8720 | 0.8720 | **0.8811** | 0.8692 | 0.8692 | **0.8829** | 0.8698 | 0.8698 |
| ALL | 0.9200 | 0.9160 | 0.9166 | 0.9188 | 0.9130 | 0.9070 | 0.9201 | 0.9140 | 0.9172 |

**The headline:** on tier=S — the segment this was designed to fix — every
specialized variant is **0.010–0.013 AUC worse** than the plain global model. The
dedicated S sub-model is a *worse* predictor of small channels than the model
trained on all tiers.

### 3.3 Routing correctness (sanity checks)

The numbers confirm the wrapper routes correctly, so the negative result is real,
not a bug:

1. **`tier_full` and `tier_Sspec` give identical tier=S AUC** within each family
   (xgb both 0.8720, lgb both 0.8692, stack both 0.8698) — correct, because on S
   rows both variants route to the *same* S sub-model.
2. **For `tier_Sspec`, tier=M and tier=L AUC match the global exactly** (e.g.
   stack_Sspec: M=0.9182, L=0.9483) — correct, because M/L fall back to the global.

---

## 4. Interpretation — why it failed

1. **Less data outweighs more focus.** The S-specialist trains on ~30% of the
   rows (5,773 vs 19,589). Boosted trees and stacking are data-hungry; the
   variance from a smaller training set costs more than specialization buys.
2. **Cross-tier signal transfer is real and valuable.** M/L rows teach
   generalizable structure (how velocity and normalized engagement map to
   outperformance) that *also* helps S. The global model exploits it; the
   specialist discards it. The tier=S gap was therefore **never** the global model
   being "diluted" by M/L — small channels are *intrinsically* noisier. The
   baseline harvester pulls a uniform last-30 videos per channel regardless of
   tier, so this is not a matter of collecting *less* baseline data for small
   channels; rather, each small-channel video's engagement rate swings far more in
   relative terms, so the median of those 30 baseline videos is a less stable
   estimate of "typical" performance. Partitioning the data cannot manufacture
   signal that is not there.
3. **Pooled AUC pays a calibration tax.** Routing mixes probabilities from
   independently-trained sub-models on slightly different scales, degrading
   cross-tier ranking and lowering the global AUC. Most visible in
   `lgb_tier_Sspec` (0.9070). This was a known risk going in; it showed up
   empirically.

---

## 5. ⚠️ Caveat — hyperparameters were transferred, not tuned

**This comparison is not a fully fair test of a tier=S specialist.** Each
sub-model inherited the **global models' hyperparameters** via `clone()` — params
tuned in notebook 04 for ~20k rows (e.g. XGB `n_estimators=500, max_depth=6`).
Those are very likely **too complex for ~5.8k S rows**, so some of the S-specialist
deficit is overfitting from transferred params rather than the partition itself.

A **tuned** S-specialist (shallower, more regularized, fewer estimators) is the
only variant that could realistically narrow the gap and was **not** evaluated
here. Expectations should stay modest, though: tuning addresses over-complexity,
but not the data-volume (point 1) or signal-transfer (point 2) disadvantages, so
its realistic ceiling is roughly *matching* the global's 0.8821 on S, not beating
it. If the tier=S blind spot is revisited, **tuning the S sub-model is the
prerequisite experiment before any firm conclusion that per-tier models cannot
help.**

---

## 6. Conclusion & recommendation

- **Do not persist** the tiered models as a model version: every variant is worse
  on tier=S *and* worse globally — strictly dominated.
- **Re-target the tier=S gap with signal-preserving methods** instead of
  data-partitioning, in rough order of promise:
  1. **Sample weighting / oversampling S inside the global model** — keeps all
     19,589 rows and the cross-tier transfer while up-weighting S. *(Tried — also
     a negative result; see §9.)*
  2. **Per-tier threshold tuning** — will not move AUC (rank-based) but can improve
     S's precision/recall trade-off at deployment.
  3. **Robust low-volume features** — encode baseline *reliability* for channels
     with few baseline videos, the apparent root cause of the noise.
  4. **Accept it** — tier=S at ~0.882 is still strong and may be the intrinsic
     ceiling for noisy small-channel data; a defensible project conclusion.
- **Keep** `modeling/tier_routing.py`, the comparison cells, and this document.
  "We tried per-tier specialization and it backfired, and here is the mechanism"
  is a substantive negative result for the project narrative.

---

## 7. Reproduce

In `notebooks/03_model_training_results.ipynb`, run top-to-bottom through the
**"Tier-specialized models (exploratory)"** section with any `RUN_MODE` (it needs
the trained globals in `run.models` and `df_val_seg` from the segment-audit cell).
`RUN_MODE = "OLD_MODEL_OLD_DATA"` reproduces the run above with no GCS writes.

## 8. Preserving these results in GCS

Although the models are not promoted to a version, the run can be preserved for
later comparison via [`utils/snapshot_experiment.py`](../src/capstone/utils/snapshot_experiment.py),
kept deliberately separate from the canonical `models/` lineage so it never
pollutes the cross-version trajectory plots.

- **Layout:**
  - `experiments/tier_split/results.jsonl` — append-only, one JSON line per run
    (run-timestamped), carrying pooled metrics, the per-tier segment audit, and
    provenance (data version, basis model, per-tier row counts, the
    hyperparameter-transfer caveat).
  - `experiments/tier_split/models/{run_id}/` — the 9 fitted sub-models
    (`{family}_{tier}.pkl`), the fitted `scaler.pkl`, and a `manifest.json` with
    feature-column order and reconstruction notes.
- **Write it:** in notebook 03's *"Preserve the experiment (GCS)"* cell, set
  `SAVE_TIER_EXPERIMENT = True` (default `False`; independent of `RUN_MODE`).
- **Read it back:** `load_tier_split_experiments()` returns one row per
  (run, model). Reconstruct a router from a saved run with
  `modeling.tier_routing.TierRoutedClassifier` plus the manifest's feature order
  and `scaler.pkl`.

---

## 9. Follow-up: tier=S sample weighting (also rejected)

**Date:** 2026-06-02
**Status:** ❌ Negative result — does not improve tier=S.
**Code:** [`modeling/tier_weighting.py`](../src/capstone/modeling/tier_weighting.py),
tests [`tests/tier_weighting_test.py`](../tests/tier_weighting_test.py), notebook 03
*"Tier=S sample weighting (exploratory)"* section.

§6 proposed sample weighting as the signal-preserving alternative to the split:
clone each v6.2 global, refit on **all** 19,589 rows (volume + cross-tier transfer
intact) but pass a `sample_weight` that up-weights tier=S. The result is a single
global model — no routing — so it is directly comparable to the v6.2 globals. We
swept S-weight multipliers {2, 3, 5} for XGB / LGB / stacking.

### Result

Tier=S AUC is flat-to-worse at every weight, while global AUC falls monotonically:

| family | tier=S AUC (x1 -> x2 -> x3 -> x5)    | global AUC (x1 -> x5) |
|--------|--------------------------------------|-----------------------|
| xgb    | 0.8821 -> 0.8808 -> 0.8784 -> 0.8801 | 0.9200 -> 0.9140      |
| lgb    | 0.8811 -> 0.8805 -> 0.8797 -> 0.8809 | 0.9188 -> 0.9138      |
| stack  | 0.8829 -> 0.8804 -> 0.8783 -> 0.8786 | 0.9201 -> 0.9140      |

The segment auditor's tier=S "drop" *shrinks* at higher weights (e.g. xgb
0.0379 → 0.0338), but this is misleading: the gap closes because the **global
ceiling falls to meet S**, not because S rises. Lowering the top is the opposite
of the goal — the pooled table is the metric that matters.

### Why it can't help

`sample_weight` changes *which errors the loss penalizes* (shifting the decision
boundary / class emphasis for S), but **ROC-AUC is rank-based and
threshold-independent** — emphasizing S rows does not make their signal more
separable. The threshold-dependent check agrees: tier=S **accuracy** is also flat
(xgb 0.7965 → 0.7913 → 0.7959 → 0.7881), so there is no operating-point benefit
either.

### Combined conclusion

Two model-side interventions have now failed for the same underlying reason — the
**per-tier split** (§1–6: less data + lost cross-tier transfer) and **sample
weighting** (§9: full data, but reweighting can't manufacture separability).
Together they establish that **tier=S ≈ 0.882 is an intrinsic ceiling**: the
limitation lives in the data/signal for small channels (high-variance per-video
engagement, so noisier normalized rates and a less stable channel median — and,
for genuinely new channels, sometimes fewer than the 30 baseline videos the
pipeline targets), not in how the model allocates capacity or attention.

**Remaining options:** (a) **feature-side** — encode baseline *reliability* for
low-volume channels, the apparent root cause; or (b) **accept** ~0.882 as the
floor, which is a defensible project conclusion given two failed model-side fixes.
No further model-side reweighting/partitioning is worth pursuing.
