# Capstone — Final Conclusions

*Predicting whether a YouTube video will beat its channel's own 7-day median
engagement, using only upload-time and 24-hour signals.*

This document is the narrative companion to notebooks
[`03_model_training_results.ipynb`](../src/capstone/notebooks/03_model_training_results.ipynb)
and [`05_final_model_selection.ipynb`](../src/capstone/notebooks/05_final_model_selection.ipynb).
It restates the headline results, but its real purpose is to draw the
project-level conclusion the full body of work points to: **performance here is
bounded by a structural ceiling in the signal, not by model choice or tuning.**
It also documents the scale of the data and modeling effort behind that finding,
and the deliberate architectural decision to harvest an entirely separate body of
data at the end purely to test generalization.

---

## 1. The question and the answer

**Problem.** Supervised binary classification: will a freshly uploaded video go on
to beat its channel's own 7-day median engagement rate? The target is
channel-relative by design (each video is judged against its *own* channel's
history, not a global popularity threshold), and is naturally ~50/50 balanced, so
the modeling challenge is ranking quality and error trade-offs rather than class
imbalance. The constraint that makes it useful — and hard — is that only
**upload-time and 24-hour** signals may be used to predict the **7-day** outcome.

**Answer.** The hypothesis holds. The best models rank above/below-baseline videos
at **~0.92 ROC-AUC in-distribution** and **~0.88 ROC-AUC** on verticals they never
trained on — far above the 0.5 random baseline, and stable across the entire
version history. Early-engagement signals (especially likes-per-hour at 24h),
normalized against channel scale and baseline history, genuinely predict 7-day
relative performance.

**Final model: XGBoost.** See §4.

---

## 2. The data harvesting effort

The dataset is not a static download — it was **harvested continuously** by a
purpose-built collection system running on Google Cloud, then versioned into
immutable snapshots. Four independent Cloud Run services
([`src/capstone/data_collection/`](../src/capstone/data_collection/)) cooperate:

| Service | Role |
|---------|------|
| **discovery** | Finds channels to track, tier-aware (Small / Medium / Large per subscriber size), with upload-velocity filtering (≥ ~1 video/week) and a candidate buffer so a mid-run quota crash never loses validated channels. |
| **harvester** | The core engine. Runs **every 3 hours**, scanning tracked channels for new uploads and polling each video at **three points in its life** — `upload` (0–6h), `24h` (20–30h), and `7d` (156–180h) — writing a snapshot row at each. Captures view/like/comment counts, subscriber count at poll time, thumbnail features (faces, brightness, colorfulness), duration, category, full description, and YouTube's AI-generated-media flag. |
| **baselines** | Gathers the last ~30 lifetime videos per channel to compute the **channel-relative baseline medians** the target is defined against — explicitly excluding any video already being tracked, to prevent leakage. |
| **validation** | A health monitor that checks completion rates across all three poll stages and flags gaps. |

The harvesting is genuinely **longitudinal**: a single usable training row only
exists once a video has been observed at all three poll points spanning a full
week, on top of YouTube Data API quota management (retries disabled, consecutive-
error bail-out, per-run quota accounting). The data accrued steadily for months at
roughly **~300 new videos/day across ~970+ channels**.

**Volume that reached the model (snapshot `v3.5_real`):**

- **106,626** raw poll rows harvested →
- **32,571** videos with all three poll points complete →
- **32,126** rows in the final modeling table (after baseline-join and cleaning),
  each with **85** engineered columns.
- Channel baselines (`v5.0`): **40,841** baseline videos across **1,375** channels.

So **over 30,000 fully-observed, week-long video trajectories** underpin the
results — every one of them the product of multiple timed API polls rather than a
one-shot scrape.

---

## 3. The modeling effort

The modeling was equally extensive and was tracked as an explicit, immutable
version lineage (see [`docs/versioning.md`](./versioning.md)):

- **51 distinct model snapshots persisted to GCS across 13 versions** (v1.0 →
  v6.2). Early versions carried 3 families (Logistic Regression, Random Forest,
  XGBoost); the family was progressively expanded to **7** by v6.1/v6.2 — adding
  LightGBM, an MLP, a soft-voting ensemble, and a stacking ensemble
  (RF + XGB + LGB → logistic meta-learner, 5-fold CV).
- That count of 51 **does not include** the per-version retrains during
  feature-engineering iteration, the **100-iteration × 5-fold random
  hyperparameter searches** for XGB / LGB / MLP (thousands of underlying fits), or
  the **tier-specialized explorations** (9 per-tier sub-models plus several
  sample-weighting sweeps — §5.2), all of which were trained, evaluated, and in
  the case of the tiered sub-models, archived to a dedicated `experiments/` namespace.
- Every model was scored on a **locked validation holdout** (fixed video IDs in
  GCS, stratified across an 18-cell vertical × tier × class key) so that
  comparisons across versions are honest and reproducible, not artifacts of a
  lucky re-split.

In short: this conclusion rests on multiple rounds of feature engineering, dozens of independently trained models, an extensive hyperparameter search on the leading candidates, and two distinct architectural attempts to beat the most difficult segment — all
converging on the same place.

---

## 4. Final model selection: XGBoost

The four strongest models — stacking ensemble, XGBoost, LightGBM, and the voting
ensemble — finished the v6.2 bake-off **within ~0.003 ROC-AUC of one another** on
the locked holdout, a spread well inside run-to-run noise. In-distribution, no
single metric separates them decisively: stacking and XGB lead ROC-AUC by a hair,
while LightGBM edges ahead on the threshold-dependent metrics (accuracy, macro and
per-class F1).

The **out-of-bounds holdout broke the tie in favor of XGBoost** (see §6 and
notebook 05):

- **Best out-of-distribution ranking** — XGB posts the highest ROC-AUC on the
  unseen Music/Sports videos.
- **Smallest generalization gap** — XGB degrades least from in-distribution to
  out-of-bounds; it is the most reliable when pushed onto content it never saw.
- **LightGBM's in-distribution edge does not transfer** — the model that looked
  best on the locked holdout's threshold metrics is the *weakest* of the three
  out of distribution, with the largest gap. Its advantage was specific to the
  trained verticals.
- **The stacking ensemble's performance does not justify its complexity** — it merely matches XGB
  in-distribution and is at best level with it out-of-bounds, while costing three
  base learners plus a meta-learner in training time, inference latency, and
  interpretability.

XGBoost is therefore the **strongest generalizer, the simplest performant option,
and fully interpretable** via feature importances — selected as the final
deployment model. LightGBM remains a credible in-distribution alternative;
stacking is retained only as evidence that ensembling does not pay off here.

**What the model learned.** Across XGB, LGB, and RF the dominant predictor is
`like_rate_24h` (likes-per-hour at the 24-hour mark) — the cleanest early momentum
signal — followed by channel-context features: baseline video count, `tier_encoded`,
baseline-normalized view/like ratios, and velocity ratios. This confirms the
project thesis: *channel-relative, contextual* signals outrank raw absolute counts.

---

## 5. The structural ceiling

The central conclusion of the project is that **the work has mapped out a ceiling
that is a property of the signal, not of the modeling.** Four independent lines of
evidence converge on it.

### 5.1 The global ceiling is real and stable

In-distribution ROC-AUC plateaus at **~0.92** and has barely moved since the
gradient-boosting models matured, despite more data, more features, more model
families, and extensive tuning. XGBoost and LightGBM — two independently
implemented gradient-boosting libraries — **converge to nearly identical scores**
(Δ ≈ 0.001 AUC). When two different implementations of the same idea land in the
same place, and stacking them on top of each other adds essentially zero
(Δ ≈ 0.0001 AUC), the limiting factor is no longer the model — it is how much
separable signal the features contain.

### 5.2 The hardest segment, `tier=S`, is a *proven* intrinsic ceiling

Every tree model — including XGBoost — drops **~0.04 ROC-AUC on small
channels (`tier=S`)** relative to its global score (e.g. XGB ~0.88 vs ~0.92). This
is the single consistent blind spot across models (verticals, by contrast, show none). Tier-specific performance improvements were explored: **two separate model-side interventions were
built, tested, and rejected** (full write-up in
[`docs/tier_split_models.md`](./tier_split_models.md)):

1. **Per-tier specialization** — dedicated S/M/L sub-models with prediction-time
   routing. This *backfired*: splitting the data cost training volume and
   cross-tier signal transfer, scoring worse on S **and** globally.
2. **`tier=S` sample up-weighting** — refitting on all rows but weighting small-
   channel rows more heavily. Also failed: `tier=S` AUC stayed flat-to-worse while
   global AUC fell monotonically. The apparent "narrowing" of the blind-spot gap
   was an illusion — the global ceiling fell to meet S, the opposite of the goal.

The reason: ROC-AUC is rank-based and threshold-independent, so
*caring more* about small-channel rows cannot manufacture separability that the
features do not contain. Small channels simply carry **noisier signal** (e.g. unstable normalized engagement rates). The `tier=S` ceiling of
**~0.88 is intrinsic to the data for low-volume channels**, and the only remaining
lever with real upside is feature-side (encoding baseline *reliability*), not
model-side.

### 5.3 The linear ceiling

Logistic Regression sits **~0.15 ROC-AUC below** the tree models and has not
closed that gap in any version, even with stronger regularization. The signal is
fundamentally non-linear — velocity ratios, subscriber-normalized rates, tier
interactions, and the `is_short × like_rate_24h` cross — so a linear boundary has
its own, lower ceiling. LR is retained as an interpretable baseline, not a
candidate.

### 5.4 The out-of-distribution ceiling

On the OOB holdout (§6), all candidates settle at **~0.88 ROC-AUC** — a graceful,
consistent ~0.04 drop from in-distribution, not a collapse. The model transfers,
but there is a clear, repeatable upper bound on how well early signal predicts
late outcome once you leave the trained verticals.

**Taken together:** different model families, different complexities, two targeted
fixes for the worst segment, and an unseen-data test all bump into the same
boundary. This seems indicative of a *signal-limited* problem. The most important
result of all this training is therefore a negative-but-valuable one: **we now know
where the ceiling is and why it is there**, which is itself a defensible scientific
conclusion and a clearer pathway for future explorations (more feature collection + feature engineering,
not more models).

---

## 6. The out-of-bounds validation architecture

A deliberate architectural decision shaped the end of the
project: **harvesting an entirely separate body of data for the sole purpose of
testing generalization.**

The models are trained and validated only on three verticals —
**Tech, Lifestyle, Education** (`MODELING_VERTICALS`). But the data-collection
system was also pointed at two **held-out verticals it would never train on —
Music and Sports** (`GEN_VERTICALS`). The pipeline's `DataSplitter` routes every
video from a non-modeling vertical into a dedicated generalization set
(`run.df_gen` / `X_gen` / `y_gen`), entirely walled off from train/test/val.

This is a stronger test than a normal holdout. A standard validation set asks
"does the model work on *unseen videos* from *seen channels and topics*?" The OOB
set asks the harder question: **"does the model work on entire content categories
it has never encountered?"** Music and Sports have different engagement dynamics,
upload cadences, and audience behavior from the trained verticals — so strong OOB
performance is real evidence the model learned *transferable structure* about early
engagement, not vertical-specific quirks.

Executing the OOB validation (notebook 05) confirmed both halves of the story: Our model **does**
generalize (~0.88 AUC on 2,447 never-seen Music/Sports videos), and it does so with
a **bounded, predictable degradation** — which is exactly what reinforced the
structural-ceiling conclusion above and served as the tiebreaker that selected
XGBoost as the final model.

---

## 7. Known limitations

- **`tier=S` blind spot** (~0.04 AUC below global) — intrinsic to low-volume
  channel data; proven not fixable model-side (§5.2).
- **Out-of-distribution drop** (~0.04 AUC) — real and graceful; verticals far from
  the trained three should be treated as lower-confidence until represented in
  training.
- **Margins between top candidates are small** (~0.002–0.004 AUC). The XGBoost
  recommendation is *directionally* robust (it wins OOB ranking **and**
  generalization gap), but these are point estimates — bootstrap confidence
  intervals would firm them up.
- **Possible residual synthetic content** — YouTube's AI-media flag is captured and
  filtered, but undetected AI-generated content may remain in the data. It is unclear whether AI-generated content has different engagement dynamics.

---

## 8. Future work

- **Exploration of additional feature capture**: 23 raw features are captured prior to feature engineering. There are additional signals available from the YouTube v3 API which could be explored for enhanced training runs.
- **Feature-side reliability encoding** for low-volume channels (variance /
  dispersion of the baseline rates, not just their median) — the one lever §5.2
  leaves open for the `tier=S` ceiling.
- **Operating-threshold tuning**, e.g. a lower decision threshold for `tier=S`
  predictions, since the limitation is separability rather than calibration.
- **Broaden OOB coverage** as the Music/Sports set grows, and fold additional
  verticals into training once enough data accrues. (A further ~1k real triplets
  are expected in the near future, which will support the next snapshot cycle.)
- **Confidence intervals** on the validation and OOB tables, so the final
  comparison rests on intervals rather than point estimates.

---

## 9. Closing

The project set out to test whether early engagement predicts channel-relative
7-day performance, and the answer is a confident yes — ~0.92 AUC in-distribution,
~0.88 on entirely unseen content categories, from a single interpretable XGBoost
model. But the more durable contribution is what the **51 models across 13
versions, the extensive tuning, the two rejected
tier-specialization strategies, and the dedicated out-of-bounds harvest** together
establish: a clear, well-evidenced **structural ceiling**. Multiple model
families, multiple complexity levels, two targeted fixes for the worst segment, and
a true generalization test all arrive at the same boundary. Knowing precisely where
that ceiling sits — and that it lives in the signal rather than the model — is a strong and defensible conclusion for this project.