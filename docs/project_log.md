# Capstone Project Log

### [Week of 2026_03_23]:

**Overview:**
- Harvesting pipeline is stable; videos are accruing at a steady clip (~300 new videos / day across 972 channels); data initial processing + modelling pipeline is funcional.

**Data Collection:**

- Added AI-generated (`synthetic_media`) flag to the YouTube API data collection, so that in the future I can check if there is significant AI-generated content in the data (there still may be, if it wasn't detected and flagged by YouTube).

**Modelling:**

- Finished with all of the preliminary feature engineering, in order to get to a point where I could actively model (and validate whether my hypothesis "holds water" at all).
- Implemented Synthetic Data generation to augment the real data I am collecting through the data harvester
- Bolstered the data processing pipeline and implemented Logistic Regression, Random Forrest, and XGBClassifier.
- Ensured that there is snapshot functionality for raw 
- Validated that the approach should work -- preliminary results from all 3 models indicate accuracy > 0.5 (better than random), and varying precision, recall and ROC-AUC scores. XGBoost is doing the best:

```
============================================================
XGBoost — Evaluated on Real Data Only
============================================================

ROC-AUC: 0.7971

                precision    recall  f1-score   support

Below Baseline       0.71      0.64      0.67       356
Above Baseline       0.74      0.80      0.77       458

      accuracy                           0.73       814
     macro avg       0.73      0.72      0.72       814
  weighted avg       0.73      0.73      0.73       814

Confusion Matrix:
[[228 128]
 [ 92 366]]
```

**Next Steps:**

1. Continue to let data accrue
1. Determine what additional feature engineering I can perform to boost model performance
1. Evaluate if I should add additional models
1. Determine if I should also add a Linear Regression prediction to the project for engagement rate -- outside of the scope of what I originally set out to do, but it's an interesting thought :-) Not sure if I will have time.