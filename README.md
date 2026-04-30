# UC Berkeley Professional Certificate in Machine Learning and Artificial Intelligence - Capstone Project

## Repository Summary
This is my Capstone Project for the 6-month "Professional Certificate in Machine Learning and Artificial Intelligence" program. The program includes instruction from the UC Berkeley School of Engineering and UC Berkeley Haas School of Business.

## Project Overview

**Research Question**

My research question is: *"Can we predict whether a YouTube video will achieve above-median engagement within 7 days of publication, using only signals observable at upload time or within the first 24 hours?"* 

Engagement is measured as likes and comments relative to views, benchmarked against that channel's own historical median.

---

**Data Sources**

All data is collected via the public YouTube Data API v3 (free tier). I built a custom harvesting pipeline that collects video metadata and metrics in three snapshots per video: at upload time, at \~24 hours post-publish, and at 7 days post-publish. The dataset currently spans \~965 channels across three verticals (Education, Lifestyle, Tech) and three subscriber tiers (1K–100K, 100K–1M, and 1M–10M), with over 10,000 video snapshot triplets collected to date and stored in Google BigQuery. No third-party datasets are used.

---

**Techniques**

This is a binary classification problem. My target variable is whether a given video's 7-day engagement rate exceeds that channel's own historical median. Using the historical median is a design choice that naturally balances the classes. Features include upload-time metadata (title length, tags, duration, category, thumbnail characteristics like brightness and face count expressed as numbers) as well as early-signal velocity features derived from the 24-hour snapshot (\`view\` and \`like\` growth rate in the first day). I am evaluating three model families:

* Logistic Regression with L1 (LASSO) regularization, for interpretability and feature selection  
* Random Forest, for handling non-linear relationships and providing feature importance rankings  
* XGBoost (gradient boosting), for predictive performance

I also plan to test a simple ensemble (averaged RF \+ XGBoost predictions) and am exploring segment-specific models broken out by vertical and subscriber tier.

---

**Expected Results**

I expect tree-based models (Random Forest, XGBoost) to outperform logistic regression, which would suggest that engagement dynamics are driven by non-linear interactions between features. For example, the effect of title length on engagement may differ depending on channel size or content category. ROC-AUC is my primary metric. E

Early results support this hypothesis, with tree models reaching \~0.80 AUC versus \~0.65 for logistic regression. Feature importance rankings — particularly for early-signal velocity features and upload-time metadata like video duration and whether the video qualifies as a YouTube Short — should to provide useful insight that will inform actionable takeaways.

---

**Why This Question Matters**

For individual YouTube creators, the days immediately following a video upload are critical. If a video isn't gaining traction early, there's often a narrow window to act — adjusting the title, updating the thumbnail, pushing the video in community posts or on social media, or even choosing to re-upload. Most creators are flying blind during this window, relying on gut instinct rather than data.

For a video platform at scale, this question matters even more. Early engagement prediction could power smarter recommendation seeding, better support tools for creators, and more accurate forecasting of which content will drive long-term retention. If a model can flag — within the first 24 hours — that a video is likely to underperform relative to a creator's own baseline, that signal is immediately useful to both the creator and the platform, without needing to wait a week to find out.

The framing around a channel's *own* historical baseline (rather than a global threshold) is a deliberate design choice that makes this question relevant across creator sizes: a 5,000-subscriber education channel and a 5-million-subscriber tech channel face very different engagement landscapes, and my model takes this into account.

## Project Documents

- [Project Overview](docs/jelani_gouldbailey_capstone_project_overview.pdf)
- [Design Docs](docs/)

## Design Notes & Code Organization

As the project size grew I found it helpful to split the code into more modular units. Instead of a single notebook for the entire project, I have multiple, 
purpose-built notebooks. For example EDA was conducted in one notebook, while 
modeling and hyperparameter tuning each have their own notebooks. Data, models and train/test/validation results were snapshotted to Google Cloud Storage for reuse. This separation of concerns allowed for cleaner adjustment of parameters, kept the implementation code separate from the write-up and results, and in some cases was necessary for time and performance management. For example, an early version of hyperparameter tuning took 30+ mins each time the notebook ran.

## Capstone Project Week 20 Check-In
For the Capstone check-in (Week 20 of the program), please review the EDA notebook linked [here](src/capstone/notebooks/eda.ipynb):

## Capstone Project week 24 Final:
pending.

## Project Code
- Notebooks:
  - [Exploratory Data Analysis](src/capstone/notebooks/eda.ipynb)
  - [Model Training](src/capstone/notebooks/retrain_models.ipynb)
  - [Hyper Parameter Tuning](src/capstone/notebooks/hyperparam_tuning.ipynb)
- [pipeline + utility code](src/capstone/)
- [unit tests](tests/)