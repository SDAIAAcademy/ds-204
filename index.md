---
course: "Machine Learning for Tabular Data: Baselines and Validation"
code: "DS-204"
version: "2027.1.0"
language: "en"
view: "trainee"
source: "https://sdaiaacademy.github.io/ds-204/index.html"
---

# Machine Learning for Tabular Data: Baselines and Validation

تعلم الآلة للبيانات الجدولية: خطوط الأساس والتحقق من الصحة

Data Science Track · 4 teaching days + capstone day · 26 hours · Level 200

Take a tabular prediction from the decision it serves to a model card: a logged baseline ladder, validation that mirrors use, a leakage audit, a threshold that fits capacity and errors by subgroup with intervals

Participants predict which trainees of a fictional vocational training fund are likely not to complete, so that advisers can offer support early. On synthetic enrolment data with repeated trainees and cohorts, they climb a baseline ladder to tuned gradient-boosted trees, test an automated or pretrained challenger, choose validation that mirrors use, audit for leakage, set a threshold, break errors down by subgroup and write a model card. Deployment is left to AE-205 and AE-305.

**Who it is for:** Practitioners. Graduates and early-career analysts who have completed the FN courses or passed the placement test, and analysts moving into data science.

**Format:** Intensive: 4 teaching days of four sessions and a capstone day, with guided pre-work and a tracked follow-up task

## The case study

**Sullam Vocational Training Fund.** Sullam Vocational Training Fund is a fictional fund that pays for vocational programmes at 60 training providers in 13 regions. Participants work on 45,000 synthetic enrolments of 30,000 trainees over five yearly cohorts to build Mubakkir, a score that at the end of week 2 puts the trainees most likely not to complete on an adviser's list. All data are synthetic, written with a fixed seed, with repeated trainees, cohort drift, four planted leaks and registers that under-record absence in two regions.

## The course

### Day 1: Frame and baseline

Slides: https://sdaiaacademy.github.io/ds-204/slides/day1.md

- **Session 1: A prediction serves a decision**
  - State the target, the unit and the prediction time from the decision a score serves
  - Separate the columns known at the prediction time from those recorded at exit
  - Compute the break-even probability from the cost of each error
  - [Lab 1.1: Frame the non-completion question (25 min)](https://sdaiaacademy.github.io/ds-204/labs/D1_1_frame.ipynb)
- **Session 2: The baseline ladder**
  - Explain why every model is scored on the same folds and logged
  - Score a constant predictor and say what its average precision is
  - Write a three-rule baseline and read its score and fold spread
  - [Lab 1.2: A constant and three rules (25 min)](https://sdaiaacademy.github.io/ds-204/labs/D1_2_baselines.ipynb)
- **Session 3: Linear models first**
  - Build a pipeline that imputes, encodes and scales inside each fold
  - Fit a regularised logistic model and log its score and fold spread
  - Read the largest coefficients and say what they do not prove
  - [Lab 1.3: A logistic model in a pipeline (25 min)](https://sdaiaacademy.github.io/ds-204/labs/D1_3_logistic_pipeline.ipynb)
- **Session 4: Features from history**
  - Explain why an earlier enrolment counts only when its outcome was known
  - Build two history features from records that ended before the prediction time
  - Run a time-travel test that fails when a feature reads the future
  - [Lab 1.4: Two history features (25 min)](https://sdaiaacademy.github.io/ds-204/labs/D1_4_history_features.ipynb)

### Day 2: Trees and challengers

Slides: https://sdaiaacademy.github.io/ds-204/slides/day2.md

- **Session 1: Gradient-boosted trees**
  - Describe how boosting adds small trees to correct the errors so far
  - Compare training and validation scores and read the gap
  - Read importance scores and name two of their limits
  - [Lab 2.1: Boosted trees with defaults (25 min)](https://sdaiaacademy.github.io/ds-204/labs/D2_1_boosted_defaults.ipynb)
- **Session 2: Tuning under a budget**
  - Name the settings of boosted trees that are worth a budget
  - Run a random search whose choices are made on inner folds
  - Report the gain over defaults with its fold spread and its cost in time
  - [Lab 2.2: Twenty trials in nested folds (25 min)](https://sdaiaacademy.github.io/ds-204/labs/D2_2_nested_search.ipynb)
- **Session 3: Automated and pretrained challengers**
  - Describe what an automated tabular tool and a tabular foundation model each do
  - Check a challenger's licence and where the data would be sent before using it
  - Run a challenger on the same folds and the same time budget as the champion
  - [Lab 2.3: A challenger on equal terms (25 min)](https://sdaiaacademy.github.io/ds-204/labs/D2_3_challenger.ipynb)
- **Session 4: Is the challenger better**
  - Explain why fold scores of two models are compared as pairs
  - Compute the mean paired difference with an interval
  - Decide against a minimum effect stated in advance, in one line
  - [Lab 2.4: The paired difference and the decision (25 min)](https://sdaiaacademy.github.io/ds-204/labs/D2_4_paired_difference.ipynb)

### Day 3: Validation and leakage

Slides: https://sdaiaacademy.github.io/ds-204/slides/day3.md

- **Session 1: The split decides the score**
  - Choose the split that mirrors how a model will be used
  - Implement random, grouped and time-ordered splits on the same rows
  - Explain each gap between the three scores from the structure of the data
  - [Lab 3.1: One model, three splits (25 min)](https://sdaiaacademy.github.io/ds-204/labs/D3_1_three_splits.ipynb)
- **Session 2: Kinds of leakage**
  - Distinguish target leakage, future information, group or duplicate contamination and preparation leakage
  - Identify the kind of leak from a description of how the data were collected
  - Match each case to the validation scheme its use requires
  - [Lab 3.2: Six leaks from other projects (25 min)](https://sdaiaacademy.github.io/ds-204/labs/D3_2_six_leaks.ipynb)
- **Session 3: Hunting planted leaks**
  - Use the time each column is recorded to find features written after the outcome
  - Find duplicated rows that sit on both sides of a split
  - Use importance scores to choose which features to question first
  - [Lab 3.3: Four planted leaks (25 min)](https://sdaiaacademy.github.io/ds-204/labs/D3_3_leak_hunt.ipynb)
- **Session 4: Reading someone else's notebook**
  - Read a notebook in the order rows, split, preparation, tuning, metric
  - List the faults that make a reported score untrustworthy
  - Measure which fault moves the score most by repairing one at a time
  - [Lab 3.4: A colleague's notebook (25 min)](https://sdaiaacademy.github.io/ds-204/labs/D3_4_colleague_notebook.ipynb)

### Day 4: Errors, groups and documentation

Slides: https://sdaiaacademy.github.io/ds-204/slides/day4.md

- **Session 1: Thresholds for rare outcomes**
  - Read precision and recall off a precision-recall curve for a rare outcome
  - Choose a threshold that fits the advisers' capacity and report its precision and recall
  - Explain what happens to a fixed threshold when the next cohort differs
  - [Lab 4.1: A threshold that fits capacity (25 min)](https://sdaiaacademy.github.io/ds-204/labs/D4_1_threshold_capacity.ipynb)
- **Session 2: Calibration**
  - Compare expected and observed rates by decile and read a reliability plot
  - Calibrate a model on rows it was not trained on, with sigmoid and isotonic methods
  - Explain why a weighted model ranks well and still cannot be read as a probability
  - [Lab 4.2: Calibrate and compare by decile (25 min)](https://sdaiaacademy.github.io/ds-204/labs/D4_2_calibration.ipynb)
- **Session 3: Errors by subgroup**
  - Compare the training data with the eligible population by region, age band and gender
  - Report held-out error rates by subgroup with intervals that reflect slice size
  - Relate each finding to a bias type of the SDAIA AI Bias Reference Guide and to one action
  - [Lab 4.3: Errors by region, gender and age band (25 min)](https://sdaiaacademy.github.io/ds-204/labs/D4_3_subgroup_errors.ipynb)
- **Session 4: The model card**
  - Assemble a model card from measured results
  - State the limits of the validation and the uses the model is not for
  - Prepare the hand-over a deployment team needs and frame the capstone problem
  - [Lab 4.4: The model card and the capstone problem (25 min)](https://sdaiaacademy.github.io/ds-204/labs/D4_4_model_card.ipynb)

### Capstone: Capstone: a validated tabular model and its defence

Slides: https://sdaiaacademy.github.io/ds-204/slides/capstone.md

## Capstone

A tabular problem from a shortlist of synthetic and public cases taken from framing to model card, in a public repository whose README reports the baseline ladder, validation, leakage audit, threshold, subgroup errors and a paired comparison; eight-minute defence

**What you submit:** A link to a public GitHub repository: the code with a fixed seed and a lock file, the run log, the leakage audit, the subgroup results and the model card, and a README that reports the baseline ladder, the validation scheme, the leakage audit, the threshold, errors by subgroup and a paired comparison; prompts and agent transcripts are committed with it

**Rubric:**

- Framing and baseline: 15%
- Validation and leakage: 25%
- Models and comparison: 20%
- Threshold, calibration and subgroup errors: 25%
- Model card and reproducibility: 15%

## Outcomes

- **LO1** Identify the leakage type and the fitting validation scheme in short tabular modelling cases, given six described problems and their data-collection process (Awareness) Evidence: a table naming the leak and the scheme for each, at least five of six correct
- **LO2** Apply a baseline ladder from a constant predictor to tuned gradient-boosted trees, logging every run, given identical folds and a fixed tuning budget (Applied) Evidence: a run log with score and fold spread per rung
- **LO3** Implement validation that matches the data structure and measure how far a random split overstates performance, given repeated trainees across five yearly cohorts (Applied) Evidence: random, grouped and time-ordered scores with the gap explained
- **LO4** Test an automated or pretrained challenger against the tuned model under identical folds and time budget, given the champion model and a challenger configuration (Applied) Evidence: a paired comparison over folds with an interval and a one-line decision
- **LO5** Analyse calibration, the decision threshold and held-out errors by subgroup, with intervals that reflect slice size, given predictions joined to region, gender, age band and provider, and adviser capacity (Applied) Evidence: a calibration plot, a threshold with its precision and recall, and a subgroup error table
- **LO6** Build a validated tabular model from framing to model card and defend its validation scheme and limits, given a synthetic case or an approved public tabular problem (Performance) Evidence: a public repository with run log, leakage audit, subgroup results and model card, defended live

## Tools

- **Analysis environment and notebooks:** Python with uv and JupyterLab or marimo, R with Quarto
- **Machine learning library with splitters and pipelines:** scikit-learn, tidymodels
- **Gradient-boosted trees:** LightGBM, XGBoost, CatBoost
- **Automated or pretrained challenger:** AutoGluon, a tabular foundation model used within its licence
- **Subgroup metrics:** Fairlearn, grouped metrics written in pandas

## Prerequisites

- FN-101 Python for Data and AI Work
- FN-103 Mathematics for Machine Learning: Vectors, Gradients and Probability
- FN-105 How Machine Learning and Language Models Work
- DS-201 Statistical Inference by Simulation
- Joins and groups tables with a dataframe library
- Explains overfitting and the purpose of a held-out set

## Sources

### National references

- [Principles of AI Ethics (SDAIA)](https://sdaia.gov.sa/en/SDAIA/about/Documents/ai-principles.pdf)
- [AI Bias Reference Guide, first edition (SDAIA, May 2026)](https://sdaia.gov.sa/en/MediaCenter/KnowledgeCenter/ResearchLibrary/EN-AI-BIAS.pdf)
- [Personal Data Protection Law (SDAIA)](https://sdaia.gov.sa/en/SDAIA/about/Documents/PersonalDataProtectionLaw.pdf)
- [Implementing Regulation of the Personal Data Protection Law (SDAIA)](https://sdaia.gov.sa/en/SDAIA/about/Documents/ImplementingRegulationPersonalDataProtectionLaw.pdf)

### Validation, pipelines and leakage

- [scikit-learn user guide: Cross-validation: evaluating estimator performance](https://scikit-learn.org/stable/modules/cross_validation.html)
- [scikit-learn user guide: Common pitfalls and recommended practices](https://scikit-learn.org/stable/common_pitfalls.html)
- [scikit-learn user guide: Pipelines and composite estimators](https://scikit-learn.org/stable/modules/compose.html)
- [scikit-learn example: Nested versus non-nested cross-validation](https://scikit-learn.org/stable/auto_examples/model_selection/plot_nested_cross_validation_iris.html)
- [Kapoor and Narayanan (2023). Leakage and the reproducibility crisis in machine-learning-based science](https://arxiv.org/abs/2207.07048)

### Models and challengers

- [LightGBM documentation: Parameters tuning](https://lightgbm.readthedocs.io/en/stable/Parameters-Tuning.html)
- [Grinsztajn, Oyallon and Varoquaux (2022). Why do tree-based models still outperform deep learning on tabular data?](https://arxiv.org/abs/2207.08815)
- [Bergstra and Bengio (2012). Random search for hyper-parameter optimization](https://www.jmlr.org/papers/v13/bergstra12a.html)
- [AutoGluon documentation: Tabular essentials](https://auto.gluon.ai/stable/tutorials/tabular/tabular-essentials.html)
- [Hollmann and others (2025). Accurate predictions on small data with a tabular foundation model](https://www.nature.com/articles/s41586-024-08328-6)
- [TabPFN repository: licence and usage terms (Prior Labs)](https://github.com/PriorLabs/TabPFN)

### Thresholds, calibration and importance

- [scikit-learn user guide: Probability calibration](https://scikit-learn.org/stable/modules/calibration.html)
- [scikit-learn user guide: Tuning the decision threshold for class prediction](https://scikit-learn.org/stable/modules/classification_threshold.html)
- [scikit-learn user guide: Permutation feature importance](https://scikit-learn.org/stable/modules/permutation_importance.html)
- [Saito and Rehmsmeier (2015). The precision-recall plot is more informative than the ROC plot when evaluating binary classifiers on imbalanced datasets](https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0118432)

### Subgroups and documentation

- [Mitchell and others (2019). Model cards for model reporting](https://arxiv.org/abs/1810.03993)
- [Fairlearn user guide: Assessment (MetricFrame)](https://fairlearn.org/main/user_guide/assessment/index.html)
- [SciPy manual: proportion_ci, confidence intervals for a proportion (Wilson method)](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats._result_classes.BinomTestResult.proportion_ci.html)

### Public tables for the capstone

- [UCI Machine Learning Repository: Predict Students' Dropout and Academic Success](https://archive.ics.uci.edu/dataset/697/predict+students+dropout+and+academic+success)
- [UCI Machine Learning Repository: Bank Marketing](https://archive.ics.uci.edu/dataset/222/bank+marketing)
- [UCI Machine Learning Repository: Default of Credit Card Clients](https://archive.ics.uci.edu/dataset/350/default+of+credit+card+clients)

---

SDAIA Academy · Data Science Track · Machine Learning for Tabular Data: Baselines and Validation

Course version 2027.1.0 · engine 1.0.0 · built 2026-11-15 · commit 6acdecd
