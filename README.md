<!-- academy:title -->
# Machine Learning for Tabular Data: Baselines and Validation

تعلّم الآلة للبيانات الجدولية: خطوط الأساس والتحقق من الصحة

Data Science Track · SDAIA Academy · DS-204 · Level 200

Take a tabular prediction from the decision it serves to a model card: a logged baseline ladder, validation that mirrors use, a leakage audit, a threshold that fits capacity and errors by subgroup with intervals

Participants predict which trainees of a fictional vocational training fund are likely not to complete, so that advisers can offer support early. On synthetic enrolment data with repeated trainees and cohorts, they climb a baseline ladder to tuned gradient-boosted trees, test an automated or pretrained challenger, choose validation that mirrors use, audit for leakage, set a threshold, break errors down by subgroup and write a model card. Deployment is left to AE-205 and AE-305.

**Format:** Intensive: 4 teaching days of four sessions and a capstone day, with guided pre-work and a tracked follow-up task

**Course page:** https://sdaiaacademy.github.io/ds-204/
<!-- /academy:title -->

## Outcomes

| | Outcomes | Level | Evidence |
|---|---|---|---|
| LO1 | Identify the leakage type and the fitting validation scheme in short tabular modelling cases, given six described problems and their data-collection process | Awareness | a table naming the leak and the scheme for each, at least five of six correct |
| LO2 | Apply a baseline ladder from a constant predictor to tuned gradient-boosted trees, logging every run, given identical folds and a fixed tuning budget | Applied | a run log with score and fold spread per rung |
| LO3 | Implement validation that matches the data structure and measure how far a random split overstates performance, given repeated trainees across five yearly cohorts | Applied | random, grouped and time-ordered scores with the gap explained |
| LO4 | Test an automated or pretrained challenger against the tuned model under identical folds and time budget, given the champion model and a challenger configuration | Applied | a paired comparison over folds with an interval and a one-line decision |
| LO5 | Analyse calibration, the decision threshold and held-out errors by subgroup, with intervals that reflect slice size, given predictions joined to region, gender, age band and provider, and adviser capacity | Applied | a calibration plot, a threshold with its precision and recall, and a subgroup error table |
| LO6 | Build a validated tabular model from framing to model card and defend its validation scheme and limits, given a synthetic case or an approved public tabular problem | Performance | a public repository with run log, leakage audit, subgroup results and model card, defended live |

<!-- academy:units -->
## Units

| Unit | Title | Sessions | Slides |
|---|---|---|---|
| Day 1 | Frame and baseline | A prediction serves a decision; The baseline ladder; Linear models first; Features from history | [`slides/day1.html`](slides/day1.html) |
| Day 2 | Trees and challengers | Gradient-boosted trees; Tuning under a budget; Automated and pretrained challengers; Is the challenger better | [`slides/day2.html`](slides/day2.html) |
| Day 3 | Validation and leakage | The split decides the score; Kinds of leakage; Hunting planted leaks; Reading someone else's notebook | [`slides/day3.html`](slides/day3.html) |
| Day 4 | Errors, groups and documentation | Thresholds for rare outcomes; Calibration; Errors by subgroup; The model card | [`slides/day4.html`](slides/day4.html) |
| Capstone | Capstone: a validated tabular model and its defence |  | [`slides/capstone.html`](slides/capstone.html) |
<!-- /academy:units -->

## Case study

**Sullam Vocational Training Fund (صندوق سُلَّم للتدريب المهني).** A fictional fund that pays for vocational programmes at 60 training providers in 13 regions. Its lead adviser asks which trainees are likely not to complete, so that advisers can offer support early. Participants build **Mubakkir (مبكّر)**, a score that at the end of week 2 of a programme ranks the trainees by the risk of not completing. The advisers can agree 800 support plans a year, so the score is judged as a ranking.

The data are synthetic and written by `labs/prepare_data.py` with a fixed seed: 45,000 enrolments of 30,000 trainees over five yearly cohorts (2021 to 2025). No row describes a real person, provider or programme. The table has repeated trainees, cohort drift, four planted leaks, registers that under-record absence in two regions and a held-out cohort that stays closed until Day 4. `labs/solutions/answer_key.md` lists what is planted and which lab finds it.

The people of the case: Dr. Haya Al-Mogren (director of trainee support, who decides whether the score is used), Ghassan Al-Fayez (lead adviser, who owns the capacity and the cost of each error), Maram Al-Rasheed (data protection officer), Hamza Al-Babtain (provider relations manager), Loay Al-Kadi (the analyst whose first notebook scores too well) and Razan Al-Angari (the data scientist: the role of the participant).

## How the course is organised

- **Pre-work (two hours):** `labs/00_prework_pipelines_cv.ipynb`, on pipelines and cross-validation, with ten self-check questions. A supervised entry quiz follows it on Day 1.
- **Four teaching days** of four sessions. Each session is about 25 minutes of concepts, one checkpoint and a 25-minute lab. Every lab starts from a supplied reference state, so no lab depends on another, and every lab has a solution.
- **One ladder, one log.** Every model is scored on the same five folds and logged with its mean score and fold spread: a constant, three rules, a regularised logistic model, boosted trees with defaults, a 20-trial search in nested folds, and an automated challenger on the same folds and time budget.
- **A capstone day:** an individual project defended in eight minutes.
- **Follow-up (tracked):** the leakage checklist of `labs/followup/` applied to one model from the participant's own work or to a public notebook.

Deployment, monitoring and retraining are not in this course: AE-205 and AE-305 start from the model card of Day 4.

### Assessment

- An AI-off supervised entry quiz and exit check with items in the style of the badge exam: 20%. Their questions are not in this repository; `assessments/bank.json` holds practice questions.
- The capstone repository, scored on the rubric: framing and baseline 15, validation and leakage 25, models and comparison 20, threshold, calibration and subgroup errors 25, model card and reproducibility 15: 60%.
- A live defence of the validation choice: 20%. The assessor swaps the split or removes a feature and the participant predicts the effect before re-running.

Passing needs 70% overall and at least 60% on both the capstone rubric and the supervised check.

### Use of AI assistants

- **AI-off:** the supervised checks, and the first attempt at each split (Lab 3.1), leakage hunt (Lab 3.3) and subgroup table (Lab 4.3).
- **AI-tutor:** library syntax in every other lab.
- **AI-full:** the capstone, with prompts and agent transcripts committed and every choice defended live.

No lab calls a language model, and nothing is downloaded except Python packages. Everything runs on a laptop processor or in a free hosted notebook.

### Responsible use

Mubakkir ranks people for support, so a wrong prediction is a harm to a trainee. Participants check whether the training data represent each region and age group, report errors by subgroup with intervals, relate each finding to a bias type of the SDAIA AI Bias Reference Guide, and state in the model card that the score informs an adviser and never decides eligibility alone.

<!-- academy:labs -->
## Labs

Every lab is a notebook in `labs/`. Open it in Google Colab from the course page, or run it locally with `uv`. Exercises raise `NotImplementedError` until they are done; the reference versions are in `labs/solutions/`.

| | Session | Lab | File |
|---|---|---|---|
| 1.1 | Day 1 · Session 1 | Frame the non-completion question | [`D1_1_frame.ipynb`](labs/D1_1_frame.ipynb) |
| 1.2 | Day 1 · Session 2 | A constant and three rules | [`D1_2_baselines.ipynb`](labs/D1_2_baselines.ipynb) |
| 1.3 | Day 1 · Session 3 | A logistic model in a pipeline | [`D1_3_logistic_pipeline.ipynb`](labs/D1_3_logistic_pipeline.ipynb) |
| 1.4 | Day 1 · Session 4 | Two history features | [`D1_4_history_features.ipynb`](labs/D1_4_history_features.ipynb) |
| 2.1 | Day 2 · Session 1 | Boosted trees with defaults | [`D2_1_boosted_defaults.ipynb`](labs/D2_1_boosted_defaults.ipynb) |
| 2.2 | Day 2 · Session 2 | Twenty trials in nested folds | [`D2_2_nested_search.ipynb`](labs/D2_2_nested_search.ipynb) |
| 2.3 | Day 2 · Session 3 | A challenger on equal terms | [`D2_3_challenger.ipynb`](labs/D2_3_challenger.ipynb) |
| 2.4 | Day 2 · Session 4 | The paired difference and the decision | [`D2_4_paired_difference.ipynb`](labs/D2_4_paired_difference.ipynb) |
| 3.1 | Day 3 · Session 1 | One model, three splits | [`D3_1_three_splits.ipynb`](labs/D3_1_three_splits.ipynb) |
| 3.2 | Day 3 · Session 2 | Six leaks from other projects | [`D3_2_six_leaks.ipynb`](labs/D3_2_six_leaks.ipynb) |
| 3.3 | Day 3 · Session 3 | Four planted leaks | [`D3_3_leak_hunt.ipynb`](labs/D3_3_leak_hunt.ipynb) |
| 3.4 | Day 3 · Session 4 | A colleague's notebook | [`D3_4_colleague_notebook.ipynb`](labs/D3_4_colleague_notebook.ipynb) |
| 4.1 | Day 4 · Session 1 | A threshold that fits capacity | [`D4_1_threshold_capacity.ipynb`](labs/D4_1_threshold_capacity.ipynb) |
| 4.2 | Day 4 · Session 2 | Calibrate and compare by decile | [`D4_2_calibration.ipynb`](labs/D4_2_calibration.ipynb) |
| 4.3 | Day 4 · Session 3 | Errors by region, gender and age band | [`D4_3_subgroup_errors.ipynb`](labs/D4_3_subgroup_errors.ipynb) |
| 4.4 | Day 4 · Session 4 | The model card and the capstone problem | [`D4_4_model_card.ipynb`](labs/D4_4_model_card.ipynb) |
<!-- /academy:labs -->

## Capstone

A tabular problem from a shortlist of synthetic and public cases, taken from framing to model card, in a public repository whose README reports the baseline ladder, validation, leakage audit, threshold, subgroup errors and a paired comparison; an eight-minute defence. The shortlist is in `labs/capstone/`: three synthetic cases (Bayader Microfinance, Awtad Equipment Rental, Nasaim Home Care) with a starter script and a README template, and three public tables that need the trainer's approval of the framing.

**Submission:** a link to a public GitHub repository: the code with a fixed seed and a lock file, the run log, the leakage audit, the subgroup results and the model card, and a README that reports the baseline ladder, the validation scheme, the leakage audit, the threshold, errors by subgroup and a paired comparison. Prompts and agent transcripts are committed with it.

## Prerequisites

- FN-101 Python for Data and AI Work
- FN-103 Mathematics for Machine Learning: Vectors, Gradients and Probability
- FN-105 How Machine Learning and Language Models Work
- DS-201 Statistical Inference by Simulation
- Joins and groups tables with a dataframe library
- Explains overfitting and the purpose of a held-out set

<!-- academy:tools -->
## Tools

Each line is a capability; the examples are interchangeable.

- **Analysis environment and notebooks:** Python with uv and JupyterLab or marimo, R with Quarto
- **Machine learning library with splitters and pipelines:** scikit-learn, tidymodels
- **Gradient-boosted trees:** LightGBM, XGBoost, CatBoost
- **Automated or pretrained challenger:** AutoGluon, a tabular foundation model used within its licence
- **Subgroup metrics:** Fairlearn, grouped metrics written in pandas
<!-- /academy:tools -->

<!-- academy:sources -->
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
<!-- /academy:sources -->

### National references

- SDAIA Principles of AI Ethics: https://sdaia.gov.sa/en/SDAIA/about/Documents/ai-principles.pdf
- SDAIA AI Bias Reference Guide (first edition, May 2026): https://sdaia.gov.sa/en/MediaCenter/KnowledgeCenter/ResearchLibrary/EN-AI-BIAS.pdf
- Personal Data Protection Law: https://sdaia.gov.sa/en/SDAIA/about/Documents/PersonalDataProtectionLaw.pdf
- Implementing Regulation of the Personal Data Protection Law: https://sdaia.gov.sa/en/SDAIA/about/Documents/ImplementingRegulationPersonalDataProtectionLaw.pdf
