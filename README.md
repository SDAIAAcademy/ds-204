<!-- academy:title -->
# Machine Learning for Tabular Data: Baselines and Validation

تعلم الآلة للبيانات الجدولية: خطوط الأساس والتحقق من الصحة

Data Science Track · SDAIA Academy · DS-204 · Level 200

REPLACE: one line under the title: what participants can do after the course

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
| Capstone | Capstone project presentations |  | [`slides/capstone.html`](slides/capstone.html) |
<!-- /academy:units -->

## Case study

**REPLACE: the name of the fictional organisation every lab works on.** REPLACE: two sentences on the case study and its synthetic data

<!-- academy:labs -->
## Labs

Every lab is a notebook in `labs/`. Open it in Google Colab from the course page, or run it locally with `uv`. Exercises raise `NotImplementedError` until they are done; the reference versions are in `labs/solutions/`.

| | Session | Lab | File |
|---|---|---|---|
| 1.1 | Day 1 · Session 1 | Write target, unit, prediction time and adviser action for the question | [`D1_1_write_target_unit.ipynb`](labs/D1_1_write_target_unit.ipynb) |
| 1.2 | Day 1 · Session 2 | Score a constant predictor and a three-rule baseline; log both | [`D1_2_score_constant_predictor.ipynb`](labs/D1_2_score_constant_predictor.ipynb) |
| 1.3 | Day 1 · Session 3 | Fit a regularised logistic model in a pipeline; log score and spread | [`D1_3_fit_regularised_logistic.ipynb`](labs/D1_3_fit_regularised_logistic.ipynb) |
| 1.4 | Day 1 · Session 4 | Add two history features from past records only; pass a time-travel test | [`D1_4_add_two_history.ipynb`](labs/D1_4_add_two_history.ipynb) |
| 2.1 | Day 2 · Session 1 | Train boosted trees with defaults; compare training and validation scores | [`D2_1_train_boosted_trees.ipynb`](labs/D2_1_train_boosted_trees.ipynb) |
| 2.2 | Day 2 · Session 2 | Run a 20-trial search in nested folds and record the gain | [`D2_2_run_20_trial.ipynb`](labs/D2_2_run_20_trial.ipynb) |
| 2.3 | Day 2 · Session 3 | Run a challenger on the same folds and time budget | [`D2_3_run_challenger_same.ipynb`](labs/D2_3_run_challenger_same.ipynb) |
| 2.4 | Day 2 · Session 4 | Compute the paired fold difference with an interval and decide | [`D2_4_compute_paired_fold.ipynb`](labs/D2_4_compute_paired_fold.ipynb) |
| 3.1 | Day 3 · Session 1 | Score one model under random, grouped and time-ordered splits | [`D3_1_score_one_model.ipynb`](labs/D3_1_score_one_model.ipynb) |
| 3.2 | Day 3 · Session 2 | Classify six leaks from other projects by type | [`D3_2_classify_six_leaks.ipynb`](labs/D3_2_classify_six_leaks.ipynb) |
| 3.3 | Day 3 · Session 3 | Find four planted leaks with timestamps, duplicates and importance scores | [`D3_3_find_four_planted.ipynb`](labs/D3_3_find_four_planted.ipynb) |
| 3.4 | Day 3 · Session 4 | List what makes a colleague's notebook score untrustworthy | [`D3_4_list_what_makes.ipynb`](labs/D3_4_list_what_makes.ipynb) |
| 4.1 | Day 4 · Session 1 | Pick a threshold on the precision-recall curve that fits adviser capacity | [`D4_1_pick_threshold_precision.ipynb`](labs/D4_1_pick_threshold_precision.ipynb) |
| 4.2 | Day 4 · Session 2 | Calibrate the boosted model; compare expected and observed by decile | [`D4_2_calibrate_boosted_model.ipynb`](labs/D4_2_calibrate_boosted_model.ipynb) |
| 4.3 | Day 4 · Session 3 | Break errors down by region, gender and age band with intervals | [`D4_3_break_errors_down.ipynb`](labs/D4_3_break_errors_down.ipynb) |
| 4.4 | Day 4 · Session 4 | Draft the model card and choose the capstone problem | [`D4_4_draft_model_card.ipynb`](labs/D4_4_draft_model_card.ipynb) |
<!-- /academy:labs -->

## Capstone

A tabular problem from a shortlist of synthetic and public cases taken from framing to model card, in a public repository whose README reports the baseline ladder, validation, leakage audit, threshold, subgroup errors and a paired comparison; eight-minute defence

**Submission:** REPLACE: what participants submit and how: a public repository link, or a shared folder for a course without code

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

- REPLACE: every source the course cites, with its address, in course.json under "sources"; this list is written from there
<!-- /academy:sources -->

### National references

- SDAIA AI Ethics Principles: REPLACE: the address of the official text
- SDAIA AI Bias Reference Guide: REPLACE: the address of the official text
- Personal Data Protection Law and its implementing regulations: REPLACE: the address of the official text
