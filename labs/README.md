# Machine Learning for Tabular Data: Baselines and Validation: labs

## Quick start

- Google Colab: open a notebook from the course page; its first cell clones this repository.
- Locally: clone the repository, then open the notebooks in `labs/` with Jupyter.

| Unit | Session | Notebook |
|---|---|---|
| Day 1 | Session 1 · A prediction serves a decision | [`D1_1_write_target_unit.ipynb`](https://colab.research.google.com/github/SDAIAAcademy/ds-204/blob/main/labs/D1_1_write_target_unit.ipynb) |
| Day 1 | Session 2 · The baseline ladder | [`D1_2_score_constant_predictor.ipynb`](https://colab.research.google.com/github/SDAIAAcademy/ds-204/blob/main/labs/D1_2_score_constant_predictor.ipynb) |
| Day 1 | Session 3 · Linear models first | [`D1_3_fit_regularised_logistic.ipynb`](https://colab.research.google.com/github/SDAIAAcademy/ds-204/blob/main/labs/D1_3_fit_regularised_logistic.ipynb) |
| Day 1 | Session 4 · Features from history | [`D1_4_add_two_history.ipynb`](https://colab.research.google.com/github/SDAIAAcademy/ds-204/blob/main/labs/D1_4_add_two_history.ipynb) |
| Day 2 | Session 1 · Gradient-boosted trees | [`D2_1_train_boosted_trees.ipynb`](https://colab.research.google.com/github/SDAIAAcademy/ds-204/blob/main/labs/D2_1_train_boosted_trees.ipynb) |
| Day 2 | Session 2 · Tuning under a budget | [`D2_2_run_20_trial.ipynb`](https://colab.research.google.com/github/SDAIAAcademy/ds-204/blob/main/labs/D2_2_run_20_trial.ipynb) |
| Day 2 | Session 3 · Automated and pretrained challengers | [`D2_3_run_challenger_same.ipynb`](https://colab.research.google.com/github/SDAIAAcademy/ds-204/blob/main/labs/D2_3_run_challenger_same.ipynb) |
| Day 2 | Session 4 · Is the challenger better | [`D2_4_compute_paired_fold.ipynb`](https://colab.research.google.com/github/SDAIAAcademy/ds-204/blob/main/labs/D2_4_compute_paired_fold.ipynb) |
| Day 3 | Session 1 · The split decides the score | [`D3_1_score_one_model.ipynb`](https://colab.research.google.com/github/SDAIAAcademy/ds-204/blob/main/labs/D3_1_score_one_model.ipynb) |
| Day 3 | Session 2 · Kinds of leakage | [`D3_2_classify_six_leaks.ipynb`](https://colab.research.google.com/github/SDAIAAcademy/ds-204/blob/main/labs/D3_2_classify_six_leaks.ipynb) |
| Day 3 | Session 3 · Hunting planted leaks | [`D3_3_find_four_planted.ipynb`](https://colab.research.google.com/github/SDAIAAcademy/ds-204/blob/main/labs/D3_3_find_four_planted.ipynb) |
| Day 3 | Session 4 · Reading someone else's notebook | [`D3_4_list_what_makes.ipynb`](https://colab.research.google.com/github/SDAIAAcademy/ds-204/blob/main/labs/D3_4_list_what_makes.ipynb) |
| Day 4 | Session 1 · Thresholds for rare outcomes | [`D4_1_pick_threshold_precision.ipynb`](https://colab.research.google.com/github/SDAIAAcademy/ds-204/blob/main/labs/D4_1_pick_threshold_precision.ipynb) |
| Day 4 | Session 2 · Calibration | [`D4_2_calibrate_boosted_model.ipynb`](https://colab.research.google.com/github/SDAIAAcademy/ds-204/blob/main/labs/D4_2_calibrate_boosted_model.ipynb) |
| Day 4 | Session 3 · Errors by subgroup | [`D4_3_break_errors_down.ipynb`](https://colab.research.google.com/github/SDAIAAcademy/ds-204/blob/main/labs/D4_3_break_errors_down.ipynb) |
| Day 4 | Session 4 · The model card | [`D4_4_draft_model_card.ipynb`](https://colab.research.google.com/github/SDAIAAcademy/ds-204/blob/main/labs/D4_4_draft_model_card.ipynb) |

The notebooks are generated from the percent-format sources in `labs/_src/` by `labs/_tools/py2nb.py` (`uv run python _tools/py2nb.py` from `labs/`), which also writes the solutions to `labs/solutions/`. Edit the sources, not the notebooks.
