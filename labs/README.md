# Machine Learning for Tabular Data: Baselines and Validation: labs

Hands-on notebooks for the SDAIA Academy course **DS-204 Machine Learning for Tabular Data: Baselines and Validation (تعلّم الآلة للبيانات الجدولية: خطوط الأساس والتحقق من الصحة)**. Every lab works on **Mubakkir**, a model that ranks the trainees of the fictional Sullam Vocational Training Fund by their risk of not completing, so that advisers can offer support early.

No lab calls a language model, needs a key or downloads a model. Everything runs on a laptop processor or on a free Colab runtime. The data are synthetic and are in the repository.

## Quick start

**Google Colab**
1. Open a notebook from the [course page](https://sdaiaacademy.github.io/ds-204/), or directly: `https://colab.research.google.com/github/SDAIAAcademy/ds-204/blob/main/labs/<notebook>.ipynb`.
2. Run the first cell. It clones this repository and installs `requirements-colab.txt`.

**Local**
```bash
git clone https://github.com/SDAIAAcademy/ds-204
cd ds-204/labs
uv sync                        # creates .venv from pyproject.toml (uv.lock pins every version)
uv run jupyter lab
```
Run `00_setup_check.ipynb` once before Day 1, and work through `00_prework_pipelines_cv.ipynb` (two hours) before the course.

**Quick mode.** `DS204_QUICK=1` shrinks the searches and repeats (4 trials in place of 20, 12 seconds a fold for the challenger). Use it on a slow machine; the numbers on the slides come from the full mode.

## Notebooks

| Day | Session | Notebook | AI policy |
|---|---|---|---|
| 0 | Setup | [`00_setup_check.ipynb`](https://colab.research.google.com/github/SDAIAAcademy/ds-204/blob/main/labs/00_setup_check.ipynb) | |
| 0 | Pre-work: pipelines and cross-validation | [`00_prework_pipelines_cv.ipynb`](https://colab.research.google.com/github/SDAIAAcademy/ds-204/blob/main/labs/00_prework_pipelines_cv.ipynb) | AI-tutor |
| 1 | A prediction serves a decision | [`D1_1_frame.ipynb`](https://colab.research.google.com/github/SDAIAAcademy/ds-204/blob/main/labs/D1_1_frame.ipynb) | AI-tutor |
| 1 | The baseline ladder | [`D1_2_baselines.ipynb`](https://colab.research.google.com/github/SDAIAAcademy/ds-204/blob/main/labs/D1_2_baselines.ipynb) | AI-tutor |
| 1 | Linear models first | [`D1_3_logistic_pipeline.ipynb`](https://colab.research.google.com/github/SDAIAAcademy/ds-204/blob/main/labs/D1_3_logistic_pipeline.ipynb) | AI-tutor |
| 1 | Features from history | [`D1_4_history_features.ipynb`](https://colab.research.google.com/github/SDAIAAcademy/ds-204/blob/main/labs/D1_4_history_features.ipynb) | AI-tutor |
| 2 | Gradient-boosted trees | [`D2_1_boosted_defaults.ipynb`](https://colab.research.google.com/github/SDAIAAcademy/ds-204/blob/main/labs/D2_1_boosted_defaults.ipynb) | AI-tutor |
| 2 | Tuning under a budget | [`D2_2_nested_search.ipynb`](https://colab.research.google.com/github/SDAIAAcademy/ds-204/blob/main/labs/D2_2_nested_search.ipynb) | AI-tutor |
| 2 | Automated and pretrained challengers | [`D2_3_challenger.ipynb`](https://colab.research.google.com/github/SDAIAAcademy/ds-204/blob/main/labs/D2_3_challenger.ipynb) | AI-tutor |
| 2 | Is the challenger better | [`D2_4_paired_difference.ipynb`](https://colab.research.google.com/github/SDAIAAcademy/ds-204/blob/main/labs/D2_4_paired_difference.ipynb) | AI-tutor |
| 3 | The split decides the score | [`D3_1_three_splits.ipynb`](https://colab.research.google.com/github/SDAIAAcademy/ds-204/blob/main/labs/D3_1_three_splits.ipynb) | AI-off first |
| 3 | Kinds of leakage | [`D3_2_six_leaks.ipynb`](https://colab.research.google.com/github/SDAIAAcademy/ds-204/blob/main/labs/D3_2_six_leaks.ipynb) | AI-tutor |
| 3 | Hunting planted leaks | [`D3_3_leak_hunt.ipynb`](https://colab.research.google.com/github/SDAIAAcademy/ds-204/blob/main/labs/D3_3_leak_hunt.ipynb) | AI-off first |
| 3 | Reading someone else's notebook | [`D3_4_colleague_notebook.ipynb`](https://colab.research.google.com/github/SDAIAAcademy/ds-204/blob/main/labs/D3_4_colleague_notebook.ipynb) | AI-tutor |
| 4 | Thresholds for rare outcomes | [`D4_1_threshold_capacity.ipynb`](https://colab.research.google.com/github/SDAIAAcademy/ds-204/blob/main/labs/D4_1_threshold_capacity.ipynb) | AI-tutor |
| 4 | Calibration | [`D4_2_calibration.ipynb`](https://colab.research.google.com/github/SDAIAAcademy/ds-204/blob/main/labs/D4_2_calibration.ipynb) | AI-tutor |
| 4 | Errors by subgroup | [`D4_3_subgroup_errors.ipynb`](https://colab.research.google.com/github/SDAIAAcademy/ds-204/blob/main/labs/D4_3_subgroup_errors.ipynb) | AI-off first |
| 4 | The model card | [`D4_4_model_card.ipynb`](https://colab.research.google.com/github/SDAIAAcademy/ds-204/blob/main/labs/D4_4_model_card.ipynb) | AI-tutor |

"AI-off first" means the first attempt at the marked tasks is made without an AI assistant; each notebook says which tasks.

## How a lab works

- Every lab starts from a **supplied reference state** (the data, the reference run log, reference fold scores or reference predictions). No lab needs the files of another.
- ✏️ marks an exercise. It raises `NotImplementedError` until it is done; the lab then continues on the reference result and says so (`↪ using the reference ...`). `✅` and `❌` lines check your answer.
- 🔍 marks something to look at before you go on. ✱ marks a stretch task.
- What a lab writes goes to `labs/artifacts/`, which is not committed: your run log (`run_log.csv`), plots, tables and notes.
- The reference versions of all exercises are in `solutions/`; `solutions/answer_key.md` is the trainer's key.

The notebooks are generated from the percent-format sources in `_src/` by `_tools/py2nb.py` (`uv run python _tools/py2nb.py` from `labs/`), which also writes the solutions. Edit the sources, not the notebooks.

## Data

All data are synthetic and written with a fixed seed by `prepare_data.py`. No row describes a real person, provider or programme.

| File | What it holds |
|---|---|
| `data/enrolments.csv.gz` | 45,000 enrolments of 30,000 trainees, cohorts 2021 to 2025 |
| `data/columns.csv` | The data dictionary: what each column means and when it is recorded |
| `data/providers.csv` | The 60 training providers: region, type, year of opening |
| `data/decision.json` | The decision the model serves: owner, action, capacity, costs |
| `data/population.csv` | The fund's eligible population by region, age band and gender |
| `data/received_table.csv.gz` | The feature table "as received" for Lab 3.3, with four planted leaks |
| `data/six_cases.json` | The six cases of Lab 3.2 |
| `data/predictions.csv.gz` | Probabilities for the cohorts of 2024 and 2025 for Day 4, written by `_tools/make_predictions.py` |
| `data/reference/` | The trainer's results of the full run: the reference run log, fold scores and tables behind every safety net |
| `capstone/data/` | The three synthetic capstone tables and their data dictionaries, written by `capstone/make_datasets.py` |

## The helper `sullam.py`

The notebooks import it as `U`.

| Group | Functions |
|---|---|
| Data | `enrolments`, `columns`, `providers`, `decision`, `population`, `received_table`, `scorable`, `modelling_table`, `dev_holdout`, `day4_predictions` |
| Validation | `folds` (the fund's five folds forward in time), `inner_folds`, `split_scores`, `time_travel_test` |
| Models | `make_pipeline`, `logistic`, `boosted`, `champion`, `champion_params` |
| Scores | `ap`, `metrics`, `recall_at`, `top_share`, `score_folds`, `score_fixed`, `paired_interval`, `wilson` |
| Run log | `log_run`, `my_log`, `reference_log`, `fold_scores`, `oof` |
| Lab mechanics | `check`, `reference`, `keep`, `save`, `savefig`, `md`, `js`, `pick` |

## Other folders

| Folder | What it holds |
|---|---|
| `colleague/` | Loay Al-Kadi's notebook for Lab 3.4, as a script with a repair switch for each block |
| `capstone/` | The brief, the shortlist, the README template, the starter and the three synthetic cases |
| `followup/` | The leakage checklist and the template of the follow-up note |

## Rebuilding everything (trainers)

```bash
uv run python prepare_data.py                 # the data, from the seed
DS204_WRITE_REF=1 DS204_STRICT=1 uv run python _src/D1_1_frame.py    # and so on, in order: writes data/reference/
uv run python _tools/make_predictions.py      # after Lab 2.2: the Day 4 predictions
uv run python capstone/make_datasets.py       # the capstone cases
uv run python _tools/py2nb.py                 # the notebooks and solutions
```
