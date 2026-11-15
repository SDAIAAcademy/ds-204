# Capstone: a validated tabular model, from framing to model card

You take one tabular problem from the shortlist through everything the four days practised, in a public
repository of your own, and defend it in eight minutes.

## The problem

Choose one in Lab 4.4. `shortlist.json` describes six: three synthetic cases written by `make_datasets.py`
(Bayader Microfinance, Awtad Equipment Rental, Nasaim Home Care) and three public tables from the UCI Machine
Learning Repository. A problem from your own work needs the trainer's approval on Day 4.

```bash
uv run python capstone/make_datasets.py     # writes capstone/data/ (already in the repository)
uv run python capstone/starter.py bayader   # reads one case and prints what you have to decide first
```

## What you submit

A link to a **public GitHub repository** with:

- the code, with a fixed seed and a lock file, that reproduces every number in the README;
- the run log, the leakage audit, the subgroup results and the model card;
- a README in the structure of `TEMPLATE_README.md`: baseline ladder, validation scheme, leakage audit,
  threshold, errors by subgroup and one paired comparison;
- the prompts and agent transcripts you used, in a folder `ai/`.

AI assistants and agents are allowed without limit (AI-full). Every choice is yours to defend live.

## The defence: eight minutes

| Minutes | What happens |
|---|---|
| 0 to 4 | You present: the decision, the validation scheme and why, the ladder, the one finding you are least comfortable with |
| 4 to 8 | The assessor names one change, a different split or one feature removed. You say what you expect to happen to the score and why, **before** it is run. Then it is run, or the assessor shows the result |

There is no separate question time: the change and your prediction are the questions.

## How it is assessed

| Criterion | Weight | The assessor looks for |
|---|---|---|
| Framing and baseline | 15 | A decision with an owner and a capacity; a prediction time; baselines on the same folds as the models |
| Validation and leakage | 25 | A split that rehearses the use; other splits measured once; an audit with one removal at a time |
| Models and comparison | 20 | Every rung logged on identical folds; a stated tuning budget; a paired difference with an interval and a decision |
| Threshold, calibration and subgroup errors | 25 | A rule for the list; expected against observed; errors by subgroup with intervals; a bias type and an action per finding |
| Model card and reproducibility | 15 | Limits stated plainly; the human decision; a fresh clone reproduces the numbers |

Passing the course needs 70% overall and at least 60% on this rubric and on the supervised check.

## Before you submit

- [ ] A fresh clone runs with two commands and gives the README's numbers
- [ ] No column in the model is recorded after the prediction time, and the README says how you know
- [ ] The held-out rows were scored once
- [ ] Every table of rates by subgroup has intervals
- [ ] The README says what the model must not be used for
