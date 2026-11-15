# (Your model's name): (the outcome) at (the organisation)

One sentence: who acts on the score, what they do, and when.

> Capstone of DS-204, SDAIA Academy. The data are synthetic or public; see "Data".

## 1 · Framing and baseline

- **Decision and owner:** who decides what, with which capacity or cost.
- **Unit and prediction time:** one row is ...; the score is made on ...; columns recorded later are not used.
- **Target:** definition, and its rate.
- **Metric:** which one, and why it fits the decision.
- **Baselines:** the constant answer and a rule a colleague could apply, on the same folds as every model.

## 2 · Validation and leakage

- **Validation scheme:** the split, and the use it rehearses. What the training part knows on the day a
  validation row is scored.
- **Other splits, measured once:** the gap to a random split (and to a grouped or time-ordered one), with one
  sentence per gap.
- **Leakage audit:** the table below, one removal at a time.

| Removal | Kind of leak | Score before | Score after |
|---|---|---|---|
| | | | |

## 3 · Models and comparison

The run log, every rung on identical folds, with the tuning budget stated before the search.

| Rung | Model | Mean score | Fold spread | Seconds |
|---|---|---|---|---|
| | | | | |

**Paired comparison:** champion against (challenger or simpler rung). Difference, 95% interval, minimum effect
stated in advance, decision in one line.

## 4 · Threshold, calibration and subgroup errors

- **Threshold or list length:** the rule, why, and its precision and recall on rows held out until the end.
- **Calibration:** expected against observed by decile; whether a calibrator is applied, and the evidence.
- **Errors by subgroup:** the error that harms people most, by at least two attributes, with intervals. Which
  gaps the intervals support. The bias type behind each finding and one action.

## 5 · Model card

Intended use · not for · data · validation · results · subgroups · limits · the human decision · what a
deployment team needs (when the model was trained, the retraining interval, what to monitor).

## Reproduce

```bash
uv sync
uv run python run.py        # seed fixed; writes the run log and every table above
```

## Data

Source and licence. For a public table: the dataset page and its citation.

## Use of AI assistants

Which assistant or agent, for what. The prompts and transcripts are in `ai/`. Every choice in this README is
mine to defend.
