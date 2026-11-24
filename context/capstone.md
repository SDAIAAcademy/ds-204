---
course: "Machine Learning for Tabular Data: Baselines and Validation"
code: "DS-204"
version: "2027.2.0"
deck: "capstone"
day: 5
language: "en"
view: "trainee"
source: "https://sdaiaacademy.github.io/ds-204/slides/capstone.html"
---

# Machine Learning for Tabular Data: Baselines and Validation

## Capstone: Capstone: a validated tabular model and its defence

Each participant defends a tabular model from framing to model card in eight minutes, including one change the assessor asks for

#### [capstone#title] Capstone: a validated tabular model and its defence

*SDAIA Academy · Machine Learning for Tabular Data: Baselines and Validation*

Capstone: Each participant defends a tabular model from framing to model card in eight minutes, including one change the assessor asks for

Data Science Track · 4 teaching days + capstone day · 8 min talk and demo · DS-204

#### [capstone#agenda] How the day runs

*Capstone*

- **Build and finish**: your problem from framing to model card; the README is the deliverable
- **Supervised exit check**: without AI assistance, items in the style of the badge exam
- **Defences**: eight minutes each, one timer, one change named by the assessor
- **Review and closing**: what the repositories showed, the follow-up task, what comes next

#### [capstone#deliverable] A public repository whose README reports six things

*What you hand in*

- **Ladder and validation.** Baselines and models on identical folds; the split and the use it rehearses.
- **Audit and comparison.** Leaks removed one at a time; one paired difference with an interval and a decision.
- **Threshold and subgroups.** The rule for the list on rows held out to the end; errors by subgroup with intervals.

- A fixed seed and a lock file: a fresh clone reproduces every number
- The model card, and the prompts and agent transcripts you used, in `ai/`

#### [capstone#rubric] The rubric

*How the repository is scored*

| Item | Value |
| --- | --- |
| Framing and baseline | 15% |
| Validation and leakage | 25% |
| Models and comparison | 20% |
| Threshold, calibration and subgroup errors | 25% |
| Model card and reproducibility | 15% |

> **A modest score that is validated beats a high one that is not.** No criterion rewards the size of the number.

#### [capstone#format] Eight minutes: four are yours, four belong to one change

*The defence*

- **Minutes 0 to 4 · You present.** The decision. The validation scheme and why. The ladder in one table. The finding you are least comfortable with.
- **Minutes 4 to 8 · One change.** The assessor swaps the split or removes a feature. You say what will happen to the score, and why, **before** it is run. Then it is run, or the result is shown.

> **One timer, no separate question time.** The change and your prediction are the questions.

#### [capstone#the-change] Predict the effect before it is run

*What the assessor may ask*

| The change | What a good answer contains |
| --- | --- |
| "Score it under a random split." | The direction, a rough size, and which rows the training part would then know |
| "Keep each unit on one side." | Whether the model can recognise a unit, and what history is worth |
| "Remove your strongest feature." | What the feature carries, and which others can stand in for it |
| "Put back the column you removed." | The kind of leak, and why the higher score cannot be repeated in use |

#### [capstone#running-order] Who defends when

*Running order*

| # | Presenter | Problem |
| --- | --- | --- |
| 1 |  |  |
| 2 |  |  |
| 3 |  |  |
| 4 |  |  |
| 5 |  |  |
| 6 |  |  |
| 7 |  |  |
| 8 |  |  |

#### [capstone#final-prep] Final prep · 20 minutes

*Before the first defence*

Clone your repository into a new folder and run it once. Have a second split ready as a switch.

#### [capstone#now-presenting] Defence, with the change at minute 4

*Now presenting*

#### [capstone#review] What the repositories showed

*Trainer review*

- **Strongest patterns.** What several repositories got right.
- **Common gaps.** What most would repair next.
- **One surprise.** A prediction that failed, and what it taught.

#### [capstone#follow-up] The follow-up: the leakage checklist on a model that is not yours

*In the next four weeks*

- Choose one model from your own work, or a public notebook
- Work through `labs/followup/leakage_checklist.md`
- Send the one-page note of `note_template.md` to your trainer

- **What the note holds.** The reported score, the split and the use, each checklist question with its answer, what you would remove or measure next. Numbers, not opinions.

#### [capstone#what-next] Keep going

*After the course*

- **Your repository stays yours:** keep it reproducible, and add the one measurement you named today
- **AE-205 Deploying AI Services:** a model like yours as a service, with containers, serving and monitoring
- **AE-305 Operating Machine Learning and Language Model Systems:** retraining, drift and operations, starting from a model card

#### [capstone#thanks] Thank you

*SDAIA Academy*

Machine Learning for Tabular Data: Baselines and Validation
