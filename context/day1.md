---
course: "Machine Learning for Tabular Data: Baselines and Validation"
code: "DS-204"
version: "2027.1.0"
deck: "day1"
day: 1
language: "en"
view: "trainee"
source: "https://sdaiaacademy.github.io/ds-204/slides/day1.html"
---

# Machine Learning for Tabular Data: Baselines and Validation

## Day 1: Frame and baseline

From the advisers' question to a number every later model has to beat

#### [day1#title] Frame and baseline

*SDAIA Academy · Machine Learning for Tabular Data: Baselines and Validation*

Day 1: From the advisers' question to a number every later model has to beat

Data Science Track · 4 teaching days + capstone day · 4 sessions · 4 labs · DS-204

#### [day1#agenda] Frame and baseline

*Today*

- **Session 1 · A prediction serves a decision**: The decision comes first: target, unit, prediction time, the action a score triggers and what each error costs.
- **Session 2 · The baseline ladder**: A ladder of models on identical folds, with a log: the constant predictor, then rules an adviser could write.
- **Session 3 · Linear models first**: The first model that learns: a regularised logistic model inside a pipeline fitted on training rows only.
- **Session 4 · Features from history**: A trainee's earlier enrolments are the strongest new information, if they are read from the past only.

#### [day1#four-days] Four days from an adviser's question to a model card

*The course*

1. **Frame and baseline**: Decision, constant, rules, linear, history (Day 1)
2. **Trees and challengers**: Boosted trees, a tuning budget, a paired decision (Day 2)
3. **Validation and leakage**: Three splits, four kinds of leak, an audit (Day 3)
4. **Errors and documentation**: Threshold, calibration, subgroups, model card (Day 4)

Day 5 is yours: a tabular problem taken from framing to model card, defended live.

#### [day1#case-sullam] Sullam Vocational Training Fund wants to offer support early

*The case study*

- **The fund.** A fictional fund that pays for programmes at 60 providers in 13 regions. 45,000 synthetic enrolments, 30,000 trainees, cohorts 2021 to 2025.
- **The problem.** One enrolment in six ends without completion. A lost place costs SAR 9,000. The advisers can run 800 support plans a year.
- **The model.** Mubakkir ranks trainees at the end of week 2 by the risk of not completing. You are Razan Al-Angari, the data scientist who builds it and defends its numbers.

> **Synthetic, and treated as personal data.** No row describes a real person or provider. Every lab handles the table as if it did.

#### [day1#setup-check] One notebook proves your setup

*Before session 1*

- **In Colab:** open `00_setup_check.ipynb` from the course page and run all cells
- **On a laptop:** in `labs/`, run `uv sync`, then `uv run jupyter lab`
- It imports every library, reads the enrolments and prints its check marks
- No key, no graphics processor, no model download

- **The pre-work.** `00_prework_pipelines_cv.ipynb`: two hours on pipelines and cross-validation with self-check questions. The supervised entry quiz follows it, without AI assistance.

### Session 1: A prediction serves a decision

#### [day1#s1] A prediction serves a decision

*Session 1*

- State the target, the unit and the prediction time from the decision a score serves
- Separate the columns known at the prediction time from those recorded at exit
- Compute the break-even probability from the cost of each error

#### [day1#decision-first] A prediction is worth what the decision it changes is worth

*Start at the end*

1. **1 · The decision** Which trainees does an adviser contact in week 3?
2. **2 · The prediction** How likely is each trainee not to complete?
3. **3 · The data** What is known about each enrolment at that moment?

> **What goes wrong.** Teams start at step 3: "we have a table, let us predict a column". The model then answers a question nobody asked, at a moment when nobody decides anything.

#### [day1#four-lines] Four lines fix the task before any code

*The framing*

| Line | For Mubakkir | What it rules out |
| --- | --- | --- |
| **Target** | `not_completed`: left without completing | A vague "risk" |
| **Unit** | One enrolment | One row per trainee |
| **Prediction time** | The end of week 2 | Anything recorded later |
| **Action** | An adviser agrees a support plan in week 3 | Removing a place |

Change one line and you have a different project with different data.

#### [day1#recorded-when] Every column has a time at which it starts to exist

*The moment decides the columns*

| Recorded | Columns in the export | At the end of week 2 |
| --- | --- | --- |
| At registration | `region`, `gender`, `education`, `aptitude_score`, `has_dependants` | known |
| At enrolment | `provider_id`, `programme`, `start_date`, `age`, `distance_km`, `stipend`, … | known |
| At the end of week 2 | `absences_w2` | known, just |
| At exit | `not_completed`, `weeks_attended`, `exit_date`, `exit_reason` | does not exist yet |

> **Ask for the data dictionary first.** If nobody can say when a column is written, treat it as written at exit until proven otherwise.

#### [day1#too-good-column] Weeks attended is the outcome, written in another unit

*One column to distrust*

**21.9 · 7.7** mean weeks attended: those who completed, and those who did not

- `weeks_attended` separates the classes almost by itself
- A model given it would look brilliant on old data
- At the end of week 2 it reads "2" for everyone
- This is **target leakage**: Day 3 hunts four planted leaks

#### [day1#population] A third of the non-completers leave before the model can see them

*Which rows*

| Item | Value |
| --- | --- |
| All 45,000 enrolments | 16.7% |
| Still enrolled after week 2: 42,556 | 11.9% |
| Left in weeks 1 and 2: 2,444 | 100% |

- No support plan in week 3 can reach a trainee who has left
- They leave the model's **population**
- Every score in the course is about the 42,556 scorable rows
- The 2,444 are a limit to state, not to hide

#### [day1#cost-of-errors] The two errors do not cost the same

*Two ways to be wrong*

| Error | What happens | Cost to the fund |
| --- | --- | --- |
| Missed non-completer | Nobody called; the place is lost | SAR 9,000 |
| Needless support plan | An adviser's time went to a trainee who would have completed | SAR 450 |

- **Support is not a cure.** One supported non-completer in four then completes (the advisers' own figure).
- **Break-even.** A plan pays when *p* × 0.25 × 9,000 > 450: when the probability of not completing is above 0.20.

#### [day1#capacity] Capacity turns a probability into a list of 800 names

*The action has a size*

**8 in 100** scorable enrolments can get a support plan in the largest cohort: 800 plans for 10,100. Smaller cohorts, from 6,800, reach 12 in 100; the run log scores the top 8%

- The advisers do not ask "who is above 0.5?"
- They ask "which 800 first?"
- So the course scores **rankings**: average precision, and the share of non-completers on the list
- Accuracy is never reported

#### [day1#minimisation] Known is not the same as justified

*Responsible use*

- The purpose is narrow: decide whom to offer support
- **Data minimisation:** use only the personal data that purpose needs
- `gender` is known and left out of the model
- It stays in the table to check the errors by group on Day 4
- `trainee_id` joins a history; it is never a feature

- **Who decides.** The data scientist proposes, with a reason per column. Maram Al-Rasheed, the data protection officer, signs. The framing sheet of Lab 1.1 is what she reads.

#### [day1#check-framing] Which framing serves the advisers?

*Checkpoint*

**Question:** The advisers can agree 800 support plans a year, starting in week 3 of a programme. Which framing fits?

- A. One row per trainee: will this trainee ever leave a programme?
- B. One row per enrolment, at exit: did the trainee complete?
- C. One row per provider and year: how many trainees will leave?
- D. One row per enrolment, at the end of week 2: will the trainee not complete?

#### [day1#lab-1-1] Frame the non-completion question

Lab 1.1 · 25 min

Write target, unit, prediction time and adviser action for the question, keep the rows that can be scored, mark the columns known at that time and price the two errors.

[D1_1_frame.ipynb](https://sdaiaacademy.github.io/ds-204/labs/D1_1_frame.ipynb)

- [ ] Read the export and its data dictionary (4 min)
- [ ] Write target, unit, prediction time and adviser action (6 min)
- [ ] Keep the rows that can be scored (4 min)
- [ ] Mark the columns known at the end of week 2 (6 min)
- [ ] Price the two errors: the break-even probability (5 min)
- [ ] ✱ Stretch: Frame the same question on the enrolment day: what is gained and what is lost

### Session 2: The baseline ladder

#### [day1#s2] The baseline ladder

*Session 2*

- Explain why every model is scored on the same folds and logged
- Score a constant predictor and say what its average precision is
- Write a three-rule baseline and read its score and fold spread

#### [day1#why-ladder] A score means nothing until you know what no model scores

*Before any model*

1. **Constant**: The same score for everyone (Lab 1.2)
2. **Rules**: Three lines an adviser could write (Lab 1.2)
3. **Linear**: A regularised logistic model (Labs 1.3, 1.4)
4. **Boosted trees**: Defaults, then 20 trials (Labs 2.1, 2.2)
5. **Challenger**: An automated tool (Lab 2.3)

Each rung has to earn its extra complexity against the rung below.

#### [day1#identical-folds] Every rung is scored on the same five folds

*The rule of the ladder*

| Fold | Validates on | Training rows | Validation rows | Base rate |
| --- | --- | --- | --- | --- |
| 1 | 2022, second half | 9,068 | 3,902 | 11.6% |
| 2 | 2023, first half | 12,960 | 4,266 | 10.8% |
| 3 | 2023, second half | 17,028 | 4,326 | 11.0% |
| 4 | 2024, first half | 21,382 | 4,649 | 13.0% |
| 5 | 2024, second half | 25,836 | 4,627 | 12.6% |

> **Cohort 2025 is closed.** 10,117 rows stay untouched until Day 4. Each fold trains on programmes due to end before its half-year began; Day 3 explains why.

#### [day1#run-log] One line per run, never edited afterwards

*The habit*

```plaintext
# file: artifacts/run_log.csv
lab  rung         folds  mean_ap  spread  min_fold  max_fold  recall_at_capacity
1.2  constant     5      0.1181   0.0098  0.1076    0.1301    0.0759
1.2  three rules  5      0.3099   0.0125  0.2899    0.3233    0.2922
```

- **Mean** over the folds, and the **spread** between them
- The five fold scores themselves, for the paired comparison of Day 2
- Seconds, features and a note: what a reviewer asks first

#### [day1#constant-floor] The floor of average precision is the base rate, not 0.5

*Rung 1*

**0.118** average precision of the same score for everyone, mean of five folds

- **Average precision:** precision averaged over every position where a non-completer is found
- A ranking with no information has the base rate as its precision everywhere
- ROC AUC is the score whose floor is 0.5
- For a rare outcome and a short list, average precision is the stricter judge

#### [day1#three-rules] Three rules on a card already sort the trainees

*Rung 2*

| Item | Value |
| --- | --- |
| No rule met: 25,487 | 6.1% |
| One rule: 5,767 | 24.4% |
| Two rules: 1,097 | 62.0% |
| Three rules: 88 | 78.4% |

- Absent two days or more in the first two weeks
- Aged 19 or under
- Entry test below 38
- Score: the number of rules met

Average precision 0.310. With no learning, 29% of non-completers are on the advisers' list.

#### [day1#spread] The spread between folds is the yardstick for every gain

*Reading a rung*

| Fold | Constant | Three rules |
| --- | --- | --- |
| 1 | 0.116 | 0.310 |
| 2 | 0.108 | 0.290 |
| 3 | 0.110 | 0.316 |
| 4 | 0.130 | 0.323 |
| 5 | 0.126 | 0.311 |
| Mean · spread | 0.118 · 0.010 | 0.310 · 0.012 |

- The rules win on all five folds
- The gain, 0.19, is fifteen times the spread
- A gain inside the spread is a question, not a result
- Lab 2.4 answers such questions with an interval

#### [day1#check-ladder] What does 0.30 mean?

*Checkpoint*

**Question:** Loay reports that his model reaches an average precision of 0.30 on a split of his own. What is the right reply?

- A. It is poor: the model is wrong 70% of the time
- B. It cannot be judged until it stands beside the constant and the rules on the same folds
- C. It is good: it is well above the base rate of 0.12
- D. It is slightly worse than the rules, which score 0.31

#### [day1#lab-1-2] A constant and three rules

Lab 1.2 · 25 min

Score a constant predictor and a three-rule baseline on the fund's five folds and log both: the first two rungs of the ladder.

[D1_2_baselines.ipynb](https://sdaiaacademy.github.io/ds-204/labs/D1_2_baselines.ipynb)

- [ ] Load the folds; read the base rate of each (4 min)
- [ ] Score the constant predictor (5 min)
- [ ] Write the three-rule baseline and score it (8 min)
- [ ] Log both runs (4 min)
- [ ] Read the log: the number to beat (4 min)
- [ ] ✱ Stretch: A fourth rule from the provider's past non-completion rate

### Session 3: Linear models first

#### [day1#s3] Linear models first

*Session 3*

- Build a pipeline that imputes, encodes and scales inside each fold
- Fit a regularised logistic model and log its score and fold spread
- Read the largest coefficients and say what they do not prove

#### [day1#linear-first] A linear model is the first model worth beating

*Rung 3*

- **Logistic regression:** a weighted sum of the inputs, turned into a probability
- Fits in seconds, on a laptop
- Its probabilities are usually close to observed rates
- Each weight can be read, with care
- On tables with mostly additive effects it is hard to beat by much

- **What it cannot do alone.** It adds effects. "Absence matters, except for employer-sponsored trainees" is an interaction: someone has to write it in, or a tree has to find it. That is Day 2.

#### [day1#pipeline] A pipeline fits every preparation step on training rows only

*Preparation is part of the model*

1. **Impute** Median, plus a "was missing" column
2. **Encode** One column per category; rare ones share one
3. **Scale** Mean 0, spread 1
4. **Model** Logistic regression

- In each fold, `fit` sees the training rows; the validation rows only pass through `transform`
- 15 columns become 101 model inputs
- The entry test is missing for 14.2% of rows, and not at random: walk-in registrations skip it

#### [day1#pipeline-code] The whole rung is one object that can be fitted and scored

*In code*

```python
# file: D1_3_logistic_pipeline.ipynb
numeric = Pipeline([("impute", SimpleImputer(strategy="median", add_indicator=True)),
                    ("scale", StandardScaler())])
categories = Pipeline([("fill", SimpleImputer(strategy="constant", fill_value="not recorded")),
                       ("onehot", OneHotEncoder(handle_unknown="infrequent_if_exist",
                                                min_frequency=30, sparse_output=False))])
prepare = ColumnTransformer([("num", numeric, NUM), ("cat", categories, CAT)])
model = Pipeline([("prepare", prepare),
                  ("model", LogisticRegression(C=0.1, max_iter=2000))])
```

> **One object, one rule.** Whatever is inside the pipeline is refitted in every fold. Whatever you compute before it has seen every row.

#### [day1#logistic-score] The logistic model wins on every fold, by far more than the spread

*The rung, measured*

| Item | Value |
| --- | --- |
| Constant | 0.118 |
| Three rules | 0.310 |
| Logistic regression | 0.501 |

- Fold scores 0.477 to 0.545; spread 0.028
- The share of non-completers on the advisers' list rises from 29% to 40%
- Fold 1, with the least training data, scores highest: folds differ in more than size

#### [day1#regularisation] The penalty hardly matters here, and choosing it has a price

*One setting*

| C (inverse penalty) | Mean | Spread |
| --- | --- | --- |
| 0.003, strong penalty | 0.490 | 0.023 |
| 0.1 | 0.501 | 0.028 |
| 10, weak penalty | 0.501 | 0.032 |

- **Regularisation** pulls weights towards zero
- With 20,000 rows and 101 inputs, only a very strong penalty costs anything
- Picking C by these scores is tuning on the validation rows
- Harmless for three values; Lab 2.2 does it properly for twenty

#### [day1#what-pipeline-protects] The harm of preparing outside the folds depends on what the step has read

*What the pipeline protects*

| Preparation | Mean score | Verdict |
| --- | --- | --- |
| Imputer and scaler inside each fold | 0.5013 | the reference |
| Imputer and scaler fitted on all rows | 0.5014 | wrong, and almost free |
| Provider rate from training rows | 0.5018 | honest |
| Provider rate from all rows | 0.5041 | the gain is false |

> **What goes wrong.** A step that never sees the outcome leaks little. A step built from the outcome (a rate per provider, a feature selection, a tuned setting) leaks in proportion to what it read.

#### [day1#coefficients] A weight ranks what the model leans on, and proves nothing about causes

*Reading the model*

| Input (fold 5) | Weight |
| --- | --- |
| Provider P23 | +0.91 |
| Provider P14 | −0.80 |
| `absences_w2` | +0.74 |
| `aptitude_score` | −0.65 |

- Per standard deviation of a scaled column, or per category, others held fixed
- Providers differ a great deal: a finding for Hamza Al-Babtain, not a verdict
- "Absence predicts leaving" is not "reducing absence prevents leaving"

#### [day1#check-pipeline] Where may the scaler be fitted?

*Checkpoint*

**Question:** You score a logistic model on five folds. On which rows is the scaler of fold 3 fitted?

- A. On the training rows of fold 3 only, each time the pipeline is fitted
- B. On all development rows, once, before the folds are made
- C. On the validation rows of fold 3, so that they are scaled correctly
- D. On the 2025 cohort, because it is the most recent

#### [day1#lab-1-3] A logistic model in a pipeline

Lab 1.3 · 25 min

Fit a regularised logistic model in a pipeline on the same folds, log score and spread, try three strengths and see what the pipeline protects.

[D1_3_logistic_pipeline.ipynb](https://sdaiaacademy.github.io/ds-204/labs/D1_3_logistic_pipeline.ipynb)

- [ ] Build the pipeline: impute, encode, scale, model (7 min)
- [ ] Score it on the five folds and log the run (5 min)
- [ ] Try three regularisation strengths; log each (5 min)
- [ ] Prepare outside the folds and compare (5 min)
- [ ] Read the largest coefficients (3 min)
- [ ] ✱ Stretch: Add one interaction by hand and log it

### Session 4: Features from history

#### [day1#s4] Features from history

*Session 4*

- Explain why an earlier enrolment counts only when its outcome was known
- Build two history features from records that ended before the prediction time
- Run a time-travel test that fails when a feature reads the future

#### [day1#repeated-trainees] One trainee in three has more than one enrolment

*The structure of the table*

| Item | Value |
| --- | --- |
| One enrolment | 19,500 |
| Two | 7,000 |
| Three | 2,500 |
| Four | 1,000 |

- 30,000 trainees, 45,000 enrolments
- The rows are **not independent**: a trainee carries habits and circumstances from one enrolment to the next
- That is information for the model today
- And a trap for validation on Day 3

#### [day1#history-signal] An earlier non-completion more than triples the rate

*What history says*

| Item | Value |
| --- | --- |
| No earlier non-completion: 40,262 rows | 10.3% |
| One: 1,937 rows | 38.9% |
| Two or more: 357 rows | 54.3% |

- `prior_enrolments`: earlier enrolments whose outcome was known
- `prior_noncompletions`: how many of those ended without completion
- No rule of Lab 1.2 moves the rate as much as this one column

#### [day1#known-outcome] An earlier enrolment counts only once its outcome is known

*The rule*

| One trainee | Starts | Ends | Outcome | Known when the April 2025 row is scored? |
| --- | --- | --- | --- | --- |
| Culinary arts | January 2021 | June 2021 | completed | yes |
| Culinary arts | July 2021 | July 2021 | left | yes |
| Culinary arts | July 2024 | July 2024 | left | yes |
| Electrical installation | April 2025 | September 2025 | ? | this is the row to score |

> **What goes wrong.** "Earlier" is read as "started earlier". 1,423 scorable rows have an earlier enrolment that was still running at the prediction time: its outcome is in the table and did not exist that day.

#### [day1#time-travel-test] The time-travel test hides the future and checks that nothing changes

*A test, not a promise*

1. **1 · Pick a past date** For example 1 March 2023
2. **2 · Compute twice** From all records, and from the records ended by that date
3. **3 · Compare** For every row predicted before that date

- A feature that reads the past only gives the same value both times
- The "started earlier" version changes for about 90 rows at each date: it fails
- The test checks the features, not the model, and takes seconds

#### [day1#history-gain] History helps on every fold, and only for those who have one

*Rung 3, with history*

| Fold | Logistic | With history | Gain |
| --- | --- | --- | --- |
| 1 | 0.545 | 0.557 | +0.012 |
| 2 | 0.483 | 0.514 | +0.031 |
| 3 | 0.515 | 0.530 | +0.016 |
| 4 | 0.477 | 0.509 | +0.031 |
| 5 | 0.486 | 0.512 | +0.026 |
| Mean | 0.501 | 0.524 | +0.023 |

| Validation rows | Without | With |
| --- | --- | --- |
| New to the fund: 14,455 | 0.484 | 0.484 |
| Returning: 7,315 | 0.523 | 0.579 |

#### [day1#check-history] Which history feature reads the future?

*Checkpoint*

**Question:** Four candidate features for a row scored at the end of week 2. Which one fails the time-travel test?

- A. The number of the trainee's enrolments that ended before the prediction date
- B. The trainee's entry test score
- C. The trainee's non-completion rate over all of their enrolments in the table
- D. The trainee's days absent in the first two weeks of this programme

#### [day1#lab-1-4] Two history features

Lab 1.4 · 25 min

Add two history features from past records only, pass a time-travel test and log what they add to the logistic rung.

[D1_4_history_features.ipynb](https://sdaiaacademy.github.io/ds-204/labs/D1_4_history_features.ipynb)

- [ ] Follow one trainee through the records (3 min)
- [ ] Write the two history features from past records only (9 min)
- [ ] Run the time-travel test (5 min)
- [ ] Rescore the logistic model with them; log the run (5 min)
- [ ] Compare new and returning trainees (3 min)
- [ ] ✱ Stretch: A third feature: days since the last exit

#### [day1#wrap-up] Three things to keep

*Day 1*

- **The decision frames the task.** Target, unit, prediction time, action. They fix the rows (42,556 of 45,000) and the columns (nothing recorded at exit).
- **A ladder with a log.** Constant 0.118, three rules 0.310, logistic 0.501, with history 0.524. Same folds, every run logged, spread beside every mean.
- **The past only.** Preparation inside the folds; history from records whose outcome was known; a test that fails when a feature reads the future.

#### [day1#next] Day 2 · Trees and challengers

*Next*

The number to beat is 0.524. Tomorrow: boosted trees, a budget of 20 trials, an automated challenger, and an interval that says whether a gain is real.

- Keep `artifacts/run_log.csv`: tomorrow adds three rungs
- Guess now: by how much will tuned boosted trees beat 0.524?
