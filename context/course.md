---
course: "Machine Learning for Tabular Data: Baselines and Validation"
code: "DS-204"
version: "2027.1.0"
scope: "course"
language: "en"
view: "trainee"
source: "https://sdaiaacademy.github.io/ds-204/"
---

# Machine Learning for Tabular Data: Baselines and Validation

Take a tabular prediction from the decision it serves to a model card: a logged baseline ladder, validation that mirrors use, a leakage audit, a threshold that fits capacity and errors by subgroup with intervals

Participants predict which trainees of a fictional vocational training fund are likely not to complete, so that advisers can offer support early. On synthetic enrolment data with repeated trainees and cohorts, they climb a baseline ladder to tuned gradient-boosted trees, test an automated or pretrained challenger, choose validation that mirrors use, audit for leakage, set a threshold, break errors down by subgroup and write a model card. Deployment is left to AE-205 and AE-305.

- **Audience:** Practitioners. Graduates and early-career analysts who have completed the FN courses or passed the placement test, and analysts moving into data science.
- **Prerequisites:** FN-101 Python for Data and AI Work; FN-103 Mathematics for Machine Learning: Vectors, Gradients and Probability; FN-105 How Machine Learning and Language Models Work; DS-201 Statistical Inference by Simulation; Joins and groups tables with a dataframe library; Explains overfitting and the purpose of a held-out set
- **Case study:** Sullam Vocational Training Fund: Sullam Vocational Training Fund is a fictional fund that pays for vocational programmes at 60 training providers in 13 regions. Participants work on 45,000 synthetic enrolments of 30,000 trainees over five yearly cohorts to build Mubakkir, a score that at the end of week 2 puts the trainees most likely not to complete on an adviser's list. All data are synthetic, written with a fixed seed, with repeated trainees, cohort drift, four planted leaks and registers that under-record absence in two regions.

## Outcomes

- LO1 (Awareness): Identify the leakage type and the fitting validation scheme in short tabular modelling cases, given six described problems and their data-collection process
- LO2 (Applied): Apply a baseline ladder from a constant predictor to tuned gradient-boosted trees, logging every run, given identical folds and a fixed tuning budget
- LO3 (Applied): Implement validation that matches the data structure and measure how far a random split overstates performance, given repeated trainees across five yearly cohorts
- LO4 (Applied): Test an automated or pretrained challenger against the tuned model under identical folds and time budget, given the champion model and a challenger configuration
- LO5 (Applied): Analyse calibration, the decision threshold and held-out errors by subgroup, with intervals that reflect slice size, given predictions joined to region, gender, age band and provider, and adviser capacity
- LO6 (Performance): Build a validated tabular model from framing to model card and defend its validation scheme and limits, given a synthetic case or an approved public tabular problem

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

## Day 2: Trees and challengers

Boosted trees, a fixed tuning budget, an automated challenger and a paired decision

#### [day2#title] Trees and challengers

*SDAIA Academy · Machine Learning for Tabular Data: Baselines and Validation*

Day 2: Boosted trees, a fixed tuning budget, an automated challenger and a paired decision

Data Science Track · 4 teaching days + capstone day · 4 sessions · 4 labs · DS-204

#### [day2#agenda] Trees and challengers

*Today*

- **Session 1 · Gradient-boosted trees**: Boosted trees with the library's defaults: what they fit, what they memorise, and what importance scores can and cannot say.
- **Session 2 · Tuning under a budget**: A budget of 20 trials, spent inside each training part so that the reported score is not the one that chose the settings.
- **Session 3 · Automated and pretrained challengers**: An automated challenger, and what to check before a pretrained one: its licence, where the data goes, and a contest on equal terms.
- **Session 4 · Is the challenger better**: Two models scored on the same folds give paired differences: an interval on their mean, and a decision stated before looking.

#### [day2#review-day1] What did history add, and how do you know?

*Review of Day 1*

**Question:** Two history features lifted the logistic rung from 0.501 to 0.524. The fold spread of that rung is 0.028. Why is the gain still believable?

- A. Because 0.023 is large compared with the base rate of 0.12
- B. Because the table has 32,000 development rows
- C. Because the gain has the same sign on all five folds, where both models met the same rows
- D. Because the features passed the time-travel test

#### [day2#where-we-are] The features are fixed; today the model changes

*Today*

1. **Frame and baseline**: Constant 0.118, rules 0.310, logistic 0.524 (Day 1)
2. **Trees and challengers**: Boosted trees, a tuning budget, a paired decision (Day 2)
3. **Validation and leakage**: Three splits, four kinds of leak, an audit (Day 3)
4. **Errors and documentation**: Threshold, calibration, subgroups, model card (Day 4)

### Session 1: Gradient-boosted trees

#### [day2#s1] Gradient-boosted trees

*Session 1*

- Describe how boosting adds small trees to correct the errors so far
- Compare training and validation scores and read the gap
- Read importance scores and name two of their limits

#### [day2#one-tree] A tree splits the trainees into groups and predicts each group's rate

*The building block*

1. **Absent 2 days or more?** The first question splits all trainees
2. **Sponsored by an employer?** Asked only inside the "absent" group
3. **A leaf** The share that did not complete among the trainees who land here

- Every question is "is this column above or below a value?"
- A second question inside one branch is an **interaction**, found without anyone writing it
- Scaling changes nothing: trees only compare

#### [day2#boosting] Boosting adds one small tree at a time, each correcting the sum so far

*Many small trees*

- Start from the base rate
- Fit a small tree to what the current model still gets wrong
- Add a fraction of it: the **learning rate**
- Repeat a few hundred times
- The prediction is the sum of all the trees

- **Why it leads on tables.** It handles mixed columns, missing values and interactions with little preparation, and it runs on a laptop processor. On medium-sized tables, tuned tree ensembles have matched or beaten deep networks in published comparisons.

#### [day2#defaults] Defaults are one setting among thousands, chosen for nobody's table

*What ships in the box*

| Setting | LightGBM default | What it controls |
| --- | --- | --- |
| Trees | 100 | How long the sum grows |
| Leaves per tree | 31 | How many interactions one tree can hold |
| Learning rate | 0.1 | How much each tree may correct |
| Rows per leaf, at least | 20 | How small a group may be |

> **31 leaves with 20 rows each.** A tree may isolate a group of 20 trainees. With 9,000 training rows and a 12% outcome, that is two or three non-completers per leaf.

#### [day2#train-vs-valid] The training score measures memory; the gap shrinks as the rows grow

*Lab 2.1, fold by fold*

| Fold | Training rows | Score on training rows | Score on validation rows |
| --- | --- | --- | --- |
| 1 | 9,068 | 0.849 | 0.501 |
| 2 | 12,960 | 0.786 | 0.473 |
| 3 | 17,028 | 0.739 | 0.521 |
| 4 | 21,382 | 0.707 | 0.507 |
| 5 | 25,836 | 0.685 | 0.515 |

> **What goes wrong.** A report quotes 0.85. It is the model's score on rows it has already seen. Only the validation column goes into the log.

#### [day2#more-trees] Past 50 trees, each new tree helps the training rows and hurts the others

*Follow the trees*

| Trees | Training | Validation |
| --- | --- | --- |
| 10 | 0.519 | 0.492 |
| 25 | 0.559 | 0.507 |
| 50 | 0.615 | 0.517 |
| 100 | 0.685 | 0.515 |
| 200 | 0.775 | 0.508 |
| 400 | 0.897 | 0.500 |

- Fold 5, default trees of 31 leaves
- The training score never stops rising
- The validation score peaks early and declines
- The peak depends on the other settings: smaller trees peak later

#### [day2#defaults-vs-logistic] With defaults, boosted trees score below the logistic model

*The rung, measured*

| Fold | Logistic + history | Boosted, defaults | Difference |
| --- | --- | --- | --- |
| 1 | 0.557 | 0.501 | −0.056 |
| 2 | 0.514 | 0.473 | −0.041 |
| 3 | 0.530 | 0.521 | −0.009 |
| 4 | 0.509 | 0.507 | −0.002 |
| 5 | 0.512 | 0.515 | +0.004 |
| Mean | 0.524 | 0.503 | −0.021 |

- A more flexible model is not automatically a rung up
- The loss is largest where training rows are fewest
- It goes into the log as it is
- The honest next step: tune, under a budget

#### [day2#importance] Two importance scores, two different answers for the same column

*What does the model use?*

| Column | Share of gain | Score lost when shuffled |
| --- | --- | --- |
| `absences_w2` | 39% | 0.251 |
| `aptitude_score` | 16% | 0.076 |
| `prior_noncompletions` | 5% | 0.041 |
| `provider_id` | 11% | 0.029 |
| `distance_km` | 8% | 0.007 |

- **Gain** is counted on training rows: it rewards columns with many values
- **Permutation** is measured on validation rows, one column at a time
- Columns that carry the same information share the credit
- Neither gives a direction or a cause

#### [day2#check-training-score] What does a high training score tell you?

*Checkpoint*

**Question:** A boosted model scores 0.85 on its 9,068 training rows and 0.50 on the validation rows of the same fold. What do you conclude?

- A. The model has memorised part of the training rows; 0.50 is the number to log
- B. The true score is about halfway, near 0.67
- C. The validation rows must contain a leak
- D. The validation fold is unusually hard and should be replaced

#### [day2#lab-2-1] Boosted trees with defaults

Lab 2.1 · 25 min

Train boosted trees with defaults, compare training and validation scores on the fund's folds and log the rung.

[D2_1_boosted_defaults.ipynb](https://sdaiaacademy.github.io/ds-204/labs/D2_1_boosted_defaults.ipynb)

- [ ] Fit with defaults on one fold: training against validation (5 min)
- [ ] Score the five folds and log the run (5 min)
- [ ] Follow both scores as trees are added (7 min)
- [ ] Set the rung beside the logistic model (4 min)
- [ ] Read two kinds of importance (4 min)
- [ ] ✱ Stretch: Stop adding trees early, without touching the validation fold

### Session 2: Tuning under a budget

#### [day2#s2] Tuning under a budget

*Session 2*

- Name the settings of boosted trees that are worth a budget
- Run a random search whose choices are made on inner folds
- Report the gain over defaults with its fold spread and its cost in time

#### [day2#settings-that-matter] A few settings trade flexibility against stability

*What to tune*

- **How much is learnt.** The number of trees and the learning rate. They work as a pair: a smaller rate needs more trees.
- **How complex each tree is.** Leaves per tree, and the fewest rows a leaf may hold. Fewer leaves and larger leaves: less to memorise.
- **How much is held back.** Each tree sees a share of the rows and of the inputs; a penalty shrinks the leaf values.

Lab 2.2 searches seven settings with 2,592 combinations. The budget is 20 of them.

#### [day2#budget] State the budget before the search, and draw the trials at random

*The rule*

- A **budget**: 20 trials, fixed in advance, with a seed
- **Random search**: each trial draws every setting independently
- With a few settings that matter and several that hardly do, random draws cover the important ones better than a grid
- Every trial is kept, the poor ones too

- **Why a budget is about honesty.** Each trial is another look at the validation rows. A search that stops "when the score looks good" has no defined number of looks, and its best score cannot be interpreted.

#### [day2#choosing-flatters] A score that chose the winner flatters the winner

*Why tuning needs its own rows*

| Rows used to choose among 20 trials | Winner's score there | Same model, fresh rows | Flattery |
| --- | --- | --- | --- |
| 10,885 | 0.521 | 0.522 | −0.001 |
| 2,000 | 0.529 | 0.519 | +0.010 |
| 500 | 0.540 | 0.515 | +0.025 |

> **What goes wrong.** The best of 20 is partly the best and partly the luckiest. The fewer rows choose, and the more trials compete, the larger the lucky part.

#### [day2#nested] Inner folds choose, the outer fold reports

*Nested validation*

1. **1 · Outer fold** Set its validation half-year aside
2. **2 · Inner folds** Inside the training part, score 20 trials on two later half-years
3. **3 · Refit** The best settings, on the whole training part
4. **4 · Report** One score on the half-year no trial has seen

- Repeated for each of the five outer folds: five searches, five scores
- What is validated is the **procedure** "search 20 trials, then fit"
- The inner folds are built like the outer ones: later half-years

#### [day2#nested-result] Twenty trials gain 0.026 over defaults, on every fold

*Lab 2.2, measured*

| Fold | Defaults | Tuned (nested) | Gain | Settings chosen: trees · leaves · rate |
| --- | --- | --- | --- | --- |
| 1 | 0.501 | 0.542 | +0.041 | 200 · 8 · 0.05 |
| 2 | 0.473 | 0.517 | +0.044 | 400 · 4 · 0.05 |
| 3 | 0.521 | 0.540 | +0.019 | 400 · 4 · 0.05 |
| 4 | 0.507 | 0.519 | +0.013 | 400 · 8 · 0.05 |
| 5 | 0.515 | 0.526 | +0.011 | 400 · 4 · 0.10 |
| Mean · spread | 0.503 · 0.019 | 0.529 · 0.012 | +0.026 | small trees, every time |

Every fold chose trees of 4 or 8 leaves. The default is 31.

#### [day2#tuned-vs-logistic] The tuned trees are 0.005 above the logistic model

*The ladder so far*

| Item | Value |
| --- | --- |
| Three rules | 0.310 |
| Logistic + history | 0.524 |
| Boosted, defaults | 0.503 |
| Boosted, 20 trials | 0.529 |

- Against the logistic rung: ahead on four folds, behind by 0.015 on fold 1
- A mean gain of 0.005 with fold spreads of 0.012 and 0.020
- Is that a gain? Session 4 gives the tool to answer

#### [day2#check-nested] Which score may be reported?

*Checkpoint*

**Question:** An analyst tries 200 settings, scores each on the same five folds and reports the highest mean, 0.547. What is wrong with the number?

- A. 200 trials are too many for a table of this size
- B. Random search is unreliable; a grid would give the true score
- C. Five folds are too few to score any model
- D. The folds that chose the winner also score it, so part of 0.547 is luck

#### [day2#lab-2-2] Twenty trials in nested folds

Lab 2.2 · 25 min

Run a 20-trial search in nested folds, record the gain over defaults and see how far the inner score flatters the choice.

[D2_2_nested_search.ipynb](https://sdaiaacademy.github.io/ds-204/labs/D2_2_nested_search.ipynb)

- [ ] Read the search space and the budget (3 min)
- [ ] Write the search over inner folds (8 min)
- [ ] Run it inside each outer fold (6 min)
- [ ] Log the tuned rung; record the gain (4 min)
- [ ] Compare inner and outer scores (4 min)
- [ ] ✱ Stretch: Spend 40 trials: what does the second half of the budget buy?

### Session 3: Automated and pretrained challengers

#### [day2#s3] Automated and pretrained challengers

*Session 3*

- Describe what an automated tabular tool and a tabular foundation model each do
- Check a challenger's licence and where the data would be sent before using it
- Run a challenger on the same folds and the same time budget as the champion

#### [day2#two-challengers] One automates the search; the other arrives already trained

*Two kinds of challenger*

- **Automated machine learning.** You give the training rows, a metric and a time limit. The tool prepares the columns, trains several model families, tunes them and blends the best. Example: AutoGluon Tabular. It trains on your machine, on your rows only.
- **A tabular foundation model.** A network pretrained on millions of synthetic tables. It reads your training rows as context and predicts the new rows in one pass, with no tuning. Example: TabPFN. Published results are strongest on small tables.

> **Both are challengers, not shortcuts.** Each has to stand on the same folds as the champion, in the same log.

#### [day2#licence-check] Three questions decide whether a challenger may be used at all

*Before any score*

| Question | AutoGluon Tabular | TabPFN (checked November 2026) |
| --- | --- | --- |
| **Licence of the code** | Apache 2.0 | Apache 2.0 |
| **Licence of the weights** | None: it trains from your rows | Separate from the code; recent versions are released for non-commercial use |
| **Where do the rows go?** | Nowhere | Nowhere when run locally; to the provider when its hosted service is used |

> **What goes wrong.** A pretrained model wins the contest, and three weeks later someone reads the licence of its weights. Read it first, on the official page, on the day you use it.

#### [day2#fair-contest] Same folds, same features, same time, written down first

*A fair contest*

- **Folds:** the fund's five, forward in time
- **Features:** the champion's 17, no others
- **Time:** what the champion's search used, 140 seconds for five folds, so 28 seconds a fold
- **Metric:** average precision, told to the tool
- **Configuration:** one preset, fixed before the run

- **What a contest without rules measures.** Give one side an hour and the other a minute, or let one side see a column the other did not, and the result says who was favoured, not which approach is better.

#### [day2#tool-own-score] The tool's own score comes from a split it chose by itself

*Read what it built*

| Built on fold 5 | Its own validation score |
| --- | --- |
| Weighted blend | 0.491 |
| LightGBM, three variants | 0.483 to 0.484 |
| Random forests, two | 0.447 to 0.455 |
| Extra trees, two | 0.443 to 0.445 |

- It set aside a **random** part of the training rows to choose and weight its models
- The fund validates forward in time
- On the fund's fold 5 the same blend scores 0.518
- Log your score, never the tool's

#### [day2#challenger-result] On equal terms the challenger lands close to the champion

*Lab 2.3, measured*

| Fold | Champion | Challenger | Challenger − champion |
| --- | --- | --- | --- |
| 1 | 0.542 | 0.534 | −0.009 |
| 2 | 0.517 | 0.497 | −0.021 |
| 3 | 0.540 | 0.541 | +0.001 |
| 4 | 0.519 | 0.521 | +0.001 |
| 5 | 0.526 | 0.518 | −0.008 |
| Mean | 0.529 | 0.522 | −0.007 |

Ahead on two folds, behind on three. Not yet a decision.

#### [day2#check-contest] What makes this contest unfair?

*Checkpoint*

**Question:** A team reports that an automated tool beats their tuned model, 0.561 to 0.529. Which detail makes the comparison worthless?

- A. The tool blends several models and the champion is one model
- B. The tool's 0.561 is its own internal validation score; the champion's 0.529 is on the fund's folds
- C. The tool was told to optimise average precision
- D. Both ran with a limit of 28 seconds a fold

#### [day2#lab-2-3] A challenger on equal terms

Lab 2.3 · 25 min

Run an automated challenger on the same five folds with the time budget the tuned search used, and log it beside the champion.

[D2_3_challenger.ipynb](https://sdaiaacademy.github.io/ds-204/labs/D2_3_challenger.ipynb)

- [ ] Read the challenger configuration and its licence (3 min)
- [ ] Write the function that runs it on one fold (8 min)
- [ ] Run the five folds on the same time budget (7 min)
- [ ] Log the run; read what it built (4 min)
- [ ] Set the two rows of the log side by side (3 min)
- [ ] ✱ Stretch: Give the challenger the fund's inner split in place of its own

### Session 4: Is the challenger better

#### [day2#s4] Is the challenger better

*Session 4*

- Explain why fold scores of two models are compared as pairs
- Compute the mean paired difference with an interval
- Decide against a minimum effect stated in advance, in one line

#### [day2#pairs] Two models on the same folds give pairs, and pairs are compared by their differences

*The key idea*

| Item | Value |
| --- | --- |
| Spread of the champion's folds | 0.012 |
| Spread of the challenger's folds | 0.017 |
| Spread of the five differences | 0.009 |

- A hard half-year lowers both models
- The difference within a fold removes that shared part
- So the differences vary less than either score
- Compare two means with their spreads and you throw the pairing away

#### [day2#t-interval] Five differences give an honest, wide interval

*An interval on the mean difference*

**−0.007** mean difference, challenger minus champion; 95% interval −0.018 to +0.004

- Mean ± *t* × (spread of the differences ÷ √5)
- With 5 folds the multiplier is 2.78, not 1.96
- The interval contains zero: no difference is shown
- The folds share training rows, so the true uncertainty is, if anything, larger

#### [day2#paired-bootstrap] The paired bootstrap resamples validation rows, not folds

*A second opinion*

- Both models scored the same 21,770 validation rows
- Draw rows again with replacement, inside each fold
- Score **both** models on the same draw
- Keep the difference; repeat 1,000 times

| Method | 95% interval of the difference |
| --- | --- |
| *t* over five folds | −0.018 to +0.004 |
| Paired bootstrap over rows | −0.013 to −0.002 |

> **Two questions.** The first asks what other half-years would show, the second what other trainees in these half-years would show. Report both when they differ.

#### [day2#minimum-effect] State the smallest gain worth having before you look

*The decision rule*

- **Adopt the challenger.** The whole interval lies above the minimum effect.
- **Keep the champion.** The whole interval lies below the minimum effect.
- **Not shown.** The interval straddles it. The rule says what happens then: the simpler model stays.

For the fund: +0.01 average precision, agreed with Dr. Haya Al-Mogren before the contest.

#### [day2#decision-line] Difference, interval, minimum effect, decision

*The one-line decision*

> **The line of Lab 2.4.** On the same five folds the challenger differs from the champion by −0.007 average precision (95% interval −0.018 to +0.004; paired bootstrap −0.013 to −0.002). Minimum effect stated in advance: +0.01. Decision: keep the champion.

- Four parts, in this order, in the capstone README
- "Keep the champion" is shown here: the whole interval is below +0.01
- It is a statement about this contest: 28 seconds a fold, one preset

#### [day2#champion-vs-logistic] The tuned trees are not shown to beat the logistic model

*The same tool, pointed at our own ladder*

**+0.005** champion minus logistic with history; 95% interval −0.010 to +0.019

- The interval contains zero and the minimum effect
- Verdict by the rule: **not shown**
- The course carries the trees forward; the model card says a linear model does as well
- A result worth reporting, not hiding

#### [day2#check-interval] What does this interval decide?

*Checkpoint*

**Question:** A challenger's mean paired difference is +0.006, with a 95% interval from −0.004 to +0.016. The minimum effect stated in advance is +0.01. What is the decision?

- A. Adopt the challenger: its mean difference is positive
- B. The two models are shown to be equal
- C. Not shown: the interval straddles the minimum effect, so the champion stays
- D. Adopt it, with the minimum effect lowered to +0.005

#### [day2#lab-2-4] The paired difference and the decision

Lab 2.4 · 25 min

Compute the paired fold difference between champion and challenger with an interval, check it with a paired bootstrap and write the one-line decision.

[D2_4_paired_difference.ipynb](https://sdaiaacademy.github.io/ds-204/labs/D2_4_paired_difference.ipynb)

- [ ] Line up the fold scores of the two models (3 min)
- [ ] Compute the paired differences and their mean (4 min)
- [ ] Put a t interval on the mean difference (6 min)
- [ ] Check it with a paired bootstrap over rows (7 min)
- [ ] Write the decision in one line (5 min)
- [ ] ✱ Stretch: The same comparison between the champion and the logistic rung

#### [day2#wrap-up] Three things to keep

*Day 2*

- **Defaults are a rung, not the model.** Boosted trees with defaults scored 0.503, below the logistic rung at 0.524. Twenty trials in nested folds reached 0.529.
- **Choose and report on different rows.** A winner picked on 500 rows was flattered by 0.025. Inner folds choose; the outer fold reports.
- **A difference needs an interval.** Challenger −0.007 (−0.018 to +0.004): keep the champion. Champion over logistic +0.005 (−0.010 to +0.019): not shown.

#### [day2#next] Day 3 · Validation and leakage

*Next*

Every score so far used the fund's folds. Tomorrow you find out why they are built that way, what a random split would have told you, and how to audit a score that is too good.

- Loay Al-Kadi's notebook reports a score above 0.9 on this table
- Tomorrow's first attempts at the split and the leak hunt are without AI assistance

## Day 3: Validation and leakage

The split that mirrors use, the four kinds of leakage and how to audit a score

#### [day3#title] Validation and leakage

*SDAIA Academy · Machine Learning for Tabular Data: Baselines and Validation*

Day 3: The split that mirrors use, the four kinds of leakage and how to audit a score

Data Science Track · 4 teaching days + capstone day · 4 sessions · 4 labs · DS-204

#### [day3#agenda] Validation and leakage

*Today*

- **Session 1 · The split decides the score**: The same model scores differently under random, grouped and time-ordered splits, and each gap has a cause you can name.
- **Session 2 · Kinds of leakage**: Four kinds of leakage, each with the question that catches it, practised on cases from other sectors.
- **Session 3 · Hunting planted leaks**: Three instruments for an audit: when each column is written, whether rows repeat, and which features carry the score.
- **Session 4 · Reading someone else's notebook**: Reviewing a notebook you did not write: an order to read it in, and the things that make its headline number untrustworthy.

#### [day3#review-day2] What did the challenger's interval decide?

*Review of Day 2*

**Question:** The challenger differed from the champion by −0.007 (95% interval −0.018 to +0.004). The minimum effect was +0.01. Why was the decision "keep the champion" and not "not shown"?

- A. Because the mean difference is negative
- B. Because the whole interval lies below the minimum effect
- C. Because the interval excludes zero
- D. Because the champion won on three of the five folds

#### [day3#where-we-are] The model is fixed; today the score itself is questioned

*Today*

1. **Frame and baseline**: The decision, the rows, three baselines (Day 1)
2. **Trees and challengers**: Champion 0.529; challenger not adopted (Day 2)
3. **Validation and leakage**: Three splits, four kinds of leak, an audit (Day 3)
4. **Errors and documentation**: Threshold, calibration, subgroups, model card (Day 4)

### Session 1: The split decides the score

#### [day3#s1] The split decides the score

*Session 1*

- Choose the split that mirrors how a model will be used
- Implement random, grouped and time-ordered splits on the same rows
- Explain each gap between the three scores from the structure of the data

#### [day3#rehearsal] A validation split is a rehearsal of how the model will be used

*The principle*

- **Who is new in use?** New rows of known people, or people the model has never seen? If people are new, they must be new in validation.
- **What is later in use?** The model is trained once and scores what comes afterwards. Then validation rows come after training rows.
- **What is known on that day?** An outcome that does not exist yet cannot train a model. Training rows are those whose outcome was known.

The split is chosen from the use, before any score is seen.

#### [day3#three-splits] Random, grouped and time-ordered splits answer different questions

*Three ways to cut the same rows*

| Split | How it cuts | The question it answers |
| --- | --- | --- |
| **Random** | Each row lands in a fold by chance | How well does the model do on more rows like these, from the same period? |
| **Grouped** | All rows of one trainee stay on one side | How well does it do on a trainee it has never seen? |
| **Time-ordered** | Train on the past, validate on a later half-year | How well does it do on the next intake? |

> **None is wrong in itself.** A split is wrong for a use. A random split is right when rows are independent and the model is used on rows like them.

#### [day3#what-training-knows] A random split trains on the future and on outcomes nobody knew yet

*Lab 3.1, before any score*

| For the day a validation row is scored | Random | Grouped | Time-ordered |
| --- | --- | --- | --- |
| Validation trainees who also have a training row | 40% | 0% | 30% |
| Training rows that start later than that day | 47% | 47% | 0% |
| Training rows whose programme was not yet due to end | 58% | 58% | 0% |
| Training rows per fold | 25,951 | 25,951 | 17,254 |

> **What goes wrong.** A split is judged by its score only. Describe first what the training part knows; the table costs ten lines and no model.

#### [day3#due-to-end] Train on programmes that were due to end, not on those that had ended

*A detail that decides the folds*

- On 1 July 2024 a 36-week programme that began in March is still running
- Its early leavers already have an outcome; its other trainees have none
- "Every row with a known outcome" keeps the leavers and drops their classmates
- The model then learns that recent long programmes always fail

- **The fund's rule.** A row trains a fold only when its programme's **planned end** lies before the validation half-year begins. The whole class enters together, or not at all.

#### [day3#three-scores] For the champion, the random split does not overstate; hiding the history does

*Lab 3.1, measured*

| Item | Value |
| --- | --- |
| Random | 0.521 |
| Grouped | 0.520 |
| Grouped, history hidden | 0.500 |
| Time-ordered | 0.525 |

- Random against time-ordered: −0.004, inside the fold spreads of 0.021 and 0.012
- Grouping alone changes nothing: the champion does not recognise a trainee
- A trainee the fund has never seen has no history: −0.020

#### [day3#new-and-returning] One score averages two kinds of trainees

*The grouped gap, explained*

| In the fund's validation folds | Share of rows | Average precision |
| --- | --- | --- |
| New to the fund | 66% | 0.491 |
| Returning | 34% | 0.578 |

- The 0.529 of the ladder is partly earned on returning trainees
- For a programme where nobody has a history, quote the grouped score with history hidden: 0.500
- Both numbers go into the model card

#### [day3#ageing] A model one year old scores 0.016 below a fresh one

*What only time order can show*

| Age of the model | Against a fresh model | Pairs |
| --- | --- | --- |
| Fresh | 0.000 | 5 |
| Half a year | −0.006 | 4 |
| One year | −0.016 | 3 |
| One and a half years | −0.020 | 2 |

- Each fold's model, scored on the later half-years too
- A random split trains and validates on one period: its model never ages
- The older model also had fewer rows: both are what "not retraining" means
- The number sets the retraining interval

#### [day3#when-random-overstates] When does a random split overstate badly?

*Beyond this table*

- **Many rows per unit.** Thousands of readings per machine, dozens of visits per patient. The model recognises the unit. Here: at most four rows per trainee.
- **Features built from other rows.** A rate or a count that read the validation rows or the future. The next three sessions.
- **A world that moves fast.** Prices, fraud patterns, policies. The faster the change, the more a model's age costs.

> **The habit.** Choose the split from the use. Then measure the other splits once, and write the gaps down with their causes.

#### [day3#check-split] Which split for which use?

*Checkpoint*

**Question:** The fund wants to offer Mubakkir to a partner fund in another country. None of its trainees has ever enrolled with Sullam. Which score from Lab 3.1 comes closest to what the partner should expect?

- A. The time-ordered score, 0.525
- B. The random score, 0.521
- C. The grouped score with history hidden, 0.500, and even that with caution
- D. The score of returning trainees, 0.578

#### [day3#lab-3-1] One model, three splits

Lab 3.1 · 25 min

Score one model under random, grouped and time-ordered splits, measure how far the random split overstates and explain each gap.

[D3_1_three_splits.ipynb](https://sdaiaacademy.github.io/ds-204/labs/D3_1_three_splits.ipynb)

- [ ] Build the three splits on the same rows (5 min)
- [ ] Score the same model under each (7 min)
- [ ] Explain the gap: new and returning trainees (4 min)
- [ ] What only time order shows: let the model age (5 min)
- [ ] Write the gap table with one sentence per gap (4 min)
- [ ] ✱ Stretch: A model that memorises: does the grouped gap grow?

### Session 2: Kinds of leakage

#### [day3#s2] Kinds of leakage

*Session 2*

- Distinguish target leakage, future information, group or duplicate contamination and preparation leakage
- Identify the kind of leak from a description of how the data were collected
- Match each case to the validation scheme its use requires

#### [day3#four-kinds] Leakage comes in four kinds

*A taxonomy*

- **Target leakage.** A column written because of the outcome, or together with it.
- **Future information.** A value computed from records later than the prediction time.
- **Group or duplicate contamination.** The same person, machine or row on both sides of the split.
- **Preparation leakage.** A step fitted or chosen on rows that later score the model.

All four give a score in validation that the model cannot repeat in use.

#### [day3#target-leak] Target leakage: the column knows the answer

*Kind 1*

- Recorded when the outcome happens: `exit_reason`, `weeks_attended`
- A consequence of the outcome: calls from the collections team after a missed payment
- Someone's answer to the same question: an antibiotic ordered because the doctor suspects the infection

- **The question.** When is this column written, and by whom? Would a person who already knew the outcome be the one filling it in?

> **What goes wrong.** The model looks brilliant and the top feature is one nobody questioned because it "obviously matters".

#### [day3#future-leak] Future information: right column, wrong date

*Kind 2*

- A history that counts later records: all non-completions of a trainee, whenever they happened
- A yearly average attached to a row from March
- A status column overwritten in place: today's value on an old row

- **The question.** On the day of the prediction, what would the value of this feature have been? Rebuild it with later records hidden and compare.

#### [day3#group-leak] Group or duplicate contamination: the model has met the test rows

*Kind 3*

- Many rows of one unit, split at random: hours of one pump, visits of one patient
- The same record under two keys after two systems were merged
- Copies made on purpose: classes "balanced" by duplicating rows before the split

- **The question.** What is the unit the model meets in use? Can the same real-world event appear in the table more than once?

#### [day3#preparation-leak] Preparation leakage: something was fitted before the split

*Kind 4*

- Scaling or imputing with statistics of all rows
- Encoding a category by its outcome rate over all rows
- Selecting the "best" features with the outcome of all rows
- Choosing settings on the rows that report the score

- **The question.** Was any step fitted or chosen before the split? Everything that learns from data belongs inside the pipeline, fitted on training rows only.

#### [day3#scheme-from-use] The validation scheme follows from the use

*The second half of each case*

| The model will be used on | Scheme |
| --- | --- |
| More independent rows like these, same period | Random |
| Units it has never seen: new pumps, new patients, a new site | Grouped by that unit |
| Later records of a population that changes | Time-ordered |
| New units, later | Grouped and time-ordered |

In Lab 3.2 the last sentence of each case states the use.

#### [day3#check-kind] Name the leak

*Checkpoint*

**Question:** A bank predicts which customers will close their account next month. One feature is the number of calls the customer made to the retention desk. Customers are transferred to that desk when they ask to close the account. Which kind of leak is this?

- A. Target leakage
- B. Preparation leakage
- C. Group contamination
- D. No leak: call behaviour is a legitimate feature

#### [day3#lab-3-2] Six leaks from other projects

Lab 3.2 · 25 min

Classify six leaks from other projects by type, name the validation scheme each case needs and check the table against the reference.

[D3_2_six_leaks.ipynb](https://sdaiaacademy.github.io/ds-204/labs/D3_2_six_leaks.ipynb)

- [ ] Read the six cases and how their data were collected (5 min)
- [ ] Name the kind of leak in each (8 min)
- [ ] Name the validation scheme each needs (6 min)
- [ ] Check the table; read the explanations (3 min)
- [ ] Write the question that would have caught each (3 min)
- [ ] ✱ Stretch: Add a seventh case from your own work

### Session 3: Hunting planted leaks

#### [day3#s3] Hunting planted leaks

*Session 3*

- Use the time each column is recorded to find features written after the outcome
- Find duplicated rows that sit on both sides of a split
- Use importance scores to choose which features to question first

#### [day3#too-good] A score far above your ladder is the start of an audit

*The trigger*

**1.000** average precision of the table Loay Al-Kadi sent, on his random five-fold split

- The honest ladder ends at 0.529
- The outcome depends on months of a trainee's life after week 2
- No table of week-2 columns can know that perfectly
- So some column is not a week-2 column

#### [day3#instrument-timestamps] Timestamps: when is each column written?

*Instrument 1*

| Column | Recorded at |
| --- | --- |
| `aptitude_score` | Registration |
| `absences_w2` | End of week 2 |
| `weeks_attended` | Exit |
| `provider_rate` | Not in the dictionary |
| `trainee_noncompletions` | Not in the dictionary |

- The data dictionary answers for recorded columns
- Removing `weeks_attended`: 1.000 falls to 0.882
- A column missing from the dictionary was **computed**: ask from which rows

#### [day3#instrument-duplicates] Duplicates: a new key on an old row

*Instrument 2*

- 33,491 rows, 33,491 different enrolment keys
- 2,104 rows agree with another row on trainee, provider, programme and start date
- Two provider systems were merged; 1,052 enrolments were entered twice
- Removing the copies: 0.882 becomes 0.881

- **How to look.** Do not test the key. Test the columns that describe the real event. Then ask the owner of the data why they repeat.

#### [day3#instrument-importance] Importance scores: question the feature that carries the score

*Instrument 3*

| Item | Value |
| --- | --- |
| trainee_noncompletions | 80% |
| absences_w2 | 8% |
| aptitude_score | 5% |
| provider_rate | 2% |

- Share of gain after the first two removals
- One computed column carries four fifths of the model
- Importance does not prove a leak: it says where to look first

#### [day3#rebuild] Rebuild the column and see which rows it must have read

*Testing a computed column*

| Computed column | It equals, on every row | Verdict |
| --- | --- | --- |
| `trainee_noncompletions` | All non-completions of the trainee, this enrolment and later ones included. It equals the past-only count on 84% of rows. | **Leak**: future information |
| `provider_rate` | The provider's rate over all rows, 2021 to 2025 | **Leak**: preparation |
| `earlier_enrolments` | The trainee's enrolments that started earlier, without their outcome | **Keep** |

> **An audit also clears columns.** The third one reads only what the fund knows on the prediction day.

#### [day3#one-at-a-time] One removal at a time, with the score before and after

*The discipline*

| Removal | Before | After | Kind |
| --- | --- | --- | --- |
| `weeks_attended` | 1.000 | 0.882 | Target leakage |
| The copied rows | 0.882 | 0.881 | Duplicate contamination |
| `trainee_noncompletions` | 0.881 | 0.503 | Future information |
| `provider_rate` | 0.503 | 0.501 | Preparation leakage |

- Each leak hid the ones behind it
- On the fund's folds the cleaned table scores 0.505; the champion, with a past-only history, 0.529

#### [day3#size-and-seriousness] The size of a leak is measured; whether it is a leak is decided by how the column was made

*A judgment*

- `provider_rate` moved the score by 0.002
- Each provider has several hundred rows: one row's own outcome hardly changes the rate
- With five rows per provider the same column would dominate
- And it read the 2025 cohort, the holdout

- **Why remove a small leak.** The measured effect belongs to this table and this model. The fault belongs to the procedure, and the procedure will be reused on the next table.

#### [day3#check-instrument] Which instrument first?

*Checkpoint*

**Question:** A received table scores far above your ladder. Its data dictionary lists when every recorded column is written. Twelve further columns were computed by a colleague. Where do you start?

- A. Tune the model again on the received table
- B. Drop all twelve computed columns
- C. Switch to a time-ordered split and compare
- D. Read the dictionary for columns recorded after the prediction time, then rank the rest by importance

#### [day3#lab-3-3] Four planted leaks

Lab 3.3 · 25 min

Find four planted leaks in a received feature table with timestamps, duplicates and importance scores, removing one at a time with the score before and after.

[D3_3_leak_hunt.ipynb](https://sdaiaacademy.github.io/ds-204/labs/D3_3_leak_hunt.ipynb)

- [ ] Score the received table as it came (4 min)
- [ ] Timestamps: when is each column written? (5 min)
- [ ] Duplicates: the same enrolment under two keys (5 min)
- [ ] Importance scores: question the top features (7 min)
- [ ] Write the audit note: before and after each removal (4 min)
- [ ] ✱ Stretch: Rebuild the provider rate from past cohorts only

### Session 4: Reading someone else's notebook

#### [day3#s4] Reading someone else's notebook

*Session 4*

- Read a notebook in the order rows, split, preparation, tuning, metric
- List the faults that make a reported score untrustworthy
- Measure which fault moves the score most by repairing one at a time

#### [day3#first-questions] Three questions to any reported score

*Before any code*

- **On which rows?** How many test rows, from which period, with what share of the outcome? Do they look like the population the model will score?
- **Against what?** What does a constant answer score on the same rows? What did the previous model score?
- **Chosen how?** How many versions were tried, and on which rows was this one chosen?

"95% accuracy" answers none of them.

#### [day3#accuracy-rare] For a rare outcome, accuracy rewards saying "nothing happens"

*The metric*

| Item | Value |
| --- | --- |
| "Everyone completes", all enrolments | 0.83 |
| "Everyone completes", scorable rows of 2024 H2 | 0.87 |
| Loay's notebook, all five blocks repaired | 0.89 |

- Accuracy counts both classes alike and depends on their share
- It needs a threshold, and 0.5 is nobody's decision
- The run log uses average precision and recall at capacity for this reason

#### [day3#review-order] Rows, split, preparation, tuning, metric

*An order to read in*

1. **1 · Rows** Who is in the table? Anyone the model never scores?
2. **2 · Split** Does it rehearse the use?
3. **3 · Preparation** What was fitted, copied or encoded before the split?
4. **4 · Tuning** On which rows was the model chosen?
5. **5 · Metric** Does the number mean what the message says?

The earlier the fault, the more of what follows it spoils.

#### [day3#balancing] "Balancing the classes" before the split copies test rows into training

*The fault that did the damage*

- Each non-completer is copied four times, then the table is split at random
- Most test non-completers have a copy in training
- A tree with five rows per leaf recognises them
- The test set is half non-completers: no cohort looks like it

- **If you balance at all.** Inside the training part only, as a step of the pipeline. The test rows keep the real share. Tomorrow shows that a ranking does not need it.

#### [day3#repairs] Repair one fault at a time, then all together

*Lab 3.4, measured*

| Run | Accuracy | "Everyone completes" | Average precision |
| --- | --- | --- | --- |
| As written | 0.948 | 0.499 | 0.980 |
| Repair the rows only | 0.978 | 0.490 | 0.994 |
| Repair the balancing only | 0.865 | 0.835 | 0.569 |
| Repair the split only | 0.743 | 0.475 | 0.843 |
| Repair the encoding or the tuning only | 0.95 | 0.499 | 0.98 |
| All five repaired | 0.890 | 0.874 | 0.494 |

One repair makes the number go up. Faults interact.

#### [day3#review-note] A review note names numbers, not character

*What you send back*

> **Five lines.** The number as reported. The fault that matters most, with its measured effect. The other faults. The honest number. What to do next.

- "Your split is wrong" starts an argument
- "With the copies kept out of the test set the accuracy is 0.87, and a constant answer scores 0.84" ends one
- Offer the next step: one pipeline, the fund's folds, a line in the run log

#### [day3#check-review] What do you ask first?

*Checkpoint*

**Question:** A vendor's slide says: "Our model predicts non-completion with 91% accuracy." Which question tells you most, fastest?

- A. Which algorithm does the model use?
- B. How many rows was it trained on?
- C. What share of the test rows completed, and what does "everyone completes" score on them?
- D. What is its area under the curve?

#### [day3#lab-3-4] A colleague's notebook

Lab 3.4 · 25 min

List what makes a colleague's notebook score untrustworthy, check the list, repair the faults one at a time and write the review note.

[D3_4_colleague_notebook.ipynb](https://sdaiaacademy.github.io/ds-204/labs/D3_4_colleague_notebook.ipynb)

- [ ] Run the notebook as it is; read the reported number (3 min)
- [ ] Read the code and mark what you find (9 min)
- [ ] Check your list against the reference (3 min)
- [ ] Repair one fault at a time and rescore (6 min)
- [ ] Write a five-line review note (4 min)
- [ ] ✱ Stretch: Rewrite the notebook as one pipeline scored on the fund's folds

#### [day3#wrap-up] Three things to keep

*Day 3*

- **The split rehearses the use.** For the champion: random 0.521, time-ordered 0.525, new trainees 0.500, one year old −0.016. Gaps are measured and written down with their causes.
- **Four kinds, four questions.** When is it written? What was its value that day? Can the unit be on both sides? Was anything fitted before the split?
- **One removal at a time.** 1.000, 0.882, 0.881, 0.503, 0.501. And 95% accuracy became 0.89 against a constant 0.87.

#### [day3#next] Day 4 · Errors, groups and documentation

*Next*

The score is trusted. Tomorrow it becomes a decision: where to cut the list for 800 support plans, whether the probabilities hold, for whom the model fails, and what goes into the model card.

- The 2025 cohort is opened, once
- The first attempt at the subgroup table is without AI assistance
- You choose your capstone problem in the last lab

## Day 4: Errors, groups and documentation

Where to cut the ranking, whether its probabilities hold, for whom it fails, and what to write down

#### [day4#title] Errors, groups and documentation

*SDAIA Academy · Machine Learning for Tabular Data: Baselines and Validation*

Day 4: Where to cut the ranking, whether its probabilities hold, for whom it fails, and what to write down

Data Science Track · 4 teaching days + capstone day · 4 sessions · 4 labs · DS-204

#### [day4#agenda] Errors, groups and documentation

*Today*

- **Session 1 · Thresholds for rare outcomes**: A score becomes a decision at a threshold, and the threshold comes from capacity and costs, never from 0.5.
- **Session 2 · Calibration**: Calibration: measured by decile, repaired on unseen rows, lost again under drift.
- **Session 3 · Errors by subgroup**: Errors are harms to people: who is represented, who is missed, and how sure a small slice lets you be.
- **Session 4 · The model card**: The model card is the hand-over: what the model is for, how it was validated, where it fails and what it must never decide.

#### [day4#review-day3] Which kind of leak was it?

*Review of Day 3*

**Question:** In the received table, `trainee_noncompletions` equalled the past-only count on 84% of the rows and carried 80% of the model's gain. Which kind of leak is it?

- A. Group or duplicate contamination
- B. Preparation leakage
- C. No leak: a history of non-completion is a fair feature
- D. Future information

#### [day4#where-we-are] From a trusted score to a decision and a page that travels with it

*Today*

1. **Frame and baseline**: The decision, the rows, three baselines (Day 1)
2. **Trees and challengers**: Champion 0.529; challenger not adopted (Day 2)
3. **Validation and leakage**: Gaps measured; four leaks removed (Day 3)
4. **Errors and documentation**: Threshold, calibration, subgroups, model card (Day 4)

### Session 1: Thresholds for rare outcomes

#### [day4#s1] Thresholds for rare outcomes

*Session 1*

- Read precision and recall off a precision-recall curve for a rare outcome
- Choose a threshold that fits the advisers' capacity and report its precision and recall
- Explain what happens to a fixed threshold when the next cohort differs

#### [day4#score-to-decision] A model gives a score; a threshold turns it into an action

*The step nobody trains*

1. **Score** 0.47 for this trainee, at the end of week 2
2. **Threshold** On the list when the score is at or above the cut
3. **Action** An adviser calls in week 3

- Average precision judged the ranking over every possible threshold
- The fund runs one threshold, and lives with its two kinds of error
- The threshold is a decision of the fund, informed by the model

#### [day4#precision-recall] Precision speaks for the advisers, recall for the trainees

*Two sentences for Ghassan*

- **476 of 800** plans on the 2024 list go to a trainee who then left: precision 0.60
- **476 of 1,190** trainees who left were on the list: recall 0.40

- A random list of 800 would find about 103
- Six in ten trainees who left were **not** on the list
- Say both sentences; either one alone misleads

#### [day4#three-thresholds] Three thresholds, three different lists

*Lab 4.1, measured*

| Threshold | On the list | Precision | Recall | Net value, thousand riyals |
| --- | --- | --- | --- | --- |
| 0.5, the library default | 582 | 0.65 | 0.32 | 589 |
| 0.41, fits 800 plans | 800 | 0.60 | 0.40 | 711 |
| 0.20, break-even | 1,679 | 0.41 | 0.58 | 790 |

- 0.5 leaves 218 plans unused: nothing in the decision says 0.5
- Break-even: a plan costs 450, prevents one loss of 9,000 in four
- Capacity picks the middle row

#### [day4#weights] Class weights move the scores; they do not choose a threshold

*A common remedy*

| Model, cut at 0.5 | On the list | Average precision |
| --- | --- | --- |
| Champion | 582 | 0.516 |
| Champion with class weights | 2,916 | 0.520 |

- "The outcome is rare, so weight it": the ranking stays as good
- The scores shift upward, so 0.5 now flags a third of the cohort
- Rarity is handled by the metric and the threshold, not by changing what the scores mean

#### [day4#holdout] The same threshold gives a different list on the next cohort

*The 2025 cohort, opened once*

| Rule | On the list | Precision | Recall |
| --- | --- | --- | --- |
| 2024, threshold 0.41, where it was chosen | 800 | 0.60 | 0.40 |
| 2025, the same threshold | 897 | 0.61 | 0.40 |
| 2025, the top 800 | 800 | 0.65 | 0.38 |

> **The model did not get worse.** The cohort is larger and its scores sit a little higher. A threshold on a probability and a capacity in plans are two different things.

#### [day4#rule] A fixed threshold or a fixed list length

*The choice to write down*

- **Fixed list length.** Always fits the advisers. The weakest name on the list varies: in a hard intake, trainees at higher risk are left out.
- **Fixed threshold.** Equal risk is treated equally across intakes. The list overruns or underuses capacity, and depends on the probabilities staying calibrated.

- Either can be defended; the fund chose the list length
- Write down which rule runs, and who may change it

#### [day4#check-threshold] What happens to the list next year?

*Checkpoint*

**Question:** The fund keeps the threshold of 0.41. Next year a new programme doubles the intake, with trainees much like this year's. What happens?

- A. The list roughly doubles to 1,600 names, twice what the advisers can support
- B. The list stays at 800, because the threshold was chosen for 800
- C. Precision halves, because there are twice as many trainees
- D. The scores fall, so fewer trainees pass the threshold

#### [day4#lab-4-1] A threshold that fits capacity

Lab 4.1 · 25 min

Pick a threshold on the precision-recall curve that fits adviser capacity, compare it with 0.5 and with break-even, and test it on the held-out cohort.

[D4_1_threshold_capacity.ipynb](https://sdaiaacademy.github.io/ds-204/labs/D4_1_threshold_capacity.ipynb)

- [ ] Draw the precision-recall curve of the validation cohort (4 min)
- [ ] Find the threshold that fits 800 support plans (6 min)
- [ ] Compare with 0.5 and with the break-even probability (5 min)
- [ ] Apply the threshold to the held-out cohort (6 min)
- [ ] Decide: a fixed threshold or a fixed list length (4 min)
- [ ] ✱ Stretch: An interval for the precision at the chosen threshold

### Session 2: Calibration

#### [day4#s2] Calibration

*Session 2*

- Compare expected and observed rates by decile and read a reliability plot
- Calibrate a model on rows it was not trained on, with sigmoid and isotonic methods
- Explain why a weighted model ranks well and still cannot be read as a probability

#### [day4#promise] A calibrated score keeps a promise: of those given 0.3, three in ten leave

*The idea*

- Calibration is about what the numbers **mean**
- Ranking is about their **order**
- A model can be good at one and bad at the other
- It matters when the number is shown to an adviser, compared with a break-even, or multiplied by a cost

- **How to measure it.** Sort the trainees by score and cut them into ten groups. In each: the mean score (expected) beside the share that left (observed). One summary number: the Brier score, the mean squared gap between score and outcome.

#### [day4#deciles] Same ranking, very different promises

*Lab 4.2, cohort 2024*

| Decile | Champion expects | Observed | Weighted model expects | Observed |
| --- | --- | --- | --- | --- |
| 1, lowest | 0.013 | 0.017 | 0.071 | 0.016 |
| 5 | 0.050 | 0.071 | 0.297 | 0.073 |
| 8 | 0.144 | 0.144 | 0.573 | 0.133 |
| 9 | 0.254 | 0.219 | 0.723 | 0.231 |
| 10, highest | 0.603 | 0.558 | 0.902 | 0.554 |

> **What goes wrong.** An adviser reads "0.90" beside a name. Of the trainees given that number, 55% left.

#### [day4#calibrators] A calibrator is a small model from score to probability

*The repair*

- **Sigmoid.** A logistic regression on the log-odds of the score: two numbers, a shift and a stretch. Works with few rows; cannot bend.
- **Isotonic.** A rising step function. Takes any rising shape; needs more rows, and trainees on one step become tied.

- Fitted on rows the scoring model was **not** trained on: here the 2024 cohort
- Tested on rows neither has seen: the 2025 cohort
- For the weighted model the sigmoid's shift is −2.0; for the champion, −0.2

#### [day4#calibration-result] Calibration repairs what the weights broke, and adds nothing to a calibrated model

*Lab 4.2, on the 2025 cohort*

| Model | Calibration | Brier | Mean expected | Average precision |
| --- | --- | --- | --- | --- |
| Champion | as delivered | 0.0852 | 0.137 | 0.554 |
| Champion | sigmoid | 0.0853 | 0.133 | 0.554 |
| Weighted | as delivered | 0.1717 | 0.399 | 0.553 |
| Weighted | sigmoid | 0.0854 | 0.131 | 0.553 |
| Weighted | isotonic | 0.0858 | 0.131 | 0.526 |

Observed: 0.135. The sigmoid never changes the order; isotonic ties cost 0.03 here.

#### [day4#drift] Calibration holds for a population; eight providers are new

*Where the promise breaks*

| Cohort 2025 | Trainees | Expected | Observed (95% interval) |
| --- | --- | --- | --- |
| Providers open before 2025 | 8,214 | 0.130 | 0.119 (0.113 to 0.127) |
| Providers that opened in 2025 | 1,903 | 0.163 | 0.204 (0.186 to 0.223) |

- For the new providers the probabilities are four points too low
- The ranking inside them is as good as elsewhere
- A provider the model does not know is scored like an average one

#### [day4#check-calibration] Which score may be read as a probability?

*Checkpoint*

**Question:** Two models have the same average precision. Model A's scores average 0.14, model B's 0.40; 13.5% of the cohort left. The fund wants to list everyone above the break-even probability, 0.20. Which statement holds?

- A. Either model can be used: they rank equally well
- B. Only model A's scores can be compared with 0.20, after checking them by decile
- C. Model B is better: it is more confident about the trainees at risk
- D. Use the mean of the two scores

#### [day4#lab-4-2] Calibrate and compare by decile

Lab 4.2 · 25 min

Calibrate the boosted model, compare expected and observed rates by decile before and after, and see where the 2025 cohort moves the calibration.

[D4_2_calibration.ipynb](https://sdaiaacademy.github.io/ds-204/labs/D4_2_calibration.ipynb)

- [ ] Expected against observed by decile, as delivered (5 min)
- [ ] Fit sigmoid and isotonic calibrators on the validation cohort (6 min)
- [ ] Compare by decile on the held-out cohort (6 min)
- [ ] Look at the providers that opened in 2025 (4 min)
- [ ] Write the calibration line of the model card (4 min)
- [ ] ✱ Stretch: Calibrate inside the time-ordered folds

### Session 3: Errors by subgroup

#### [day4#s3] Errors by subgroup

*Session 3*

- Compare the training data with the eligible population by region, age band and gender
- Report held-out error rates by subgroup with intervals that reflect slice size
- Relate each finding to a bias type of the SDAIA AI Bias Reference Guide and to one action

#### [day4#two-harms] Two errors, two different harms

*Errors are not symmetric*

- **Missed.** A trainee leaves and was not on the list. Nobody called. The harm falls on the trainee. Measured among those who left: the **miss rate**, 0.62 for the 2025 list.
- **Listed without need.** A trainee on the list would have completed anyway. An adviser's hour is spent; the trainee gets a friendly call. Measured on the list: one minus **precision**.

- The first harm leads the audit
- The question: is the miss rate the same for everyone?

#### [day4#represented] Who is in the data the model learnt from?

*Before any error*

| Group | In training data | In eligible population | Ratio |
| --- | --- | --- | --- |
| Age 35 and over | 4.7% | 20% | 0.24 |
| Age 17 to 19 | 6.8% | 16% | 0.42 |
| Najran | 2.2% | 3.6% | 0.61 |
| Jazan | 4.2% | 6.8% | 0.62 |
| Women | 43% | 52% | 0.82 |
| Age 20 to 24 | 45% | 31% | 1.44 |

Every validation score of this week describes the people who enrolled, not the people the fund serves.

#### [day4#intervals] A rate from a small slice needs an interval that knows its size

*The tool*

| Slice | Left | Missed | 95% interval |
| --- | --- | --- | --- |
| Riyadh | 444 | 0.58 | 0.53 to 0.62 |
| Jazan | 86 | 0.73 | 0.63 to 0.82 |
| Qassim | 21 | 0.81 | 0.60 to 0.92 |
| Al-Baha | 12 | 0.75 | 0.47 to 0.91 |

- The miss rate is a share of those who left: that count sets the width
- The Wilson interval stays inside 0 and 1 and behaves for small counts
- Read the width before the position

#### [day4#by-region] By the rule stated in advance, two regional gaps are supported

*Lab 4.3, measured*

- **The rule:** a gap is supported when the whole interval lies above or below the cohort's miss rate, 0.62
- **Jazan:** 0.73, interval 0.63 to 0.82: above
- **Riyadh:** 0.58, interval 0.530 to 0.622: below 0.623, by a hair
- Najran, 0.75: its interval starts at 0.61
- Qassim, 0.81: too few cases to tell

- **Nineteen slices.** Thirteen regions, two genders, four age bands. With that many, one can cross the line by chance. A rule fixed in advance limits the damage; a cause you can check does more.

#### [day4#gender-age] A five-point gap between women and men is not supported, and not dismissed

*The other two attributes*

| Slice | Left | Missed | 95% interval |
| --- | --- | --- | --- |
| Women | 625 | 0.65 | 0.61 to 0.69 |
| Men | 744 | 0.60 | 0.57 to 0.64 |
| Age 17 to 19 | 91 | 0.55 | 0.45 to 0.65 |
| Age 35 and over | 123 | 0.57 | 0.48 to 0.65 |

- Both gender intervals contain 0.62
- Women are also under-represented in the training data
- So this row is read first on the next cohort's table

#### [day4#cause] Where attendance is kept on paper, the model's best signal is weakest

*A gap with a cause*

| Registers | Left | Missed (95% interval) | Recorded absences, mean | Non-completion |
| --- | --- | --- | --- | --- |
| Electronic, eleven regions | 1,232 | 0.61 (0.58 to 0.64) | 0.54 | 13% |
| Paper, Jazan and Najran | 137 | 0.74 (0.66 to 0.80) | 0.27 | 23% |

- More trainees leave there, and fewer early absences are recorded
- The model is right about what the registers say
- Retraining changes nothing: the fault is in the measurement

#### [day4#bias-types] Four findings, four bias types of the SDAIA guide, one action each

*Naming what was found*

| Finding | Bias type | One action |
| --- | --- | --- |
| Trainees who leave in Jazan and Najran are missed more | **Measurement** | Electronic registers; advisers there do not rely on the list alone |
| Trainees of 35 and over are a quarter as present as in the population | **Coverage** | State it in the card; validate again before use on programmes for this group |
| The highest miss rate rests on 21 cases | **Sample size** | Report intervals; pool small regions or cohorts |
| One miss rate, 0.62, was all that had been reported | **Aggregation** | Report by region, gender and age band with every cohort |

#### [day4#not-blind] Leaving a column out does not make a model blind to it

*A common belief*

- Gender is not a feature of Mubakkir
- Other features differ between women and men: programme, employment, distance
- And who is in the data differs: 43% against 52%
- So errors can still differ by gender

- **What follows.** The attribute stays in the table, outside the model, so that errors can be audited. Under the Personal Data Protection Law personal data serve a stated purpose: here the audit of errors, agreed with the fund's data protection officer.

#### [day4#check-slice] Reading a small slice

*Checkpoint*

**Question:** In one region 12 trainees left and 9 of them were not on the list: a miss rate of 0.75, interval 0.47 to 0.91. The cohort's rate is 0.62. What do you report?

- A. The model fails this region: 0.75 is far above 0.62
- B. The region is fine: its interval contains 0.62
- C. Leave the region out of the table: the slice is too small
- D. The rate with its interval, and that 12 cases cannot tell; pool with later cohorts

#### [day4#lab-4-3] Errors by region, gender and age band

Lab 4.3 · 25 min

Break held-out errors down by region, gender and age band with intervals, check who is represented and name the bias type behind each finding.

[D4_3_subgroup_errors.ipynb](https://sdaiaacademy.github.io/ds-204/labs/D4_3_subgroup_errors.ipynb)

- [ ] Compare the training data with the eligible population (5 min)
- [ ] Miss rate and precision by region, with intervals (7 min)
- [ ] The same by gender and by age band (5 min)
- [ ] Decide which gaps the intervals support (4 min)
- [ ] Name the bias type and one action per finding (4 min)
- [ ] ✱ Stretch: The same table by provider: how many slices can carry an interval?

### Session 4: The model card

#### [day4#s4] The model card

*Session 4*

- Assemble a model card from measured results
- State the limits of the validation and the uses the model is not for
- Prepare the hand-over a deployment team needs and frame the capstone problem

#### [day4#card-holds] A model card is the one page that travels with the model

*The hand-over*

- **Measured: filled from results.** Intended use · data · validation scheme · results on held-out rows · calibration · errors by subgroup.
- **Written: no number writes them.** What the model is not for · where the scores do not hold · who decides · what the next team needs.

- Its readers were not in the room: an adviser, an auditor, the deployment team
- Every number in it can be traced to a run

#### [day4#honest-results] The results section says what was not shown

*What the card admits*

- Champion 0.529 on the folds; against the logistic rung +0.005, interval −0.010 to +0.019: **an advantage is not shown**
- New trainees 0.491, returning trainees 0.578: the headline is an average
- Expected 13.7%, observed 13.5% overall; four points too low at new providers

> **Why write it down.** The next team may need a model an adviser can read, or one that is cheaper to run. The card tells them a simpler one does as well within the uncertainty.

#### [day4#limits] A useful limit names a situation and what to do in it

*The written half*

| Weak | Useful |
| --- | --- |
| "May not generalise to new data." | "At a provider that opened after training the probabilities are about four points too low: use the rank, not the number." |
| "Performance may degrade over time." | "A model one year old scored 0.016 lower: retrain every half-year, on programmes due to end." |
| "Use with care in some regions." | "Where attendance is kept on paper, a trainee who is not on the list is not thereby low-risk." |

A reader should recognise the situation when they are in it.

#### [day4#human] The score puts a name on a list; a person decides

*The sentence to keep*

- An adviser decides whether and how to contact a trainee
- An adviser may add any trainee who is not on the list
- No trainee is refused, ranked for funding or told a score
- Providers are not ranked by it

- **What the score does not answer.** It says who is likely to leave. It does not say whether a support plan helps, or whom it helps most. That needs a comparison of supported and unsupported trainees, which the fund has not run.

#### [day4#handover] What a deployment team cannot guess

*After this course*

- The code version, the seed and the lock file
- The date of training and the rule for training rows
- Each feature, when it is recorded, and how history is built
- The list rule, and who may change it
- What to monitor each intake, and when to retrain

- **Where it continues.** Serving, monitoring and retraining are not in this course. AE-205 deploys a model as a service; AE-305 operates it. Both start from a card like this one.

#### [day4#capstone-brief] The capstone: one problem, the whole path, eight minutes

*Day 5*

- Six problems on the shortlist: three synthetic cases, three public tables
- A public repository of your own, AI-full, with prompts and transcripts committed
- A README that reports ladder, validation, leakage audit, threshold, subgroup errors and one paired comparison

- **The defence.** Four minutes are yours. Then the assessor swaps the split or removes a feature, and you say what will happen to the score, and why, before it is run.

#### [day4#check-card] Which sentence belongs in the card?

*Checkpoint*

**Question:** You may add one sentence to the limits of Mubakkir's card. Which one helps its readers most?

- A. "The model uses gradient-boosted trees with 400 trees of 8 leaves."
- B. "Results may vary on new data."
- C. "Not validated for trainees in their first two weeks: they have no early-absence record."
- D. "The model achieves an average precision of 0.55 on held-out data."

#### [day4#lab-4-4] The model card and the capstone problem

Lab 4.4 · 25 min

Draft the model card of Mubakkir from the measured results, write its limits and hand-over list, and choose and frame the capstone problem.

[D4_4_model_card.ipynb](https://sdaiaacademy.github.io/ds-204/labs/D4_4_model_card.ipynb)

- [ ] Collect the measured numbers (4 min)
- [ ] Fill the card: use, data, validation, results, subgroups (8 min)
- [ ] Write the limits and the human-decision statement (5 min)
- [ ] List what the deployment team needs (3 min)
- [ ] Choose the capstone problem and write its framing (5 min)
- [ ] ✱ Stretch: Exchange cards with a neighbour: one question each

#### [day4#wrap-up] Three things to keep

*Day 4*

- **The decision sets the threshold.** 800 plans: precision 0.65, recall 0.38 on the 2025 cohort. The same threshold listed 897. Write down the rule.
- **Check what the numbers mean.** The champion's probabilities hold by decile; the weighted model's do not; at new providers they run four points low.
- **An average hides people.** Miss rate 0.62 overall, 0.74 where registers are on paper. Intervals for every slice, a bias type and an action for every finding.

#### [day4#next] Day 5 · Capstone

*Next*

Your problem, your repository, the same path: frame, ladder, validation, audit, threshold, subgroups, card. Then eight minutes in which a change is named and you predict its effect.

- Start from your framing of Lab 4.4 and the README template
- Hold some rows back now; score them once

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
