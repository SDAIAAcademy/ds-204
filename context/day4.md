---
course: "Machine Learning for Tabular Data: Baselines and Validation"
code: "DS-204"
version: "2027.1.0"
deck: "day4"
day: 4
language: "en"
view: "trainee"
source: "https://sdaiaacademy.github.io/ds-204/slides/day4.html"
---

# Machine Learning for Tabular Data: Baselines and Validation

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
