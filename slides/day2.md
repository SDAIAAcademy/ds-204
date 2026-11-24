---
course: "Machine Learning for Tabular Data: Baselines and Validation"
code: "DS-204"
version: "2027.2.0"
deck: "day2"
day: 2
language: "en"
view: "trainee"
source: "https://sdaiaacademy.github.io/ds-204/slides/day2.html"
---

# Machine Learning for Tabular Data: Baselines and Validation

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
