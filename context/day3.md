---
course: "Machine Learning for Tabular Data: Baselines and Validation"
code: "DS-204"
version: "2027.1.0"
deck: "day3"
day: 3
language: "en"
view: "trainee"
source: "https://sdaiaacademy.github.io/ds-204/slides/day3.html"
---

# Machine Learning for Tabular Data: Baselines and Validation

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
