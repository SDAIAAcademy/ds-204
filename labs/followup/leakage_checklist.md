# Leakage checklist

For a model you did not build, or built before this course. Work through it in order: each part can spoil
everything after it. Write the answer and how you know, not "yes" or "no".

## Before any code: the reported score

- [ ] What is the number, on how many rows, from which period, with what share of the outcome?
- [ ] What does a constant answer score on the same rows? What did the previous model or rule score?
- [ ] How many versions were tried, and on which rows was this one chosen?

## 1 · Rows

- [ ] Is every row one the model would really score, at the time it would score it?
- [ ] Are rows that should stay closed (a final test period) outside everything that was fitted or chosen?
- [ ] Can the same real-world event appear more than once? Test the columns that describe the event, not the key.

## 2 · Split

- [ ] How will the model be used: on more rows like these, on units it has never seen, on later records?
- [ ] Does the split rehearse that use? What does the training part know on the day a validation row is scored:
      which share starts later, and which share had no outcome yet?
- [ ] For an outcome that takes time to settle: are training rows chosen by when the outcome was due, and not
      by whether it had already happened?
- [ ] Were the other splits measured once, and the gaps written down with a cause?

## 3 · Columns

- [ ] For every recorded column: when is it written, and by whom? Would a person who already knew the outcome
      be the one filling it in?
- [ ] For every computed column: from which rows was it computed? Rebuild it with later records hidden and
      compare.
- [ ] Which feature carries most of the score? Does its meaning allow that?

## 4 · Preparation and tuning

- [ ] Was anything fitted or chosen before the split: scaling, imputing, encoding by outcome, selecting
      features, balancing classes?
- [ ] Is everything that learns from data inside one pipeline, fitted on training rows only?
- [ ] Were settings chosen on rows other than the ones that report the score?

## 5 · Metric

- [ ] Does the metric fit the decision and the rarity of the outcome?
- [ ] If a threshold is used: where does it come from?
- [ ] Is a difference between two models reported with an interval, on the same rows?

## The audit

Remove **one thing at a time** and write the score before and after.

| Removal | Kind of leak | Score before | Score after |
|---|---|---|---|
| | | | |

Kinds: target leakage · future information · group or duplicate contamination · preparation leakage.

The audit ends when the score is in a range that honest baselines explain, and every remaining feature has an
answer to "when is it written" or "from which rows was it computed".
