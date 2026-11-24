# Trainer's key

The solution notebooks in this folder hold the reference code of every exercise. This page holds what is not
in a notebook: the numbers to expect, the planted features of the data, and the key of the capstone cases.
All numbers are from the full run with the fixed seed (`DS204_QUICK` not set).

## The run log (average precision, the fund's five folds)

| Lab | Rung | Mean | Fold spread |
|---|---|---|---|
| 1.2 | constant | 0.118 | 0.010 |
| 1.2 | three rules | 0.310 | 0.013 |
| 1.3 | logistic | 0.501 | 0.028 |
| 1.4 | logistic + history | 0.524 | 0.020 |
| 2.1 | boosted, defaults | 0.503 | 0.019 |
| 2.2 | boosted, 20 trials (nested) | 0.529 | 0.012 |
| 2.3 | challenger (AutoGluon Tabular, 28 s a fold) | 0.522 | 0.017 |

Decisions: challenger minus champion −0.007 (95% interval −0.018 to +0.004), minimum effect +0.01: keep the
champion. Champion minus logistic +0.005 (−0.010 to +0.019): not shown. The run times and the challenger's
scores vary a little between machines; the decisions do not.

## What is planted in the data (`prepare_data.py`)

| Planted | Where it shows |
|---|---|
| Columns recorded at exit: `weeks_attended`, `exit_date`, `exit_reason` | Lab 1.1 (the data dictionary), Lab 3.3 |
| Trainees who repeat (35% of all trainees have more than one row; 28% within the development rows) | Labs 1.4 and 3.1 |
| Outcomes that take 12 to 36 weeks to exist | The fold rule "due to end" (Lab 3.1, slide "due-to-end") |
| A fund that changes: provider quality drifts, an employer mandate, a stipend rule, blended delivery | The ageing table of Lab 3.1; the new providers of Lab 4.2 |
| Eight providers that open in 2025 | Lab 4.2, task 4 |
| Paper registers in Jazan and Najran that under-record early absences | Lab 4.3 |
| Under-representation of trainees of 35 and over, of some regions and of women | Lab 4.3, task 1 |

## Lab 1.3, stretch: segments with k-means

Four columns of the 32,439 development rows (`age`, `aptitude_score`, `distance_km`, `absences_w2`), four
segments. Share of the total variance: aptitude 0.557, distance 0.380, age 0.062, absence 0.002. Separation of
the segment means in standard deviations, unscaled against scaled: age 0.05 and 1.81, aptitude 2.68 and 1.19,
distance 2.84 and 2.75, absence 0.59 and 2.16. Unscaled, the segments are cut by aptitude and distance (non-
completion 0.04 to 0.23); scaled, a segment of 4,079 frequent absentees (1.95 days absent) appears, with non-
completion 0.40, and a segment of older trainees (mean age 31.5). The rates are read after the segments are
made. The worked example of the slides and the practice question come from `_tools/segments_example.py`.

## Lab 3.1

Random 0.521, grouped 0.520, grouped with history hidden 0.500, time-ordered 0.525. The random split does **not**
overstate for this model; say so plainly. What the split costs here: 0.020 for trainees with no history, and
0.016 for a model one year old. Stretch: a model with leaves of one row scores 0.463 under the random split
and 0.462 under the grouped one. No gap opens, because seven trainees in ten have a single row.

## Lab 3.2

| Case | Kind | Scheme |
|---|---|---|
| 1 Microfinance | target | time |
| 2 Pumps | group | grouped |
| 3 Dates | preparation | random |
| 4 Appointments | future | time |
| 5 Claims | group (duplicates) | grouped |
| 6 Infection | target | grouped+time |

Accept a reasoned second kind: case 6 is also future information (the column covers the whole stay); case 5
can be argued as time-ordered after the copies are removed; case 6 as grouped alone, if the participant argues that the
new hospitals' patients are the only change that matters.

## Lab 3.3: the four planted leaks of `data/received_table.csv.gz`

| Leak | Kind | Instrument | Score before | Score after |
|---|---|---|---|---|
| `weeks_attended` | target | the data dictionary | 1.000 | 0.882 |
| 1,052 enrolments entered twice (1,350 over all cohorts) | duplicate | repeated rows | 0.882 | 0.881 |
| `trainee_noncompletions` | future information | importance, then rebuilding | 0.881 | 0.503 |
| `provider_rate` | preparation | rebuilding | 0.503 | 0.501 |

`earlier_enrolments` is honest and stays.

## Lab 3.4: Loay's notebook

| Block | Fault | Repaired alone: accuracy |
|---|---|---|
| 1 rows | early leavers and the 2025 holdout are in the table | 0.978 |
| 2 balance | non-completers copied before the split | 0.865 |
| 3 encode | provider rate over all rows | 0.950 |
| 4 split | random | 0.743 |
| 5 tune | best of six settings on the test set | 0.948 |
| 6 report | accuracy on a test set balanced by copying | |

As written 0.948. All five repaired: accuracy 0.890 against 0.874 for a constant answer; average precision 0.494.

## Day 4

- Lab 4.1: threshold 0.413 for 800 plans on the 2024 cohort (precision 0.595, recall 0.400). On 2025 the same
  threshold lists 897; the top 800 have precision 0.645 and recall 0.377.
- Lab 4.2: Brier on 2025: champion 0.0852 as delivered, 0.0853 with a sigmoid; weighted model 0.1717 as
  delivered, 0.0854 with a sigmoid. New providers: expected 0.163, observed 0.204.
- Lab 4.3: miss rate 0.623. Supported by the rule: Jazan (above) and Riyadh (below). Paper registers: 0.737
  (0.658 to 0.804) against 0.610. Bias types: A measurement, B coverage, C sample size, D aggregation.

## The capstone cases (`capstone/make_datasets.py`)

Each synthetic case has the same structure under different names.

| | Bayader | Awtad | Nasaim |
|---|---|---|---|
| Recorded after the outcome (target leakage) | `collection_calls` | `late_fee_sar` | `closure_code` |
| Counts every record of the unit, this one and later ones (future information) | `borrower_missed_count` | `customer_late_count` | `patient_discontinued_count` |
| A rate over all rows of the site (preparation leakage) | `branch_missed_rate` | `depot_late_rate` | `team_discontinued_rate` |
| Honest computed column | `earlier_records` | `earlier_records` | `earlier_records` |
| Rows entered twice under new keys | none | 600 | none |
| A count under-recorded in two regions (Jazan, Tabuk) | `late_utility_bills` | `open_contracts` | `missed_first_visits` |
| An effect that reverses in 2025 | `loan_purpose` | `equipment_class` | `referral_source` |

With honest columns and a forward split, average precision is about 0.32 to 0.37 on 2025, at outcome rates of
11% to 13%. A repository that reports 0.8 or more has kept a computed column; 0.99 has kept the first row of
the table above. Units repeat (up to about 30 records), so a history built from past records is the strongest
honest addition.

Changes worth asking for in the defence: a random split in place of the forward one (small effect unless a
computed column is still in); the unit kept on one side (shows what history is worth); the strongest honest
feature removed; the under-recorded count examined by region.
