# /// script
# requires-python = ">=3.11"
# dependencies = ["numpy>=2.0", "pandas>=2.2", "scipy>=1.14", "scikit-learn>=1.6", "lightgbm>=4.5", "matplotlib>=3.9", "tabulate>=0.9"]
# ///
# %% [markdown]
# # Lab 1.1 · Frame the non-completion question
#
# Day 1 · Session 1 (A prediction serves a decision) · 25 min.
#
# Ghassan Al-Fayez leads the advisers of Sullam Vocational Training Fund. His question: "Which trainees are
# likely not to complete, so that we can offer support early?" Before any model you write down what is predicted,
# for what, when, and what an adviser does with it. Then you find out which rows and which columns that leaves.
#
# **AI policy: AI-tutor.** An assistant may explain pandas syntax. The framing is yours.
#
# **Reference state.** `U.enrolments()`, the export of 45,000 enrolments, and `U.columns()`, its data dictionary.
#
# **Tasks**
# - [ ] 1 · Read the export and its data dictionary (4')
# - [ ] 2 · Write target, unit, prediction time and adviser action (6')
# - [ ] 3 · Keep the rows that can be scored (4')
# - [ ] 4 · Mark the columns known at the end of week 2 (6')
# - [ ] 5 · Price the two errors: the break-even probability (5')
# - [ ] ✱ Stretch · Frame the same question on the enrolment day: what is gained and what is lost

# %%
import os, sys, subprocess
from pathlib import Path

IN_COLAB = "google.colab" in sys.modules
if IN_COLAB:
    REPO_URL = "https://github.com/SDAIAAcademy/ds-204.git"
    if not Path("/content/ds-204").exists():
        subprocess.run(["git", "clone", "-q", REPO_URL, "/content/ds-204"], check=True)
    os.chdir("/content/ds-204/labs")
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "-r", "requirements-colab.txt"], check=True)
else:
    while not Path("sullam.py").exists() and Path.cwd() != Path.cwd().parent:
        os.chdir("..")  # solutions/ notebooks run from labs/
sys.path.insert(0, os.getcwd())
import numpy as np
import pandas as pd
import sullam as U

raw = U.enrolments()
dictionary = U.columns()
decision = U.decision()

# %% [markdown]
# ## 1 · Read the export and its data dictionary (4')

# %%
print(f"{len(raw):,} enrolments of {raw.trainee_id.nunique():,} trainees at {raw.provider_id.nunique()} providers, "
      f"cohorts {raw.cohort_year.min()} to {raw.cohort_year.max()}")
print(f"did not complete: {raw.not_completed.mean():.1%}")
print(raw.head(3).T.to_string())
print()
print(dictionary.to_string(index=False))

# %% [markdown]
# 🔍 **Look at** the column `recorded_at` of the dictionary. Four moments appear: registration, enrolment, the end
# of week 2 and exit. Hold on to that: it decides everything in task 4.
#
# ## 2 · Write target, unit, prediction time and adviser action (6')
#
# ✏️ **Exercise** · Choose one option for each of the four lines. The options are printed below.

# %%
OPTIONS = {
    "target": ["a risk score from 0 to 100", "not_completed: 1 when the trainee left without completing",
               "weeks_attended", "the provider's completion rate"],
    "unit": ["one trainee", "one provider", "one enrolment", "one cohort year"],
    "prediction_time": ["the enrolment day", "the end of week 2", "the last week of the programme", "the exit day"],
    "action": ["the trainee loses the place", "the provider is paid less",
               "an adviser contacts the trainee in week 3 and agrees a support plan",
               "the trainee is moved to a shorter programme"],
}
for line, options in OPTIONS.items():
    print(line + ":")
    for k, o in enumerate(options):
        print(f"   {k}  {o}")


def framing() -> dict:
    """The number of the option you choose for each line, for example {"target": 0, ...}."""
    # BEGIN SOLUTION
    return {"target": 1, "unit": 2, "prediction_time": 1, "action": 2}
    # END SOLUTION


# %%
KEY = {"target": 1, "unit": 2, "prediction_time": 1, "action": 2}
try:
    frame = framing()
except NotImplementedError:
    print("↪ using the reference framing")
    frame = dict(KEY)
for line in OPTIONS:
    print(f"{line:16s} {OPTIONS[line][frame[line]]}")
U.check(frame == KEY, "the framing matches the decision: one enrolment, at the end of week 2, for an adviser's list")

# %% [markdown]
# Why these. The **unit** is the enrolment: the same trainee can be at risk in one programme and safe in the next,
# and the adviser acts on one enrolment. The **prediction time** is the end of week 2: early enough for a support
# plan in week 3, late enough for the first two weeks of the register to exist. The **action** is support. A score
# that removed a place would be a different project, and not one this course builds.
#
# ## 3 · Keep the rows that can be scored (4')
#
# A trainee who has already left by the end of week 2 cannot be supported in week 3.
#
# ✏️ **Exercise** · Return the rows of `raw` that are still enrolled at the end of week 2.

# %%
def scorable_rows(raw: pd.DataFrame) -> pd.DataFrame:
    # BEGIN SOLUTION
    return raw[raw.weeks_attended > 2]
    # END SOLUTION


# %%
try:
    rows = scorable_rows(raw)
except NotImplementedError:
    print("↪ using the reference rows")
    rows = raw[raw.weeks_attended > 2]
early = raw[raw.weeks_attended <= 2]
population = {"exported": len(raw), "scorable": len(rows), "left_in_weeks_1_2": len(early),
              "rate_exported": float(raw.not_completed.mean()), "rate_scorable": float(rows.not_completed.mean()),
              "share_of_non_completers_gone_by_week_2": float(len(early) / raw.not_completed.sum())}
U.keep("d1_1_population", population)
print(f"scorable rows: {len(rows):,} of {len(raw):,}; {len(early):,} trainees had left by the end of week 2")
print(f"non-completion: {population['rate_exported']:.1%} of all enrolments, {population['rate_scorable']:.1%} of the scorable rows")
print(f"{population['share_of_non_completers_gone_by_week_2']:.0%} of all non-completers leave before the model can see them")
U.check(len(rows) == len(U.scorable(raw)), "the scorable rows are the ones still enrolled after week 2")
U.check(early.not_completed.all(), "everyone who left in weeks 1 and 2 is a non-completer the model cannot help")

# %% [markdown]
# 🔍 **Look at** the two rates. Every score in this course is about the scorable rows. The base rate of the whole
# export is the wrong floor, and the early leavers are a limit to write into the model card on Day 4.
#
# In live use, "still enrolled" is read from the register on that day. In the export it has to be read from
# `weeks_attended`, a column written at exit. It may choose the rows. It may never be a feature.
#
# ## 4 · Mark the columns known at the end of week 2 (6')
#
# ✏️ **Exercise** · From the dictionary, return the columns that may be features: known at the end of week 2,
# and neither a key nor the target. Then name the one known column you would leave out, and why.

# %%
KEYS = ["enrolment_id", "trainee_id"]


def candidate_features(dictionary: pd.DataFrame) -> list:
    # BEGIN SOLUTION
    known = dictionary[dictionary.known_at_end_of_week_2 == "yes"].column
    return [c for c in known if c not in KEYS]
    # END SOLUTION


LEFT_OUT = "gender"        # ✏️ the known column you leave out of the model (the reference answer is gender)
REASON = "Support is offered on need, not on gender; it is kept to check the errors by group on Day 4."

# %%
try:
    candidates = candidate_features(dictionary)
except NotImplementedError:
    candidates = U.reference("d1_1_candidates")
U.keep("d1_1_candidates", candidates)
after = dictionary[dictionary.known_at_end_of_week_2 == "no"].column.tolist()
print("candidates:", ", ".join(candidates))
print("recorded at exit, never a feature:", ", ".join(after))
U.check(not set(candidates) & set(after), "no candidate is recorded at exit")
U.check("absences_w2" in candidates, "the register of the first two weeks is a candidate")
U.check(set(candidates) - {LEFT_OUT, "start_date", "cohort_year"} >= set(U.BASE) - {"start_month"},
        "the course's feature list is inside your candidates")
tell = raw.groupby("not_completed").weeks_attended.mean()
print(f"mean weeks attended: completed {tell[0]:.1f}, not completed {tell[1]:.1f}")

# %% [markdown]
# 🔍 **Look at** the last line. `weeks_attended` separates the two classes almost by itself, because it is the
# outcome written in another unit. A model given it would look brilliant on old data and be useless at the end of
# week 2. Day 3 hunts columns like it.
#
# **Known is not the same as justified.** `gender` is known at registration. The course leaves it out of the model
# and keeps it for the error table of Lab 4.3. `trainee_id` joins a trainee's history in Lab 1.4 and is never a
# feature itself. Maram Al-Rasheed, the data protection officer, signs the list.
#
# ## 5 · Price the two errors: the break-even probability (5')
#
# A support plan costs the fund money. A lost place costs more. Support does not always work.
#
# ✏️ **Exercise** · Above which probability of non-completion does a support plan pay for itself?

# %%
print(json_text := "\n".join(f"{k}: {v}" for k, v in decision.items()))


def break_even(decision: dict) -> float:
    # BEGIN SOLUTION
    saved = decision["share_of_supported_non_completers_who_then_complete"] * decision["cost_of_one_lost_place_sar"]
    return decision["cost_of_one_support_plan_sar"] / saved
    # END SOLUTION


# %%
try:
    p_star = break_even(decision)
except NotImplementedError:
    p_star = U.reference("d1_1_break_even")
U.keep("d1_1_break_even", p_star)
capacity = decision["adviser_capacity_per_year"]
per_year = rows.groupby("cohort_year").size()
print(f"a support plan pays when p x 0.25 x 9,000 > 450, so when p > {p_star:.2f}")
print(f"capacity: {capacity} plans a year; scorable rows a year: {per_year.min():,} to {per_year.max():,}, "
      f"so about {capacity / per_year.mean():.0%} of trainees")
U.check(abs(p_star - 0.20) < 1e-9, "the break-even probability is 0.20")
sheet = ("# Framing sheet: Mubakkir\n\n" + "\n".join(f"- **{k}**: {OPTIONS[k][frame[k]]}" for k in OPTIONS)
         + f"\n- **population**: enrolments still active at the end of week 2 ({len(rows):,} of {len(raw):,})"
         + f"\n- **left out**: {LEFT_OUT}. {REASON}"
         + f"\n- **break-even probability**: {p_star:.2f}; **capacity**: {capacity} support plans a year\n")
U.save("D1_1_framing.md", sheet)

# %% [markdown]
# 🔍 **Look at** the two numbers together. The base rate of the scorable rows is below the break-even: supporting
# everyone would cost more than it saves, and the advisers could not do it anyway. The fund needs a ranking.
#
# ## ✱ Stretch · x
#
# Frame the same question on the enrolment day. Which candidate column disappears? How many more trainees could
# be reached? Write two lines under the framing sheet.

# %% [markdown]
# ## Checkpoint
#
# 1. Why is the unit one enrolment and not one trainee?
#    <details><summary>Answer</summary>The adviser acts on one enrolment in week 3, and the same trainee can be at
#    risk in one programme and not in another. A row per trainee would also mix outcomes of different years.</details>
# 2. `exit_reason` is blank for everyone who completed. Why is it worse than useless as a feature?
#    <details><summary>Answer</summary>It is written at exit, so at the end of week 2 it is blank for everyone.
#    On old data it would hand the model the answer; in use it would say nothing.</details>
# 3. The base rate is below the break-even probability. What does that tell the fund?
#    <details><summary>Answer</summary>Supporting everyone loses money. Support pays only for trainees whose own
#    probability is above 0.20, so the fund needs a score that separates them.</details>
#
# **Carry forward.** `artifacts/D1_1_framing.md`: target, unit, prediction time, action, population and
# break-even. The model card of Lab 4.4 opens with these lines.
