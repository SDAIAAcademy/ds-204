# /// script
# requires-python = ">=3.11"
# dependencies = ["numpy>=2.0", "pandas>=2.2", "scipy>=1.14", "scikit-learn>=1.6", "lightgbm>=4.5", "matplotlib>=3.9", "tabulate>=0.9"]
# ///
# %% [markdown]
# # Lab 3.3 · Four planted leaks
#
# Day 3 · Session 3 (Hunting planted leaks) · 25 min.
#
# Loay Al-Kadi sends a feature table he prepared from the fund's records, with a note: "random five-fold split,
# average precision close to 1". Four leaks are planted in it. You find them with three instruments, remove one
# at a time, and record the score before and after each removal.
#
# **AI policy: AI-off for the first attempt at tasks 2 to 4.** The hunt is yours. After that, AI-tutor.
#
# **Reference state.** `U.received_table()` (the table as received), `U.columns()` (the data dictionary of the
# fund's records) and the champion's settings (`U.champion_params()`).
#
# **Tasks**
# - [ ] 1 · Score the received table as it came (4')
# - [ ] 2 · Timestamps: when is each column written? (5')
# - [ ] 3 · Duplicates: the same enrolment under two keys (5')
# - [ ] 4 · Importance scores: question the top features (7')
# - [ ] 5 · Write the audit note: before and after each removal (4')
# - [ ] ✱ Stretch · Rebuild the provider rate from past cohorts only

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
from sklearn.model_selection import StratifiedKFold

received = U.received_table()
received = received[received.cohort_year <= 2024].reset_index(drop=True)      # the development cohorts
NEW = ["weeks_attended", "provider_rate", "trainee_noncompletions", "earlier_enrolments"]
print(f"{len(received):,} rows, {received.enrolment_id.nunique():,} enrolment keys")
print("columns beyond the 15 basic features:", NEW)
audit = []


def score(table: pd.DataFrame, features: list) -> float:
    """Loay's validation: a random five-fold split, here with the champion's settings."""
    split = list(StratifiedKFold(U.pick(5, 3), shuffle=True, random_state=U.SEED).split(table, table[U.TARGET]))
    make = lambda: U.make_pipeline(U.boosted(**U.champion_params()), features, scale=False)
    return U.split_scores(make, table, split, features)["mean"]


def record(step: str, before: float, after: float, finding: str) -> float:
    audit.append({"step": step, "before": before, "after": after, "change": after - before, "finding": finding})
    print(f"{step}: {before:.3f} -> {after:.3f}")
    return after


# %% [markdown]
# ## 1 · Score the received table as it came (4')

# %%
features = U.BASE + NEW
current = score(received, features)
print(f"as received: average precision {current:.3f}")
U.keep("d3_3_as_received", current)

# %% [markdown]
# 🔍 **Look at** the number. The honest ladder of Day 2 ended at 0.529. A score this far above everything you
# measured is not good news. It is the start of an audit.
#
# ## 2 · Timestamps: when is each column written? (5')
#
# First instrument: the data dictionary says when each column is written.
#
# ✏️ **Exercise** · Return the features of the received table that the dictionary marks as **not** known at the
# end of week 2 (column `known_at_end_of_week_2` of `U.columns()`).

# %%
def written_too_late(features: list, dictionary: pd.DataFrame) -> list:
    # BEGIN SOLUTION
    late = set(dictionary.column[dictionary.known_at_end_of_week_2 == "no"])
    return [f for f in features if f in late]
    # END SOLUTION


# %%
dictionary = U.columns()
try:
    late = written_too_late(features, dictionary)
except NotImplementedError:
    print("↪ using the reference answer")
    late = ["weeks_attended"]
print("written after the prediction time:", late)
print("not in the dictionary at all:", [f for f in features if f not in set(dictionary.column) and f != "start_month"])
U.check(late == ["weeks_attended"], "one feature is recorded at exit")
features = [f for f in features if f not in late]
current = record("1 remove weeks_attended", current, score(received, features), "target leakage: recorded at exit")

# %% [markdown]
# 🔍 **Look at** the second list. Three columns have no entry in the dictionary: somebody computed them. For a
# computed column the question "when is it written?" becomes "from which rows was it computed?". Keep them for
# task 4.
#
# ## 3 · Duplicates: the same enrolment under two keys (5')
#
# Second instrument: does the same real enrolment appear twice? A key can be new while the row is old.
#
# ✏️ **Exercise** · Return a boolean array: `True` for every row that shares trainee, provider, programme and
# start date with another row.

# %%
def twins(table: pd.DataFrame) -> np.ndarray:
    # BEGIN SOLUTION
    return table.duplicated(["trainee_id", "provider_id", "programme", "start_date"], keep=False).to_numpy()
    # END SOLUTION


# %%
try:
    twin = np.asarray(twins(received))
except NotImplementedError:
    print("↪ using the reference answer")
    twin = received.duplicated(["trainee_id", "provider_id", "programme", "start_date"], keep=False).to_numpy()
print(f"{twin.sum():,} rows are one of a pair; enrolment keys are unique: {received.enrolment_id.is_unique}")
pair = received[twin].sort_values(["trainee_id", "start_date"]).head(4)
print(pair[["enrolment_id", "trainee_id", "provider_id", "start_date", "absences_w2", U.TARGET]].to_string(index=False))
U.check(twin.sum() == 2 * (len(received) - len(received.drop_duplicates(["trainee_id", "provider_id", "programme", "start_date"]))),
        "every twin row has exactly one partner")

# how does the model do on the twins? one random split, the current features
split = list(StratifiedKFold(5, shuffle=True, random_state=U.SEED).split(received, received[U.TARGET]))
p, yr = np.zeros(len(received)), received[U.TARGET].to_numpy()
for tr, va in split:
    model = U.make_pipeline(U.boosted(**U.champion_params()), features, scale=False)
    p[va] = U.predict(model.fit(received.iloc[tr][features], yr[tr]), received.iloc[va][features])
on_twins = {"twin rows": U.ap(yr[twin], p[twin]), "all other rows": U.ap(yr[~twin], p[~twin])}
U.keep("d3_3_twins", {"rows": int(twin.sum()), **on_twins})
print("average precision:", {k: round(v, 3) for k, v in on_twins.items()})
deduplicated = received.drop_duplicates(["trainee_id", "provider_id", "programme", "start_date"]).reset_index(drop=True)
current = record("2 remove the copied rows", current, score(deduplicated, features), "duplicate contamination: about 3 rows in 100 are copies")
received = deduplicated

# %% [markdown]
# 🔍 **Look at** the size of the change. It is small, because about three rows in a hundred were copies, and because
# another leak is still drowning it. A small change is not a reason to keep a leak: the copies also count twice in every
# rate and every interval you compute later.
#
# ## 4 · Importance scores: question the top features (7')
#
# Third instrument: which features carry the score? A feature that carries far more than its meaning allows is
# the first suspect.

# %%
model = U.make_pipeline(U.boosted(**U.champion_params()), features, scale=False).fit(received[features], received[U.TARGET])
gain = pd.Series(model[-1].booster_.feature_importance("gain"), index=model[:-1].get_feature_names_out())
gain = (gain / gain.sum()).sort_values(ascending=False)
U.keep("d3_3_gain", gain.head(6))
print(gain.head(6).round(3).to_string())

# %% [markdown]
# One computed column carries most of the gain. Test all three computed columns the same way: build each again
# from the records and see which rows it must have read.

# %%
raw = U.enrolments()
honest = U.add_history(received, raw)                                     # past records only, as in Lab 1.4
KEY = ["trainee_id", "provider_id", "programme", "start_date"]
everything = U.received_table().drop_duplicates(KEY)                      # all cohorts, 2025 included
whole_life = received.trainee_id.map(everything.groupby("trainee_id")[U.TARGET].sum())
all_years = received.provider_id.map(everything.groupby("provider_id")[U.TARGET].mean().round(4))
started_before = received.groupby("trainee_id").start_date.rank(method="first") - 1
tests = pd.DataFrame({
    "equals": ["all enrolments of the trainee, this one and later ones included", "all rows of the provider, 2021 to 2025",
               "enrolments of the trainee that started earlier"],
    "match": [(received.trainee_noncompletions == whole_life).mean(), np.isclose(received.provider_rate, all_years).mean(),
              (received.earlier_enrolments == started_before).mean()],
    "equals the past-only version": [(received.trainee_noncompletions == honest.prior_noncompletions).mean(), np.nan, np.nan],
}, index=["trainee_noncompletions", "provider_rate", "earlier_enrolments"])
U.keep("d3_3_tests", tests)
print(tests.round(3).to_string())

# %% [markdown]
# ✏️ **Exercise** · Give your verdict on each computed column: `"leak"` or `"keep"`.

# %%
def verdicts() -> dict:
    # BEGIN SOLUTION
    return {"trainee_noncompletions": "leak", "provider_rate": "leak", "earlier_enrolments": "keep"}
    # END SOLUTION


# %%
reference_verdicts = {"trainee_noncompletions": "leak", "provider_rate": "leak", "earlier_enrolments": "keep"}
try:
    mine = verdicts()
except NotImplementedError:
    print("↪ using the reference verdicts")
    mine = reference_verdicts
U.check(mine == reference_verdicts, "two computed columns leak and one is honest")
features = [f for f in features if f != "trainee_noncompletions"]
current = record("3 remove trainee_noncompletions", current, score(received, features),
                 "future information: counts this enrolment and later ones")
features = [f for f in features if f != "provider_rate"]
current = record("4 remove provider_rate", current, score(received, features),
                 "preparation leakage: a rate fitted on all rows, validation and 2025 included")

# %% [markdown]
# 🔍 **Look at** `earlier_enrolments`. It counts enrolments that **started** earlier, without their outcome. On the
# prediction day the fund knows that a trainee enrolled before, so the column is honest, and it stays. An audit
# also clears columns.
#
# `provider_rate` moved the score very little: each provider has several hundred rows, so one row's own outcome
# hardly changes its rate. It is still a leak, and it read the 2025 cohort, which nobody was to open before
# Day 4. The size of a leak is measured; whether it is a leak is decided by how the column was made.
#
# ## 5 · Write the audit note: before and after each removal (4')

# %%
note = pd.DataFrame(audit).set_index("step")
fund = U._calendar(received)
on_fund_folds = U.split_scores(lambda: U.make_pipeline(U.boosted(**U.champion_params()), features, scale=False),
                               fund, U.folds(fund), features)["mean"]
U.keep("d3_3_audit", note)
U.keep("d3_3_fund_folds", on_fund_folds)
print(note.round(4).to_string())
line = (f"The received table scored {note.before.iloc[0]:.3f}. With four leaks removed it scores {current:.3f} on the "
        f"same random split and {on_fund_folds:.3f} on the fund's folds. The champion with past-only history: "
        f"{U.fold_scores('boosted, 20 trials').mean():.3f}.")
print(line)
U.save("D3_3_audit.md", "# Lab 3.3: audit of the received table\n\n" + U.md(note, 4) + "\n\n" + line + "\n")
if not U.QUICK:
    U.check(note.after.iloc[-1] < 0.6 < note.before.iloc[0], "the audit brings the score back to the honest range")

# %% [markdown]
# 🔍 **Look at** the order of the removals. Two leaks account for almost all of the false score, and each hid the
# ones behind it: while `weeks_attended` was in the table, nothing else seemed to matter. That is why an audit
# removes **one thing at a time** and writes the score down each time.
#
# ## ✱ Stretch · Rebuild the provider rate from past cohorts only
#
# Build an honest provider rate: for each row, the non-completion rate of its provider among the enrolments that
# had **ended** before the row's prediction date. Add it and score on the fund's folds. Does it help beyond
# `provider_id`?

# %% [markdown]
# ## Checkpoint
#
# 1. Which instrument found which leak?
#    <details><summary>Answer</summary>The dictionary's timestamps found `weeks_attended`. The search for repeated
#    rows found the copies. Importance pointed at `trainee_noncompletions`, and rebuilding the computed columns
#    showed which rows it and `provider_rate` had read.</details>
# 2. Removing `provider_rate` changed the score by about 0.001. Why remove it?
#    <details><summary>Answer</summary>It is fitted on rows that score the model, and on the holdout cohort. The
#    effect is small with hundreds of rows per provider and would be large with five. A leak is decided by how
#    the column was made, not by its measured effect.</details>
# 3. Why keep `earlier_enrolments`?
#    <details><summary>Answer</summary>It uses only what the fund knows on the prediction day: that the trainee
#    enrolled before. It reads no outcome.</details>
#
# **Carry forward.** `artifacts/D3_3_audit.md`: the audit note with the score before and after each removal. The
# capstone README needs one like it.
