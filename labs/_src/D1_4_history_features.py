# /// script
# requires-python = ">=3.11"
# dependencies = ["numpy>=2.0", "pandas>=2.2", "scipy>=1.14", "scikit-learn>=1.6", "lightgbm>=4.5", "matplotlib>=3.9", "tabulate>=0.9"]
# ///
# %% [markdown]
# # Lab 1.4 · Two history features
#
# Day 1 · Session 4 (Features from history) · 25 min.
#
# One trainee in three rows is not new to the fund. What happened in the earlier enrolments is the strongest
# information the table has not used yet, and the easiest to get wrong: a history feature must read the past only.
# You write two, and a test that fails when a feature reads the future.
#
# **AI policy: AI-tutor.**
#
# **Reference state.** `U.enrolments()` (the records) and `U.scorable()` (the rows to score, with
# `prediction_date`, 14 days after the start).
#
# **Tasks**
# - [ ] 1 · Follow one trainee through the records (3')
# - [ ] 2 · Write the two history features from past records only (9')
# - [ ] 3 · Run the time-travel test (5')
# - [ ] 4 · Rescore the logistic model with them; log the run (5')
# - [ ] 5 · Compare new and returning trainees (3')
# - [ ] ✱ Stretch · A third feature: days since the last exit

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
rows = U.scorable(raw)
times = raw.groupby("trainee_id").size().value_counts().sort_index()
print("enrolments per trainee:", times.to_dict())

# %% [markdown]
# ## 1 · Follow one trainee through the records (3')

# %%
busy = raw.groupby("trainee_id").size()
example = raw[raw.trainee_id == busy[busy == 4].index[3]].sort_values("start_date")
print(example[["enrolment_id", "programme", "start_date", "exit_date", "not_completed"]].to_string(index=False))

# %% [markdown]
# 🔍 **Look at** the dates. When the model scores an enrolment, 14 days after its start, which of the other rows
# have an `exit_date` before that day? Only those have a known outcome. An enrolment that **started** earlier and
# is still running tells nothing yet.
#
# ## 2 · Write the two history features from past records only (9')
#
# ✏️ **Exercise** · Write `history(rows, records)`. For every row of `rows` add two columns:
#
# - `prior_enrolments`: how many enrolments of the same trainee in `records` have an `exit_date` before the row's
#   `prediction_date`;
# - `prior_noncompletions`: how many of those ended without completion.
#
# Return `rows` with the two columns, same order, same index. A merge on `trainee_id`, a filter on the two dates
# and a group-by are enough.

# %%
def history(rows: pd.DataFrame, records: pd.DataFrame) -> pd.DataFrame:
    # BEGIN SOLUTION
    past = records[["trainee_id", "exit_date", U.TARGET]].rename(columns={"exit_date": "past_exit", U.TARGET: "past_y"})
    pairs = rows[["enrolment_id", "trainee_id", "prediction_date"]].merge(past, on="trainee_id", how="left")
    pairs = pairs[pairs.past_exit < pairs.prediction_date]          # the past only: the outcome was known
    counts = pairs.groupby("enrolment_id").past_y.agg(["size", "sum"])
    out = rows.copy()
    out["prior_enrolments"] = out.enrolment_id.map(counts["size"]).fillna(0).astype(int)
    out["prior_noncompletions"] = out.enrolment_id.map(counts["sum"]).fillna(0).astype(int)
    return out
    # END SOLUTION


# %%
try:
    history(rows.head(50), raw)
    mine = history
except NotImplementedError:
    print("↪ using the reference history features, U.add_history")
    mine = U.add_history
table = mine(rows, raw)
print(table.groupby("prior_enrolments").size().to_dict())
by_prior = table.groupby(table.prior_noncompletions.clip(upper=2))[U.TARGET].agg(["size", "mean"])
by_prior.index = ["none", "one", "two or more"]
U.keep("d1_4_by_prior", by_prior)
print(by_prior.round(3))
ref = U.add_history(rows, raw)
U.check((table[U.HISTORY].to_numpy() == ref[U.HISTORY].to_numpy()).all(), "your two columns equal the reference columns")

# %% [markdown]
# 🔍 **Look at** the non-completion rate by the number of earlier non-completions. One column moves the rate more
# than any rule of Lab 1.2.
#
# ## 3 · Run the time-travel test (5')
#
# The test: for three dates in the past, compute the features of the rows predicted before that date twice. Once
# from all records, once from the records that had ended by that date. A feature that reads the past only gives
# the same answer both times. Run it on your function, then on a tempting wrong version.

# %%
def tempting(rows: pd.DataFrame, records: pd.DataFrame) -> pd.DataFrame:
    """Counts the trainee's other enrolments that STARTED earlier, whatever their end. Looks right. Is not."""
    past = records[["trainee_id", "start_date", U.TARGET]].rename(columns={"start_date": "past_start", U.TARGET: "past_y"})
    pairs = rows[["enrolment_id", "trainee_id", "start_date"]].merge(past, on="trainee_id", how="left")
    pairs = pairs[pairs.past_start < pairs.start_date]
    counts = pairs.groupby("enrolment_id").past_y.agg(["size", "sum"])
    out = rows.copy()
    out["prior_enrolments"] = out.enrolment_id.map(counts["size"]).fillna(0).astype(int)
    out["prior_noncompletions"] = out.enrolment_id.map(counts["sum"]).fillna(0).astype(int)
    return out


passed = U.time_travel_test(mine, rows, raw)
U.check(passed, "your history features pass the time-travel test")
print("the tempting version:")
failed = not U.time_travel_test(tempting, rows, raw)
U.check(failed, "the version that counts by start date fails it")
running = (tempting(rows, raw).prior_enrolments != table.prior_enrolments).sum()
U.keep("d1_4_running", int(running))
print(f"{running:,} rows have an earlier enrolment that was still running at the prediction time")

# %% [markdown]
# 🔍 **Look at** what failed. The tempting version reads the outcome of an enrolment that had not ended yet. On
# old data that outcome is in the table. At the end of week 2 it does not exist.
#
# ## 4 · Rescore the logistic model with them; log the run (5')

# %%
dev, _ = U.dev_holdout(table)
folds = U.folds(dev)
r_base = U.score_folds(lambda: U.make_pipeline(U.logistic(), U.BASE), dev, folds, U.BASE)
r_hist = U.score_folds(lambda: U.make_pipeline(U.logistic()), dev, folds, U.FEATURES)
U.log_run("1.4", "logistic + history", r_hist, model="logistic regression, C=0.1", features=U.FEATURES)
U.keep_oof("logistic", dev, r_hist["oof"])
both = pd.DataFrame({"logistic": r_base["ap"], "logistic + history": r_hist["ap"]}, index=range(1, 6)).rename_axis("fold")
both["difference"] = both["logistic + history"] - both["logistic"]
U.keep("d1_4_both", both)
print(both.round(4))
print(f"mean gain {both.difference.mean():+.4f}; recall at capacity {r_base['per_fold'].recall_at_capacity.mean():.3f} "
      f"-> {r_hist['per_fold'].recall_at_capacity.mean():.3f}")
U.check((both.difference > 0).all(), "history helps on every fold")

# %% [markdown]
# ## 5 · Compare new and returning trainees (3')
#
# The gain is an average over two kinds of trainees. For one kind the new columns are always zero.

# %%
y = dev[U.TARGET].to_numpy()
seen = np.isfinite(r_hist["oof"])                       # rows that were in a validation fold
returning = (dev.prior_enrolments > 0).to_numpy()
parts = {}
for name, mask in [("new to the fund", seen & ~returning), ("returning", seen & returning)]:
    parts[name] = {"rows": int(mask.sum()), "base_rate": float(y[mask].mean()),
                   "ap_without_history": U.metrics(y[mask], r_base["oof"][mask])["ap"],
                   "ap_with_history": U.metrics(y[mask], r_hist["oof"][mask])["ap"]}
parts = pd.DataFrame(parts).T
U.keep("d1_4_parts", parts)
print(parts.round(3))
line = (f"Two history features from past records only lift the logistic rung from {r_base['mean']:.3f} to "
        f"{r_hist['mean']:.3f}. The gain comes from the {parts.loc['returning', 'rows']:,.0f} validation rows of "
        f"returning trainees; for a trainee new to the fund the columns are zero.")
print(line)
U.save("D1_4_history.md", "# Lab 1.4: history features\n\n" + U.md(both, 4) + "\n\n" + U.md(parts) + "\n\n" + line + "\n")

# %% [markdown]
# ## ✱ Stretch · A third feature: days since the last exit
#
# Add `days_since_last_exit` (a large number for a trainee with no earlier exit). Run the time-travel test on it
# before you look at its score.

# %% [markdown]
# ## Checkpoint
#
# 1. An earlier enrolment started in January and ended in September. The row to score starts in April. Does the
#    earlier enrolment count? <details><summary>Answer</summary>No. On the prediction day in April its outcome
#    was not known. It counts for rows predicted after its exit date.</details>
# 2. What does the time-travel test prove, and what does it not prove?
#    <details><summary>Answer</summary>That hiding later records does not change the features, so they read the
#    past only. It says nothing about how useful they are, and nothing about the other columns.</details>
# 3. For whom will Mubakkir be weaker, and where do you write that down?
#    <details><summary>Answer</summary>For trainees new to the fund, who have no history. It goes into the limits
#    of the model card, with the two scores of task 5.</details>
#
# **Carry forward.** `U.modelling_table()` returns the scorable rows with these two features. Every lab from here
# on starts from it.
