# /// script
# requires-python = ">=3.11"
# dependencies = ["numpy>=2.0", "pandas>=2.2", "scipy>=1.14", "scikit-learn>=1.6", "lightgbm>=4.5", "matplotlib>=3.9", "tabulate>=0.9"]
# ///
# %% [markdown]
# # Lab 1.2 · A constant and three rules
#
# Day 1 · Session 2 (The baseline ladder) · 25 min.
#
# The first two rungs of the ladder need no learning: a constant predictor, and three rules an adviser could
# write on a card. You score both on the fund's five folds and open the run log that every later model joins.
#
# **AI policy: AI-tutor.**
#
# **Reference state.** `U.modelling_table(history=False)`, the scorable rows; `U.folds(dev)`, the fund's folds.
#
# **Tasks**
# - [ ] 1 · Load the folds; read the base rate of each (4')
# - [ ] 2 · Score the constant predictor (5')
# - [ ] 3 · Write the three-rule baseline and score it (8')
# - [ ] 4 · Log both runs (4')
# - [ ] 5 · Read the log: the number to beat (4')
# - [ ] ✱ Stretch · A fourth rule from the provider's past non-completion rate

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

table = U.modelling_table(history=False)
dev, holdout = U.dev_holdout(table)        # cohort 2025 stays closed until Day 4
folds = U.folds(dev)

# %% [markdown]
# ## 1 · Load the folds; read the base rate of each (4')
#
# Each fold validates on one half-year and trains on the rows whose outcome was known before that half-year began.
# Day 3 explains why the folds are built this way. Today they are simply the same for every model.

# %%
y = dev[U.TARGET].to_numpy()
fold_table = pd.DataFrame([{
    "fold": k, "validates_on": U.block_name(int(dev.block.iloc[va[0]])), "train_rows": len(tr), "valid_rows": len(va),
    "base_rate": y[va].mean()} for k, (tr, va) in enumerate(folds, 1)]).set_index("fold")
U.keep("d1_2_folds", fold_table)
print(f"development rows {len(dev):,} (cohorts 2021 to 2024), holdout rows {len(holdout):,} (cohort 2025, closed)")
print(fold_table.round(3))

# %% [markdown]
# 🔍 **Look at** the base rates. They are not equal: a score on one fold is not comparable with a score on another.
# Models are compared fold by fold, on the same rows.
#
# ## 2 · Score the constant predictor (5')
#
# The constant predictor gives every trainee the same score: the non-completion rate of the training rows.
#
# ✏️ **Exercise** · Write it. It receives the training rows and the validation rows of one fold and returns one
# score per validation row.

# %%
def constant(train: pd.DataFrame, valid: pd.DataFrame) -> np.ndarray:
    # BEGIN SOLUTION
    return np.full(len(valid), train[U.TARGET].mean())
    # END SOLUTION


# %%
try:
    r_constant = U.score_fixed(constant, dev, folds)
except NotImplementedError:
    print("↪ using the reference constant predictor")
    r_constant = U.score_fixed(lambda tr, va: np.full(len(va), tr[U.TARGET].mean()), dev, folds)
print(r_constant["per_fold"][["base_rate", "ap", "auc", "recall_at_capacity"]].round(3))
U.check(np.allclose(r_constant["per_fold"].ap, r_constant["per_fold"].base_rate),
        "the average precision of a constant score is the base rate of the fold")

# %% [markdown]
# **Average precision** summarises a ranking for a rare outcome: the precision averaged over every position at
# which a non-completer is found. Its floor is the base rate, not 0.5. **Recall at capacity** is the share of
# non-completers among the 8 trainees in 100 that the advisers can support: with no information it is about 8%.
#
# ## 3 · Write the three-rule baseline and score it (8')
#
# ✏️ **Exercise** · Write three rules from the columns in `dev`, each true or false per trainee. The score is the
# number of rules a trainee meets, from 0 to 3. Use what an adviser would say: absence in the first two weeks,
# age, the entry test, distance, work beside the programme.

# %%
def three_rules(train: pd.DataFrame, valid: pd.DataFrame) -> np.ndarray:
    # BEGIN SOLUTION
    absent = valid.absences_w2 >= 2
    young = valid.age <= 19
    low_test = valid.aptitude_score < 38
    return absent.astype(int) + young.astype(int) + low_test.astype(int)
    # END SOLUTION


# %%
def reference_rules(train, valid):
    return (valid.absences_w2 >= 2).astype(int) + (valid.age <= 19).astype(int) + (valid.aptitude_score < 38).astype(int)


try:
    r_rules = U.score_fixed(three_rules, dev, folds)
except NotImplementedError:
    print("↪ using the reference rules: two or more absences, 19 or younger, entry test below 38")
    r_rules = U.score_fixed(reference_rules, dev, folds)
print(r_rules["per_fold"][["base_rate", "ap", "auc", "recall_at_capacity"]].round(3))
U.check((r_rules["per_fold"].ap > r_constant["per_fold"].ap).all(), "the rules beat the constant on every fold")
by_score = dev.assign(score=reference_rules(dev, dev)).groupby("score")[U.TARGET].agg(["size", "mean"])
U.keep("d1_2_by_score", by_score)
print(by_score.round(3))

# %% [markdown]
# 🔍 **Look at** the last table (the reference rules on all development rows): the non-completion rate by the
# number of rules met. A card with three lines already sorts trainees.
#
# ## 4 · Log both runs (4')
#
# One line per run: which rung, the mean score over the folds, the spread between folds, the time it took. A line
# is never edited afterwards.

# %%
U.log_run("1.2", "constant", r_constant, model="training base rate for everyone", features=[])
U.log_run("1.2", "three rules", r_rules, model="count of three rules", features=["absences_w2", "age", "aptitude_score"])
print(U.my_log()[["lab", "rung", "mean_ap", "spread", "min_fold", "max_fold", "recall_at_capacity"]].to_string(index=False))

# %% [markdown]
# ## 5 · Read the log: the number to beat (4')

# %%
summary = pd.DataFrame({"constant": r_constant["per_fold"].ap, "three rules": r_rules["per_fold"].ap})
summary.loc["mean"] = summary.mean()
U.keep("d1_2_summary", summary)
print(summary.round(3))
gain = r_rules["mean"] - r_constant["mean"]
line = (f"With no learning, three rules reach an average precision of {r_rules['mean']:.3f} "
        f"(spread {r_rules['spread']:.3f} over five folds) against {r_constant['mean']:.3f} for a constant, and put "
        f"{r_rules['per_fold'].recall_at_capacity.mean():.0%} of non-completers on the advisers' list. "
        f"Every model from here on has to beat {r_rules['mean']:.3f} by more than the folds disagree.")
print(line)
U.save("D1_2_baselines.md", "# Lab 1.2: the first two rungs\n\n" + U.md(summary) + "\n\n" + line + "\n")
U.check(gain > 2 * r_rules["spread"], "the gain of the rules over the constant is larger than twice the fold spread")

# %% [markdown]
# ## ✱ Stretch · x
#
# Add a rule from the provider: the provider's non-completion rate in the **training rows** is in the top quarter.
# Compute the rate inside the function, from `train` only. Does a fourth rule earn its line in the log?

# %% [markdown]
# ## Checkpoint
#
# 1. A colleague reports an average precision of 0.30. Is that good?
#    <details><summary>Answer</summary>Not known until it stands beside the log: the constant scores about 0.12
#    and three rules about 0.29 on these folds. And only if it was scored on the same folds.</details>
# 2. Why does the constant predictor take its rate from the training rows and not from the validation rows?
#    <details><summary>Answer</summary>At prediction time the outcomes of the validation rows do not exist. A
#    baseline obeys the same rule as a model.</details>
# 3. The fold spread of the rules is about 0.015. Model B beats model A by 0.004 on the mean. What do you say?
#    <details><summary>Answer</summary>Nothing yet. A difference smaller than the folds disagree needs the paired
#    comparison of Lab 2.4 before anyone calls it a gain.</details>
#
# **Carry forward.** `artifacts/run_log.csv` holds your two lines. Later labs start from the reference log
# (`U.reference_log()`), so nothing depends on this file.
