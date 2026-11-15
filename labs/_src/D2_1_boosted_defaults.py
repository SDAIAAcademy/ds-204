# /// script
# requires-python = ">=3.11"
# dependencies = ["numpy>=2.0", "pandas>=2.2", "scipy>=1.14", "scikit-learn>=1.6", "lightgbm>=4.5", "matplotlib>=3.9", "tabulate>=0.9"]
# ///
# %% [markdown]
# # Lab 2.1 · Boosted trees with defaults
#
# Day 2 · Session 1 (Gradient-boosted trees) · 25 min.
#
# The next rung is gradient-boosted trees with the settings the library ships with. You compare the score on the
# rows the model was trained on with the score on the rows it has not seen, follow both as trees are added, and
# set the rung beside the logistic model in the log.
#
# **AI policy: AI-tutor.**
#
# **Reference state.** `U.modelling_table()` with the two history features, the fund's folds and the reference
# run log of Day 1.
#
# **Tasks**
# - [ ] 1 · Fit with defaults on one fold: training against validation (5')
# - [ ] 2 · Score the five folds and log the run (5')
# - [ ] 3 · Follow both scores as trees are added (7')
# - [ ] 4 · Set the rung beside the logistic model (4')
# - [ ] 5 · Read two kinds of importance (4')
# - [ ] ✱ Stretch · Stop adding trees early, without touching the validation fold

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
from sklearn.inspection import permutation_importance

table = U.modelling_table()
dev, _ = U.dev_holdout(table)
folds = U.folds(dev)
X, y = dev[U.FEATURES], dev[U.TARGET].to_numpy()
make = lambda **settings: U.make_pipeline(U.boosted(**settings), scale=False)     # trees need no scaling
print("LightGBM defaults: 100 trees, 31 leaves each, learning rate 0.1, at least 20 rows per leaf")

# %% [markdown]
# ## 1 · Fit with defaults on one fold: training against validation (5')
#
# ✏️ **Exercise** · Fit `make()` on the training rows of fold 5 and return two average precisions: on those same
# training rows, and on the validation rows. `U.predict(model, rows)` gives probabilities, `U.ap(y, p)` the score.

# %%
tr, va = folds[-1]


def train_and_validation(model) -> tuple:
    # BEGIN SOLUTION
    model.fit(X.iloc[tr], y[tr])
    return U.ap(y[tr], U.predict(model, X.iloc[tr])), U.ap(y[va], U.predict(model, X.iloc[va]))
    # END SOLUTION


# %%
one = make()
try:
    ap_train, ap_valid = train_and_validation(one)
except NotImplementedError:
    print("↪ using the reference scores of fold 5")
    one.fit(X.iloc[tr], y[tr])
    ap_train, ap_valid = U.ap(y[tr], U.predict(one, X.iloc[tr])), U.ap(y[va], U.predict(one, X.iloc[va]))
U.keep("d2_1_fold5", {"train": ap_train, "valid": ap_valid})
print(f"fold 5: average precision on the training rows {ap_train:.3f}, on the validation rows {ap_valid:.3f}")
U.check(ap_train > ap_valid + 0.05, "the training score is well above the validation score")

# %% [markdown]
# 🔍 **Look at** the gap. The training score says how much the model can memorise. It says nothing about how the
# model will do on next half-year's trainees. Only the validation score goes into the log.
#
# ## 2 · Score the five folds and log the run (5')

# %%
r_default = U.score_folds(make, dev, folds, train_scores=True)
print(r_default["per_fold"][["n_train", "train_ap", "ap", "recall_at_capacity"]].round(3))
U.log_run("2.1", "boosted, defaults", r_default, model="LightGBM, library defaults", features=U.FEATURES)
U.keep_oof("boosted_defaults", dev, r_default["oof"])

# %% [markdown]
# 🔍 **Look at** `train_ap` down the folds. The more training rows, the less the model can memorise each one, and
# the smaller the gap.
#
# ## 3 · Follow both scores as trees are added (7')
#
# Boosting adds one small tree at a time, each correcting the errors of the sum so far. A model fitted with 400
# trees can predict with only its first `k`: `model[-1].predict_proba(prepared, num_iteration=k)`.
#
# ✏️ **Exercise** · Complete `curve`: for each `k` in `steps`, the training and the validation average precision
# of the first `k` trees.

# %%
steps = [10, 25, 50, 100, 200, 400]
long = make(n_estimators=400).fit(X.iloc[tr], y[tr])
prepared_tr, prepared_va = long[:-1].transform(X.iloc[tr]), long[:-1].transform(X.iloc[va])


def curve(steps) -> pd.DataFrame:
    rows = []
    for k in steps:
        # BEGIN SOLUTION
        p_tr = long[-1].predict_proba(prepared_tr, num_iteration=k)[:, 1]
        p_va = long[-1].predict_proba(prepared_va, num_iteration=k)[:, 1]
        rows.append({"trees": k, "train": U.ap(y[tr], p_tr), "validation": U.ap(y[va], p_va)})
        # END SOLUTION
    return pd.DataFrame(rows).set_index("trees")


# %%
try:
    learning = curve(steps)
except NotImplementedError:
    learning = U.reference("d2_1_curve")
U.keep("d2_1_curve", learning)
print(learning.round(3))
best_k = int(learning.validation.idxmax())
U.check(learning.train.is_monotonic_increasing, "the training score rises with every batch of trees")
U.check(best_k < 400, f"the validation score peaks earlier, at {best_k} trees, and then falls")
plt = U.style()
fig, ax = plt.subplots()
learning.plot(ax=ax, marker="o", logx=True)
ax.set(xlabel="trees (log scale)", ylabel="average precision", title="Fold 5: more trees, two different stories")
U.savefig(fig, "D2_1_trees.png")

# %% [markdown]
# 🔍 **Look at** where the two lines part. Past the peak, every further tree fits the training rows better and
# the validation rows worse. The default of 100 trees is one point on this curve, and nobody chose it for this
# table.
#
# ## 4 · Set the rung beside the logistic model (4')

# %%
logit = U.fold_scores("logistic + history")
beside = pd.DataFrame({"logistic + history": logit, "boosted, defaults": r_default["ap"]}, index=range(1, 6)).rename_axis("fold")
beside["difference"] = beside["boosted, defaults"] - beside["logistic + history"]
U.keep("d2_1_beside", beside)
print(beside.round(4))
print(f"mean difference {beside.difference.mean():+.4f}; boosted trees win on {(beside.difference > 0).sum()} of 5 folds; "
      f"seconds: {r_default['seconds']:.0f}")

# %% [markdown]
# 🔍 **Look at** the sign of the difference. A more flexible model is not automatically a rung up. With default
# settings the trees overfit the smaller folds, and a model that only adds effects keeps its lead.
#
# ## 5 · Read two kinds of importance (4')
#
# Two ways to ask "what does the model use?". **Gain**: how much each input reduced the training loss, summed
# over the trees. **Permutation**: how much the validation score falls when one original column is shuffled.

# %%
names = [n.split("__")[-1] for n in one[:-1].get_feature_names_out()]
gain = pd.Series(one[-1].booster_.feature_importance(importance_type="gain"), index=names)
column_of = lambda n: next((c for c in U.LOW + U.HIGH if n.startswith(c + "_")), n.replace("missingindicator_", ""))
gain_by_column = (gain.groupby(column_of).sum() / gain.sum()).sort_values(ascending=False)
perm = permutation_importance(one, X.iloc[va], y[va], scoring="average_precision", n_repeats=U.pick(5, 2),
                              random_state=U.SEED, n_jobs=1)
permutation = pd.Series(perm.importances_mean, index=U.FEATURES).sort_values(ascending=False)
importance = pd.DataFrame({"gain_share": gain_by_column, "permutation_drop": permutation}).sort_values(
    "permutation_drop", ascending=False)
U.keep("d2_1_importance", importance)
print(importance.round(3))
U.check(importance.index[0] == "absences_w2", "shuffling the register of the first two weeks costs the most")
line = (f"Boosted trees with defaults: {r_default['mean']:.3f} (spread {r_default['spread']:.3f}), against "
        f"{logit.mean():.3f} for the logistic rung; training score on fold 5 {ap_train:.3f}, validation {ap_valid:.3f}.")
print(line)
U.save("D2_1_boosted.md", "# Lab 2.1: boosted trees with defaults\n\n" + line + "\n\n" + U.md(beside, 4) + "\n\n"
       + U.md(learning) + "\n\n" + U.md(importance) + "\n")

# %% [markdown]
# 🔍 **Look at** `provider_id` in both columns. Its 60 values take a large share of the gain, because many small
# splits add up, and shuffling it costs the validation score much less than shuffling the absences. Three limits
# to remember: gain is measured on training rows; two columns that carry the same information share their
# importance; and neither number says in which direction a column acts, or that changing it would change the
# outcome.
#
# ## ✱ Stretch · Stop adding trees early, without touching the validation fold
#
# Choose the number of trees without touching the validation rows of fold 5: hold out the last half-year of the
# **training** rows, find the peak there, refit on all training rows with that number, and score fold 5 once.

# %% [markdown]
# ## Checkpoint
#
# 1. The training average precision is 0.75 and the validation 0.51. Which number do you report?
#    <details><summary>Answer</summary>The validation score. The training score measures memory. The gap between
#    them is worth a line in the notes: it says the model is more flexible than the data can support.</details>
# 2. Boosted trees with defaults score below the logistic rung. Is that a reason to drop trees?
#    <details><summary>Answer</summary>No. Defaults are one setting among many and were not chosen for this
#    table. It is a reason to tune under a budget, which is Lab 2.2, and to keep the logistic rung in view.</details>
# 3. `provider_id` has the second-largest gain. Can you tell Hamza Al-Babtain which providers cause trainees to
#    leave? <details><summary>Answer</summary>No. Importance says the model uses the column, not why. A provider
#    may enrol trainees with harder circumstances. That needs a different study.</details>
#
# **Carry forward.** The rung "boosted, defaults" in the log. Lab 2.2 tunes it.
