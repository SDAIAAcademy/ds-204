# /// script
# requires-python = ">=3.11"
# dependencies = ["numpy>=2.0", "pandas>=2.2", "scipy>=1.14", "scikit-learn>=1.6", "lightgbm>=4.5", "matplotlib>=3.9", "tabulate>=0.9"]
# ///
# %% [markdown]
# # Lab 3.1 · One model, three splits
#
# Day 3 · Session 1 (The split decides the score) · 25 min.
#
# Every score of Days 1 and 2 came from the fund's folds. Today you find out what the other two common splits
# would have said about the same model, and why. One model, the champion, and three ways to cut the same
# 32,439 development rows.
#
# **AI policy: AI-off for the first attempt at tasks 1 and 2.** Write the two splits and read the first table
# yourself. After that, AI-tutor.
#
# **Reference state.** `U.modelling_table()`, the fund's folds (`U.folds`), the champion with the settings of
# Lab 2.2 (`U.champion()`) and its reference validation-fold probabilities (`U.oof()`).
#
# **Tasks**
# - [ ] 1 · Build the three splits on the same rows (5')
# - [ ] 2 · Score the same model under each (7')
# - [ ] 3 · Explain the gap: new and returning trainees (4')
# - [ ] 4 · What only time order shows: let the model age (5')
# - [ ] 5 · Write the gap table with one sentence per gap (4')
# - [ ] ✱ Stretch · A model that memorises: does the grouped gap grow?

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
from sklearn.model_selection import GroupKFold, StratifiedKFold

table = U.modelling_table()
dev, _ = U.dev_holdout(table)
y = dev[U.TARGET].to_numpy()
per_trainee = dev.groupby("trainee_id").size()
print(f"{len(dev):,} rows of {dev.trainee_id.nunique():,} trainees; {(per_trainee > 1).mean():.0%} of the trainees have more than one row")

# %% [markdown]
# ## 1 · Build the three splits on the same rows (5')
#
# A split is a list of five pairs `(training positions, validation positions)`.
#
# ✏️ **Exercise** · Write the two splits the fund does **not** use.
#
# - `random_split`: five folds at random, with the same share of non-completers in each
#   (`StratifiedKFold(5, shuffle=True, random_state=U.SEED)`).
# - `grouped_split`: five folds in which all rows of one trainee stay on the same side (`GroupKFold(5)`, with
#   `groups=dev.trainee_id`).

# %%
def random_split(dev: pd.DataFrame) -> list:
    # BEGIN SOLUTION
    return list(StratifiedKFold(5, shuffle=True, random_state=U.SEED).split(dev, dev[U.TARGET]))
    # END SOLUTION


def grouped_split(dev: pd.DataFrame) -> list:
    # BEGIN SOLUTION
    return list(GroupKFold(5).split(dev, dev[U.TARGET], groups=dev.trainee_id))
    # END SOLUTION


# %%
try:
    splits = {"random": random_split(dev), "grouped": grouped_split(dev)}
except NotImplementedError:
    print("↪ using the reference splits")
    splits = {"random": list(StratifiedKFold(5, shuffle=True, random_state=U.SEED).split(dev, y)),
              "grouped": list(GroupKFold(5).split(dev, y, groups=dev.trainee_id))}
splits["time-ordered"] = U.folds(dev)


def describe(split) -> dict:
    """What the training part knows that it could not know on the day a validation row is scored."""
    seen, later, running = [], [], []
    for tr, va in split:
        day = dev.start_date.iloc[va].to_numpy()                       # the start of each validation row
        starts = np.sort(dev.start_date.iloc[tr].to_numpy())
        ends = np.sort(dev.planned_end.iloc[tr].to_numpy())
        seen.append(dev.trainee_id.iloc[va].isin(set(dev.trainee_id.iloc[tr])).mean())
        later.append(1 - np.searchsorted(starts, day, side="right").mean() / len(tr))
        running.append(1 - np.searchsorted(ends, day, side="left").mean() / len(tr))
    return {"training rows per fold": int(np.mean([len(tr) for tr, _ in split])),
            "validation trainees also in training": float(np.mean(seen)),
            "training rows that start later": float(np.mean(later)),
            "training rows not yet due to end": float(np.mean(running))}


structure = pd.DataFrame({name: describe(s) for name, s in splits.items()}).T
U.keep("d3_1_structure", structure)
print(structure.round(3).to_string())
U.check(structure.loc["grouped", "validation trainees also in training"] == 0, "the grouped split keeps every trainee on one side")
U.check(structure.loc["random", "training rows that start later"] > 0.3, "the random split trains on rows from the future")
U.check(structure.loc["time-ordered", "training rows not yet due to end"] == 0,
        "the time-ordered split trains only on programmes that were due to end")

# %% [markdown]
# 🔍 **Look at** the three rows. The last two columns are averages over the validation rows: for the day a row is
# scored, the share of the training rows that start later, and the share whose programme was not yet due to end.
# The random split trains half on the future, and mostly on outcomes nobody knew yet. The grouped split removes the shared trainees and keeps the
# future. Only the time-ordered split describes a day on which the fund could really have trained this model.
#
# ## 2 · Score the same model under each (7')
#
# The same model, the champion, under each split. For the grouped split there are two readings. A trainee who
# is in no training row is, for the fund, a trainee it has never seen: such a trainee has no history. So the
# grouped split is scored twice: with the history columns as they are, and with the history of the validation
# rows hidden.

# %%
scores = {
    "random": U.split_scores(U.champion, dev, splits["random"]),
    "grouped": U.split_scores(U.champion, dev, splits["grouped"]),
    "grouped, history hidden": U.split_scores(U.champion, dev, splits["grouped"], new_trainees=True),
    "time-ordered (the fund's folds)": U.split_scores(U.champion, dev, splits["time-ordered"]),
}
three = pd.DataFrame({k: {"mean_ap": v["mean"], "spread": v["spread"], "auc": v["auc"]} for k, v in scores.items()}).T
three["against time-ordered"] = three.mean_ap - three.mean_ap.iloc[-1]
U.keep("d3_1_three", three)
print(three.round(4).to_string())

# %% [markdown]
# 🔍 **Look at** the table before you read on, and write down what you expected. Many analysts expect the random
# split to score far above the others. Here it does not: the random and the time-ordered score are inside each
# other's spread, and grouping alone changes nothing. The only clear gap appears when the history is hidden.
# A gap between splits is something you **measure**. It depends on the table and on the model.
#
# ## 3 · Explain the gap: new and returning trainees (4')
#
# Grouping alone changed nothing: the champion does not recognise a trainee from the other columns. Hiding the
# history did. So the score of the fund's folds is an average over two kinds of trainees.

# %%
oof = U.oof(("champion",)).merge(dev[["enrolment_id", "prior_enrolments"]], on="enrolment_id")
oof = oof[oof.champion.notna()]
kinds = {}
for name, part in [("new to the fund", oof[oof.prior_enrolments == 0]), ("returning", oof[oof.prior_enrolments > 0])]:
    kinds[name] = {"validation rows": len(part), "share": len(part) / len(oof), "base_rate": part[U.TARGET].mean(),
                   "ap": U.ap(part[U.TARGET], part.champion), "recall_at_capacity": U.recall_at(part[U.TARGET], part.champion)}
kinds = pd.DataFrame(kinds).T
U.keep("d3_1_kinds", kinds)
print(kinds.round(3).to_string())

# %% [markdown]
# 🔍 **Look at** the share of rows that are new to the fund. For them the random split, and the fund's own folds,
# report a score that is partly earned on returning trainees. If Mubakkir were moved to a programme where
# nobody has a history, the grouped score with hidden history is the one to quote.
#
# ## 4 · What only time order shows: let the model age (5')
#
# What can only a time-ordered split show? A random split trains and validates on the same period, so its model
# is always as fresh as its data. In use, a model is trained once and then gets older. The five models of the
# fund's folds were each trained for one half-year. Score each of them on the later half-years too.

# %%
folds = splits["time-ordered"]
blocks = sorted(dev.block.unique())[-U.N_FOLDS:]
rows = []
for k, (tr, _) in enumerate(folds):
    model = U.champion().fit(dev.iloc[tr][U.FEATURES], y[tr])
    for later in blocks[k:]:
        va = np.flatnonzero(dev.block == later)
        rows.append({"trained for": U.block_name(blocks[k]), "scored on": U.block_name(later),
                     "age in half-years": later - blocks[k], "ap": U.ap(y[va], U.predict(model, dev.iloc[va][U.FEATURES]))})
ageing = pd.DataFrame(rows)
grid = ageing.pivot(index="trained for", columns="scored on", values="ap")
print(grid.round(3).to_string(na_rep=""))
fresh = ageing[ageing["age in half-years"] == 0].set_index("scored on").ap
ageing["against a fresh model"] = ageing.ap - ageing["scored on"].map(fresh)
by_age = ageing.groupby("age in half-years")["against a fresh model"].agg(["mean", "count"])
U.keep("d3_1_grid", grid)
U.keep("d3_1_by_age", by_age)
print(by_age.round(4).to_string())
if not U.QUICK:
    U.check(by_age["mean"].iloc[2] < 0, "a model two half-years old scores below a fresh one")

# %% [markdown]
# 🔍 **Look at** each column of the grid from the bottom up: the same half-year, scored by older and older
# models. The older model also had fewer rows to learn from. Both are part of what "not retraining" means, and
# no random split can show either.
#
# ## 5 · Write the gap table with one sentence per gap (4')
#
# ✏️ **Exercise** · Complete the three sentences with your numbers. They go into the validation section of the
# model card.

# %%
r, g, h, t = (three.mean_ap[k] for k in three.index)
gaps = pd.DataFrame([
    {"comparison": "random against time-ordered", "gap": r - t, "cause": "none shown: inside the fold spread"},
    {"comparison": "grouped against random", "gap": g - r, "cause": "none: the model does not recognise trainees"},
    {"comparison": "history hidden against random", "gap": h - r, "cause": "a new trainee has no history"},
    {"comparison": "one year old against fresh", "gap": by_age["mean"].iloc[2], "cause": "the fund changes; the model does not"},
]).set_index("comparison")
U.keep("d3_1_gaps", gaps)
print(gaps.round(4).to_string())
def my_sentences() -> list:
    # BEGIN SOLUTION
    return [
        f"For a fresh model and the fund's mix of trainees, a random split reports {r:.3f} and the time-ordered folds {t:.3f}: no overstatement is shown.",
        f"For a trainee new to the fund a random split overstates by {r - h:.3f}: with no history the score is {h:.3f}.",
        f"A model used one year after training scores {-by_age['mean'].iloc[2]:.3f} lower than a fresh one, which only the time-ordered split can show.",
    ]
    # END SOLUTION


try:
    sentences = my_sentences()
except NotImplementedError:
    sentences = ["(your three sentences)"]
for s in sentences:
    print("-", s)
U.save("D3_1_three_splits.md", "# Lab 3.1: one model, three splits\n\n" + U.md(three, 4) + "\n\n" + U.md(gaps, 4) + "\n\n"
       + "\n".join("- " + s for s in sentences) + "\n")

# %% [markdown]
# ## ✱ Stretch · A model that memorises: does the grouped gap grow?
#
# Replace the champion by a model that can memorise: `U.make_pipeline(U.boosted(num_leaves=255,
# min_child_samples=1, n_estimators=300), scale=False)`. Score it under the random and the grouped split, history
# as it is. Does a gap open? What does the share of trainees with more than one row tell you about why?

# %% [markdown]
# ## Checkpoint
#
# 1. The random split and the time-ordered folds give the same score here. Why does the fund still use the
#    time-ordered folds? <details><summary>Answer</summary>Because they describe a day on which the model could
#    really have been trained: only programmes that were due to end, only the past. That the two agree is a
#    measured result for this model and this table. It does not hold for the next feature someone adds.</details>
# 2. Which score would you quote for a new vocational programme whose trainees have never enrolled before?
#    <details><summary>Answer</summary>The grouped score with the history hidden. Those trainees look like the
#    validation rows of that split.</details>
# 3. A model one year old loses about 0.015. Where does that number go?
#    <details><summary>Answer</summary>Into the model card, as the reason for a retraining interval, and into the
#    hand-over list: the deployment team has to know when the model was trained.</details>
#
# **Carry forward.** `artifacts/D3_1_three_splits.md`: the gap table with one sentence per gap. It is the
# evidence for the third course outcome.
