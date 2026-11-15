# /// script
# requires-python = ">=3.11"
# dependencies = ["numpy>=2.0", "pandas>=2.2", "scipy>=1.14", "scikit-learn>=1.6", "lightgbm>=4.5", "matplotlib>=3.9", "tabulate>=0.9"]
# ///
# %% [markdown]
# # Lab 2.2 · Twenty trials in nested folds
#
# Day 2 · Session 2 (Tuning under a budget) · 25 min.
#
# Defaults did not beat the logistic model. Now you spend a budget of 20 trials on the settings of the boosted
# trees. The trials are judged on folds **inside** each training part, so that the score you report was never
# used to choose anything.
#
# **AI policy: AI-tutor.**
#
# **Reference state.** `U.modelling_table()`, the fund's folds and the reference run log (the rung "boosted,
# defaults").
#
# **Tasks**
# - [ ] 1 · Read the search space and the budget (3')
# - [ ] 2 · Write the search over inner folds (8')
# - [ ] 3 · Run it inside each outer fold (6')
# - [ ] 4 · Log the tuned rung; record the gain (4')
# - [ ] 5 · Compare inner and outer scores (4')
# - [ ] ✱ Stretch · Spend 40 trials: what does the second half of the budget buy?

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
import time

table = U.modelling_table()
dev, _ = U.dev_holdout(table)
folds = U.folds(dev)
y = dev[U.TARGET].to_numpy()
N_TRIALS = U.pick(20, 4)

# %% [markdown]
# ## 1 · Read the search space and the budget (3')
#
# Seven settings, a handful of values each: 2,592 combinations. The budget is 20 of them, drawn at
# random with a fixed seed.

# %%
SPACE = {
    "n_estimators": [100, 200, 400],            # how many trees
    "learning_rate": [0.02, 0.05, 0.1],         # how much each tree may correct
    "num_leaves": [4, 8, 16, 31],               # how complex each tree is
    "min_child_samples": [20, 50, 100, 200],    # the fewest rows a leaf may hold
    "subsample": [0.7, 1.0],                    # share of rows each tree sees
    "colsample_bytree": [0.5, 0.8, 1.0],        # share of inputs each tree sees
    "reg_lambda": [0.0, 5.0, 20.0],             # penalty on leaf values
}


def sample(rng) -> dict:
    s = {k: (v[int(rng.integers(len(v)))]) for k, v in SPACE.items()}
    return {**s, "subsample_freq": 1}


print("combinations:", int(np.prod([len(v) for v in SPACE.values()])), "· budget:", N_TRIALS, "trials")
print("one draw:", sample(np.random.default_rng(0)))

# %% [markdown]
# ## 2 · Write the search over inner folds (8')
#
# `U.inner_folds(train)` builds two folds inside a training part, the same way the fund's folds are built: each
# validates on a later half-year. `U.matrices(train, inner)` prepares them once and returns, per inner fold,
# `(X_train, y_train, X_valid, y_valid)`.
#
# ✏️ **Exercise** · Complete `search`: draw `n_trials` settings, score each on the inner folds (mean average
# precision), and return the best settings with the table of all trials.

# %%
def search(train: pd.DataFrame, n_trials: int, seed: int = U.SEED):
    train = train.reset_index(drop=True)
    inner = U.matrices(train, U.inner_folds(train))
    rng = np.random.default_rng(seed)
    trials = []
    for _ in range(n_trials):
        settings = sample(rng)
        # BEGIN SOLUTION
        scores = [U.ap(yv, U.boosted(**settings).fit(Xt, yt).predict_proba(Xv)[:, 1]) for Xt, yt, Xv, yv in inner]
        trials.append({**settings, "inner_ap": float(np.mean(scores))})
        # END SOLUTION
    trials = pd.DataFrame(trials).sort_values("inner_ap", ascending=False).reset_index(drop=True)
    best = {k: trials.loc[0, k].item() for k in list(SPACE) + ["subsample_freq"]}
    return best, trials


# %%
tr, va = folds[-1]
try:
    best5, trials5 = search(dev.iloc[tr], N_TRIALS)
    mine = True
except NotImplementedError:
    mine = False
    trials5 = U.reference("d2_2_trials_fold5")
U.keep("d2_2_trials_fold5", trials5)
print(trials5.round(4).to_string())
U.check(trials5.inner_ap.iloc[0] >= trials5.inner_ap.iloc[-1], "the trials are sorted, best first")

# %% [markdown]
# 🔍 **Look at** the top and the bottom of the table. Which settings do the best trials share? Which do the worst
# share? The gap between the best and the worst trial is what the budget buys.
#
# ## 3 · Run it inside each outer fold (6')
#
# The search runs again inside each of the five outer folds. The settings it picks are refitted on the whole
# training part and scored once on the validation half-year, which no trial has seen.

# %%
def nested(n_trials: int) -> dict:
    rows, oof, t0 = [], np.full(len(dev), np.nan), time.perf_counter()
    for k, (tr, va) in enumerate(folds, 1):
        best, trials = search(dev.iloc[tr], n_trials, seed=U.SEED + k)
        model = U.make_pipeline(U.boosted(**best), scale=False).fit(dev.iloc[tr][U.FEATURES], y[tr])
        p = U.predict(model, dev.iloc[va][U.FEATURES])
        oof[va] = p
        rows.append({"fold": k, "n_train": len(tr), "n_valid": len(va), "base_rate": float(y[va].mean()),
                     **U.metrics(y[va], p), "inner_best": float(trials.inner_ap.iloc[0]),
                     "trees": best["n_estimators"], "leaves": best["num_leaves"], "rate": best["learning_rate"]})
    per = pd.DataFrame(rows).set_index("fold")
    return {"per_fold": per, "ap": per.ap.tolist(), "mean": float(per.ap.mean()), "spread": float(per.ap.std(ddof=1)),
            "seconds": time.perf_counter() - t0, "oof": oof}


if mine:
    r_tuned = nested(N_TRIALS)
    per = r_tuned["per_fold"]
else:
    per = U.reference("d2_2_per_fold")
    per.index = per.index.astype(int)
    r_tuned = None
U.keep("d2_2_per_fold", per)
print(per[["ap", "inner_best", "trees", "leaves", "rate", "recall_at_capacity"]].round(4))

# %% [markdown]
# 🔍 **Look at** the settings chosen in each fold. They need not agree, and that is fine: what is being validated
# is the **procedure** "search 20 trials, then fit", not one set of numbers.
#
# ## 4 · Log the tuned rung; record the gain (4')

# %%
default = U.fold_scores("boosted, defaults")
logit = U.fold_scores("logistic + history")
if r_tuned is not None:
    U.log_run("2.2", "boosted, 20 trials", r_tuned, model=f"LightGBM, {N_TRIALS}-trial random search in nested folds",
              features=U.FEATURES, note="settings chosen on inner folds")
    U.keep_oof("champion", dev, r_tuned["oof"])
    U.keep("d2_2_seconds", r_tuned["seconds"])
gain = pd.DataFrame({"logistic + history": logit, "boosted, defaults": default, "boosted, tuned": per.ap.to_numpy()},
                    index=range(1, 6)).rename_axis("fold")
gain["tuned - defaults"] = gain["boosted, tuned"] - gain["boosted, defaults"]
gain["tuned - logistic"] = gain["boosted, tuned"] - gain["logistic + history"]
U.keep("d2_2_gain", gain)
print(gain.round(4))
print(f"gain of tuning over defaults: {gain['tuned - defaults'].mean():+.4f}; over the logistic rung: "
      f"{gain['tuned - logistic'].mean():+.4f}")
if not U.QUICK:
    U.check((gain["tuned - defaults"] > 0).all(), "tuning beats the defaults on every fold")

# %% [markdown]
# 🔍 **Look at** the two gains. The first is clear. The second is small: keep it in mind for Lab 2.4, where a
# difference gets an interval.
#
# ## 5 · Compare inner and outer scores (4')
#
# In each fold the best trial had an inner score (`inner_best`). It sits on other rows than the outer score, so
# the two cannot be compared as numbers: the inner score **chooses**, the outer score **reports**. What happens
# when one score does both? A final search on all development rows, judged on the fund's five folds, gives the
# settings the fund will use from here on. Then an experiment on its 20 trials: choose the best trial on a random
# part A of the validation rows, and read its score on A and on the untouched part B.

# %%
if mine:
    mats = U.matrices(dev, folds)
    rng = np.random.default_rng(U.SEED)
    final, preds = [], []
    for _ in range(N_TRIALS):
        s = sample(rng)
        p = [U.boosted(**s).fit(Xt, yt).predict_proba(Xv)[:, 1] for Xt, yt, Xv, yv in mats]
        final.append({**s, "ap": float(np.mean([U.ap(m[3], q) for m, q in zip(mats, p)]))})
        preds.append(np.concatenate(p))
    order = np.argsort([-f["ap"] for f in final])
    final = pd.DataFrame(final).iloc[order].reset_index(drop=True)
    chosen = {k: final.loc[0, k].item() for k in list(SPACE) + ["subsample_freq"]}
    preds, yy = np.array(preds), np.concatenate([m[3] for m in mats])

    def optimism(size_a: int, repeats: int) -> dict:
        """Choose the best trial on part A; its score there, and on the rest."""
        rng, on_a, on_b = np.random.default_rng(U.SEED), [], []
        for _ in range(repeats):
            a = np.zeros(len(yy), dtype=bool)
            a[rng.choice(len(yy), size_a, replace=False)] = True
            pick = int(np.argmax([U.ap(yy[a], p[a]) for p in preds]))
            on_a.append(U.ap(yy[a], preds[pick][a]))
            on_b.append(U.ap(yy[~a], preds[pick][~a]))
        return {"rows_used_to_choose": size_a, "score_where_chosen": float(np.mean(on_a)),
                "score_on_fresh_rows": float(np.mean(on_b)), "optimism": float(np.mean(on_a) - np.mean(on_b))}

    chosen_twice = pd.DataFrame([optimism(n, U.pick(40, 5)) for n in (len(yy) // 2, 2000, 500)]).set_index("rows_used_to_choose")
    U.keep("d2_2_best", {"params": chosen, "ap_on_the_fund_folds": float(final.ap.iloc[0]),
                         "median_trial": float(final.ap.median()), "worst_trial": float(final.ap.iloc[-1])})
    U.keep("d2_2_optimism", chosen_twice)
else:
    chosen = U.reference("d2_2_best")["params"]
    chosen_twice = U.reference("d2_2_optimism")
print("the fund's settings:", chosen)
print(chosen_twice.round(4))
line = (f"A {N_TRIALS}-trial search in nested folds scores {per.ap.mean():.3f} (spread {per.ap.std(ddof=1):.3f}): "
        f"{gain['tuned - defaults'].mean():+.3f} over defaults and {gain['tuned - logistic'].mean():+.3f} over the "
        f"logistic rung.")
print(line)
U.save("D2_2_search.md", "# Lab 2.2: tuning under a budget\n\n" + line + "\n\n" + U.md(gain, 4) + "\n\n" + U.md(chosen_twice, 4) + "\n")
if not U.QUICK:
    U.check(chosen_twice.optimism.iloc[-1] > chosen_twice.optimism.iloc[0], "the fewer rows choose, the more the chosen score flatters")

# %% [markdown]
# 🔍 **Look at** the column `optimism`. A score that was used to choose is higher than the same model's score on
# fresh rows. With 20 trials and ten thousand rows the difference is small. With 500 rows it is not. More trials
# push it the same way. Nested folds remove the question whatever its size.
#
# ## ✱ Stretch · x
#
# Set `N_TRIALS = 40` and run tasks 2 to 4 again. Log it as a new line. What did the second half of the budget
# buy, and what did it cost in seconds?

# %% [markdown]
# ## Checkpoint
#
# 1. Why may the best inner score not be reported as the score of the model?
#    <details><summary>Answer</summary>It was chosen for being the highest of 20, so part of it is luck. The outer
#    fold, which no trial saw, gives the score of the procedure.</details>
# 2. The five outer folds chose five different settings. Which one is "the model"?
#    <details><summary>Answer</summary>None of them. The nested score describes the procedure. The model the fund
#    uses comes from running that procedure once on all development rows.</details>
# 3. Tuning gained 0.02 over defaults and took a few minutes. When would you stop?
#    <details><summary>Answer</summary>When the gain of more budget is inside the fold spread, or when the budget
#    stated at the start is spent. Stating it first is what keeps the score honest.</details>
#
# **Carry forward.** The rung "boosted, 20 trials" is the **champion**. `U.champion()` builds it with the settings
# of `data/reference/d2_2_best.json` for every later lab.
