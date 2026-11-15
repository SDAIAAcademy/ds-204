# /// script
# requires-python = ">=3.11"
# dependencies = ["numpy>=2.0", "pandas>=2.2", "scipy>=1.14", "scikit-learn>=1.6", "lightgbm>=4.5", "matplotlib>=3.9", "tabulate>=0.9"]
# ///
# %% [markdown]
# # Lab 2.3 · A challenger on equal terms
#
# Day 2 · Session 3 (Automated and pretrained challengers) · 25 min.
#
# The champion is the tuned boosted model of Lab 2.2. The challenger is an automated tool: you hand it the
# training rows and a time limit, and it trains several model families and blends them. A fair contest gives it
# the same five folds and the time the champion's search used, no more.
#
# **AI policy: AI-tutor.**
#
# **Reference state.** `U.modelling_table()`, the fund's folds and the reference run log (the rung "boosted,
# 20 trials" and the seconds its search took).
#
# **Tasks**
# - [ ] 1 · Read the challenger configuration and its licence (3')
# - [ ] 2 · Write the function that runs it on one fold (8')
# - [ ] 3 · Run the five folds on the same time budget (7')
# - [ ] 4 · Log the run; read what it built (4')
# - [ ] 5 · Set the two rows of the log side by side (3')
# - [ ] ✱ Stretch · Give the challenger the fund's inner split in place of its own

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
import shutil, tempfile, time, logging

table = U.modelling_table()
dev, _ = U.dev_holdout(table)
folds = U.folds(dev)
y = dev[U.TARGET].to_numpy()

# %% [markdown]
# ## 1 · Read the challenger configuration and its licence (3')
#
# The configuration is fixed before the contest and written down. So is the licence check: an open-source tool
# that trains on your machine sends no data anywhere. A hosted model or a pretrained one would need two more
# lines here: the licence of its weights, and where the rows travel.

# %%
champion_seconds = float(U.reference_log().query("rung == 'boosted, 20 trials'").seconds.iloc[-1])
BUDGET = int(U.pick(max(20, round(champion_seconds / len(folds))), 12))      # seconds per fold
CONFIG = {
    "tool": "AutoGluon Tabular (open source, Apache License 2.0), run on this machine",
    "preset": "medium_quality",
    "metric": "average_precision",
    "time_limit_per_fold_seconds": BUDGET,
    "features": "the 17 features of the champion, no others",
    "data_leaves_the_machine": False,
}
for k, v in CONFIG.items():
    print(f"{k:30s} {v}")
print(f"the champion's search took {champion_seconds:.0f} s for five folds")

# %% [markdown]
# ## 2 · Write the function that runs it on one fold (8')
#
# ✏️ **Exercise** · Complete `run_challenger`. Create a `TabularPredictor` with `label=U.TARGET`,
# `eval_metric="average_precision"`, `path=folder` and `verbosity=0`; call `.fit` on the training rows (features
# and target in one frame) with `time_limit=seconds` and `presets="medium_quality"`; return the probability of
# class 1 for the validation rows as an array, and the predictor.

# %%
def run_challenger(train: pd.DataFrame, valid: pd.DataFrame, seconds: int, folder: str):
    from autogluon.tabular import TabularPredictor
    # BEGIN SOLUTION
    predictor = TabularPredictor(label=U.TARGET, eval_metric="average_precision", path=folder, verbosity=0)
    predictor.fit(train[U.FEATURES + [U.TARGET]], time_limit=seconds, presets="medium_quality")
    return predictor.predict_proba(valid[U.FEATURES])[1].to_numpy(), predictor
    # END SOLUTION


# %% [markdown]
# ## 3 · Run the five folds on the same time budget (7')
#
# The tool writes its models to a folder; a temporary one is used and removed. If the tool is not installed, or
# the exercise is not done, the lab continues on the trainer's reference run.

# %%
def contest() -> dict:
    logging.disable(logging.CRITICAL)         # the tool reports every model family that is not installed
    rows, oof, built, t0 = [], np.full(len(dev), np.nan), None, time.perf_counter()
    for k, (tr, va) in enumerate(folds, 1):
        folder = tempfile.mkdtemp(prefix="ds204_")
        try:
            p, predictor = run_challenger(dev.iloc[tr], dev.iloc[va], BUDGET, folder)
            if k == len(folds):
                built = predictor.leaderboard(silent=True)[["model", "score_val", "fit_time"]]
        finally:
            shutil.rmtree(folder, ignore_errors=True)
        oof[va] = p
        rows.append({"fold": k, "n_train": len(tr), "n_valid": len(va), "base_rate": float(y[va].mean()),
                     **U.metrics(y[va], p)})
        print(f"fold {k}: average precision {rows[-1]['ap']:.3f}")
    per = pd.DataFrame(rows).set_index("fold")
    return {"per_fold": per, "ap": per.ap.tolist(), "mean": float(per.ap.mean()), "spread": float(per.ap.std(ddof=1)),
            "seconds": time.perf_counter() - t0, "oof": oof, "built": built}


try:
    r_challenger = contest()
    per, built = r_challenger["per_fold"], r_challenger["built"]
except (NotImplementedError, ImportError) as reason:
    logging.disable(logging.NOTSET)
    print(f"↪ using the reference run of the challenger ({type(reason).__name__})")
    r_challenger = None
    per, built = U.reference("d2_3_per_fold"), U.reference("d2_3_built")
    per.index = per.index.astype(int)
logging.disable(logging.NOTSET)
U.keep("d2_3_per_fold", per)
U.keep("d2_3_built", built)
print(per[["n_train", "ap", "auc", "recall_at_capacity"]].round(4))

# %% [markdown]
# ## 4 · Log the run; read what it built (4')

# %%
if r_challenger is not None:
    U.log_run("2.3", "challenger", r_challenger, model=f"AutoGluon Tabular, medium_quality, {BUDGET} s per fold",
              features=U.FEATURES, note="same folds and time budget as the champion")
    U.keep_oof("challenger", dev, r_challenger["oof"])
print("what it built on fold 5 (score_val is on its own internal validation rows):")
print(built.round(4).to_string(index=False))

# %% [markdown]
# 🔍 **Look at** the list. Several model families and a weighted blend on top. `score_val` comes from validation
# rows the tool set aside **by itself**, at random, inside the training part. The fund's folds validate forward
# in time. The tool's own number is therefore not the one to report; yours, from task 3, is.
#
# ## 5 · Set the two rows of the log side by side (3')

# %%
champion = U.fold_scores("boosted, 20 trials")
side = pd.DataFrame({"champion": champion, "challenger": per.ap.to_numpy()}, index=range(1, 6)).rename_axis("fold")
side["challenger - champion"] = side.challenger - side.champion
U.keep("d2_3_side", side)
print(side.round(4))
print(f"means: champion {side.champion.mean():.4f}, challenger {side.challenger.mean():.4f}; "
      f"the challenger is ahead on {(side['challenger - champion'] > 0).sum()} of 5 folds")
U.save("D2_3_challenger.md", "# Lab 2.3: the challenger\n\n" + "\n".join(f"- {k}: {v}" for k, v in CONFIG.items())
       + "\n\n" + U.md(side, 4) + "\n")

# %% [markdown]
# 🔍 **Look at** the last column and resist a verdict. Two means and five differences are not yet a decision.
# Lab 2.4 turns them into one.
#
# ## ✱ Stretch · x
#
# The tool chooses its blend on a random part of the training rows. Pass it the fund's inner split in place of
# its own: `tuning_data=` the last half-year of the training part, `train_data=` the rest. Same budget. Does
# its score on the outer fold move?

# %% [markdown]
# ## Checkpoint
#
# 1. The challenger ran with a 30-minute limit per fold; the champion's search took one minute in all. Is the
#    comparison fair? <details><summary>Answer</summary>No. A contest fixes the folds and the budget for both
#    sides. Otherwise it measures who was given more time.</details>
# 2. Why is the tool's own `score_val` not logged?
#    <details><summary>Answer</summary>It is measured on a random part of the training rows that the tool chose,
#    and it was used to pick and weight the models. The log holds scores on the fund's folds only.</details>
# 3. A colleague proposes a pretrained tabular model through a hosted service. Which two questions come before
#    any score? <details><summary>Answer</summary>Does the licence of the model allow this use, and may the
#    trainee rows leave the fund's machines at all? Either answer can end the contest before it starts.</details>
#
# **Carry forward.** The rung "challenger" in the log, with its five fold scores, for Lab 2.4.
