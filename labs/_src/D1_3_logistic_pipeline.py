# /// script
# requires-python = ">=3.11"
# dependencies = ["numpy>=2.0", "pandas>=2.2", "scipy>=1.14", "scikit-learn>=1.6", "lightgbm>=4.5", "matplotlib>=3.9", "tabulate>=0.9"]
# ///
# %% [markdown]
# # Lab 1.3 · A logistic model in a pipeline
#
# Day 1 · Session 3 (Linear models first) · 25 min.
#
# The third rung is the first model that learns: a regularised logistic model. You build it inside a pipeline, so
# that every number the preparation needs (a median, a scale, a list of categories) comes from training rows only.
#
# **AI policy: AI-tutor.** scikit-learn syntax is fair game for an assistant.
#
# **Reference state.** `U.modelling_table(history=False)`, the fund's folds and the reference run log.
#
# **Tasks**
# - [ ] 1 · Build the pipeline: impute, encode, scale, model (7')
# - [ ] 2 · Score it on the five folds and log the run (5')
# - [ ] 3 · Try three regularisation strengths; log each (5')
# - [ ] 4 · Prepare outside the folds and compare (5')
# - [ ] 5 · Read the largest coefficients (3')
# - [ ] ✱ Stretch · Add one interaction by hand and log it

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
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

table = U.modelling_table(history=False)
dev, _ = U.dev_holdout(table)
folds = U.folds(dev)
NUM, CAT = U.NUMERIC, U.LOW + U.HIGH
print("numeric:", NUM)
print("categories:", CAT)
print("missing values, %:", (dev[U.BASE].isna().mean() * 100).round(1)[lambda s: s > 0].to_dict())

# %% [markdown]
# ## 1 · Build the pipeline: impute, encode, scale, model (7')
#
# ✏️ **Exercise** · Write `build(C)`. Numeric columns: fill a missing value with the median and add a 0/1 column
# that says it was missing (`add_indicator=True`), then scale. Categories: fill with the words "not recorded",
# then one column per value; values seen fewer than 30 times share one column (`min_frequency=30`,
# `handle_unknown="infrequent_if_exist"`). Last step: `LogisticRegression(C=C, max_iter=2000)`.

# %%
def build(C: float = 0.1) -> Pipeline:
    # BEGIN SOLUTION
    numeric = Pipeline([("impute", SimpleImputer(strategy="median", add_indicator=True)),
                        ("scale", StandardScaler())])
    categories = Pipeline([("fill", SimpleImputer(strategy="constant", fill_value="not recorded")),
                           ("onehot", OneHotEncoder(handle_unknown="infrequent_if_exist", min_frequency=30,
                                                    sparse_output=False))])
    prepare = ColumnTransformer([("num", numeric, NUM), ("cat", categories, CAT)])
    return Pipeline([("prepare", prepare), ("model", LogisticRegression(C=C, max_iter=2000))])
    # END SOLUTION


# %%
try:
    build()
    make = build
except NotImplementedError:
    print("↪ using the reference pipeline, U.make_pipeline(U.logistic(C), U.BASE)")
    make = lambda C=0.1: U.make_pipeline(U.logistic(C), U.BASE)
tr, va = folds[-1]
one = make().fit(dev.iloc[tr][U.BASE], dev.iloc[tr][U.TARGET])
width = one[:-1].transform(dev.iloc[va][U.BASE]).shape[1]
print(f"fold 5: {len(U.BASE)} columns become {width} model inputs")
U.check(isinstance(one, Pipeline) and width > len(U.BASE), "the pipeline prepares and fits in one object")

# %% [markdown]
# ## 2 · Score it on the five folds and log the run (5')

# %%
r_logit = U.score_folds(lambda: make(0.1), dev, folds, U.BASE)
print(r_logit["per_fold"][["n_train", "base_rate", "ap", "auc", "recall_at_capacity"]].round(3))
U.log_run("1.3", "logistic", r_logit, model="logistic regression, C=0.1", features=U.BASE)
rules = U.fold_scores("three rules")
print("three rules, fold by fold:", rules.round(3), "· logistic:", np.round(r_logit["ap"], 3))
U.check((np.array(r_logit["ap"]) > rules).all(), "the logistic model beats the three rules on every fold")

# %% [markdown]
# 🔍 **Look at** both rows fold by fold. The model wins on each of the five, by far more than the folds disagree.
# That is what a real step up the ladder looks like.
#
# ## 3 · Try three regularisation strengths; log each (5')
#
# `C` is the inverse of the penalty: small `C`, strong penalty, coefficients pulled towards zero.

# %%
strengths = {}
for C in U.pick([0.003, 0.1, 10.0], [0.003, 0.1]):
    r = r_logit if C == 0.1 else U.score_folds(lambda: make(C), dev, folds, U.BASE)
    strengths[C] = {"mean_ap": r["mean"], "spread": r["spread"]}
    if C != 0.1:
        U.log_run("1.3", f"logistic C={C}", r, model=f"logistic regression, C={C}", features=U.BASE)
strengths = pd.DataFrame(strengths).T.rename_axis("C")
U.keep("d1_3_strengths", strengths)
print(strengths.round(4))

# %% [markdown]
# 🔍 **Look at** the three means against the spread. With tens of thousands of rows and about a hundred inputs,
# the penalty hardly matters unless it is very strong. Choosing `C` by these same fold scores would be tuning on
# the validation rows: harmless for three values, not for three hundred. Lab 2.2 does it properly.
#
# ## 4 · Prepare outside the folds and compare (5')
#
# What does the pipeline protect? Two experiments on the same folds. First, the preparation is fitted once on
# **all** development rows, the validation rows included. Second, a new feature that depends on the outcome, the
# provider's non-completion rate, is computed on all rows, then inside each fold.

# %%
def score_with_outside_preparation() -> float:
    full = make()
    prepare = full[:-1].fit(dev[U.BASE])                  # medians, scales and categories read every row
    X, y = prepare.transform(dev[U.BASE]), dev[U.TARGET].to_numpy()
    return float(np.mean([U.metrics(y[va], LogisticRegression(C=0.1, max_iter=2000).fit(X[tr], y[tr])
                                    .predict_proba(X[va])[:, 1])["ap"] for tr, va in folds]))


def score_with_provider_rate(inside: bool) -> float:
    scores, y = [], dev[U.TARGET].to_numpy()
    cols = U.BASE + ["provider_rate"]
    saved = list(U.NUMERIC)
    U.NUMERIC.append("provider_rate")
    try:
        for tr, va in folds:
            source = dev.iloc[tr] if inside else dev      # outside: the validation outcomes are in the rate
            rate = source.groupby("provider_id")[U.TARGET].mean()
            d = dev.assign(provider_rate=dev.provider_id.map(rate).fillna(source[U.TARGET].mean()))
            m = U.make_pipeline(U.logistic(0.1), cols).fit(d.iloc[tr][cols], y[tr])
            scores.append(U.metrics(y[va], U.predict(m, d.iloc[va][cols]))["ap"])
    finally:
        U.NUMERIC[:] = saved
    return float(np.mean(scores))


protect = pd.Series({"pipeline fitted inside each fold": r_logit["mean"],
                     "imputer and scaler fitted on all rows": score_with_outside_preparation(),
                     "provider rate from training rows": score_with_provider_rate(inside=True),
                     "provider rate from all rows": score_with_provider_rate(inside=False)}, name="mean_ap")
U.keep("d1_3_protect", protect)
print(protect.round(4))
U.check(protect["provider rate from all rows"] > protect["provider rate from training rows"],
        "a rate that has read the validation outcomes scores higher than one that has not")

# %% [markdown]
# 🔍 **Look at** the two pairs. A median or a scale read from all rows moves the score in the fourth decimal: the
# harm is small when the step never sees the outcome. A feature built from the outcome on all rows moves it far
# more, and all of that gain is false. The pipeline costs nothing and removes the question.
#
# ## 5 · Read the largest coefficients (3')

# %%
names = one[:-1].get_feature_names_out()
coef = pd.Series(one[-1].coef_[0], index=[n.split("__")[-1] for n in names]).sort_values(key=abs, ascending=False)
U.keep("d1_3_coefficients", coef.head(8))
print(coef.head(8).round(2))
line = (f"Logistic regression in a pipeline: average precision {r_logit['mean']:.3f}, spread {r_logit['spread']:.3f}, "
        f"against {rules.mean():.3f} for three rules.")
print(line)
U.save("D1_3_logistic.md", "# Lab 1.3: the logistic rung\n\n" + line + "\n\n" + U.md(strengths, 4) + "\n\n"
       + U.md(protect.to_frame(), 4) + "\n")

# %% [markdown]
# A coefficient is the change in the log-odds for one standard deviation of a scaled column, or for a category
# against the rest, **with the other inputs held fixed**. It ranks what the model leans on. It does not say that
# changing the column would change the outcome.
#
# ## ✱ Stretch · Add one interaction by hand and log it
#
# The rule of Lab 1.2 worked on absence. Does absence mean the same for everyone? Add one column by hand,
# `absences_w2` times an indicator for `enrolled_via == "employer"`, and log the run. Keep the answer for Day 2.

# %% [markdown]
# ## Checkpoint
#
# 1. Where is the median that fills a missing `distance_km` in fold 3 computed?
#    <details><summary>Answer</summary>On the training rows of fold 3, each time the pipeline is fitted. The
#    validation rows receive it; they never contribute to it.</details>
# 2. The spread over folds is about 0.03 and the gain over the rules about 0.2. What does the spread add?
#    <details><summary>Answer</summary>It is the yardstick: a gain several times the spread, with the same sign on
#    every fold, is a gain. A gain inside the spread is a question.</details>
# 3. The largest coefficient belongs to `absences_w2`. May the fund conclude that fewer absences cause completion?
#    <details><summary>Answer</summary>No. The model was built to predict. Absence may be a sign of something else
#    (work, transport, a class that is falling apart). Whether support helps is a separate question.</details>
#
# **Carry forward.** The logistic rung of the log. Lab 1.4 adds the trainee's history to it.
