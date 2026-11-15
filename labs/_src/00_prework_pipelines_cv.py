# /// script
# requires-python = ">=3.11"
# dependencies = ["numpy>=2.0", "pandas>=2.2", "scipy>=1.14", "scikit-learn>=1.6", "lightgbm>=4.5", "matplotlib>=3.9", "tabulate>=0.9"]
# ///
# %% [markdown]
# # Pre-work · Pipelines and cross-validation
#
# About two hours, before Day 1. The course assumes that you can join and group a table, explain overfitting
# and say what a held-out set is for (FN-101, FN-105, DS-201). This notebook adds the two tools every lab uses: a
# **pipeline** that prepares and fits in one object, and **cross-validation** that scores it on several folds.
# Work through the four parts, then answer the ten questions at the end without looking back. The supervised
# entry quiz on Day 1 asks questions of the same kind, without AI assistance.
#
# **AI policy: AI-tutor.** An assistant may explain syntax and concepts. Predict each result before you run it.

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
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import KFold, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

# %% [markdown]
# ## 1 · A held-out set, once more (20 min)
#
# A small practice table: 3,000 rows drawn from the fund's enrolments, five columns. The outcome is
# `not_completed`.

# %%
practice = U.modelling_table(history=False).sample(3000, random_state=1)[
    ["age", "aptitude_score", "absences_w2", "employment", "programme", U.TARGET]].reset_index(drop=True)
print(practice.head())
print("missing values:", practice.isna().sum().to_dict())
X, y = practice.drop(columns=U.TARGET), practice[U.TARGET]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=1, stratify=y)
print(len(X_train), "training rows,", len(X_test), "held-out rows; non-completion", round(y_train.mean(), 3))

# %% [markdown]
# ## 2 · A pipeline (35 min)
#
# The model cannot read a missing number or the word "part-time". Preparation turns the columns into numbers, and
# it has parameters of its own: a median, a mean and a spread, a list of categories. Those are learnt from the
# training rows, exactly like the weights of the model.
#
# ✏️ **Exercise** · Complete `make_model()`: numeric columns are imputed with the median and scaled, the two text
# columns are one-hot encoded (`handle_unknown="ignore"`), and a logistic regression follows.

# %%
NUMERIC, TEXT = ["age", "aptitude_score", "absences_w2"], ["employment", "programme"]


def make_model() -> Pipeline:
    # BEGIN SOLUTION
    numeric = Pipeline([("impute", SimpleImputer(strategy="median")), ("scale", StandardScaler())])
    prepare = ColumnTransformer([("num", numeric, NUMERIC), ("text", OneHotEncoder(handle_unknown="ignore"), TEXT)])
    return Pipeline([("prepare", prepare), ("model", LogisticRegression(max_iter=1000))])
    # END SOLUTION


# %%
try:
    model = make_model()
except NotImplementedError:
    print("↪ using a reference pipeline")
    model = Pipeline([("prepare", ColumnTransformer([
        ("num", Pipeline([("impute", SimpleImputer(strategy="median")), ("scale", StandardScaler())]), NUMERIC),
        ("text", OneHotEncoder(handle_unknown="ignore"), TEXT)])), ("model", LogisticRegression(max_iter=1000))])
model.fit(X_train, y_train)
median = model.named_steps["prepare"].named_transformers_["num"].named_steps["impute"].statistics_
print("medians learnt from the training rows:", dict(zip(NUMERIC, median.round(1))))
print("held-out accuracy:", round(model.score(X_test, y_test), 3), "· share that completed:", round(1 - y_test.mean(), 3))
U.check(model.score(X_test, y_test) > 0.8, "the pipeline fits and predicts")

# %% [markdown]
# 🔍 **Look at** the last line. The accuracy is close to the share of trainees who completed: a model that says
# "completes" for everyone would do nearly as well. Day 1 replaces accuracy with scores that suit a rare outcome.
#
# ## 3 · Cross-validation (30 min)
#
# One held-out set gives one number. Five folds give five: a mean, and a spread that says how much to trust it.
# `cross_val_score` fits a fresh copy of the whole pipeline on four fifths and scores the remaining fifth, five
# times.

# %%
cv = KFold(n_splits=5, shuffle=True, random_state=1)
scores = cross_val_score(model, X, y, cv=cv, scoring="average_precision")
print("average precision per fold:", scores.round(3))
print(f"mean {scores.mean():.3f}, spread {scores.std(ddof=1):.3f}, base rate {y.mean():.3f}")
U.check(scores.mean() > y.mean(), "the model ranks better than chance")

# %% [markdown]
# 🔍 **Look at** the spread. Two models whose means differ by less than this are not yet different. The folds
# here are random. Day 3 shows when random folds flatter a model and what to use in their place.
#
# ## 4 · Preparation inside the folds (35 min)
#
# A demonstration with data that hold **no signal at all**: 200 rows, 2,000 columns of noise and a coin-flip
# outcome. The honest average precision is about 0.5. Two ways to pick the 20 "best" columns:

# %%
rng = np.random.default_rng(1)
noise, coin = rng.normal(size=(200, 2000)), rng.integers(0, 2, 200)
outside = SelectKBest(f_classif, k=20).fit_transform(noise, coin)        # chosen with the outcome of every row
wrong = cross_val_score(LogisticRegression(max_iter=1000), outside, coin, cv=cv, scoring="average_precision")
inside = Pipeline([("select", SelectKBest(f_classif, k=20)), ("model", LogisticRegression(max_iter=1000))])
right = cross_val_score(inside, noise, coin, cv=cv, scoring="average_precision")
print(f"columns chosen before the folds: {wrong.mean():.2f}")
print(f"columns chosen inside each fold: {right.mean():.2f}")
U.check(wrong.mean() > right.mean() + 0.2, "choosing columns with the held-out outcomes invents a signal")

# %% [markdown]
# 🔍 **Look at** the two numbers. Nothing can be predicted here, and the first number says otherwise. The step
# that chose the columns had read the outcome of the rows later used for scoring. Inside the pipeline, the
# choice is made again in every fold from the training rows only.
#
# ## Ten questions
#
# Answer on paper first.
#
# 1. What does `fit` learn in a `StandardScaler`? <details><summary>Answer</summary>The mean and the standard
#    deviation of each column, from the rows it is given.</details>
# 2. On which rows is the median of the imputer computed during `cross_val_score`?
#    <details><summary>Answer</summary>On the training part of each fold, five times.</details>
# 3. Why `handle_unknown="ignore"` in the encoder? <details><summary>Answer</summary>A category that appears only
#    in the held-out rows would otherwise stop the prediction.</details>
# 4. A model scores 0.95 on its training rows and 0.60 on held-out rows. Name it.
#    <details><summary>Answer</summary>Overfitting: it has learnt the training rows, not the pattern.</details>
# 5. What is the held-out set for? <details><summary>Answer</summary>To estimate how the model does on rows it has
#    not seen. It stops being that the moment you choose anything by looking at it.</details>
# 6. Five fold scores are 0.31, 0.29, 0.33, 0.30, 0.32. Another model has a mean of 0.315. Better?
#    <details><summary>Answer</summary>Not shown. The difference, 0.005, is well inside the spread of 0.016.</details>
# 7. Why did the noise columns of part 4 seem to predict the coin?
#    <details><summary>Answer</summary>Among 2,000 random columns some correlate with any outcome by chance.
#    Choosing them with all rows carries that chance into the held-out rows.</details>
# 8. Accuracy is 88% and 88% of trainees complete. What has the model learnt?
#    <details><summary>Answer</summary>Possibly nothing: predicting "completes" for all gives the same accuracy.</details>
# 9. What is the difference between a parameter and a setting such as `C`?
#    <details><summary>Answer</summary>A parameter is learnt by `fit` from the training rows. A setting is chosen
#    by you before fitting, and choosing it also needs data the final score does not come from.</details>
# 10. You join a table of provider averages computed on all rows to your training rows. What have you done?
#    <details><summary>Answer</summary>Put information from the held-out rows, outcomes included if the average
#    is of the outcome, into the training data. Lab 1.3 measures it.</details>
