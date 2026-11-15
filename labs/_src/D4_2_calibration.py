# /// script
# requires-python = ">=3.11"
# dependencies = ["numpy>=2.0", "pandas>=2.2", "scipy>=1.14", "scikit-learn>=1.6", "lightgbm>=4.5", "matplotlib>=3.9", "tabulate>=0.9"]
# ///
# %% [markdown]
# # Lab 4.2 · Calibrate and compare by decile
#
# Day 4 · Session 2 (Calibration) · 25 min.
#
# An adviser reads "0.6" beside a name and understands "six in ten like her leave". Is that true? You compare
# what the boosted model expects with what happened, decile by decile: for the champion as delivered and for the
# same model trained with class weights. Then you calibrate both on the 2024 cohort and check the result on the
# 2025 cohort.
#
# **AI policy: AI-tutor.**
#
# **Reference state.** `U.day4_predictions()`: columns `p` (the champion), `p_weighted` (the champion trained
# with class weights) and `p_logistic` (the logistic rung), for the cohorts of 2024 and 2025.
#
# **Tasks**
# - [ ] 1 · Expected against observed by decile, as delivered (5')
# - [ ] 2 · Fit sigmoid and isotonic calibrators on the validation cohort (6')
# - [ ] 3 · Compare by decile on the held-out cohort (6')
# - [ ] 4 · Look at the providers that opened in 2025 (4')
# - [ ] 5 · Write the calibration line of the model card (4')
# - [ ] ✱ Stretch · Calibrate inside the time-ordered folds

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
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss

scored = U.day4_predictions()
valid = scored[scored.split == "validation"].reset_index(drop=True)
holdout = scored[scored.split == "holdout"].reset_index(drop=True)
MODELS = {"p": "boosted, as delivered", "p_weighted": "boosted, class weights"}

# %% [markdown]
# ## 1 · Expected against observed by decile, as delivered (5')
#
# ✏️ **Exercise** · Write `by_decile`. Sort the trainees by the probability, cut them into ten groups of equal
# size, and return for each group the mean probability (`expected`), the share that did not complete
# (`observed`) and the number of trainees (`n`). `pd.qcut(values.rank(method="first"), 10, labels=False)` gives
# the group of each row.

# %%
def by_decile(cohort: pd.DataFrame, column: str) -> pd.DataFrame:
    # BEGIN SOLUTION
    group = pd.qcut(cohort[column].rank(method="first"), 10, labels=False) + 1
    table = cohort.groupby(group).agg(expected=(column, "mean"), observed=(U.TARGET, "mean"), n=(column, "size"))
    return table.rename_axis("decile")
    # END SOLUTION


# %%
def reference_by_decile(cohort, column):
    group = pd.qcut(cohort[column].rank(method="first"), 10, labels=False) + 1
    return cohort.groupby(group).agg(expected=(column, "mean"), observed=(U.TARGET, "mean"), n=(column, "size")).rename_axis("decile")


try:
    by_decile(valid, "p")
    deciles = by_decile
except NotImplementedError:
    print("↪ using the reference decile table")
    deciles = reference_by_decile
delivered = pd.concat({name: deciles(valid, col)[["expected", "observed"]] for col, name in MODELS.items()}, axis=1)
U.keep("d4_2_delivered", delivered)
print(delivered.round(3).to_string())
U.check(np.allclose(deciles(valid, "p").to_numpy(), reference_by_decile(valid, "p").to_numpy()), "your decile table equals the reference")
plt = U.style()
fig, ax = plt.subplots()
for (col, name), colour in zip(MODELS.items(), ("#1f8a8a", "#d9534f")):
    t = deciles(valid, col)
    ax.plot(t.expected, t.observed, "o-", color=colour, label=name)
ax.plot([0, 1], [0, 1], color="grey", ls=":", lw=1)
ax.set(xlabel="expected: mean probability of the decile", ylabel="observed: share that did not complete",
       title="Cohort 2024: a calibrated model sits on the dotted line")
ax.legend()
U.savefig(fig, "D4_2_reliability.png")

# %% [markdown]
# 🔍 **Look at** the two pairs of columns. The model as delivered is close to what happened in every decile, a
# little high at the top. The weighted model ranks the trainees just as well (Lab 4.1) and its numbers are far
# from what happened: its top decile says nine in ten, and fewer than six in ten left. Class weights tell the
# model that non-completers are eight times as common as they are, and it answers accordingly.
#
# ## 2 · Fit sigmoid and isotonic calibrators on the validation cohort (6')
#
# A calibrator is a small model with one input, the score, fitted on rows the scoring model was **not** trained
# on. Here: the cohort of 2024. Two common ones:
#
# - **sigmoid**: a logistic regression on the log-odds of the score. Two numbers; it can shift and stretch.
# - **isotonic**: a rising step function. It can take any rising shape and needs more rows.

# %%
def log_odds(p) -> np.ndarray:
    p = np.clip(np.asarray(p, dtype=float), 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p)).reshape(-1, 1)


calibrators = {}
for col in MODELS:
    sigmoid = LogisticRegression(C=1e6).fit(log_odds(valid[col]), valid[U.TARGET])
    isotonic = IsotonicRegression(y_min=0, y_max=1, out_of_bounds="clip").fit(valid[col], valid[U.TARGET])
    calibrators[col] = {"as delivered": lambda p: np.asarray(p),
                        "sigmoid": lambda p, m=sigmoid: m.predict_proba(log_odds(p))[:, 1],
                        "isotonic": lambda p, m=isotonic: m.predict(np.asarray(p))}
    print(f"{MODELS[col]}: sigmoid slope {sigmoid.coef_[0, 0]:.2f}, intercept {sigmoid.intercept_[0]:+.2f}")

# %% [markdown]
# 🔍 **Look at** the two numbers of each sigmoid. A slope near 1 and an intercept near 0 mean "leave the scores
# as they are". Which model needed a large shift?
#
# ## 3 · Compare by decile on the held-out cohort (6')
#
# The test is the cohort of 2025, which neither the models nor the calibrators have seen. The Brier score is the
# mean squared distance between probability and outcome: lower is better, and it rewards both ranking and
# calibration.

# %%
y = holdout[U.TARGET].to_numpy()
rows = []
for col, name in MODELS.items():
    for method, f in calibrators[col].items():
        q = f(holdout[col])
        top = q >= np.sort(q)[::-1][len(q) // 10 - 1]
        rows.append({"model": name, "calibration": method, "brier": brier_score_loss(y, q), "mean expected": q.mean(),
                     "observed": y.mean(), "top decile expected": q[top].mean(), "top decile observed": y[top].mean(),
                     "average precision": U.ap(y, q)})
on_holdout = pd.DataFrame(rows).set_index(["model", "calibration"])
U.keep("d4_2_holdout", on_holdout.reset_index())
print(on_holdout.round(4).to_string())
weighted = on_holdout.loc["boosted, class weights"]
U.check(weighted.brier["sigmoid"] < weighted.brier["as delivered"] / 1.5, "calibration repairs the weighted model")

# %% [markdown]
# 🔍 **Look at** three things.
#
# - The weighted model, calibrated, is as good as the model as delivered. Calibration repaired what the weights
#   broke; it added nothing beyond that.
# - The model as delivered does not improve. A model trained on the real class shares with a log-loss objective is
#   often calibrated already. Calibrating it again spends the 2024 cohort for nothing: you measure first.
# - The sigmoid keeps the order of the trainees exactly, so average precision does not move and the list of Lab 4.1
#   stays the same. Isotonic calibration produces steps: trainees on one step are tied, and the ranking loses
#   about 0.02. A calibrator changes what the numbers mean, never who is at risk.
#
# ## 4 · Look at the providers that opened in 2025 (4')
#
# Calibration is a property of a model **on a population**. Eight providers opened in 2025. The model has never
# seen them.

# %%
providers = U.providers()
new = holdout.provider_id.isin(providers.provider_id[providers.opened_year == 2025]).to_numpy()
rows = {}
for name, part in (("providers open before 2025", holdout[~new]), ("providers that opened in 2025", holdout[new])):
    k, n = int(part[U.TARGET].sum()), len(part)
    low, high = U.wilson(k, n)
    rows[name] = {"trainees": n, "expected": part.p.mean(), "observed": k / n, "observed low": low, "observed high": high,
                  "average precision": U.ap(part[U.TARGET], part.p)}
drift = pd.DataFrame(rows).T
U.keep("d4_2_new_providers", drift)
print(drift.round(3).to_string())
U.check(drift.expected.iloc[1] < drift["observed low"].iloc[1], "at the new providers the model expects fewer non-completions than happened")

# %% [markdown]
# 🔍 **Look at** the second row. The expected rate lies below the whole interval of the observed rate: for the new
# providers the probabilities are too low by about four points, while the ranking inside them is as good as
# elsewhere. A provider the model does not know is scored like an average one. No calibrator fitted on 2024
# could have known.
#
# ## 5 · Write the calibration line of the model card (4')
#
# ✏️ **Exercise** · Write the calibration line of the model card: how the probabilities of the delivered model
# compare with what happened on the 2025 cohort, and where they must not be trusted.

# %%
def my_line() -> str:
    # BEGIN SOLUTION
    a = on_holdout.loc[("boosted, as delivered", "as delivered")]
    return (f"On the 2025 cohort the model expects {a['mean expected']:.1%} non-completion and {a.observed:.1%} happened; in its top "
            f"decile it expects {a['top decile expected']:.0%} and {a['top decile observed']:.0%} happened. No calibrator is applied: "
            f"none improved the Brier score ({a.brier:.4f}). At providers that opened after training the probabilities are about "
            f"{drift.observed.iloc[1] - drift.expected.iloc[1]:.0%} too low: use the rank there, not the number.")
    # END SOLUTION


# %%
try:
    line = my_line()
except NotImplementedError:
    line = "(your calibration line)"
print(line)
U.save("D4_2_calibration.md", "# Lab 4.2: calibration\n\n" + U.md(delivered) + "\n\n" + U.md(on_holdout.reset_index(), 4) + "\n\n"
       + U.md(drift) + "\n\n" + line + "\n")

# %% [markdown]
# ## ✱ Stretch · x
#
# The calibrators here were fitted on one cohort. Fit a sigmoid calibrator for the weighted model inside each of
# the fund's five folds instead, on the last half-year of each training part, and score the calibrated
# probabilities on the validation half-year. How much do the five slopes differ?

# %% [markdown]
# ## Checkpoint
#
# 1. Two models have the same average precision. One is calibrated, one is not. For which uses does the
#    difference matter? <details><summary>Answer</summary>Not for a list of fixed length: the ranking is the same.
#    It matters whenever the number itself is read or used: shown to an adviser, compared with a break-even
#    probability, multiplied by a cost.</details>
# 2. Why is a calibrator fitted on rows the model was not trained on?
#    <details><summary>Answer</summary>A flexible model has partly memorised its training rows, so its scores fit
#    them better than they fit new rows (Lab 2.1: 0.69 against 0.52). A calibrator fitted there would learn to
#    trust the scores too much.</details>
# 3. A colleague proposes to break even at 0.20 using the weighted model's scores. What happens?
#    <details><summary>Answer</summary>Its scores average 0.40 where one in eight leaves, so most of the cohort
#    lies above 0.20. A break-even probability can only be compared with a calibrated probability.</details>
#
# **Carry forward.** `artifacts/D4_2_calibration.md`: the calibration line, for the model card.
