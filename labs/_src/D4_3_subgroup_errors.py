# /// script
# requires-python = ">=3.11"
# dependencies = ["numpy>=2.0", "pandas>=2.2", "scipy>=1.14", "scikit-learn>=1.6", "lightgbm>=4.5", "matplotlib>=3.9", "tabulate>=0.9"]
# ///
# %% [markdown]
# # Lab 4.3 · Errors by region, gender and age band
#
# Day 4 · Session 3 (Errors by subgroup) · 25 min.
#
# A trainee the model misses gets no adviser. So the question "how good is the list?" has a second half: good
# for whom? You compare the training data with the people the fund serves, break the misses of the 2025 list down
# by region, gender and age band with intervals that respect the size of each slice, and name the kind of bias
# behind each finding.
#
# **AI policy: AI-off for the first attempt at tasks 2 to 4.** The table and its reading are yours. After that,
# AI-tutor.
#
# **Reference state.** `U.day4_predictions()` (the 2025 cohort with the champion's `p`), `U.population()` (the
# fund's eligible population by region, age band and gender) and `U.modelling_table()`.
#
# **Tasks**
# - [ ] 1 · Compare the training data with the eligible population (5')
# - [ ] 2 · Miss rate and precision by region, with intervals (7')
# - [ ] 3 · The same by gender and by age band (5')
# - [ ] 4 · Decide which gaps the intervals support (4')
# - [ ] 5 · Name the bias type and one action per finding (4')
# - [ ] ✱ Stretch · The same table by provider: how many slices can carry an interval?

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
scored = U.day4_predictions()
holdout = scored[scored.split == "holdout"].reset_index(drop=True)
CAPACITY = U.decision()["adviser_capacity_per_year"]
holdout["on_list"] = U.top_share(holdout.p, CAPACITY / len(holdout))
left = holdout[holdout[U.TARGET] == 1]
OVERALL = float((~left.on_list).mean())
print(f"cohort 2025: {len(holdout):,} trainees, {len(left):,} did not complete, {int(holdout.on_list.sum())} on the list")
print(f"miss rate: {OVERALL:.3f} of the trainees who left were not on the list")

# %% [markdown]
# Two errors, two harms. A trainee who leaves and was **not** on the list got no support: the **miss rate**,
# among those who left. A trainee on the list who would have completed costs an adviser's hour: the list's
# **precision**. The first harm falls on trainees, so it leads.
#
# ## 1 · Compare the training data with the eligible population (5')
#
# Before any error: who is in the data the model learnt from? Gender is not a feature of the model. It is in the
# table so that this question can be asked.

# %%
train = U.modelling_table()
train = train[train.cohort_year <= 2024]
population = U.population()
parts = []
for attribute in ("region", "age_band", "gender"):
    eligible = population[population.attribute == attribute].set_index("value").eligible_share
    share = train[attribute].value_counts(normalize=True).reindex(eligible.index)
    parts.append(pd.DataFrame({"attribute": attribute, "in training data": share, "in eligible population": eligible,
                               "ratio": share / eligible}))
represented = pd.concat(parts).sort_values("ratio")
U.keep("d4_3_represented", represented)
print(represented.round(3).to_string())

# %% [markdown]
# 🔍 **Look at** the top of the table: the groups with the smallest ratio. A ratio of 0.25 means the group is a
# quarter as present in the training data as among the people the fund is meant to serve. The model has seen
# few of them, and so has every validation score of this week.
#
# ## 2 · Miss rate and precision by region, with intervals (7')
#
# ✏️ **Exercise** · Write `slice_table(cohort, by)`. For each value of the column `by`, return one row with:
#
# - `trainees`, and `left`: how many did not complete;
# - `missed`: how many of those were not on the list, `miss_rate = missed / left`, and its 95% interval
#   `miss_low`, `miss_high` from `U.wilson(missed, left)`;
# - `listed`: how many are on the list, and `precision`: the share of those who did not complete.

# %%
def slice_table(cohort: pd.DataFrame, by: str) -> pd.DataFrame:
    # BEGIN SOLUTION
    rows = {}
    for value, part in cohort.groupby(by):
        gone = part[part[U.TARGET] == 1]
        missed, listed = int((~gone.on_list).sum()), part[part.on_list]
        low, high = U.wilson(missed, len(gone))
        rows[value] = {"trainees": len(part), "left": len(gone), "missed": missed, "miss_rate": missed / len(gone),
                       "miss_low": low, "miss_high": high, "listed": len(listed),
                       "precision": listed[U.TARGET].mean() if len(listed) else np.nan}
    return pd.DataFrame(rows).T.rename_axis(by)
    # END SOLUTION


# %%
def reference_slice_table(cohort, by):
    rows = {}
    for value, part in cohort.groupby(by):
        gone = part[part[U.TARGET] == 1]
        missed, listed = int((~gone.on_list).sum()), part[part.on_list]
        low, high = U.wilson(missed, len(gone))
        rows[value] = {"trainees": len(part), "left": len(gone), "missed": missed, "miss_rate": missed / len(gone),
                       "miss_low": low, "miss_high": high, "listed": len(listed),
                       "precision": listed[U.TARGET].mean() if len(listed) else np.nan}
    return pd.DataFrame(rows).T.rename_axis(by)


try:
    slice_table(holdout, "gender")
    slices = slice_table
except NotImplementedError:
    print("↪ using the reference slice table")
    slices = reference_slice_table
by_region = slices(holdout, "region").sort_values("miss_rate", ascending=False)
U.keep("d4_3_region", by_region)
print(by_region.round(3).to_string())
U.check(np.allclose(slices(holdout, "region").sort_index().to_numpy(dtype=float),
                    reference_slice_table(holdout, "region").sort_index().to_numpy(dtype=float), equal_nan=True),
        "your table equals the reference")
plt = U.style()
fig, ax = plt.subplots(figsize=(7.2, 4.6))
order = by_region.sort_values("miss_rate")
ax.errorbar(order.miss_rate, range(len(order)), xerr=[order.miss_rate - order.miss_low, order.miss_high - order.miss_rate],
            fmt="o", color="#1f8a8a", capsize=3)
ax.axvline(OVERALL, color="grey", ls=":", lw=1)
ax.set_yticks(range(len(order)), [f"{r} ({int(n)})" for r, n in zip(order.index, order.left)])
ax.set(xlabel="miss rate with 95% interval; in brackets: trainees who left", title="Cohort 2025: who was not on the list?")
U.savefig(fig, "D4_3_regions.png")

# %% [markdown]
# 🔍 **Look at** the widths before the positions. The region with the highest miss rate rests on about twenty
# trainees who left: its interval covers almost everything. Then look for regions whose whole interval lies to
# one side of the dotted line.
#
# ## 3 · The same by gender and by age band (5')

# %%
by_gender = slices(holdout, "gender")
by_age = slices(holdout, "age_band")
U.keep("d4_3_gender", by_gender)
U.keep("d4_3_age", by_age)
print(by_gender.round(3).to_string(), "\n")
print(by_age.round(3).to_string())

# %% [markdown]
# 🔍 **Look at** the two gender rows. The miss rates differ by about five points. Do the intervals overlap? What
# would you say to someone who quotes only the two rates?
#
# ## 4 · Decide which gaps the intervals support (4')
#
# A rule, stated before reading the answer: a gap is **supported** when the whole interval of the slice lies
# above, or below, the miss rate of the cohort.
#
# ✏️ **Exercise** · Return the names of the slices, from all three tables, whose gap is supported.

# %%
everything = pd.concat([by_region, by_gender, by_age])


def supported(table: pd.DataFrame, overall: float) -> list:
    # BEGIN SOLUTION
    return sorted(table.index[(table.miss_low > overall) | (table.miss_high < overall)])
    # END SOLUTION


# %%
reference_supported = sorted(everything.index[(everything.miss_low > OVERALL) | (everything.miss_high < OVERALL)])
try:
    mine = sorted(supported(everything, OVERALL))
except NotImplementedError:
    print("↪ using the reference answer")
    mine = reference_supported
print("gaps the intervals support:", mine)
U.check(mine == reference_supported, "the supported gaps match the rule")

# two regions report early absences on paper registers; together they form one slice with a reason behind it
paper = holdout.region.isin(["Jazan", "Najran"])
registers = slices(holdout.assign(register=np.where(paper, "paper registers (Jazan, Najran)", "electronic registers")), "register")
absences = holdout.groupby(np.where(paper, "paper registers (Jazan, Najran)", "electronic registers")).agg(
    mean_absences_w2=("absences_w2", "mean"), non_completion=(U.TARGET, "mean"))
registers = registers.join(absences)
U.keep("d4_3_registers", registers)
print(registers.round(3).to_string())

# %% [markdown]
# 🔍 **Look at** the last table. Hamza Al-Babtain confirms that providers in two regions still keep attendance on
# paper and send it late. Their trainees leave more often than elsewhere and are recorded with fewer early
# absences. The model's strongest signal is weakest exactly where the need is highest. With 19 slices on the
# table, one can cross the line by chance: a gap with a **cause** you can check deserves more trust than a gap
# without one.
#
# ## 5 · Name the bias type and one action per finding (4')
#
# The SDAIA AI Bias Reference Guide names kinds of bias. Four of them fit what you measured:
#
# - **measurement bias**: a feature or label is recorded differently for different groups;
# - **coverage bias**: a group is less present in the data than in the population the system serves;
# - **sample size bias**: a conclusion rests on too few cases;
# - **aggregation bias**: one overall figure hides groups for whom it does not hold.
#
# ✏️ **Exercise** · Name the kind behind each finding.

# %%
FINDINGS = {
    "A": "Trainees who leave in Jazan and Najran are missed more often; their early absences are under-recorded.",
    "B": "Trainees of 35 and over are a quarter as present in the training data as in the eligible population.",
    "C": "The region with the highest miss rate has about twenty trainees who left; its interval spans 0.60 to 0.92.",
    "D": "The cohort's miss rate of 0.62 was the only figure reported before today.",
}


def my_kinds() -> dict:
    # BEGIN SOLUTION
    return {"A": "measurement", "B": "coverage", "C": "sample size", "D": "aggregation"}
    # END SOLUTION


# %%
KEY = {"A": ("measurement", "Electronic registers at those providers; until then advisers there do not rely on the list alone."),
       "B": ("coverage", "State it in the card; validate again before the model is used on programmes aimed at this age group."),
       "C": ("sample size", "Report the interval, pool small regions or cohorts, and draw no conclusion from the rate alone."),
       "D": ("aggregation", "Report the miss rate by region, gender and age band with every cohort, beside the overall figure.")}
try:
    kinds = my_kinds()
except NotImplementedError:
    print("↪ using the reference answers")
    kinds = {k: v[0] for k, v in KEY.items()}
findings = pd.DataFrame({"finding": FINDINGS, "my kind": kinds, "bias type": {k: v[0] for k, v in KEY.items()},
                         "one action": {k: v[1] for k, v in KEY.items()}})
print(findings.to_string())
U.check(sum(kinds.get(k) == v[0] for k, v in KEY.items()) >= 3, "at least three of four bias types match the reference")
U.save("D4_3_subgroups.md", "# Lab 4.3: errors by subgroup, cohort 2025\n\n" + f"Miss rate of the cohort: {OVERALL:.3f}\n\n"
       + U.md(by_region) + "\n\n" + U.md(by_gender) + "\n\n" + U.md(by_age) + "\n\n" + U.md(registers) + "\n\n"
       + U.md(represented) + "\n\n" + findings.drop(columns="my kind").to_markdown() + "\n")

# %% [markdown]
# Gender is not on the list of findings: the gap of five points is not supported by the intervals. It is not
# dismissed either. Women are 43% of the training data and 52% of the eligible population, so the next cohort's
# table is read with that row first.
#
# ## ✱ Stretch · The same table by provider: how many slices can carry an interval?
#
# Build the same table by provider. How many of the 60 providers have at least 30 trainees who left? What does
# that say about a provider league table built from this model's errors?

# %% [markdown]
# ## Checkpoint
#
# 1. Qassim has the highest miss rate, 0.81. Why is it not a finding?
#    <details><summary>Answer</summary>It rests on 21 trainees who left. Its interval runs from 0.60 to 0.92 and
#    contains the cohort's rate. The honest sentence is: too few cases to tell.</details>
# 2. Gender is not a feature. Can the model still treat women and men differently?
#    <details><summary>Answer</summary>Yes: through features that differ between them, and through who is in the
#    data. That is why gender is kept for the audit. Leaving a column out does not make a model blind to
#    it.</details>
# 3. What would retraining change about finding A?
#    <details><summary>Answer</summary>Nothing. The model is right about what the registers say. The fault is in
#    the measurement, so the action is with the providers, not with the model.</details>
#
# **Carry forward.** `artifacts/D4_3_subgroups.md`: the subgroup tables and the four findings with a bias type
# and an action each. It is the evidence for the fifth course outcome, with Labs 4.1 and 4.2.
