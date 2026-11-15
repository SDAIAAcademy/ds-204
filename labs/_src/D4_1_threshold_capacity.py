# /// script
# requires-python = ">=3.11"
# dependencies = ["numpy>=2.0", "pandas>=2.2", "scipy>=1.14", "scikit-learn>=1.6", "lightgbm>=4.5", "matplotlib>=3.9", "tabulate>=0.9"]
# ///
# %% [markdown]
# # Lab 4.1 · A threshold that fits capacity
#
# Day 4 · Session 1 (Thresholds for rare outcomes) · 25 min.
#
# The champion ranks trainees. Ghassan Al-Fayez's advisers can write 800 support plans a year. You turn the
# ranking into a list: a threshold chosen on the 2024 cohort, compared with two thresholds people reach for, and
# then tested once on the 2025 cohort, which nobody has opened until now.
#
# **AI policy: AI-tutor.**
#
# **Reference state.** `U.day4_predictions()`: the champion's probabilities `p` for the cohort of 2024
# (`split == "validation"`) and of 2025 (`split == "holdout"`), each from a model trained only on programmes due
# to end before that year. `U.decision()`: capacity and costs.
#
# **Tasks**
# - [ ] 1 · Draw the precision-recall curve of the validation cohort (4')
# - [ ] 2 · Find the threshold that fits 800 support plans (6')
# - [ ] 3 · Compare with 0.5 and with the break-even probability (5')
# - [ ] 4 · Apply the threshold to the held-out cohort (6')
# - [ ] 5 · Decide: a fixed threshold or a fixed list length (4')
# - [ ] ✱ Stretch · An interval for the precision at the chosen threshold

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
from sklearn.metrics import precision_recall_curve

scored = U.day4_predictions()
valid = scored[scored.split == "validation"].reset_index(drop=True)
decision = U.decision()
CAPACITY = decision["adviser_capacity_per_year"]
print(f"validation cohort 2024: {len(valid):,} trainees, {valid[U.TARGET].mean():.1%} did not complete; capacity {CAPACITY} plans")


def at(cohort: pd.DataFrame, threshold: float, column: str = "p") -> dict:
    """The list a threshold produces: its length, its precision and its recall."""
    y, on_list = cohort[U.TARGET].to_numpy(), cohort[column].to_numpy() >= threshold
    found = int(y[on_list].sum())
    return {"threshold": float(threshold), "on the list": int(on_list.sum()), "found": found,
            "precision": found / max(1, on_list.sum()), "recall": found / y.sum()}


# %% [markdown]
# ## 1 · Draw the precision-recall curve of the validation cohort (4')

# %%
y = valid[U.TARGET].to_numpy()
precision, recall, thresholds = precision_recall_curve(y, valid.p)
plt = U.style()
fig, ax = plt.subplots()
ax.plot(recall, precision, color="#1f8a8a")
ax.axhline(y.mean(), color="grey", ls=":", lw=1)
ax.set(xlabel="recall: share of the non-completers on the list", ylabel="precision: share of the list that did not complete",
       title=f"Cohort 2024: average precision {U.ap(y, valid.p):.3f}; dotted: a random list")
U.savefig(fig, "D4_1_curve.png")

# %% [markdown]
# 🔍 **Look at** the curve from left to right. Every point is one threshold. A high threshold gives a short list
# that is mostly right; lowering it finds more non-completers and fills the list with trainees who would have
# completed anyway. No point is "the" answer: the advisers' capacity picks one.
#
# ## 2 · Find the threshold that fits 800 support plans (6')
#
# ✏️ **Exercise** · Return the threshold that puts exactly `k` trainees on the list: the `k`-th highest
# probability.

# %%
def threshold_for(p, k: int) -> float:
    # BEGIN SOLUTION
    return float(np.sort(np.asarray(p))[::-1][k - 1])
    # END SOLUTION


# %%
try:
    cut = threshold_for(valid.p, CAPACITY)
except NotImplementedError:
    print("↪ using the reference threshold")
    cut = float(np.sort(valid.p.to_numpy())[::-1][CAPACITY - 1])
capacity_list = at(valid, cut)
print({k: round(v, 3) for k, v in capacity_list.items()})
U.check(capacity_list["on the list"] == CAPACITY, f"the threshold puts {CAPACITY} trainees on the list")

# %% [markdown]
# 🔍 **Look at** precision and recall together. Of 800 plans, how many go to a trainee who would have left? Of the
# trainees who left, how many had a plan? Both sentences go to Ghassan, in those words.
#
# ## 3 · Compare with 0.5 and with the break-even probability (5')
#
# Two other thresholds are popular. 0.5, because libraries use it. And the break-even probability of Lab 1.1:
# a plan costs 450 riyals, a lost place 9,000, and a plan prevents one non-completion in four.

# %%
cost, loss, success = (decision["cost_of_one_support_plan_sar"], decision["cost_of_one_lost_place_sar"],
                       decision["share_of_supported_non_completers_who_then_complete"])
break_even = cost / (loss * success)
rows = {"fits capacity": capacity_list, "0.5": at(valid, 0.5), "break-even": at(valid, break_even),
        "class-weighted model at 0.5": at(valid, 0.5, "p_weighted")}
compare = pd.DataFrame(rows).T
compare["net value, thousand riyals"] = (compare.found * success * loss - compare["on the list"] * cost) / 1000
U.keep("d4_1_compare", compare)
print(f"break-even probability: {break_even:.2f}")
print(compare.round(3).to_string())

# %% [markdown]
# 🔍 **Look at** each row as a proposal to the fund.
#
# - 0.5 leaves more than 200 of the 800 plans unused. Nothing in the fund's decision says 0.5.
# - Break-even says every trainee above 0.20 is worth a plan: about twice the advisers' capacity. That is an
#   argument for Dr. Haya Al-Mogren about hiring advisers, with a number attached. It is not a list the advisers
#   can work through this year.
# - The last row is the champion trained with class weights, a common remedy for a rare outcome. Its ranking is
#   as good; its 0.5 lands somewhere else entirely. Weights move the scores, they do not choose a threshold.
#   Lab 4.2 returns to it.
#
# ## 4 · Apply the threshold to the held-out cohort (6')
#
# Now the cohort of 2025, for the first and only time. The threshold was fixed on 2024.

# %%
holdout = scored[scored.split == "holdout"].reset_index(drop=True)
print(f"holdout cohort 2025: {len(holdout):,} trainees, {holdout[U.TARGET].mean():.1%} did not complete")
fixed_threshold = at(holdout, cut)
fixed_length = at(holdout, float(np.sort(holdout.p.to_numpy())[::-1][CAPACITY - 1]))
on_2025 = pd.DataFrame({"2024, where it was chosen": capacity_list, "2025, same threshold": fixed_threshold,
                        "2025, the top 800": fixed_length}).T
U.keep("d4_1_holdout", on_2025)
print(on_2025.round(3).to_string())
U.check(fixed_length["on the list"] == CAPACITY, "the fixed-length list has 800 names")

# %% [markdown]
# 🔍 **Look at** the length of the list in the second row. The same threshold on a new cohort does not give 800
# names: the cohort is larger and its scores sit a little higher. The model did not get worse. A threshold on a
# probability and a capacity in plans are two different things.
#
# ## 5 · Decide: a fixed threshold or a fixed list length (4')
#
# ✏️ **Exercise** · Choose the rule the fund should run each intake, `"fixed threshold"` or `"fixed list length"`,
# and give your reason in one sentence.

# %%
def my_rule() -> tuple:
    # BEGIN SOLUTION
    return ("fixed list length",
            "The advisers' capacity is a number of plans, and a fixed threshold gave a different number of names on the next cohort.")
    # END SOLUTION


# %%
try:
    rule, reason = my_rule()
except NotImplementedError:
    rule, reason = "(your rule)", "(your reason)"
line = (f"A threshold of {cut:.2f} fits {CAPACITY} plans on the 2024 cohort: precision {capacity_list['precision']:.2f}, "
        f"recall {capacity_list['recall']:.2f}. On the 2025 cohort the same threshold lists {fixed_threshold['on the list']} "
        f"trainees; the top {CAPACITY} have precision {fixed_length['precision']:.2f} and recall {fixed_length['recall']:.2f}. "
        f"Rule: {rule}.")
print(line)
print("reason:", reason)
U.save("D4_1_threshold.md", "# Lab 4.1: a threshold that fits capacity\n\n" + U.md(compare) + "\n\n" + U.md(on_2025) + "\n\n"
       + line + "\n\nReason: " + reason + "\n")

# %% [markdown]
# Either rule can be defended. A fixed list length always fits the advisers and lets the weakest name on the list
# vary from intake to intake. A fixed threshold treats equal risk equally and overruns or underuses capacity.
# What cannot be defended is not saying which rule runs, and who may change it.
#
# ## ✱ Stretch · An interval for the precision at the chosen threshold
#
# The precision of the 2025 list is a share of 800 trainees. Put a 95% interval on it with `U.wilson(found, 800)`.
# Is the precision of 2024 inside it?

# %% [markdown]
# ## Checkpoint
#
# 1. Why is 0.5 not a threshold for this decision?
#    <details><summary>Answer</summary>Nothing in the decision produces it. The capacity gives 800 names, the costs
#    give a break-even of 0.20. 0.5 is a library default for a different question: which class is more
#    likely.</details>
# 2. Break-even lists twice as many trainees as the advisers can support. Who decides what happens with that?
#    <details><summary>Answer</summary>The owner of the budget, Dr. Haya Al-Mogren. The model supplies the number:
#    how much value the plans beyond capacity would return. It does not change the capacity.</details>
# 3. Recall at capacity is 0.38 on the 2025 cohort. Say it to an adviser without the word "recall".
#    <details><summary>Answer</summary>Of the trainees who went on to leave, about four in ten were on your list.
#    Six in ten were not, so a trainee who is not on the list may still need you.</details>
#
# **Carry forward.** `artifacts/D4_1_threshold.md`: the rule and its numbers, for the model card.
