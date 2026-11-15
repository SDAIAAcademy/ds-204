# /// script
# requires-python = ">=3.11"
# dependencies = ["numpy>=2.0", "pandas>=2.2", "scipy>=1.14", "scikit-learn>=1.6", "lightgbm>=4.5", "matplotlib>=3.9", "tabulate>=0.9"]
# ///
# %% [markdown]
# # Lab 3.4 · A colleague's notebook
#
# Day 3 · Session 4 (Reading someone else's notebook) · 25 min.
#
# Loay Al-Kadi's message: "Balanced the classes, tuned the trees, 95% accuracy. Ready for the advisers?" His
# notebook is in `colleague/loay_notebook.py`, in six numbered blocks. You review it the way you would review a
# colleague's work: run it, read it in order, list what makes the number untrustworthy, then measure which
# fault matters most.
#
# **AI policy: AI-tutor.** Make your own list in task 2 before you ask an assistant for one.
#
# **Reference state.** `colleague/loay_notebook.py`, which has a switch for each block: `run(repair=["split"])`
# runs the notebook with that one block replaced by a sound version.
#
# **Tasks**
# - [ ] 1 · Run the notebook as it is; read the reported number (3')
# - [ ] 2 · Read the code and mark what you find (9')
# - [ ] 3 · Check your list against the reference (3')
# - [ ] 4 · Repair one fault at a time and rescore (6')
# - [ ] 5 · Write a five-line review note (4')
# - [ ] ✱ Stretch · Rewrite the notebook as one pipeline scored on the fund's folds

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
sys.path.insert(0, "colleague")
import loay_notebook as loay

# %% [markdown]
# ## 1 · Run the notebook as it is; read the reported number (3')

# %%
as_written = loay.run(quick=U.QUICK)
for k, v in as_written.items():
    print(f"{k:30s} {v:,.3f}" if isinstance(v, float) else f"{k:30s} {v:,}")
U.keep("d3_4_as_written", as_written)

# %% [markdown]
# 🔍 **Look at** the rows beside the accuracy before you read any code. How many test rows are there, when the
# fund has 45,000 enrolments and a fifth was set aside? What share of them are non-completers, when the fund's
# rate is one in six? Two numbers already do not fit.
#
# ## 2 · Read the code and mark what you find (9')
#
# Read the script in the order of a review: **rows, split, preparation, tuning, metric**. The blocks marked
# "as written" are Loay's.

# %%
source = Path("colleague/loay_notebook.py").read_text(encoding="utf-8").splitlines()
start = next(i for i, line in enumerate(source) if line.startswith("def run"))
for i, line in enumerate(source[start:], start + 1):
    print(f"{i:3d}  {line}")

# %% [markdown]
# ✏️ **Exercise** · For each of the six blocks, name the fault. Use one of: `"population"` (rows the model will
# never score, or must not see yet), `"group"` (the same row or person on both sides), `"future"`,
# `"preparation"` (fitted or chosen on rows that score the model), `"metric"` (a number that cannot mean what
# the message says), or `"none"`.

# %%
def my_review() -> dict:
    # BEGIN SOLUTION
    return {1: "population", 2: "group", 3: "preparation", 4: "future", 5: "preparation", 6: "metric"}
    # END SOLUTION


# %% [markdown]
# ## 3 · Check your list against the reference (3')

# %%
REVIEW = {
    1: (("population",), "rows", "Every enrolment on file: trainees who left before week 2, whom Mubakkir never scores, "
        "and the 2025 cohort, which is the holdout."),
    2: (("group", "preparation"), "balance", "Non-completers are copied before the split, so copies of one row sit in "
        "training and in test. The test set is also half non-completers, which no cohort is."),
    3: (("preparation",), "encode", "The provider's rate is computed over all rows, test rows and their outcomes included."),
    4: (("future", "group"), "split", "A random split: the model trains on later enrolments than it is tested on, on "
        "programmes still running, and trainees repeat on both sides."),
    5: (("preparation",), "tune", "Six settings are compared on the test set and the best test score is the one reported."),
    6: (("metric",), "report", "Accuracy at 0.5 on a test set that was balanced by copying. On real cohorts, "
        "\"everyone completes\" already scores about 0.87."),
}
try:
    mine = my_review()
except NotImplementedError:
    print("↪ task 2 is not done: the reference review follows")
    mine = {}
agree = 0
for block, (accepted, switch, text) in REVIEW.items():
    said = mine.get(block, "")
    agree += said in accepted
    print(f"block {block} ({switch}): {' or '.join(accepted)}" + (f" · you said {said}" if said else ""))
    print("   " + text)
if mine:
    print(f"\nyour review agrees with the reference on {agree} of 6 blocks")
    U.check(agree >= 5, "at least five of six blocks match the reference review")

# %% [markdown]
# 🔍 **Look at** block 4. Yesterday you might have called the random split the worst fault. Lab 3.1 measured it
# for the champion: no gap. Whether it matters **here** is a question for the next task, not for an opinion.
#
# ## 4 · Repair one fault at a time and rescore (6')
#
# ✏️ **Exercise** · Run the notebook with one block repaired at a time, then with all five. `loay.run(repair=[name],
# quick=U.QUICK)` returns the same dictionary as in task 1. Collect the results in `results`, a dictionary from
# the name of the run to that dictionary.

# %%
SWITCHES = ["rows", "balance", "encode", "split", "tune"]


def repair_one_at_a_time() -> dict:
    results = {"as written": as_written}
    # BEGIN SOLUTION
    for name in SWITCHES:
        results["repair " + name] = loay.run(repair=[name], quick=U.QUICK)
    results["all five repaired"] = loay.run(repair=SWITCHES, quick=U.QUICK)
    # END SOLUTION
    return results


# %%
try:
    repairs = pd.DataFrame(repair_one_at_a_time()).T
    U.keep("d3_4_repairs", repairs)
except NotImplementedError:
    repairs = U.reference("d3_4_repairs")
repairs["above always-completes"] = repairs.accuracy - repairs.accuracy_of_always_completes
print(repairs.round(3).to_string())
if not U.QUICK:
    U.check(repairs.accuracy["all five repaired"] < repairs.accuracy["as written"], "the repaired notebook reports a lower accuracy")

# %% [markdown]
# 🔍 **Look at** three things.
#
# - Which single repair moves the reported accuracy most? Copying rows before the split does almost all the
#   damage: a tree with five rows per leaf recognises the copies.
# - Repairing the rows alone makes the number go **up**. Faults interact, which is why they are repaired one at
#   a time and then together.
# - In the last line, compare the accuracy with the accuracy of "everyone completes". What is left of 95%?
#
# The split matters here although it did not in Lab 3.1: this model has leaves of five rows and a table full of
# copies. A fault is measured for the model in front of you.
#
# ## 5 · Write a five-line review note (4')
#
# ✏️ **Exercise** · Write the review note: five lines Loay can act on. The number as reported, the fault that
# matters most with its measured effect, the other faults, the honest number, and what you ask him to do next.

# %%
def my_note(repairs: pd.DataFrame) -> list:
    # BEGIN SOLUTION
    a, b, f = repairs.loc["as written"], repairs.loc["repair balance"], repairs.loc["all five repaired"]
    return [
        f"Reported: accuracy {a.accuracy:.2f}, on a test set that is {a.non_completers_in_test:.0%} non-completers because rows were copied before the split.",
        f"Largest fault: the copies sit on both sides of the split. Balancing nothing, the accuracy is {b.accuracy:.2f} where \"everyone completes\" scores {b.accuracy_of_always_completes:.2f}.",
        "Also: early leavers and the 2025 holdout are in the rows; the provider rate and the choice of settings both use test rows; the split is random.",
        f"With all five repaired: average precision {f.average_precision:.2f}, AUC {f.auc:.2f}, on 2024 H2. Accuracy {f.accuracy:.2f} against {f.accuracy_of_always_completes:.2f} for a constant answer.",
        "Next: rebuild it as one pipeline on the fund's folds, log it beside the ladder, and report average precision and recall at capacity.",
    ]
    # END SOLUTION


try:
    note = my_note(repairs)
except NotImplementedError:
    note = ["(your five lines)"]
for line in note:
    print("-", line)
U.save("D3_4_review.md", "# Lab 3.4: review of a colleague's notebook\n\n" + U.md(repairs) + "\n\n" + "\n".join("- " + n for n in note) + "\n")

# %% [markdown]
# A review note names numbers, not character. "Your split is wrong" starts an argument. "With the copies kept
# out of the test set the accuracy is 0.87, and a constant answer scores 0.84" ends one.
#
# ## ✱ Stretch · Rewrite the notebook as one pipeline scored on the fund's folds
#
# Rewrite the notebook as one pipeline: `U.make_pipeline(U.boosted(num_leaves=15, min_child_samples=5))` on the
# scorable rows, scored with `U.score_folds` on the fund's folds. Where does it land on the ladder of Day 2?

# %% [markdown]
# ## Checkpoint
#
# 1. Why is 95% accuracy impossible to interpret before reading any code?
#    <details><summary>Answer</summary>Accuracy depends on the share of the classes. With one non-completer in
#    six, "everyone completes" is already 83% accurate. And here the test set had been balanced by copying, so
#    the number describes no real cohort.</details>
# 2. Loay says: "But balancing the classes is standard practice." What do you answer?
#    <details><summary>Answer</summary>If classes are balanced at all, it is done inside the training part only.
#    The test set keeps the real share. Day 4 shows that for a ranking it is not needed here.</details>
# 3. In which order do you read a notebook you did not write?
#    <details><summary>Answer</summary>Rows, split, preparation, tuning, metric. The earlier the fault, the more
#    of what follows it spoils.</details>
#
# **Carry forward.** `artifacts/D3_4_review.md`. The same five lines are what an assessor will write about your
# capstone repository.
