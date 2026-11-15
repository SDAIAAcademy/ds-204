# /// script
# requires-python = ">=3.11"
# dependencies = ["numpy>=2.0", "pandas>=2.2", "scipy>=1.14", "scikit-learn>=1.6", "lightgbm>=4.5", "matplotlib>=3.9", "tabulate>=0.9"]
# ///
# %% [markdown]
# # Lab 4.4 · The model card and the capstone problem
#
# Day 4 · Session 4 (The model card) · 25 min.
#
# Mubakkir leaves your hands today. What travels with it is one page: what the model is for, how it was
# validated, where it fails and what it must never decide. You assemble that page from the numbers of the four
# days, write the parts no number can write, and then frame the problem you will take through the same path in
# the capstone.
#
# **AI policy: AI-tutor.** The limits and the human-decision statement are yours.
#
# **Reference state.** The reference results of Labs 2.2 to 4.3 in `data/reference/`, `U.decision()` and
# `capstone/shortlist.json`.
#
# **Tasks**
# - [ ] 1 · Collect the measured numbers (4')
# - [ ] 2 · Fill the card: use, data, validation, results, subgroups (8')
# - [ ] 3 · Write the limits and the human-decision statement (5')
# - [ ] 4 · List what the deployment team needs (3')
# - [ ] 5 · Choose the capstone problem and write its framing (5')
# - [ ] ✱ Stretch · Exchange cards with a neighbour: one question each

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
def result(name: str):
    """A reference result of an earlier lab, read quietly."""
    return U._unjson(U.js("reference/" + name))


# %% [markdown]
# ## 1 · Collect the measured numbers (4')

# %%
log = U.reference_log().set_index("rung")
three, by_age, kinds = result("d3_1_three"), result("d3_1_by_age"), result("d3_1_kinds")
lists, calibration = result("d4_1_holdout"), result("d4_2_holdout").set_index(["model", "calibration"])
new_providers, registers, represented = result("d4_2_new_providers"), result("d4_3_registers"), result("d4_3_represented")
versus_logistic, decision = result("d2_4_vs_logistic"), U.decision()
top = lists.loc["2025, the top 800"]
delivered = calibration.loc[("boosted, as delivered", "as delivered")]
N = {
    "ladder": ", ".join(f"{rung} {log.mean_ap[rung]:.3f}" for rung in ("constant", "three rules", "logistic + history", "boosted, 20 trials")
                        if rung in log.index),
    "champion": log.mean_ap["boosted, 20 trials"], "spread": log.spread["boosted, 20 trials"],
    "vs_logistic": f"{versus_logistic['mean']:+.3f} (95% interval {versus_logistic['low']:+.3f} to {versus_logistic['high']:+.3f})",
    "random": three.mean_ap["random"], "time": three.mean_ap.iloc[-1], "hidden": three.mean_ap["grouped, history hidden"],
    "one_year": by_age["mean"].iloc[2],
    "ap_new": kinds.ap["new to the fund"], "ap_returning": kinds.ap["returning"], "share_new": kinds.share["new to the fund"],
    "precision": top.precision, "recall": top.recall, "holdout_ap": delivered["average precision"],
    "expected": delivered["mean expected"], "observed": delivered.observed,
    "new_expected": new_providers.expected.iloc[1], "new_observed": new_providers.observed.iloc[1],
    "paper_miss": registers.miss_rate.iloc[1], "paper_low": registers.miss_low.iloc[1], "paper_high": registers.miss_high.iloc[1],
    "other_miss": registers.miss_rate.iloc[0], "older_ratio": represented.ratio["35+"], "women_ratio": represented.ratio["F"],
    "capacity": decision["adviser_capacity_per_year"],
}
for k, v in N.items():
    print(f"{k:14s} {v:.3f}" if isinstance(v, float) else f"{k:14s} {v}")

# %% [markdown]
# 🔍 **Look at** the list as a reader who was not in the room. Which of these numbers would you want first? Which
# would you least like to see left out?
#
# ## 2 · Fill the card: use, data, validation, results, subgroups (8')
#
# The measured sections of the card are filled from the numbers above. Read each sentence against its source:
# a card is only as good as the reader's ability to trace it.

# %%
MEASURED = f"""## Intended use
Mubakkir ranks the trainees of Sullam Vocational Training Fund who are still enrolled at the end of week 2 by
their risk of not completing. In week 3 the advisers led by Ghassan Al-Fayez contact the {N['capacity']} highest-ranked
trainees of the year and agree a support plan.

## Data
45,000 enrolments of 30,000 trainees, cohorts 2021 to 2025, 60 providers; synthetic. Rows: trainees still
enrolled at the end of week 2 (42,556). Features: 15 columns known at the end of week 2 and two history counts
built from past records only. Gender is recorded and is not a feature; it is used to audit errors.

## Validation
Five folds forward in time: each validates on one half-year and trains on programmes that were due to end before
it. Cohort 2025 was held out and scored once. Measured once on the development cohorts: a random split {N['random']:.3f},
the time-ordered folds {N['time']:.3f}; for trainees with no history {N['hidden']:.3f}. A leakage audit removed four
columns or row sets from a received table (Lab 3.3).

## Results
Average precision on the folds: {N['ladder']}. Champion {N['champion']:.3f} (fold spread {N['spread']:.3f}); against the
logistic rung {N['vs_logistic']}: an advantage is not shown. On cohort 2025: average precision {N['holdout_ap']:.3f};
the list of {N['capacity']} has precision {N['precision']:.2f} and recall {N['recall']:.2f}. Probabilities: expected {N['expected']:.1%},
observed {N['observed']:.1%}; no calibrator is applied.

## Subgroups (cohort 2025, list of {N['capacity']})
Of the trainees who left, {N['other_miss']:.0%} were not on the list where registers are electronic and {N['paper_miss']:.0%}
(95% interval {N['paper_low']:.0%} to {N['paper_high']:.0%}) in Jazan and Najran, where early absences are recorded on paper.
New trainees: average precision {N['ap_new']:.3f}; returning trainees: {N['ap_returning']:.3f}. Trainees of 35 and over are
{N['older_ratio']:.2f} times as present in the training data as in the eligible population; women {N['women_ratio']:.2f} times.
"""
print(MEASURED)

# %% [markdown]
# ## 3 · Write the limits and the human-decision statement (5')
#
# ✏️ **Exercise** · Write the two parts no number writes for you.
#
# - `limits`: at least four sentences. Each names a situation in which the scores above do **not** hold, or a
#   use the model is not for.
# - `human`: one or two sentences that say who decides, and what the score may never decide.

# %%
def my_limits() -> list:
    # BEGIN SOLUTION
    return [
        "Not for decisions about eligibility, funding, a place on a programme or a provider's contract.",
        "Not for trainees in their first two weeks: they have no early-absence record and were never in the validation.",
        f"At providers that opened after training, the probabilities are about {N['new_observed'] - N['new_expected']:.0%} too low: use the rank, not the number.",
        "In Jazan and Najran, and wherever attendance is kept on paper, a trainee who is not on the list is not thereby low-risk.",
        f"A model one year old scored {-N['one_year']:.3f} lower than a fresh one: retrain every half-year, on programmes due to end.",
        "The validation says nothing about another country, another fund or programmes aimed at trainees of 35 and over.",
        "The score says who is likely to leave. It does not say whether a support plan helps: that needs a comparison the fund has not run.",
    ]
    # END SOLUTION


def my_human_statement() -> str:
    # BEGIN SOLUTION
    return ("The score puts a name on an adviser's list. The adviser decides whether and how to contact the trainee, may add any "
            "trainee who is not on the list, and records the reason. No trainee is refused, ranked for funding or told a score.")
    # END SOLUTION


# %%
try:
    limits, human = my_limits(), my_human_statement()
    U.check(len(limits) >= 4, "the card states at least four limits")
    U.check(len(human) > 40, "the card says who decides")
except NotImplementedError:
    limits, human = ["(your limits)"], "(your statement)"

# %% [markdown]
# ## 4 · List what the deployment team needs (3')
#
# Mubakkir will be served and monitored by another team. The list below is what they cannot guess. Complete it
# with anything your limits imply.

# %%
HANDOVER = [
    "The code version, the seed and the lock file that produced the delivered model",
    "The date of training and the rule for training rows: programmes due to end before the scoring period",
    "The features, when each is recorded, and the two history counts with their time-travel test",
    "The list rule: the top 800 of the year, split over the intakes; who may change it",
    "What to monitor each intake: list precision once outcomes arrive, expected against observed, miss rate by region, gender and age band",
    "When to retrain: every half-year; and when to stop using the model: a new register system, a new kind of programme",
]
for item in HANDOVER:
    print("- [ ]", item)

# %%
card = (f"# Model card: Mubakkir\n\nSullam Vocational Training Fund · owner of the decision: Dr. Haya Al-Mogren · "
        f"prepared in DS-204. The fund and its data are invented for teaching.\n\n" + MEASURED
        + "\n## Limits\n" + "\n".join("- " + s for s in limits) + "\n\n## The human decision\n" + human
        + "\n\n## For the deployment team\n" + "\n".join("- " + s for s in HANDOVER) + "\n")
U.save("model_card.md", card)

# %% [markdown]
# 🔍 **Look at** the card as Maram Al-Rasheed, the fund's data protection officer, would. Does it say which
# personal data the model uses, and which it leaves out? As Ghassan Al-Fayez: does it tell an adviser what a
# missing name means?
#
# ## 5 · Choose the capstone problem and write its framing (5')
#
# The capstone starts here. Read the shortlist, choose one problem and frame it before you train anything.

# %%
shortlist = U.js("../capstone/shortlist")
for p in shortlist["problems"]:
    print(f"{p['id']:13s} {p['kind']:10s} {p['title']}")
    print(f"{'':13s} {p['decision']}")
print("\n" + shortlist["own_problem"])

# %% [markdown]
# ✏️ **Exercise** · Fill in the framing of your problem. Every line is a sentence you will defend in the
# eight minutes.

# %%
def my_framing() -> dict:
    # BEGIN SOLUTION
    return {
        "problem": "bayader",
        "decision and owner": "The head of field operations sends officers to 600 new borrowers a year before the first instalment.",
        "unit and prediction time": "One loan, scored on the approval date; columns recorded during repayment are not used.",
        "target": "missed_repayment: a missed repayment in the first six months, about one loan in nine.",
        "validation scheme": "Forward in time by half-year, training on loans whose first six months had passed; 2025 held out.",
        "metric": "Average precision, and recall among the 600 highest-ranked loans.",
        "first suspects": "collection_calls (written during repayment) and the two columns the previous analyst computed.",
    }
    # END SOLUTION


# %%
try:
    framing = my_framing()
    known = {p["id"] for p in shortlist["problems"]} | {"own"}
    U.check(framing["problem"] in known, "the problem is on the shortlist (or 'own', approved by the trainer)")
    U.check(all(len(str(v)) > 3 for v in framing.values()), "every line of the framing is filled")
except NotImplementedError:
    framing = {"problem": "(your choice)"}
U.save("capstone_framing.md", "# Capstone framing\n\n" + "\n".join(f"- **{k}:** {v}" for k, v in framing.items()) + "\n")

# %% [markdown]
# ## ✱ Stretch · x
#
# Give your card to a neighbour and take theirs. Ask one question the card does not answer. Add the answer to
# your own card.

# %% [markdown]
# ## Checkpoint
#
# 1. Which sentence of the card would you keep if only one were allowed?
#    <details><summary>Answer</summary>The human decision: what the score may never decide. Every other sentence
#    helps someone use the model well; that one prevents its worst use.</details>
# 2. The card says an advantage over the logistic model "is not shown". Why write that down?
#    <details><summary>Answer</summary>The next team may need a model an adviser can read, or one that is cheaper
#    to run. The card tells them a simpler model does as well within the uncertainty, with the interval.</details>
# 3. What makes a limit useful?
#    <details><summary>Answer</summary>It names a situation a reader can recognise, and what to do in it: "at a
#    provider that opened after training, use the rank, not the number".</details>
#
# **Carry forward.** `artifacts/model_card.md` and `artifacts/capstone_framing.md`. The first is the pattern for
# your capstone's card; the second is the first section of its README.
