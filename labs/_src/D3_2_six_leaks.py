# /// script
# requires-python = ">=3.11"
# dependencies = ["numpy>=2.0", "pandas>=2.2", "scipy>=1.14", "scikit-learn>=1.6", "lightgbm>=4.5", "matplotlib>=3.9", "tabulate>=0.9"]
# ///
# %% [markdown]
# # Lab 3.2 · Six leaks from other projects
#
# Day 3 · Session 2 (Kinds of leakage) · 25 min.
#
# Six projects from other sectors, each with a score that looked good and a fault in how it was produced. For
# each you name the kind of leak and the validation scheme the use of the model requires. There is no model to
# train: the skill is reading how the data were collected.
#
# **AI policy: AI-tutor.** Decide each case yourself before you ask an assistant to explain one.
#
# **Reference state.** `data/six_cases.json` (the cases) and `data/reference/six_cases_key.json` (the checked
# table, read only in task 4).
#
# **Tasks**
# - [ ] 1 · Read the six cases and how their data were collected (5')
# - [ ] 2 · Name the kind of leak in each (8')
# - [ ] 3 · Name the validation scheme each needs (6')
# - [ ] 4 · Check the table; read the explanations (3')
# - [ ] 5 · Write the question that would have caught each (3')
# - [ ] ✱ Stretch · Add a seventh case from your own work

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
import textwrap

book = U.js("six_cases")
cases = book["cases"]
KINDS, SCHEMES = book["kinds"], book["schemes"]

# %% [markdown]
# ## 1 · Read the six cases and how their data were collected (5')

# %%
for c in cases:
    print(f"CASE {c['id']} · {c['title']}")
    for part in ("problem", "collection"):
        print(textwrap.fill(c[part], 110, initial_indent="  ", subsequent_indent="  "))
    print(f"  Split used: {c['split_used']}\n")

# %% [markdown]
# 🔍 **Look at** each case with three questions. When is each column written? What is the unit the model meets in
# use: a new row, a new person or machine, a later period? Was anything fitted before the split?
#
# ## 2 · Name the kind of leak in each (8')
#
# The four kinds:

# %%
for k, v in KINDS.items():
    print(f"{k:12s} {v}")

# %% [markdown]
# ✏️ **Exercise** · Return your answer for each case: one of `"target"`, `"future"`, `"group"`, `"preparation"`.

# %%
def my_kinds() -> dict:
    # BEGIN SOLUTION
    return {1: "target", 2: "group", 3: "preparation", 4: "future", 5: "group", 6: "target"}
    # END SOLUTION


# %% [markdown]
# ## 3 · Name the validation scheme each needs (6')
#
# The schemes:

# %%
for k, v in SCHEMES.items():
    print(f"{k:13s} {v}")

# %% [markdown]
# ✏️ **Exercise** · Return the scheme each case needs: `"random"`, `"grouped"`, `"time"` or `"grouped+time"`.
# The scheme follows from how the model will be **used**, which each case states in its last sentence.

# %%
def my_schemes() -> dict:
    # BEGIN SOLUTION
    return {1: "time", 2: "grouped", 3: "random", 4: "time", 5: "grouped", 6: "grouped"}
    # END SOLUTION


# %% [markdown]
# ## 4 · Check the table; read the explanations (3')

# %%
key = {int(k): v for k, v in U.js("reference/six_cases_key").items()}
try:
    kinds, schemes = my_kinds(), my_schemes()
except NotImplementedError:
    print("↪ tasks 2 and 3 are not done: the table below shows the reference answers only")
    kinds, schemes = {}, {}
table = pd.DataFrame([{"case": c["id"], "title": c["title"], "my kind": kinds.get(c["id"], ""),
                       "kind": key[c["id"]]["kind"], "my scheme": schemes.get(c["id"], ""),
                       "scheme": key[c["id"]]["scheme"]} for c in cases]).set_index("case")
table["kind ok"] = table["my kind"] == table["kind"]
table["scheme ok"] = table["my scheme"] == table["scheme"]
print(table.drop(columns="title").to_string())
print(f"\nkinds right: {table['kind ok'].sum()} of 6 · schemes right: {table['scheme ok'].sum()} of 6\n")
for c in cases:
    k = key[c["id"]]
    print(f"CASE {c['id']} · {c['title']}: {k['kind']}, {k['scheme']}")
    print(textwrap.fill(k["why_kind"], 110, initial_indent="  ", subsequent_indent="  "))
    print(textwrap.fill(k["why_scheme"], 110, initial_indent="  ", subsequent_indent="  "))
if kinds:
    U.check(table["kind ok"].sum() >= 5, "at least five of six kinds match the reference")
    U.check(table["scheme ok"].sum() >= 5, "at least five of six schemes match the reference")
U.save("D3_2_six_leaks.md", "# Lab 3.2: six leaks from other projects\n\n" + table.to_markdown() + "\n")

# %% [markdown]
# 🔍 **Look at** the cases you got wrong, and at case 3. A random split is not a fault in itself: it is the right
# scheme when rows are independent and the model is used on rows like them. The fault there was a step done
# before the split.
#
# Where you disagree with the reference, write your reason: two kinds can be present in one case (case 6 is
# also future information, because the column covers the whole stay). The table asks for the main one.
#
# ## 5 · Write the question that would have caught each (3')
#
# ✏️ **Exercise** · For each case, write the one question that would have caught the leak before any model was
# trained. Then compare with the reference questions.

# %%
my_questions = {
    1: "",
    2: "",
    3: "",
    4: "",
    5: "",
    6: "",
}
for c in cases:
    print(f"case {c['id']}: {key[c['id']]['question']}")

# %% [markdown]
# These six questions are the start of the leakage checklist in `followup/leakage_checklist.md`. You use it on
# the fund's own table in Lab 3.3 and in the capstone.
#
# ## ✱ Stretch · x
#
# Describe a case from your own work in the same form: the problem, how the data were collected, the split that
# was used. Give it to a neighbour to classify.

# %% [markdown]
# ## Checkpoint
#
# 1. In case 2, would a time-ordered split have been enough?
#    <details><summary>Answer</summary>No. Every pump is in both the early and the late months, so the model would
#    still be scored on pumps it has seen. The use is new pumps: the split is by pump.</details>
# 2. In case 3 the analyst moves the selection of the 20 measurements inside each fold. The accuracy falls from
#    0.96 to 0.88. Which number describes the model?
#    <details><summary>Answer</summary>0.88. The selection is part of the model and has to be fitted on training
#    rows only, like a scaler.</details>
# 3. Why does the scheme follow from the use and not from the data alone?
#    <details><summary>Answer</summary>The same table can serve two uses. The appointments of case 4 need a
#    time-ordered split for known patients next month, and a grouped one if the model were for a new clinic's
#    patients.</details>
#
# **Carry forward.** `artifacts/D3_2_six_leaks.md`: your classification table, checked. It is the evidence for
# the first course outcome.
