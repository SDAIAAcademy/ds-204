# /// script
# requires-python = ">=3.11"
# dependencies = ["numpy>=2.0", "pandas>=2.2", "scipy>=1.14", "scikit-learn>=1.6", "lightgbm>=4.5", "matplotlib>=3.9", "tabulate>=0.9"]
# ///
# %% [markdown]
# # Setup check
#
# Run every cell once before Day 1. It imports the libraries of the labs, reads the data of Sullam Vocational
# Training Fund and fits one small model. Nothing here needs a key, a graphics processor or a model download.

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

# %%
import importlib

ok = True
for name in ["numpy", "pandas", "scipy", "sklearn", "lightgbm", "matplotlib", "tabulate", "autogluon.tabular"]:
    try:
        module = importlib.import_module(name)
        print(f"✅ {name} {getattr(module, '__version__', '')}")
    except Exception as error:                       # the challenger of Lab 2.3 is the only optional import
        ok = ok and name == "autogluon.tabular"
        print(f"❌ {name}: {error}")
print(f"Python {sys.version.split()[0]}, mode: {U.MODE}")

# %%
raw = U.enrolments()
table = U.modelling_table()
dev, holdout = U.dev_holdout(table)
folds = U.folds(dev)
U.check(len(raw) == 45_000 and raw.trainee_id.nunique() == 30_000, "the export has 45,000 enrolments of 30,000 trainees")
U.check(len(folds) == 5 and all(dev.block.iloc[tr].max() < dev.block.iloc[va].min() for tr, va in folds),
        "the five folds validate on a later half-year than they train on")
U.check(len(U.reference_log()) >= 7, "the reference run log is there")
U.check(len(U.day4_predictions()) > 19_000, "the Day 4 predictions are there")

# %%
tr, va = folds[0]
model = U.make_pipeline(U.logistic()).fit(dev.iloc[tr][U.FEATURES], dev.iloc[tr][U.TARGET])
score = U.metrics(dev.iloc[va][U.TARGET], U.predict(model, dev.iloc[va][U.FEATURES]))["ap"]
U.check(score > 0.4, f"a logistic model fits and scores on fold 1 (average precision {score:.3f})")
U.check(ok, "every required library imports")
print("Ready." if ok else "Something is missing: run `uv sync` in labs/ (or the first cell again in Colab).")
