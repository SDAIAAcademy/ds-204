"""Write data/predictions.csv.gz: the probabilities Day 4 works on (trainer tool; run once, after Lab 2.2).

    uv run python _tools/make_predictions.py

Two models are trained, each only on programmes that were due to end before the cohort it scores begins:

    split "validation": cohort 2024, scored by models trained for 1 January 2024
    split "holdout":    cohort 2025, scored by models trained for 1 January 2025

Three columns of probabilities:

    p           the champion (boosted trees with the settings of Lab 2.2)
    p_weighted  the same model trained with class weights, as many tutorials advise for a rare outcome
    p_logistic  the logistic rung with history (Lab 1.4)
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import sullam as U  # noqa: E402

table = U.modelling_table()
y = table[U.TARGET].to_numpy()
parts = []
for split, year in (("validation", 2024), ("holdout", 2025)):
    begins = pd.Timestamp(f"{year}-01-01")
    train = np.flatnonzero((table.planned_end < begins).to_numpy())
    score = np.flatnonzero((table.cohort_year == year).to_numpy())
    weight = float((y[train] == 0).sum() / (y[train] == 1).sum())
    models = {"p": U.champion(),
              "p_weighted": U.make_pipeline(U.boosted(**U.champion_params(), scale_pos_weight=weight), scale=False),
              "p_logistic": U.make_pipeline(U.logistic())}
    out = pd.DataFrame({"enrolment_id": table.enrolment_id.iloc[score].to_numpy(), "split": split})
    for name, model in models.items():
        model.fit(table.iloc[train][U.FEATURES], y[train])
        out[name] = np.round(U.predict(model, table.iloc[score][U.FEATURES]), 5)
    parts.append(out)
    print(f"{split}: trained on {len(train):,} rows, scored {len(score):,}; class weight {weight:.1f}; "
          f"average precision {U.ap(y[score], out.p):.3f}")
pd.concat(parts).to_csv(U.DATA / "predictions.csv.gz", index=False, lineterminator="\n",
                        compression={"method": "gzip", "mtime": 0})
print("wrote data/predictions.csv.gz")
