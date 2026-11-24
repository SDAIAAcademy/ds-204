"""The two worked examples of k-means and scaling (trainer tool): the numbers on the Day 1 slides
"segments-kmeans" and "kmeans-units", and the practice question "q-ladder-kmeans-units".

    uv run python _tools/segments_example.py

Both tables are synthetic, written with a fixed seed, and separate from the enrolment data the labs use.

1. Employers that sponsor trainees of the fund (slides): 240 employers in three planted kinds, which differ in
   how many trainees they sponsor a year and in the share who complete.
   Staff numbers (about 600, give or take 180) are unrelated to the kind.
2. Members of a library network (practice question): yearly loans and the share of loans that are e-books.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score
from sklearn.preprocessing import StandardScaler

OUT = Path(__file__).resolve().parent / "segments_example.json"
rng = np.random.default_rng(2041)


def kmeans(X: np.ndarray, k: int) -> np.ndarray:
    return KMeans(n_clusters=k, n_init=10, random_state=0).fit_predict(X)


def profile(frame: pd.DataFrame, labels: np.ndarray, order_by: str) -> pd.DataFrame:
    p = frame.assign(n=1).groupby(labels).agg({**{c: "mean" for c in frame.columns}, "n": "sum"})
    return p.sort_values(order_by).reset_index(drop=True)


# 1. Employers ---------------------------------------------------------------------------------------------------
kinds = rng.choice(3, size=240, p=[0.45, 0.35, 0.20])
trainees = np.array([3, 14, 28])[kinds] + rng.normal(0, 1.5, 240)
completed = np.array([0.90, 0.73, 0.56])[kinds] + rng.normal(0, 0.03, 240)
staff = rng.normal(600, 180, 240).clip(60)
employers = pd.DataFrame({"staff": staff.round(), "trainees_per_year": trainees.clip(1).round(),
                          "completed_share": completed.clip(0.3, 1.0).round(3)})

raw = kmeans(employers.to_numpy(), 3)
scaled = kmeans(StandardScaler().fit_transform(employers), 3)
var_share = (employers.var() / employers.var().sum())
# the first pair of a kind-0 and a kind-2 employer whose staff differ by 200 to 400
ia = next(i for i in range(240) if kinds[i] == 0)
ib = next(i for i in range(240) if kinds[i] == 2 and 200 <= abs(staff[i] - staff[ia]) <= 400)
a, b = employers.iloc[ia], employers.iloc[ib]
pair = {"a": a.to_dict(), "b": b.to_dict(),
        "squared_differences": ((a - b) ** 2).to_dict(),
        "scaled_squared_differences": (((a - b) / employers.std()) ** 2).to_dict()}
emp = {"n": len(employers), "describe": employers.describe().round(3).to_dict(),
       "variance_share": var_share.round(6).to_dict(),
       "pair": pair,
       "unscaled": profile(employers, raw, "staff").round(3).to_dict(orient="records"),
       "scaled": profile(employers, scaled, "completed_share").round(3).to_dict(orient="records"),
       "agreement_with_kinds": {"unscaled": round(adjusted_rand_score(kinds, raw), 3),
                                "scaled": round(adjusted_rand_score(kinds, scaled), 3)}}

# 2. Library members ---------------------------------------------------------------------------------------------
kind = rng.choice(2, size=900, p=[0.6, 0.4])
loans = np.exp(rng.normal(np.log(60), 0.8, 900)).round()
ebook = (np.array([0.15, 0.70])[kind] + rng.normal(0, 0.08, 900)).clip(0, 1).round(2)
members = pd.DataFrame({"loans_per_year": loans, "ebook_share": ebook})
m_raw = kmeans(members.to_numpy(), 3)
m_scaled = kmeans(StandardScaler().fit_transform(members), 3)
lib = {"n": len(members), "describe": members.describe().round(3).to_dict(),
       "unscaled": profile(members, m_raw, "loans_per_year").round(2).to_dict(orient="records"),
       "scaled": profile(members, m_scaled, "ebook_share").round(2).to_dict(orient="records"),
       "agreement_with_kinds": {"unscaled": round(adjusted_rand_score(kind, m_raw), 3),
                                "scaled": round(adjusted_rand_score(kind, m_scaled), 3)}}

result = {"employers": emp, "library_members": lib}
OUT.write_text(json.dumps(result, indent=1) + "\n", encoding="utf-8", newline="\n")
print(json.dumps(result, indent=1))
