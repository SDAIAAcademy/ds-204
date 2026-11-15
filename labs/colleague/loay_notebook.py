"""Loay Al-Kadi's notebook "Mubakkir v0", exported as a script for Lab 3.4.

His message: "Balanced the classes, tuned the trees, 95% accuracy. Ready for the advisers?"

Read it as it is. Do not correct it here: Lab 3.4 runs it through run(repair=...), where each name in `repair`
switches one block to a sound version, so that the effect of every fault can be measured on its own.
The blocks marked "as written" are Loay's. The blocks marked "repaired" were added by the trainer.
"""
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, average_precision_score, roc_auc_score
from sklearn.model_selection import train_test_split

import sullam as U

FEATURES = U.BASE + ["provider_rate"]
GRID = [{"num_leaves": leaves, "n_estimators": trees, "min_child_samples": 5}
        for leaves in (15, 63, 255) for trees in (100, 300)]
CUT = pd.Timestamp("2024-07-01")          # used by the repaired split only


def oversample(frame: pd.DataFrame) -> pd.DataFrame:
    """Copy the non-completers until the two classes have the same size."""
    left = frame[frame[U.TARGET] == 1]
    times = int(round((frame[U.TARGET] == 0).sum() / len(left))) - 1
    return pd.concat([frame] + [left] * times).sample(frac=1.0, random_state=1)


def fit(train: pd.DataFrame, settings: dict):
    model = U.make_pipeline(U.boosted(**settings), FEATURES, scale=False)
    return model.fit(train[FEATURES], train[U.TARGET])


def run(repair=(), quick: bool = False) -> dict:
    repair = set(repair)
    grid = GRID[:2] if quick else GRID

    # ---- 1 · rows -------------------------------------------------------------------------------------------
    if "rows" in repair:                                   # repaired: the rows Mubakkir can score, 2021 to 2024
        df, _ = U.dev_holdout(U.scorable())
    else:                                                  # as written: every enrolment on file
        df = U._calendar(U.enrolments())

    # ---- 2 · balance the classes ----------------------------------------------------------------------------
    if "balance" not in repair:                            # as written: before the split
        df = oversample(df)

    # ---- 3 · encode the provider by its non-completion rate -------------------------------------------------
    if "encode" not in repair:                             # as written: the rate over all rows
        df["provider_rate"] = df.groupby("provider_id")[U.TARGET].transform("mean")

    # ---- 4 · split ------------------------------------------------------------------------------------------
    if "split" in repair:                                  # repaired: train on the past, test on 2024 H2
        train = df[(df.planned_end < CUT)]
        test = df[(df.start_date >= CUT) & (df.cohort_year == 2024)]
    else:                                                  # as written
        train, test = train_test_split(df, test_size=0.2, random_state=1)
    train, test = train.copy(), test.copy()
    if "encode" in repair:                                 # repaired: the rate from training rows only
        rate = train.groupby("provider_id")[U.TARGET].mean()
        train["provider_rate"] = train.provider_id.map(rate)
        test["provider_rate"] = test.provider_id.map(rate).fillna(train[U.TARGET].mean())

    # ---- 5 · tune -------------------------------------------------------------------------------------------
    if "tune" in repair:                                   # repaired: choose on a part of the training rows
        inner_train, inner_valid = train_test_split(train, test_size=0.25, random_state=1)
        scores = [roc_auc_score(inner_valid[U.TARGET], U.predict(fit(inner_train, s), inner_valid[FEATURES])) for s in grid]
        model = fit(train, grid[int(np.argmax(scores))])
    else:                                                  # as written: keep the setting with the best test score
        models = [fit(train, s) for s in grid]
        scores = [roc_auc_score(test[U.TARGET], U.predict(m, test[FEATURES])) for m in models]
        model = models[int(np.argmax(scores))]

    # ---- 6 · report -----------------------------------------------------------------------------------------
    p = U.predict(model, test[FEATURES])
    y = test[U.TARGET].to_numpy()
    return {"test_rows": len(test), "non_completers_in_test": float(y.mean()),
            "accuracy": float(accuracy_score(y, p >= 0.5)),             # the number in Loay's message
            "accuracy_of_always_completes": float(1 - y.mean()),
            "auc": float(roc_auc_score(y, p)), "average_precision": float(average_precision_score(y, p))}


if __name__ == "__main__":
    print(run())
