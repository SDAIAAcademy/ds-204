"""Shared helpers for the labs of Machine Learning for Tabular Data: Baselines and Validation (SDAIA Academy, DS-204).

Sullam Vocational Training Fund is a fictional fund. Every file in data/ is synthetic and written by
prepare_data.py with a fixed seed. The model the course builds is called Mubakkir: at the end of week 2 of a
programme it ranks the trainees by the risk of not completing, so that an adviser can offer support in week 3.

    import sullam as U
    raw = U.enrolments()                   # the 45,000 enrolments as exported
    table = U.modelling_table()            # the rows Mubakkir can score, with the two history features
    dev, holdout = U.dev_holdout(table)    # cohorts 2021 to 2024 for all the work; cohort 2025 stays closed
    folds = U.folds(dev)                   # the fund's five folds: each validates on a later half-year
    result = U.score_folds(lambda: U.make_pipeline(U.logistic()), dev, folds)
    U.log_run("1.3", "logistic", result)   # one line in artifacts/run_log.csv
"""

from __future__ import annotations

import json
import os
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

for _s in (sys.stdout, sys.stderr):   # Windows consoles: print check marks
    try:
        _s.reconfigure(encoding="utf-8")
    except Exception:
        pass

pd.set_option("display.width", 170)
pd.set_option("display.max_columns", 30)
warnings.filterwarnings("ignore", message=".*does not have valid feature names.*")
warnings.filterwarnings("ignore", message=".*Found unknown categories.*")

LABS = Path(__file__).resolve().parent
DATA = LABS / "data"
REF = DATA / "reference"
ART = LABS / "artifacts"          # what a lab writes: the run log, plots, tables, notes

QUICK = os.environ.get("DS204_QUICK", "0") == "1"
MODE = "quick" if QUICK else "full"
STRICT = os.environ.get("DS204_STRICT", "0") == "1"          # solution runs: a failed check stops the run
WRITE_REF = os.environ.get("DS204_WRITE_REF", "0") == "1"    # the trainer's run that writes data/reference/
SEED = 204
JOBS = min(4, os.cpu_count() or 1)

TARGET = "not_completed"
NUMERIC = ["age", "aptitude_score", "distance_km", "absences_w2", "programme_weeks", "has_dependants", "stipend"]
LOW = ["delivery_mode", "education", "employment", "enrolled_via", "start_month"]   # few values
HIGH = ["provider_id", "programme", "region"]                                       # many values
HISTORY = ["prior_enrolments", "prior_noncompletions"]
BASE = NUMERIC + LOW + HIGH                 # known at the end of week 2, without the trainee's history
FEATURES = BASE + HISTORY
AFTER_EXIT = ["weeks_attended", "exit_date", "exit_reason"]                         # recorded at exit
NOT_USED = ["gender"]                       # known, and left out: see Lab 1.1
AGE_BANDS = ["17-19", "20-24", "25-34", "35+"]
CAPACITY_SHARE = 0.08                       # the advisers can support about 8 trainees in 100
N_FOLDS = 5


# ---------------------------------------------------------------------------------------------------------------
# Small tools every lab uses
# ---------------------------------------------------------------------------------------------------------------

def pick(full, quick):
    """The full value, or the small one in quick mode (DS204_QUICK=1)."""
    return quick if QUICK else full


def check(ok, what: str) -> bool:
    """Print a self-check. In a solution run (DS204_STRICT=1) a failed check stops the run."""
    ok = bool(ok)
    print(("✅ " if ok else "❌ ") + what)
    if STRICT and not ok:
        raise AssertionError(what)
    return ok


def _jsonable(v):
    if isinstance(v, pd.DataFrame):
        return {"__frame__": json.loads(v.to_json(orient="split"))}
    if isinstance(v, pd.Series):
        return {"__series__": json.loads(v.to_json(orient="split"))}
    if isinstance(v, dict):
        return {str(k): _jsonable(x) for k, x in v.items()}
    if isinstance(v, (list, tuple)):
        return [_jsonable(x) for x in v]
    if isinstance(v, np.bool_):
        return bool(v)
    if isinstance(v, np.integer):
        return int(v)
    if isinstance(v, np.floating):
        return float(v)
    if isinstance(v, np.ndarray):
        return [_jsonable(x) for x in v.tolist()]
    return v


def _unjson(v):
    if isinstance(v, dict) and "__frame__" in v:
        f = v["__frame__"]
        return pd.DataFrame(f["data"], index=f["index"], columns=f["columns"])
    if isinstance(v, dict) and "__series__" in v:
        f = v["__series__"]
        return pd.Series(f["data"], index=f["index"], name=f.get("name"))
    if isinstance(v, dict):
        return {k: _unjson(x) for k, x in v.items()}
    return v


def keep(name: str, value):
    """Record a result of the full solution run as the reference for the safety nets (trainer run only)."""
    if WRITE_REF and not QUICK:
        REF.mkdir(parents=True, exist_ok=True)
        (REF / f"{name}.json").write_text(json.dumps(_jsonable(value), indent=1, ensure_ascii=False) + "\n",
                                          encoding="utf-8", newline="\n")
    return value


def reference(name: str):
    """The reference result of an exercise, for a participant who has not done it yet."""
    print(f"↪ using the reference result for {name} (data/reference/{name}.json)")
    return _unjson(json.loads((REF / f"{name}.json").read_text(encoding="utf-8")))


def save(name: str, text: str) -> Path:
    """Write a lab's output (a table, a note, a card) to labs/artifacts/."""
    ART.mkdir(exist_ok=True)
    path = ART / name
    path.write_text(text, encoding="utf-8", newline="\n")
    print("wrote artifacts/" + name)
    return path


def js(name: str):
    """A JSON file from labs/data/."""
    return json.loads((DATA / f"{name}.json").read_text(encoding="utf-8"))


def md(frame: pd.DataFrame, digits: int = 3) -> str:
    """A dataframe as a Markdown table."""
    return frame.round(digits).to_markdown()


def style():
    """Matplotlib with the course's defaults; returns pyplot."""
    import matplotlib
    if "ipykernel" not in sys.modules:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"figure.figsize": (7.2, 4.2), "figure.dpi": 110, "axes.spines.top": False,
                         "axes.spines.right": False, "axes.grid": True, "grid.alpha": 0.25, "font.size": 10})
    return plt


def savefig(fig, name: str) -> Path:
    ART.mkdir(exist_ok=True)
    path = ART / name
    fig.savefig(path, bbox_inches="tight")
    print("wrote artifacts/" + name)
    return path


# ---------------------------------------------------------------------------------------------------------------
# The data
# ---------------------------------------------------------------------------------------------------------------

def enrolments() -> pd.DataFrame:
    """The export: 45,000 enrolments, one row each, as the fund's system writes it."""
    return pd.read_csv(DATA / "enrolments.csv.gz", parse_dates=["start_date", "exit_date"],
                       keep_default_na=True, na_values=[""])


def providers() -> pd.DataFrame:
    return pd.read_csv(DATA / "providers.csv")


def columns() -> pd.DataFrame:
    """The data dictionary: when each column of the export is recorded."""
    return pd.read_csv(DATA / "columns.csv")


def decision() -> dict:
    """Adviser capacity and the cost of each error, as the lead adviser states them."""
    return js("decision")


def population() -> pd.DataFrame:
    """The people the fund may serve, by region, age band and gender (a synthetic reference table)."""
    return pd.read_csv(DATA / "population.csv")


def received_table() -> pd.DataFrame:
    """The feature table a colleague prepared (Lab 3.3). Four leaks are planted in it."""
    t = pd.read_csv(DATA / "received_table.csv.gz", parse_dates=["start_date"])
    return _calendar(t)


def age_band(age) -> pd.Series:
    return pd.cut(pd.Series(age), [0, 19, 24, 34, 120], labels=AGE_BANDS).astype(str)


def _calendar(t: pd.DataFrame) -> pd.DataFrame:
    t = t.copy()
    t["start_month"] = t.start_date.dt.month.astype(str)
    t["prediction_date"] = t.start_date + pd.Timedelta(days=14)
    t["planned_end"] = t.start_date + pd.to_timedelta(t.programme_weeks * 7, unit="D")
    t["block"] = (t.start_date.dt.year - 2021) * 2 + (t.start_date.dt.month >= 7).astype(int)   # half-years, 0 to 9
    t["age_band"] = age_band(t.age).to_numpy()
    return t


def scorable(raw: pd.DataFrame | None = None) -> pd.DataFrame:
    """The rows Mubakkir can score: trainees still enrolled at the end of week 2. Adds the calendar columns.

    In live use "still enrolled at the end of week 2" is read from the register on that day. In the export it
    is read from weeks_attended, which is used here to choose the rows and never as a feature."""
    raw = enrolments() if raw is None else raw
    t = raw[raw.weeks_attended > 2].copy()
    return _calendar(t).sort_values(["start_date", "enrolment_id"]).reset_index(drop=True)


def add_history(t: pd.DataFrame, raw: pd.DataFrame | None = None) -> pd.DataFrame:
    """The two history features, from past records only (the reference version of Lab 1.4).

    For each row: how many enrolments of the same trainee had ENDED before this row's prediction date, and how
    many of those ended without completion. An enrolment that started earlier and is still running does not
    count: its outcome is not known yet."""
    raw = enrolments() if raw is None else raw
    past = raw[["trainee_id", "exit_date", TARGET]].rename(columns={"exit_date": "past_exit", TARGET: "past_y"})
    j = t[["enrolment_id", "trainee_id", "prediction_date"]].merge(past, on="trainee_id", how="left")
    j = j[j.past_exit < j.prediction_date]
    g = j.groupby("enrolment_id").past_y.agg(["size", "sum"])
    out = t.copy()
    out["prior_enrolments"] = out.enrolment_id.map(g["size"]).fillna(0).astype(int)
    out["prior_noncompletions"] = out.enrolment_id.map(g["sum"]).fillna(0).astype(int)
    return out


def modelling_table(history: bool = True) -> pd.DataFrame:
    """The scorable rows with calendar columns and, unless history=False, the two history features."""
    raw = enrolments()
    t = scorable(raw)
    return add_history(t, raw) if history else t


def dev_holdout(t: pd.DataFrame):
    """Cohorts 2021 to 2024 for all the work; cohort 2025 is the holdout, opened on Day 4."""
    return (t[t.cohort_year <= 2024].reset_index(drop=True), t[t.cohort_year == 2025].reset_index(drop=True))


def folds(dev: pd.DataFrame, n: int = N_FOLDS, first: int | None = None):
    """The fund's folds: fold k validates on one half-year and trains on the enrolments whose programme was due to
    end before that half-year began. Returns a list of (train positions, validation positions).

    "Due to end", not "ended": an enrolment that is still running has no outcome yet, and keeping only the ones
    that already ended would keep the early leavers and drop their classmates. With the development cohorts the
    five validation half-years are 2022 H2 to 2024 H2."""
    block = dev.block.to_numpy()
    last = int(block.max())
    first = last - n + 1 if first is None else first
    out = []
    for b in range(first, last + 1):
        begins = dev.start_date[block == b].min()
        train = np.flatnonzero((block < b) & (dev.planned_end < begins).to_numpy())
        valid = np.flatnonzero(block == b)
        out.append((train, valid))
    return out


def inner_folds(train: pd.DataFrame, n: int = 2):
    """Folds inside one training part, built the same way, for choosing settings (nested validation)."""
    return folds(train.reset_index(drop=True), n=n)


def block_name(b: int) -> str:
    return f"{2021 + b // 2} H{1 + b % 2}"


# ---------------------------------------------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------------------------------------------

def make_pipeline(model, features=None, scale: bool = True):
    """Impute, encode and (for linear models) scale, then the model. Everything is fitted on training rows only."""
    from sklearn.compose import ColumnTransformer
    from sklearn.impute import SimpleImputer
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import OneHotEncoder, StandardScaler
    features = FEATURES if features is None else list(features)
    cat = [c for c in features if c in LOW + HIGH]
    num = [c for c in features if c not in cat]
    steps = [("impute", SimpleImputer(strategy="median", add_indicator=True))]
    if scale:
        steps.append(("scale", StandardScaler()))
    encode = Pipeline([("fill", SimpleImputer(strategy="constant", fill_value="not recorded")),
                       ("onehot", OneHotEncoder(handle_unknown="infrequent_if_exist", min_frequency=30,
                                                sparse_output=False))])
    prep = ColumnTransformer([("num", Pipeline(steps), num), ("cat", encode, cat)], verbose_feature_names_out=False)
    return Pipeline([("prepare", prep), ("model", model)])


def logistic(C: float = 0.1):
    from sklearn.linear_model import LogisticRegression
    return LogisticRegression(C=C, max_iter=2000)


def boosted(**params):
    """Gradient-boosted trees (LightGBM) with the library's defaults unless a setting is given."""
    from lightgbm import LGBMClassifier
    base = dict(random_state=SEED, n_jobs=JOBS, verbose=-1)
    base.update(params)
    return LGBMClassifier(**base)


def champion_params() -> dict:
    """The settings the trainer's 20-trial search of Lab 2.2 chose (data/reference/d2_2_best.json)."""
    return json.loads((REF / "d2_2_best.json").read_text(encoding="utf-8"))["params"]


def champion():
    """The tuned boosted model in its pipeline: the champion of Days 2 to 4."""
    return make_pipeline(boosted(**champion_params()), scale=False)


# ---------------------------------------------------------------------------------------------------------------
# Scores
# ---------------------------------------------------------------------------------------------------------------

def top_share(p, share: float = CAPACITY_SHARE) -> np.ndarray:
    """True for the rows with the highest scores: as many as the advisers can support."""
    p = np.asarray(p, dtype=float)
    k = max(1, int(round(share * len(p))))
    flag = np.zeros(len(p), dtype=bool)
    flag[np.argsort(-p, kind="stable")[:k]] = True
    return flag


def recall_at(y, p, share: float = CAPACITY_SHARE) -> float:
    """Of the trainees who did not complete, the share that is on the advisers' list."""
    y = np.asarray(y)
    return float(y[top_share(p, share)].sum() / max(1, y.sum()))


def metrics(y, p) -> dict:
    from sklearn.metrics import average_precision_score, roc_auc_score
    y, p = np.asarray(y), np.asarray(p, dtype=float)
    return {"ap": float(average_precision_score(y, p)), "auc": float(roc_auc_score(y, p)) if np.ptp(p) > 0 else 0.5,
            "recall_at_capacity": recall_at(y, p)}


def predict(model, X) -> np.ndarray:
    return model.predict_proba(X)[:, 1]


def score_folds(make_model, table: pd.DataFrame, fold_list, features=None, train_scores: bool = False) -> dict:
    """Fit a fresh model on each fold's training rows and score its validation rows.

    make_model is a function with no argument that returns an unfitted model (a pipeline). Returns the scores per
    fold (average precision first), their mean and spread, the seconds used and the out-of-fold probabilities."""
    features = FEATURES if features is None else list(features)
    y = table[TARGET].to_numpy()
    rows, oof, t0 = [], np.full(len(table), np.nan), time.perf_counter()
    for k, (tr, va) in enumerate(fold_list, 1):
        model = make_model()
        model.fit(table.iloc[tr][features], y[tr])
        p = predict(model, table.iloc[va][features])
        oof[va] = p
        row = {"fold": k, "n_train": len(tr), "n_valid": len(va), "base_rate": float(y[va].mean()), **metrics(y[va], p)}
        if train_scores:
            row["train_ap"] = metrics(y[tr], predict(model, table.iloc[tr][features]))["ap"]
        rows.append(row)
    per_fold = pd.DataFrame(rows).set_index("fold")
    return {"per_fold": per_fold, "ap": per_fold.ap.tolist(), "mean": float(per_fold.ap.mean()),
            "spread": float(per_fold.ap.std(ddof=1)), "seconds": time.perf_counter() - t0, "oof": oof}


def score_fixed(score_fn, table: pd.DataFrame, fold_list) -> dict:
    """The same as score_folds for a baseline that is a function: score_fn(train, valid) returns one score per
    validation row."""
    y = table[TARGET].to_numpy()
    rows, oof, t0 = [], np.full(len(table), np.nan), time.perf_counter()
    for k, (tr, va) in enumerate(fold_list, 1):
        p = np.asarray(score_fn(table.iloc[tr], table.iloc[va]), dtype=float)
        oof[va] = p
        rows.append({"fold": k, "n_train": len(tr), "n_valid": len(va), "base_rate": float(y[va].mean()),
                     **metrics(y[va], p)})
    per_fold = pd.DataFrame(rows).set_index("fold")
    return {"per_fold": per_fold, "ap": per_fold.ap.tolist(), "mean": float(per_fold.ap.mean()),
            "spread": float(per_fold.ap.std(ddof=1)), "seconds": time.perf_counter() - t0, "oof": oof}


# ---------------------------------------------------------------------------------------------------------------
# The run log
# ---------------------------------------------------------------------------------------------------------------

LOG_COLUMNS = ["lab", "rung", "model", "features", "folds", "mean_ap", "spread", "min_fold", "max_fold",
               "recall_at_capacity", "seconds", "fold_scores", "note"]


def log_run(lab: str, rung: str, result: dict, model: str = "", features=None, note: str = "") -> pd.DataFrame:
    """Append one run to artifacts/run_log.csv and return the line. One line per run, never edited afterwards."""
    per = result["per_fold"]
    line = pd.DataFrame([{
        "lab": lab, "rung": rung, "model": model or rung, "features": len(features) if features is not None else "",
        "folds": len(per), "mean_ap": round(result["mean"], 4), "spread": round(result["spread"], 4),
        "min_fold": round(float(per.ap.min()), 4), "max_fold": round(float(per.ap.max()), 4),
        "recall_at_capacity": round(float(per.recall_at_capacity.mean()), 4),
        "seconds": round(result["seconds"], 1), "fold_scores": " ".join(f"{v:.4f}" for v in per.ap), "note": note,
    }], columns=LOG_COLUMNS)
    ART.mkdir(exist_ok=True)
    path = ART / "run_log.csv"
    line.to_csv(path, mode="a", header=not path.exists(), index=False, lineterminator="\n")
    if WRITE_REF and not QUICK:
        REF.mkdir(parents=True, exist_ok=True)
        ref = REF / "run_log.csv"
        old = pd.read_csv(ref) if ref.exists() else pd.DataFrame(columns=LOG_COLUMNS)
        old = old[~((old.lab.astype(str) == lab) & (old.rung == rung))]
        pd.concat([old, line]).to_csv(ref, index=False, lineterminator="\n")
    print(f"logged {lab} · {rung}: mean {result['mean']:.3f}, spread {result['spread']:.3f} over {len(per)} folds")
    return line


def reference_log() -> pd.DataFrame:
    """The trainer's run log of the whole ladder (data/reference/run_log.csv): the supplied state of every lab."""
    log = pd.read_csv(REF / "run_log.csv", dtype={"lab": str}).fillna("")
    return log.sort_values("lab", kind="stable").reset_index(drop=True)


def my_log() -> pd.DataFrame:
    path = ART / "run_log.csv"
    return pd.read_csv(path, dtype={"lab": str}).fillna("") if path.exists() else pd.DataFrame(columns=LOG_COLUMNS)


def fold_scores(rung: str) -> np.ndarray:
    """The five fold scores of one rung of the reference log."""
    log = reference_log()
    return np.array([float(v) for v in log[log.rung == rung].fold_scores.iloc[-1].split()])


# ---------------------------------------------------------------------------------------------------------------
# Intervals
# ---------------------------------------------------------------------------------------------------------------

def paired_interval(a, b, level: float = 0.95) -> dict:
    """The mean of the paired differences a - b over folds, with a t interval."""
    from scipy import stats
    d = np.asarray(a, dtype=float) - np.asarray(b, dtype=float)
    half = stats.t.ppf(0.5 + level / 2, len(d) - 1) * d.std(ddof=1) / np.sqrt(len(d))
    return {"mean": float(d.mean()), "low": float(d.mean() - half), "high": float(d.mean() + half),
            "differences": d.round(4).tolist()}


def wilson(k: int, n: int, level: float = 0.95) -> tuple[float, float]:
    """A Wilson interval for a proportion k of n: honest for small slices and for rates near 0 or 1."""
    from scipy import stats
    if n == 0:
        return (float("nan"), float("nan"))
    z = stats.norm.ppf(0.5 + level / 2)
    p = k / n
    centre = (p + z * z / (2 * n)) / (1 + z * z / n)
    half = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return (float(max(0.0, centre - half)), float(min(1.0, centre + half)))


def day4_predictions() -> pd.DataFrame:
    """The champion's probabilities for Day 4 (data/predictions.csv.gz), joined to the rows they describe.

    split "validation": cohort 2024, scored by a model trained on programmes due to end before 2024.
    split "holdout": cohort 2025, scored by a model trained on programmes due to end before 2025."""
    p = pd.read_csv(DATA / "predictions.csv.gz")
    t = modelling_table()
    return p.merge(t, on="enrolment_id", how="left")


# ---------------------------------------------------------------------------------------------------------------
# Tests and fold-wise helpers
# ---------------------------------------------------------------------------------------------------------------

def time_travel_test(history_fn, table: pd.DataFrame, raw: pd.DataFrame, cutoffs=("2022-06-01", "2023-03-01", "2024-08-01")) -> bool:
    """Does a history function read the future? For each cutoff date, the features of the rows predicted before it
    are computed twice: from all records, and from the records that had ended by then. A function that reads past
    records only gives the same answer both times."""
    full = history_fn(table, raw)
    ok = True
    for c in cutoffs:
        c = pd.Timestamp(c)
        rows = table[table.prediction_date <= c]
        then = history_fn(rows, raw[raw.exit_date < c])
        now = full.loc[rows.index]
        for col in HISTORY:
            same = bool((now[col].to_numpy() == then[col].to_numpy()).all())
            if not same:
                n = int((now[col].to_numpy() != then[col].to_numpy()).sum())
                print(f"   {col}: {n:,} rows predicted before {c.date()} change when later records are hidden")
            ok = ok and same
    return ok


def split_scores(make_model, table: pd.DataFrame, splits, features=None, new_trainees: bool = False) -> dict:
    """Score a model under any list of (train positions, validation positions).

    new_trainees=True hides the history of the validation rows: under a grouped split the validation trainees
    have no record in the training part, so the fund knows nothing of their past."""
    features = FEATURES if features is None else list(features)
    y = table[TARGET].to_numpy()
    rows = []
    for k, (tr, va) in enumerate(splits, 1):
        valid = table.iloc[va].copy()
        if new_trainees:
            valid[HISTORY] = 0
        model = make_model().fit(table.iloc[tr][features], y[tr])
        rows.append({"fold": k, **metrics(y[va], predict(model, valid[features]))})
    per = pd.DataFrame(rows).set_index("fold")
    return {"per_fold": per, "mean": float(per.ap.mean()), "spread": float(per.ap.std(ddof=1)),
            "auc": float(per.auc.mean())}


def matrices(table: pd.DataFrame, fold_list, features=None):
    """For a search that fits many models on the same folds: prepare each fold once.

    Returns one (X_train, y_train, X_valid, y_valid) per fold; the preparation is fitted on the training rows."""
    features = FEATURES if features is None else list(features)
    y = table[TARGET].to_numpy()
    out = []
    for tr, va in fold_list:
        prepare = make_pipeline(boosted(), features, scale=False)[:-1]
        out.append((prepare.fit_transform(table.iloc[tr][features]), y[tr],
                    prepare.transform(table.iloc[va][features]), y[va]))
    return out


def ap(y, p) -> float:
    """Average precision: the score of the run log."""
    from sklearn.metrics import average_precision_score
    return float(average_precision_score(y, p))


def keep_oof(name: str, table: pd.DataFrame, oof) -> None:
    """Record the validation-fold probabilities of a rung (trainer run only), for the paired bootstrap of Lab 2.4."""
    if WRITE_REF and not QUICK:
        REF.mkdir(parents=True, exist_ok=True)
        seen = np.isfinite(oof)
        frame = pd.DataFrame({"enrolment_id": table.enrolment_id[seen], "p": np.round(np.asarray(oof)[seen], 5)})
        frame.to_csv(REF / f"oof_{name}.csv.gz", index=False, lineterminator="\n",
                     compression={"method": "gzip", "mtime": 0})


def oof(names=("champion", "challenger", "logistic")) -> pd.DataFrame:
    """The reference validation-fold probabilities of the named rungs, with the outcome and the fold of each row."""
    dev, _ = dev_holdout(modelling_table())
    out = dev[["enrolment_id", TARGET, "block"]].copy()
    out["fold"] = out.block - (int(out.block.max()) - N_FOLDS)
    for n in names:
        p = pd.read_csv(REF / f"oof_{n}.csv.gz")
        out[n] = out.enrolment_id.map(p.set_index("enrolment_id").p)
    return out.dropna(subset=list(names)).drop(columns="block").reset_index(drop=True)
