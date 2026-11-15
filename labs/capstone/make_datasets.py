"""Write the three synthetic capstone tables and their data dictionaries into capstone/data/.

    uv run python capstone/make_datasets.py

The organisations, people and figures are invented for teaching. Each table has the structure the course
practised on: units that repeat, a date, an outcome that is rare, a world that changes from year to year, and
two columns that a careful audit removes. Which two is for the participant to find: the data dictionary says
when every recorded column is written, and says of two columns only that the previous analyst computed them.
The trainer's key is in solutions/answer_key.md.
"""
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
OUT = HERE / "data"
REGIONS = ["Riyadh", "Makkah", "Eastern Province", "Madinah", "Asir", "Qassim", "Jazan", "Tabuk"]
REGION_SHARE = [0.27, 0.24, 0.15, 0.08, 0.08, 0.06, 0.07, 0.05]

CASES = {
    "bayader_loans": {
        "seed": 2041, "rows": 22_000, "units": 13_500, "sites": 30, "intercept": -2.75,
        "key": "loan_id", "unit": "borrower_id", "site": "branch_id", "date": "approval_date", "target": "missed_repayment",
        "x1": ("instalment_to_income", "Monthly instalment as a share of declared monthly income", "approval"),
        "x2": ("months_in_business", "Months the borrower's business has been running; blank when not declared", "approval"),
        "x3": ("late_utility_bills", "Late utility bills in the 12 months before the application, from the credit bureau", "approval"),
        "c1": ("loan_purpose", ["stock", "equipment", "working capital"], "Purpose stated on the application", "approval"),
        "c2": ("sector", ["retail", "food", "services", "crafts"], "Sector of the business", "approval"),
        "after": ("collection_calls", "Calls made to the borrower by the collections team", "during repayment"),
        "computed": ("borrower_missed_count", "branch_missed_rate"),
        "under_recorded": "Credit bureau coverage is thin in two regions: late bills are under-reported there",
        "duplicates": 0,
    },
    "awtad_rentals": {
        "seed": 2042, "rows": 20_000, "units": 9_000, "sites": 24, "intercept": -2.55,
        "key": "contract_id", "unit": "customer_id", "site": "depot_id", "date": "contract_date", "target": "returned_late",
        "x1": ("rental_days", "Days of rental agreed in the contract (standardised)", "contract"),
        "x2": ("customer_years", "Years since the customer's first contract; blank for walk-in customers", "contract"),
        "x3": ("open_contracts", "Other contracts of the customer that are open on the contract date", "contract"),
        "c1": ("equipment_class", ["lifting", "earthmoving", "power"], "Class of the equipment rented", "contract"),
        "c2": ("customer_type", ["contractor", "facility", "event", "individual"], "Type of customer", "contract"),
        "after": ("late_fee_sar", "Late fee invoiced for the contract, in riyals", "return"),
        "computed": ("customer_late_count", "depot_late_rate"),
        "under_recorded": "Two regions migrated depot systems late: open contracts are under-counted there",
        "duplicates": 600,
    },
    "nasaim_care_plans": {
        "seed": 2043, "rows": 18_000, "units": 10_500, "sites": 20, "intercept": -2.4,
        "key": "plan_id", "unit": "patient_id", "site": "team_id", "date": "plan_start", "target": "plan_discontinued",
        "x1": ("visits_per_week", "Home visits per week in the care plan (standardised)", "plan start"),
        "x2": ("carer_at_home_hours", "Hours a family carer is at home per day; blank when not assessed", "plan start"),
        "x3": ("missed_first_visits", "Visits missed in the first week of the plan", "end of week 1"),
        "c1": ("referral_source", ["hospital", "clinic", "family"], "Who referred the patient", "plan start"),
        "c2": ("care_type", ["wound care", "physiotherapy", "chronic care", "post-surgery"], "Type of care", "plan start"),
        "after": ("closure_code", "Code entered when a plan is closed (0 when not entered)", "plan closure"),
        "computed": ("patient_discontinued_count", "team_discontinued_rate"),
        "under_recorded": "Two regions record visits on paper: missed first visits are under-recorded there",
        "duplicates": 0,
    },
}


def make(name: str, c: dict) -> None:
    rng = np.random.default_rng(c["seed"])
    n, units, sites = c["rows"], c["units"], c["sites"]
    # units repeat: a third of the rows belong to a unit seen before
    unit = np.concatenate([np.arange(units), rng.choice(units, n - units, p=rng.dirichlet(np.full(units, 0.3)))])
    rng.shuffle(unit)
    frailty = rng.normal(0, 0.9, units)[unit]
    unit_region = rng.choice(len(REGIONS), units, p=REGION_SHARE)[unit]
    unit_gender = rng.choice(["F", "M"], units, p=[0.42, 0.58])[unit]
    unit_age = rng.integers(21, 68, units)[unit]
    day = np.sort(rng.integers(0, 4 * 365, n))
    date = pd.Timestamp("2022-01-01") + pd.to_timedelta(day, unit="D")
    year = date.year.to_numpy()
    site = (unit_region * 7 + rng.integers(0, sites, n)) % sites
    walk = np.cumsum(rng.normal(0, 0.22, (4, sites)), axis=0) + rng.normal(0, 0.35, sites)
    x1 = rng.normal(0, 1, n) + 0.3 * frailty
    x2 = np.clip(rng.gamma(2.0, 2.0, n) - 0.4 * frailty, 0, None)
    x3 = rng.poisson(np.exp(-0.9 + 0.55 * frailty + 0.25 * x1))
    c1 = rng.choice(3, n, p=[0.5, 0.3, 0.2])
    c2 = rng.choice(4, n, p=[0.35, 0.3, 0.2, 0.15])
    c1_effect = np.array([0.0, 0.45, -0.3])[c1] * np.where(year == 2025, -1.0, 1.0)      # one effect flips in 2025
    logit = (c["intercept"] + 0.55 * x1 - 0.16 * x2 + 0.5 * x3 + c1_effect + np.array([0, 0.2, -0.1, 0.35])[c2]
             + walk[year - 2022, site] + 0.8 * frailty + 0.012 * (unit_age - 40) * (unit_age > 55))
    y = (rng.random(n) < 1 / (1 + np.exp(-logit))).astype(int)

    t = pd.DataFrame({
        c["key"]: [f"{c['key'][0].upper()}{k:06d}" for k in rng.permutation(n) + 100_000],
        c["unit"]: [f"U{k:05d}" for k in unit + 10_000],
        c["site"]: [f"S{k:02d}" for k in site + 1],
        c["date"]: date.strftime("%Y-%m-%d"),
        "region": np.array(REGIONS)[unit_region], "gender": unit_gender, "age": unit_age,
        c["x1"][0]: np.round(x1, 3),
        c["x2"][0]: np.where(rng.random(n) < 0.12, np.nan, np.round(x2, 1)),
        c["x3"][0]: np.where(np.isin(unit_region, [6, 7]), rng.binomial(x3, 0.45), x3),   # under-recorded in two regions
        c["c1"][0]: np.array(c["c1"][1])[c1], c["c2"][0]: np.array(c["c2"][1])[c2],
        c["after"][0]: np.where(y == 1, rng.poisson(2.6, n) + 1, rng.poisson(0.04, n)) * (55 if "fee" in c["after"][0] else 1),
        c["target"]: y,
    })
    t[c["computed"][0]] = t.groupby(c["unit"])[c["target"]].transform("sum")             # every record of the unit
    t[c["computed"][1]] = t.groupby(c["site"])[c["target"]].transform("mean").round(4)   # every row of the site
    t["earlier_records"] = t.groupby(c["unit"]).cumcount()                               # honest: no outcome in it
    if c["duplicates"]:
        twins = t.sample(c["duplicates"], random_state=c["seed"]).copy()
        twins[c["key"]] = [f"{c['key'][0].upper()}{k:06d}" for k in np.arange(len(twins)) + 900_000]
        t = pd.concat([t, twins]).sample(frac=1.0, random_state=c["seed"]).reset_index(drop=True)

    OUT.mkdir(exist_ok=True)
    t.to_csv(OUT / f"{name}.csv.gz", index=False, lineterminator="\n", compression={"method": "gzip", "mtime": 0})
    rows = [(c["key"], "Key of the record", "creation"), (c["unit"], "Key of the unit; a unit can have several records", "registration"),
            (c["site"], "Site that handled the record", "creation"), (c["date"], "Date of the record: the prediction is made on this day"
             + (" (for the first-week column: seven days later)" if "week" in c["x3"][2] else ""), "creation"),
            ("region", "Region of the unit", "registration"), ("gender", "F or M, as registered; kept for the audit of errors", "registration"),
            ("age", "Age in years on the date of the record", "creation")]
    rows += [c[k][:1] + c[k][-2:] for k in ("x1", "x2", "x3")] + [(c[k][0], c[k][2], c[k][3]) for k in ("c1", "c2")]
    rows += [c["after"], (c["target"], "TARGET: 1 when the outcome happened", "after the outcome"),
             (c["computed"][0], "Computed by the previous analyst; no documentation", "unknown"),
             (c["computed"][1], "Computed by the previous analyst; no documentation", "unknown"),
             ("earlier_records", "Computed by the previous analyst: records of the unit with an earlier date", "unknown")]
    pd.DataFrame(rows, columns=["column", "meaning", "recorded_at"]).to_csv(OUT / f"{name}_columns.csv", index=False, lineterminator="\n")
    print(f"{name}: {len(t):,} rows, {t[c['unit']].nunique():,} units, outcome rate {t[c['target']].mean():.1%}; note: {c['under_recorded']}")


if __name__ == "__main__":
    for name, case in CASES.items():
        make(name, case)
