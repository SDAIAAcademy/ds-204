# /// script
# requires-python = ">=3.11"
# dependencies = ["numpy>=2.0", "pandas>=2.2"]
# ///
"""Write the synthetic data of Sullam Vocational Training Fund into labs/data/.

    uv run prepare_data.py

Sullam is a fictional fund. No row describes a real person, provider or programme. The same bytes are written on
every run (fixed seed). What is planted, and which lab finds it, is listed in solutions/answer_key.md.

Files written
    data/enrolments.csv.gz      45,000 enrolments of 30,000 trainees, five yearly cohorts, 60 providers
    data/providers.csv          the 60 training providers
    data/columns.csv            the data dictionary: when each column is recorded
    data/decision.json          adviser capacity and the cost of each error
    data/population.csv         the eligible population by region, age band and gender (for representativeness)
    data/received_table.csv.gz  the feature table "as received" for Lab 3.3, with the four planted leaks
"""

from __future__ import annotations

import gzip
import io
import json
from pathlib import Path

import numpy as np
import pandas as pd

SEED = 204
N_TRAINEES = 30_000
PER_TRAINEE = {1: 19_500, 2: 7_000, 3: 2_500, 4: 1_000}      # 45,000 enrolments
YEARS = [2021, 2022, 2023, 2024, 2025]
INTAKE_MONTHS = [1, 4, 7, 10]                                 # four intakes a year
INTERCEPT = -3.10
Z_EFFECT = 1.25
DATA = Path(__file__).resolve().parent / "data"

REGIONS = {                    # share of trainees; providers that existed from 2021; providers opened in 2025
    "Riyadh": (0.265, 12, 3), "Makkah": (0.235, 11, 2), "Eastern Province": (0.150, 8, 1),
    "Madinah": (0.070, 4, 0), "Asir": (0.070, 4, 1), "Qassim": (0.050, 3, 0), "Jazan": (0.042, 2, 1),
    "Tabuk": (0.030, 2, 0), "Hail": (0.028, 2, 0), "Najran": (0.022, 1, 0), "Al-Jouf": (0.016, 1, 0),
    "Al-Baha": (0.012, 1, 0), "Northern Borders": (0.010, 1, 0),
}
PAPER_REGISTERS = ["Jazan", "Najran"]        # absences kept on paper and typed in late: under-recorded
PROGRAMMES = {                 # weeks, effect on the log-odds of not completing, may be blended from 2024
    "Electrical installation": (24, 0.00, False), "Air-conditioning maintenance": (24, 0.05, False),
    "Welding": (36, 0.20, False), "Vehicle service": (36, 0.15, False), "Culinary arts": (24, -0.10, False),
    "IT support": (12, -0.15, True), "Logistics operations": (12, 0.00, True),
    "Retail and sales": (12, 0.10, True), "Health support assistant": (24, -0.20, False),
    "Digital marketing": (12, -0.05, True),
}
EDUCATION = ["below secondary", "secondary", "diploma", "bachelor"]
REASONS = ["found work", "family", "transport", "health", "not stated"]


def first_sunday(year: int, month: int) -> pd.Timestamp:
    d = pd.Timestamp(year=year, month=month, day=1)
    return d + pd.Timedelta(days=int((6 - d.dayofweek) % 7))


def main() -> None:
    rng = np.random.default_rng(SEED)
    intakes = [first_sunday(y, m) for y in YEARS for m in INTAKE_MONTHS]      # 20 intakes

    # ---- providers ------------------------------------------------------------------------------------------
    rows = []
    for region, (_, old, new) in REGIONS.items():
        rows += [(region, 2021)] * old + [(region, 2025)] * new
    order = sorted(range(len(rows)), key=lambda i: (rows[i][1], i))           # the eight new ones get P53 to P60
    providers = pd.DataFrame([rows[i] for i in order], columns=["region", "opened_year"])
    providers.insert(0, "provider_id", [f"P{i + 1:02d}" for i in range(len(providers))])
    providers["provider_type"] = rng.choice(["public institute", "private centre", "employer academy"],
                                            len(providers), p=[0.40, 0.42, 0.18])
    prov_effect = rng.normal(0, 0.38, len(providers))
    prov_effect[providers.opened_year.to_numpy() == 2025] += 0.35           # new providers start badly
    names = list(PROGRAMMES)
    offers = [list(rng.choice(names, rng.integers(3, 6), replace=False)) for _ in range(len(providers))]
    assert len(providers) == 60

    # ---- trainees -------------------------------------------------------------------------------------------
    n = N_TRAINEES
    z = rng.normal(0, 1, n)                                                   # persistence: never observed
    region = rng.choice(list(REGIONS), n, p=[v[0] for v in REGIONS.values()])
    gender = rng.choice(["F", "M"], n, p=[0.43, 0.57])
    age0 = np.clip(np.round(rng.gamma(3.2, 2.6, n) + 16.5), 17, 49).astype(int)   # age at the first enrolment
    education = rng.choice(EDUCATION, n, p=[0.14, 0.52, 0.22, 0.12])
    aptitude = np.clip(np.round(55 + 9 * z + rng.normal(0, 13, n), 1), 5, 99)
    dependants = (rng.random(n) < np.where(age0 >= 27, 0.55, 0.12)).astype(int)
    remote = np.round(np.clip(rng.lognormal(np.log(9), 0.75, n), 0.5, 140), 1)   # km to the home provider
    home = np.empty(n, dtype=int)
    for r in REGIONS:
        idx = np.flatnonzero(region == r)
        old = providers.index[(providers.region == r) & (providers.opened_year == 2021)].to_numpy()
        home[idx] = rng.choice(old, len(idx))
    counts = np.repeat(list(PER_TRAINEE), list(PER_TRAINEE.values()))
    rng.shuffle(counts)
    trainee_id = np.array([f"T{i:06d}" for i in rng.permutation(np.arange(100_000, 100_000 + n))])

    # ---- enrolments -----------------------------------------------------------------------------------------
    w_intake = np.repeat([0.80, 0.90, 1.00, 1.10, 1.20], 4)
    w_intake = w_intake / w_intake.sum()
    rec = []
    for i in range(n):
        picks = np.sort(rng.choice(20, counts[i], replace=False, p=w_intake))
        for k, t in enumerate(picks):
            rec.append((i, k, t))
    e = pd.DataFrame(rec, columns=["i", "k", "t"])
    m = len(e)
    assert m == 45_000
    e["start_date"] = [intakes[t] for t in e.t]
    e["cohort_year"] = e.start_date.dt.year
    i = e.i.to_numpy()

    # the provider: the home provider seven times in ten; from 2025 some go to a provider that just opened
    prov = home[i].copy()
    other = rng.random(m) < 0.30
    for r in REGIONS:
        old = providers.index[(providers.region == r) & (providers.opened_year == 2021)].to_numpy()
        new = providers.index[(providers.region == r) & (providers.opened_year == 2025)].to_numpy()
        sel = np.flatnonzero(other & (region[i] == r))
        prov[sel] = rng.choice(old, len(sel))
        if len(new):
            sel = np.flatnonzero((region[i] == r) & (e.cohort_year.to_numpy() == 2025) & (rng.random(m) < 0.24))
            prov[sel] = rng.choice(new, len(sel))
    e["provider_idx"] = prov
    e["programme"] = [offers[p][rng.integers(len(offers[p]))] for p in prov]
    weeks = e.programme.map(lambda p: PROGRAMMES[p][0]).to_numpy()
    year = e.cohort_year.to_numpy()
    blended_ok = e.programme.map(lambda p: PROGRAMMES[p][2]).to_numpy()
    blended = blended_ok & (year >= 2024) & (rng.random(m) < 0.55)
    age = age0[i] + (year - e.groupby("i").cohort_year.transform("min").to_numpy())
    dist = np.round(remote[i] * np.where(prov == home[i], 1.0, rng.uniform(0.7, 2.2, m)), 1)
    employment = rng.choice(["not working", "part-time", "full-time"], m, p=[0.66, 0.22, 0.12])
    shift = rng.random(m) < np.where(year >= 2024, 0.10, 0.0)                 # more part-time workers lately
    employment = np.where(shift & (employment == "not working"), "part-time", employment)
    via = rng.choice(["self", "employer", "social support referral"], m, p=[0.62, 0.20, 0.18])
    stipend = (rng.random(m) < np.where(via == "employer", 0.25, 0.70)).astype(int)

    # the class: trainees who start the same programme at the same provider on the same day share a shock
    class_key = e.provider_idx.astype(str) + "|" + e.programme + "|" + e.t.astype(str)
    class_code, class_names = pd.factorize(class_key)
    class_shock = rng.normal(0, 0.42, len(class_names))[class_code]

    # earlier non-completions matter beyond persistence: filled in time order below
    zi = z[i]
    young, older = (age <= 19).astype(float), (age >= 35).astype(float)
    dist_slope = np.where(year <= 2022, 0.020, 0.007)                         # a transport allowance from 2023
    trap = ((dependants[i] == 1) & (dist > 25) & (stipend == 0)).astype(float)
    prov_year = prov_effect[:, None] + np.cumsum(rng.normal(0, 0.45, (len(providers), 5)), axis=1)   # providers change
    far = dist > 20
    long_prog = weeks >= 24
    prog = e.programme.to_numpy()
    prog_effect = e.programme.map(lambda p: PROGRAMMES[p][1]).to_numpy()
    # cohort drift: four things change while the table is being collected
    prog_effect = np.where(np.isin(prog, ["Welding", "Vehicle service"]) & (year >= 2024), -0.35, prog_effect)  # new workshops
    prog_effect = np.where((prog == "Retail and sales") & (year >= 2024), 0.60, prog_effect)
    employer_effect = np.select([year <= 2022, year == 2023], [-0.45, -0.10], 0.45)      # an employer mandate from 2024
    stipend_effect = np.where(year <= 2022, -0.30, -0.75)                                 # paid on attendance from 2023
    base = (INTERCEPT + np.array([0.00, 0.06, 0.10, 0.02, 0.04])[year - 2021]
            - Z_EFFECT * zi + prov_year[prov, year - 2021] + class_shock + prog_effect
            + dist_slope * np.minimum(dist, 80) + 0.50 * young - 0.20 * ((age >= 28) & (age <= 34)) + 0.15 * older
            + 0.60 * (young.astype(bool) & long_prog)
            + np.where(gender[i] == "F", 0.55, 0.10) * dependants[i] + 1.10 * trap
            + np.select([education[i] == "below secondary", education[i] == "bachelor"], [0.35, -0.20], 0.0)
            + 0.75 * (aptitude[i] < 38) + 0.45 * ((aptitude[i] < 38) & long_prog)
            + np.select([employment == "part-time", employment == "full-time"], [0.15, 0.10], 0.0)
            + 0.65 * ((employment == "part-time") & long_prog & ~blended)
            + np.where(via == "employer", employer_effect, np.where(via == "social support referral", 0.30, 0.0))
            + stipend_effect * stipend + np.where(blended, np.where(far, -0.40, 0.90), 0.0))
    absences = np.minimum(rng.poisson(np.exp(-1.25 + 0.45 * (base - INTERCEPT) - 0.10 * zi)), 10)
    older_mask = age >= 35                                                    # their exits are not preceded by absence
    step = np.select([absences == 0, absences == 1, absences == 2], [0.0, 0.15, 0.80], 1.35 + 0.12 * (np.minimum(absences, 7) - 3))
    # early absence means less for three groups: older trainees, employer-sponsored trainees (absent for work, and
    # they finish) and blended classes (the register covers the in-person days only)
    weight = np.select([older_mask, via == "employer", blended], [0.0, 0.0, 0.25], 1.0)
    logit = (base + weight * 1.45 * step + 0.75 * older_mask + 0.50 * (via == "employer")
             + 0.45 * ((absences >= 2) & far) - 0.40)

    e["end_planned"] = e.start_date + pd.to_timedelta(weeks * 7, unit="D")
    e = e.assign(not_completed=0, weeks_attended=weeks, prior_fail=0)
    y = np.zeros(m, dtype=int)
    weeks_att = weeks.copy()
    exit_date = e.end_planned.to_numpy().copy()
    start = e.start_date.to_numpy()
    u_out, u_week = rng.random(m), rng.random(m)
    for i_tr, rows_i in e.groupby("i", sort=False).indices.items():           # rows are in time order per trainee
        for r in rows_i:
            earlier = [q for q in rows_i if start[q] < start[r] and exit_date[q] < start[r]]
            bump = 1.20 if any(y[q] == 1 for q in earlier) else 0.0
            p = 1 / (1 + np.exp(-(logit[r] + bump)))
            if u_out[r] < p:
                y[r] = 1
                w = 1 + int((weeks[r] - 1) * u_week[r] ** 1.9)              # exits are front-loaded
                weeks_att[r] = w
                exit_date[r] = start[r] + np.timedelta64(7 * w, "D")
    e["not_completed"], e["weeks_attended"], e["exit_date"] = y, weeks_att, exit_date

    # what the registers show at the end of week 2
    recorded = absences.copy()
    paper = np.isin(providers.region.to_numpy()[prov], PAPER_REGISTERS)
    recorded[paper] = rng.binomial(absences[paper], 0.35)
    weekly = (e.start_date.to_numpy() < np.datetime64("2022-07-01")) & ~paper    # registers typed in weekly until mid-2022
    recorded[weekly] = rng.binomial(absences[weekly], 0.65)
    apt = aptitude[i].astype(float).copy()
    walk_in = rng.random(m) < np.where(providers.provider_type.to_numpy()[prov] == "private centre", 0.22, 0.05)
    apt[walk_in] = np.nan                                                     # no entry test for walk-in registration
    dist_rec = dist.astype(float).copy()
    dist_rec[rng.random(m) < 0.05] = np.nan
    edu = education[i].astype(object)
    edu[rng.random(m) < 0.03] = None
    reason = np.where(y == 1, rng.choice(REASONS, m, p=[0.24, 0.22, 0.18, 0.08, 0.28]), "")
    reason = np.where((y == 1) & older_mask & (rng.random(m) < 0.5), "family", reason)

    out = pd.DataFrame({
        "trainee_id": trainee_id[i], "provider_id": providers.provider_id.to_numpy()[prov],
        "programme": e.programme, "programme_weeks": weeks, "delivery_mode": np.where(blended, "blended", "in person"),
        "start_date": e.start_date.dt.strftime("%Y-%m-%d"), "cohort_year": year,
        "region": region[i], "gender": gender[i], "age": age, "education": edu,
        "aptitude_score": apt, "has_dependants": dependants[i], "distance_km": dist_rec,
        "employment": employment, "enrolled_via": via, "stipend": stipend, "absences_w2": recorded,
        "weeks_attended": weeks_att, "exit_date": pd.to_datetime(exit_date).strftime("%Y-%m-%d"),
        "exit_reason": reason, "not_completed": y,
    })
    out = out.sample(frac=1.0, random_state=SEED).reset_index(drop=True)      # the export has no useful order
    out.insert(0, "enrolment_id", [f"E{k:06d}" for k in rng.permutation(np.arange(200_000, 200_000 + m))])
    write_csv_gz(out, DATA / "enrolments.csv.gz")
    providers.drop(columns=[]).to_csv(DATA / "providers.csv", index=False, lineterminator="\n")

    write_dictionary()
    write_decision()
    write_population(rng)
    write_received(out, rng)
    scorable = out[out.weeks_attended > 2]
    print(f"enrolments {len(out):,}, trainees {out.trainee_id.nunique():,}, providers {out.provider_id.nunique()}")
    print(f"not completed: all {out.not_completed.mean():.3f}, still enrolled after week 2 {scorable.not_completed.mean():.3f}"
          f" ({len(scorable):,} rows)")
    print(scorable.groupby("cohort_year").not_completed.agg(["size", "mean"]).round(3).to_string())


def write_csv_gz(df: pd.DataFrame, path: Path) -> None:
    """A gzip file with no time stamp inside, so the bytes are the same on every run."""
    path.parent.mkdir(parents=True, exist_ok=True)
    text = df.to_csv(index=False, lineterminator="\n")
    with open(path, "wb") as raw, gzip.GzipFile(fileobj=raw, mode="wb", mtime=0, filename="") as f:
        f.write(text.encode("utf-8"))


def write_dictionary() -> None:
    rows = [
        ("enrolment_id", "Key of the enrolment", "enrolment", "yes"),
        ("trainee_id", "Key of the trainee; the same trainee can enrol several times", "registration", "yes"),
        ("provider_id", "Training provider (60); see providers.csv", "enrolment", "yes"),
        ("programme", "Programme of study (10)", "enrolment", "yes"),
        ("programme_weeks", "Planned length in weeks: 12, 24 or 36", "enrolment", "yes"),
        ("delivery_mode", "in person, or blended (offered from 2024)", "enrolment", "yes"),
        ("start_date", "First day of the programme", "enrolment", "yes"),
        ("cohort_year", "Year of the start date: 2021 to 2025", "enrolment", "yes"),
        ("region", "Region of residence (13)", "registration", "yes"),
        ("gender", "F or M, as registered", "registration", "yes"),
        ("age", "Age in years at the start date", "enrolment", "yes"),
        ("education", "Highest level completed; blank when not recorded", "registration", "yes"),
        ("aptitude_score", "Entry test, 0 to 100; blank for walk-in registrations", "registration", "yes"),
        ("has_dependants", "1 when the trainee declared dependants", "registration", "yes"),
        ("distance_km", "Road distance from home to the provider; blank when the address did not geocode", "enrolment", "yes"),
        ("employment", "not working, part-time or full-time, at the start date", "enrolment", "yes"),
        ("enrolled_via", "self, employer, or social support referral", "enrolment", "yes"),
        ("stipend", "1 when the trainee receives the fund's monthly stipend", "enrolment", "yes"),
        ("absences_w2", "Days absent in the first two weeks (0 to 10), from the provider's register", "end of week 2", "yes"),
        ("weeks_attended", "Weeks attended before leaving or finishing", "exit", "no"),
        ("exit_date", "Day the trainee left or finished", "exit", "no"),
        ("exit_reason", "Reason given for leaving; blank for those who completed", "exit", "no"),
        ("not_completed", "TARGET: 1 when the trainee left without completing", "exit", "no"),
    ]
    pd.DataFrame(rows, columns=["column", "meaning", "recorded_at", "known_at_end_of_week_2"]).to_csv(
        DATA / "columns.csv", index=False, lineterminator="\n")


def write_decision() -> None:
    decision = {
        "owner": "Ghassan Al-Fayez, lead adviser",
        "action": "An adviser contacts the trainee in week 3 and agrees a support plan",
        "prediction_time": "the end of week 2 of the programme",
        "adviser_capacity_per_year": 800,
        "cost_of_one_support_plan_sar": 450,
        "cost_of_one_lost_place_sar": 9000,
        "share_of_supported_non_completers_who_then_complete": 0.25,
        "note": "The score puts a trainee on an adviser's list. It never decides eligibility, funding or a place.",
    }
    (DATA / "decision.json").write_text(json.dumps(decision, indent=1) + "\n", encoding="utf-8", newline="\n")


def write_population(rng) -> None:
    """The people the fund may serve (a synthetic reference table), to compare with who is in the training data."""
    region_share = {"Riyadh": 0.235, "Makkah": 0.225, "Eastern Province": 0.140, "Madinah": 0.068, "Asir": 0.072,
                    "Qassim": 0.045, "Jazan": 0.068, "Tabuk": 0.030, "Hail": 0.025, "Najran": 0.036,
                    "Al-Jouf": 0.022, "Al-Baha": 0.018, "Northern Borders": 0.016}
    age_share = {"17-19": 0.16, "20-24": 0.31, "25-34": 0.33, "35+": 0.20}
    gender_share = {"F": 0.52, "M": 0.48}
    rows = [("region", k, v) for k, v in region_share.items()]
    rows += [("age_band", k, v) for k, v in age_share.items()] + [("gender", k, v) for k, v in gender_share.items()]
    pd.DataFrame(rows, columns=["attribute", "value", "eligible_share"]).to_csv(
        DATA / "population.csv", index=False, lineterminator="\n")


def write_received(raw: pd.DataFrame, rng) -> None:
    """The feature table a colleague prepared, for Lab 3.3. Four leaks are planted in it."""
    t = raw[raw.weeks_attended > 2].copy()
    # leak 1: weeks_attended stays in the table (recorded at exit)
    # leak 2: a provider rate computed over all five years, each row's own outcome included
    t["provider_rate"] = t.groupby("provider_id").not_completed.transform("mean").round(4)
    # leak 3: a history count that reads every enrolment of the trainee, later ones and this one included
    t["trainee_noncompletions"] = t.groupby("trainee_id").not_completed.transform("sum")
    # an honest neighbour of leak 3, so that the hunt has to tell the two apart
    t = t.sort_values(["trainee_id", "start_date"])
    t["earlier_enrolments"] = t.groupby("trainee_id").cumcount()
    t = t.drop(columns=["exit_date", "exit_reason"])
    # leak 4: 1,350 rows copied under a new key when two provider systems were merged
    dup = t.sample(1350, random_state=SEED).copy()
    dup["enrolment_id"] = [f"E{k:06d}" for k in rng.permutation(np.arange(300_000, 300_000 + len(dup)))]
    t = pd.concat([t, dup]).sample(frac=1.0, random_state=SEED + 1).reset_index(drop=True)
    write_csv_gz(t, DATA / "received_table.csv.gz")


if __name__ == "__main__":
    main()
