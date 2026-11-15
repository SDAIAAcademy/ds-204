# /// script
# requires-python = ">=3.11"
# dependencies = ["numpy>=2.0", "pandas>=2.2", "scipy>=1.14", "scikit-learn>=1.6", "lightgbm>=4.5", "matplotlib>=3.9", "tabulate>=0.9"]
# ///
# %% [markdown]
# # Lab 2.4 · The paired difference and the decision
#
# Day 2 · Session 4 (Is the challenger better) · 25 min.
#
# Champion and challenger were scored on the same five folds. That makes their scores **pairs**, and the
# question "is the challenger better?" a question about five differences. You put an interval on their mean,
# check it a second way, and write a decision against a minimum effect that was stated before the contest.
#
# **AI policy: AI-tutor.**
#
# **Reference state.** The reference run log (`U.reference_log()`) and the reference validation-fold
# probabilities of both models (`U.oof()`).
#
# **Tasks**
# - [ ] 1 · Line up the fold scores of the two models (3')
# - [ ] 2 · Compute the paired differences and their mean (4')
# - [ ] 3 · Put a t interval on the mean difference (6')
# - [ ] 4 · Check it with a paired bootstrap over rows (7')
# - [ ] 5 · Write the decision in one line (5')
# - [ ] ✱ Stretch · The same comparison between the champion and the logistic rung

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
from scipy import stats

MIN_EFFECT = 0.01     # stated before the contest: the challenger must gain at least this much average precision
print(f"Minimum effect, agreed with Dr. Haya Al-Mogren before the contest: +{MIN_EFFECT} average precision.")
print("Below that, a tool with several models inside is not worth replacing one model the team can read.")

# %% [markdown]
# ## 1 · Line up the fold scores of the two models (3')

# %%
log = U.reference_log()
print(log[["lab", "rung", "mean_ap", "spread", "seconds", "fold_scores"]].to_string(index=False))
champion, challenger = U.fold_scores("boosted, 20 trials"), U.fold_scores("challenger")
pairs = pd.DataFrame({"champion": champion, "challenger": challenger}, index=range(1, 6)).rename_axis("fold")
print(pairs)

# %% [markdown]
# 🔍 **Look at** the two columns fold by fold. Both models find fold 2 harder than fold 1. Most of the spread of
# each column is shared between the models: it is the fold, not the model.
#
# ## 2 · Compute the paired differences and their mean (4')
#
# ✏️ **Exercise** · Return the five differences, challenger minus champion.

# %%
def differences(pairs: pd.DataFrame) -> np.ndarray:
    # BEGIN SOLUTION
    return (pairs.challenger - pairs.champion).to_numpy()
    # END SOLUTION


# %%
try:
    d = np.asarray(differences(pairs), dtype=float)
except NotImplementedError:
    print("↪ using the reference differences")
    d = (pairs.challenger - pairs.champion).to_numpy()
print("differences:", d.round(4), f"· mean {d.mean():+.4f}")
print(f"spread of the champion's folds {champion.std(ddof=1):.4f}, of the challenger's {challenger.std(ddof=1):.4f}, "
      f"of the differences {d.std(ddof=1):.4f}")
U.check(np.allclose(d, challenger - champion), "the differences are challenger minus champion, fold by fold")

# %% [markdown]
# 🔍 **Look at** the three spreads. The differences vary less than either model's scores, because the difficulty
# of the fold cancels. That is the whole point of pairing.
#
# ## 3 · Put a t interval on the mean difference (6')
#
# With five differences, the interval for their mean uses the t distribution with 4 degrees of freedom:
# mean ± t × (standard deviation of the differences / √5).
#
# ✏️ **Exercise** · Compute the 95% interval. `stats.t.ppf(0.975, df)` gives the multiplier.

# %%
def t_interval(d: np.ndarray, level: float = 0.95) -> tuple:
    # BEGIN SOLUTION
    half = stats.t.ppf(0.5 + level / 2, len(d) - 1) * d.std(ddof=1) / np.sqrt(len(d))
    return float(d.mean() - half), float(d.mean() + half)
    # END SOLUTION


# %%
try:
    low, high = t_interval(d)
except NotImplementedError:
    ref = U.paired_interval(challenger, champion)
    low, high = ref["low"], ref["high"]
    print("↪ using the reference interval")
U.keep("d2_4_t", {"mean": float(d.mean()), "low": low, "high": high, "differences": d.round(4).tolist()})
print(f"mean difference {d.mean():+.4f}, 95% interval {low:+.4f} to {high:+.4f} (multiplier {stats.t.ppf(0.975, 4):.2f})")
ref = U.paired_interval(challenger, champion)
U.check(abs(low - ref["low"]) < 1e-9 and abs(high - ref["high"]) < 1e-9, "the interval matches the reference")

# %% [markdown]
# 🔍 **Look at** the multiplier: 2.78, not 1.96. Five folds are few, and the interval says so. One more caution:
# the five folds share training rows, so the differences are not quite independent and the true uncertainty is,
# if anything, a little larger.
#
# ## 4 · Check it with a paired bootstrap over rows (7')
#
# A second opinion that does not lean on five numbers. Both models scored the same 21,770 validation rows. Draw
# those rows again with replacement, inside each fold, score both models on the same draw, and keep the
# difference of the mean scores. This is the paired bootstrap of DS-201.

# %%
both = U.oof(("champion", "challenger"))
print(f"{len(both):,} validation rows, each with the outcome and two probabilities")


def paired_bootstrap(both: pd.DataFrame, a: str, b: str, repeats: int, seed: int = U.SEED) -> np.ndarray:
    rng = np.random.default_rng(seed)
    groups = [g[[U.TARGET, a, b]].to_numpy() for _, g in both.groupby("fold")]
    out = np.empty(repeats)
    for r in range(repeats):
        diffs = []
        for g in groups:
            s = g[rng.integers(0, len(g), len(g))]          # the same rows for both models
            diffs.append(U.ap(s[:, 0], s[:, 2]) - U.ap(s[:, 0], s[:, 1]))
        out[r] = np.mean(diffs)
    return out


boot = paired_bootstrap(both, "champion", "challenger", U.pick(1000, 100))
b_low, b_high = np.percentile(boot, [2.5, 97.5])
U.keep("d2_4_boot", {"mean": float(boot.mean()), "low": float(b_low), "high": float(b_high),
                     "share_above_zero": float((boot > 0).mean()), "share_above_min_effect": float((boot > MIN_EFFECT).mean())})
print(f"paired bootstrap: mean {boot.mean():+.4f}, 95% interval {b_low:+.4f} to {b_high:+.4f}")
print(f"draws in which the challenger is ahead: {(boot > 0).mean():.0%}; ahead by the minimum effect: {(boot > MIN_EFFECT).mean():.0%}")
plt = U.style()
fig, ax = plt.subplots()
ax.hist(boot, bins=40, color="#1f8a8a")
ax.axvline(0, color="black", lw=1)
ax.axvline(MIN_EFFECT, color="#d9534f", lw=1.5, ls="--")
ax.set(xlabel="challenger minus champion, average precision", ylabel="bootstrap draws",
       title="Paired bootstrap: zero (black) and the minimum effect (dashed)")
U.savefig(fig, "D2_4_bootstrap.png")

# %% [markdown]
# 🔍 **Look at** the two intervals side by side. They answer slightly different questions: the t interval asks
# what other half-years would show, the bootstrap what other trainees in these half-years would show. When both
# say the same thing, say it. When they disagree, report both.
#
# ## 5 · Write the decision in one line (5')
#
# ✏️ **Exercise** · Complete `decide`. Return `"adopt the challenger"` when the whole interval is above the minimum
# effect, `"keep the champion"` when the whole interval is below it, and `"not shown: keep the champion"`
# otherwise.

# %%
def decide(low: float, high: float, min_effect: float) -> str:
    # BEGIN SOLUTION
    if low > min_effect:
        return "adopt the challenger"
    if high < min_effect:
        return "keep the champion"
    return "not shown: keep the champion"
    # END SOLUTION


# %%
def reference_decide(low, high, m):
    return "adopt the challenger" if low > m else "keep the champion" if high < m else "not shown: keep the champion"


try:
    verdict = decide(low, high, MIN_EFFECT)
except NotImplementedError:
    verdict = reference_decide(low, high, MIN_EFFECT)
U.check(verdict == reference_decide(low, high, MIN_EFFECT), "the decision follows the rule stated in advance")
seconds = log.set_index("rung").seconds
line = (f"On the same five folds the challenger differs from the champion by {d.mean():+.4f} average precision "
        f"(95% interval {low:+.4f} to {high:+.4f}; paired bootstrap {b_low:+.4f} to {b_high:+.4f}). "
        f"Minimum effect stated in advance: +{MIN_EFFECT}. Decision: {verdict}.")
print(line)
U.keep("d2_4_decision", {"verdict": verdict, "line": line})
U.save("D2_4_decision.md", "# Lab 2.4: is the challenger better?\n\n" + U.md(pairs.assign(difference=d), 4) + "\n\n" + line + "\n")

# %% [markdown]
# A decision line has four parts: the difference, its interval, the minimum effect, the decision. "Not shown" is
# a legitimate result. It is not the same as "shown to be equal", and the rule says in advance what happens
# then: the simpler model stays.
#
# ## ✱ Stretch · The same comparison between the champion and the logistic rung
#
# Run the same comparison between the champion and the rung "logistic + history" (`U.oof()` holds its
# probabilities as `logistic`). What would you tell Dr. Haya Al-Mogren about the value of the boosted trees?

# %%
logit = U.fold_scores("logistic + history")
extra = U.paired_interval(champion, logit)
U.keep("d2_4_vs_logistic", extra)
print(f"champion minus logistic: {extra['mean']:+.4f}, 95% interval {extra['low']:+.4f} to {extra['high']:+.4f}")

# %% [markdown]
# ## Checkpoint
#
# 1. Why compare differences fold by fold and not the two means with their spreads?
#    <details><summary>Answer</summary>Both models meet the same folds, so a hard fold lowers both. The
#    differences remove that shared part and vary much less than the scores.</details>
# 2. The interval runs from -0.004 to +0.012 and the minimum effect is +0.01. What is the decision?
#    <details><summary>Answer</summary>Not shown: the interval contains values below and above the minimum
#    effect. By the rule stated in advance the champion stays.</details>
# 3. Why state the minimum effect before the contest?
#    <details><summary>Answer</summary>Stated afterwards it tends to land just below whatever was measured. The
#    same reason as for a budget of trials: a rule chosen after the result is not a rule.</details>
#
# **Carry forward.** `artifacts/D2_4_decision.md`: the decision line. The capstone README needs one like it.
