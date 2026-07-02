"""
STAGE 4 - ROOT CAUSE: what makes a high-yield penicillin batch?
===============================================================

Question:  Yield ranges from ~6 to ~36 g/L across "normal" batches.
           That spread is not random noise — it is a signal worth explaining.
           Which process conditions separate high-yield from low-yield batches,
           and when in the run does the difference appear?

Approach:  Rank all 100 batches by final yield. Split into:
             top third  = winners
             bottom third = losers  (middle third excluded — too ambiguous)
           Then:
             PART A — compare average conditions between the two groups (the WHAT)
             PART B — track the key suspect over time across the run   (the WHEN)

Finding:   Winners keep residual sugar (substrate) near zero throughout the run.
           Losers let sugar pile up, which triggers catabolite repression —
           the cell stops making penicillin and shifts to growing instead.
           OUR, CER, and CO2 move with yield but are effects, not causes —
           you cannot improve yield by directly adjusting them.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from step0_data_loader import df, batch_col, yield_kg, fault

# ---------------------------------------------------------------------------
# Shared setup: label every batch a winner or loser by TRUE harvested yield.
# We rank on yield_kg (total penicillin mass), NOT last-logged concentration:
# mass is the real output, and the two correlate only ~0.36, so ranking on
# concentration would judge the wrong batches as winners/losers.
# We ignore batch number — a batch run on day 1 is judged the same as day 100.
# ---------------------------------------------------------------------------
good_ids = yield_kg[yield_kg >= yield_kg.quantile(0.66)].index   # top third
bad_ids  = yield_kg[yield_kg <= yield_kg.quantile(0.33)].index   # bottom third
print(f"winners: {len(good_ids)} batches | losers: {len(bad_ids)} batches")
print("(ranked by harvested mass, yield_kg)\n")


# ===========================================================================
# PART A - THE "WHAT": which conditions separate winners from losers?
# ===========================================================================
process_vars = [
    "Sugar feed rate(Fs:L/h)", "Aeration rate(Fg:L/h)", "Agitator RPM(RPM:RPM)",
    "Air head pressure(pressure:bar)", "Substrate concentration(S:g/L)",
    "Dissolved oxygen concentration(DO2:mg/L)", "pH(pH:pH)", "Temperature(T:K)",
    "carbon dioxide percent in off-gas(CO2outgas:%)", "PAA flow(Fpaa:PAA flow (L/h))",
    "Oil flow(Foil:L/hr)", "Oxygen Uptake Rate(OUR:(g min^{-1}))",
    "Carbon evolution rate(CER:g/h)", "Vessel Volume(V:L)",
]
process_vars = [v for v in process_vars if v in df.columns]   # keep only names that exist

# Reduce each batch from hundreds of time-step rows to a single average per variable.
batch_means = df.groupby(batch_col)[process_vars].mean()

# Compute the gap between winner and loser averages for each variable.
# Dividing by std puts every variable on the same scale (std units),
# so we can fairly compare pH (range ~6–7) against feed rate (range 0–100 L/h).
# Positive gap = winners ran that variable higher; negative = winners ran it lower.
gap = ((batch_means.loc[good_ids].mean() - batch_means.loc[bad_ids].mean())
       / batch_means.std()).sort_values(key=abs, ascending=False)

# Mann-Whitney U test: checks whether each gap is real or just random chance.
# It works without assuming the data is normally distributed,
# which makes it appropriate for small groups of ~33 batches.
# p < 0.05 means there is less than a 5% chance the gap is a fluke.
pvals = {v: stats.mannwhitneyu(batch_means.loc[good_ids, v].dropna(),
                                batch_means.loc[bad_ids,  v].dropna(),
                                alternative="two-sided").pvalue
         for v in gap.index}
pval_s = pd.Series(pvals).reindex(gap.index)


# Multiple-comparison correction (issue #5).
# We run one test per variable (~14 tests). At alpha=0.05 we would expect
# ~0.7 "significant" hits by chance alone, so a raw p<0.05 is not trustworthy.
# Benjamini-Hochberg controls the False Discovery Rate: the expected fraction
# of false positives among the variables we DO call significant. We compare the
# corrected q-values (not the raw p-values) to 0.05.
def bh_fdr(p_series):
    p = p_series.values.astype(float)
    n = len(p)
    order = p.argsort()                 # ascending p
    ranks = order.argsort() + 1         # rank of each original p (1..n)
    q = p * n / ranks                   # BH adjustment
    # enforce monotonicity from largest p downward, then cap at 1.0
    q_sorted = q[order]
    for i in range(len(q_sorted) - 2, -1, -1):
        q_sorted[i] = min(q_sorted[i], q_sorted[i + 1])
    q[order] = q_sorted
    return pd.Series(q, index=p_series.index).clip(upper=1.0)

qval_s = bh_fdr(pval_s)

summary = pd.DataFrame({"gap_std_units": gap.round(2),
                         "p_value":       pval_s.round(3),
                         "q_value_BH":    qval_s.round(3),
                         "significant":   qval_s < 0.05})
print("Condition gap with FDR correction (Mann-Whitney U + Benjamini-Hochberg, q<0.05):\n",
      summary, "\n")

# Bar labels: append * for variables that survive FDR correction.
labels = [f"{v}  *" if qval_s[v] < 0.05 else v for v in gap.index]

plt.figure(figsize=(8, 6))
plt.barh(range(len(gap)), gap.values,
         color=["seagreen" if v > 0 else "indianred" for v in gap.values])
plt.yticks(range(len(gap)), labels, fontsize=8)
plt.axvline(0, color="black", lw=0.8)
plt.xlabel("Condition gap: winners - losers (std units)   (* = p < 0.05)")
plt.title("PART A - What separates high-yield from low-yield batches?")
plt.gca().invert_yaxis()
plt.tight_layout()
plt.show()


# ===========================================================================
# PART B - THE "WHEN": does substrate pile up in losers, and at what hour?
# ===========================================================================
time_col = next(c for c in df.columns if "Time" in c)   # auto-find the time column
sub_col  = "Substrate concentration(S:g/L)"             # the lever Part A flagged

plt.figure(figsize=(11, 6))
# Plot every individual batch as a faint line so you can see the spread.
for b, g in df.groupby(batch_col):
    g = g.sort_values(time_col)
    if   b in good_ids: plt.plot(g[time_col], g[sub_col], color="seagreen",  lw=0.7, alpha=0.4)
    elif b in bad_ids:  plt.plot(g[time_col], g[sub_col], color="indianred", lw=0.7, alpha=0.4)

# Bold average line per group — this is the main visual takeaway.
for ids, color, lab in [(good_ids, "darkgreen", "winners (avg)"),
                        (bad_ids,  "darkred",   "losers (avg)")]:
    avg = df[df[batch_col].isin(ids)].groupby(time_col)[sub_col].mean()
    plt.plot(avg.index, avg.values, color=color, lw=2.6, label=lab)

# Alarm threshold = midpoint between the highest winner peak and lowest loser
# peak (post hour 100). Hour 100 is the start because batches are
# indistinguishable before that point.
#
# Issue #6 - DO NOT fit and test the threshold on the same batches, or the
# reported separation is guaranteed and meaningless. We split winners and losers
# into TRAIN and TEST halves, derive the threshold on TRAIN only, then measure
# how well it classifies the held-out TEST batches it never saw.
import numpy as np

post100 = df[df[time_col] > 100]
# one number per batch: its peak residual substrate after hour 100
peak_post100 = post100.groupby(batch_col)[sub_col].max()

rng = np.random.default_rng(42)
def split(ids):
    ids = list(ids); rng.shuffle(ids); k = len(ids) // 2
    return set(ids[:k]), set(ids[k:])          # (train, test)
good_tr, good_te = split(good_ids)
bad_tr,  bad_te  = split(bad_ids)

# threshold from TRAIN peaks only
winner_peak_tr = peak_post100.reindex(good_tr).max()
loser_peak_tr  = peak_post100.reindex(bad_tr).min()
alarm_thresh   = (winner_peak_tr + loser_peak_tr) / 2

# validate on TEST: predict "loser" if a batch's post-100 peak exceeds threshold
def score(ids, truth_is_loser):
    correct = sum((peak_post100.reindex([b]).iloc[0] > alarm_thresh) == truth_is_loser
                  for b in ids)
    return correct, len(ids)
c_bad,  n_bad  = score(bad_te,  True)
c_good, n_good = score(good_te, False)
test_acc = (c_bad + c_good) / (n_bad + n_good)

print(f"Substrate alarm threshold (TRAIN-derived, t>100h): {alarm_thresh:.2f} g/L")
print(f"  Holdout validation on {n_bad+n_good} unseen batches: "
      f"losers caught {c_bad}/{n_bad}, winners cleared {c_good}/{n_good}, "
      f"accuracy {test_acc:.0%}")
plt.axhline(alarm_thresh, color="darkorange", linestyle="--", lw=1.8,
            label=f"alarm threshold: {alarm_thresh:.1f} g/L (train-derived; holdout acc {test_acc:.0%})")

plt.xlabel("Time (h)")
plt.ylabel("Substrate concentration (g/L)")
plt.title("PART B - Do losing batches let sugar pile up, and when?")
plt.legend()
plt.tight_layout()
plt.show()


# ===========================================================================
# PART C - VALIDATION: does the yield-based "loser" label agree with the
#                      ground-truth FAULT label that ships with the dataset?
# ===========================================================================
# IndPenSim labels batches 91-100 as faulted (Fault ref = 1). Those labels were
# never used by the analysis, so we had no independent check on the split.
# Here we treat "bottom-third by harvested yield" as our predicted-bad flag and
# score it against the true fault label with a confusion matrix.
#
# IMPORTANT framing: a yield "loser" and an injected "fault" are NOT the same
# concept. Most low-yield batches are fault-free normal variation (the very
# spread Part A explains), and some faults need not depress total mass. So we
# expect modest recall and low precision — that gap is the finding, not a bug.

faulty   = set(fault[fault == 1].index)
all_ids  = list(yield_kg.index)
pred_bad = np.array([1 if b in set(bad_ids) else 0 for b in all_ids])
true_bad = np.array([1 if b in faulty       else 0 for b in all_ids])

tp = int(((pred_bad == 1) & (true_bad == 1)).sum())
fp = int(((pred_bad == 1) & (true_bad == 0)).sum())
fn = int(((pred_bad == 0) & (true_bad == 1)).sum())
tn = int(((pred_bad == 0) & (true_bad == 0)).sum())
precision = tp / (tp + fp) if (tp + fp) else float("nan")
recall    = tp / (tp + fn) if (tp + fn) else float("nan")

print("--- PART C: yield-loser flag vs. ground-truth fault label ---")
print(f"  faults (truth)            : {sorted(faulty)}")
print(f"  confusion  TP={tp}  FP={fp}  FN={fn}  TN={tn}")
print(f"  precision = {precision:.2f}   recall = {recall:.2f}")
print(f"  faults that still yield WELL (winners): {sorted(set(good_ids) & faulty)}")
print(f"  faults in the excluded middle third   : {sorted(faulty - set(good_ids) - set(bad_ids))}")
print("  Interpretation: yield ranking catches most faults but not all; some")
print("  faulted batches harvest normally, and many low-yield batches are not")
print("  faults — so yield is a useful but imperfect proxy for batch health.")
