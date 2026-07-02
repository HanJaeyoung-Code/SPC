import numpy as np
from scipy import stats
from step0_data_loader import df, batch_col, final_pen, yield_kg

# Check for accuracy in real batch grouping number
n = df[batch_col].nunique()
print(f"Grouping by: {batch_col!r}  ->  {n} batches")
assert n >= 90, f"Only {n} groups found — that is NOT ~100 batches. Stop and check!"

print("sample batch IDs:", sorted(df[batch_col].unique())[:5])

# Sanity check whether the crucial variables are in appropriate range
for c in ["Penicillin concentration(P:g/L)", "pH(pH:pH)",
          "Temperature(T:K)", "Dissolved oxygen concentration(DO2:mg/L)"]:
    if c in df.columns:
        print(f"{c[:42]:42}  min={df[c].min():.2f}  max={df[c].max():.2f}")
    else:
        print(f"{c[:42]:42}  *** not found ***")

print(f"\nFinal-concentration range : {final_pen.min():.2f} to {final_pen.max():.2f} g/L")
print(f"Harvested-yield range     : {yield_kg.min():.3e} to {yield_kg.max():.3e} (true output target)")

# ---------------------------------------------------------------------------
# Process capability against the 20 g/L minimum-CONCENTRATION spec.
# ---------------------------------------------------------------------------
# NOTE on scope: capability is computed on final CONCENTRATION because the only
# documented spec (LSL = 20 g/L) is a concentration limit. The ranking target
# elsewhere is harvested mass (yield_kg), which has no defined spec, so a
# capability index cannot be computed for it without one.
#
# Two indices, deliberately distinguished:
#   Ppk (process PERFORMANCE) uses the OVERALL standard deviation — total spread.
#   Cpk (process CAPABILITY)  uses the WITHIN-subgroup sigma, estimated here as
#                             MRbar / d2 (d2 = 1.128 for a moving range of 2).
# The previous version computed (mean - LSL)/(3 * overall_std) but called it
# "Cpk" — that quantity is actually Ppk. Both are reported below.
#
# CAVEAT: Cpk/Ppk assume an approximately NORMAL, in-control process. We test
# normality (Shapiro-Wilk) and flag the result; if non-normal, the indices are
# descriptive only and must not be read as a precise % out of spec.
lsl   = 20.0
x     = final_pen.sort_index().values
mean  = x.mean()

sd_overall = x.std(ddof=1)                       # for Ppk
mr_bar     = np.abs(np.diff(x)).mean()           # mean moving range (span 2)
d2         = 1.128
sd_within  = mr_bar / d2                          # for Cpk

ppk = (mean - lsl) / (3 * sd_overall)
cpk = (mean - lsl) / (3 * sd_within)

W, p_norm = stats.shapiro(x)

print(f"\nProcess capability vs. LSL = {lsl} g/L (on final concentration)")
print(f"  mean={mean:.2f}  sd_overall={sd_overall:.2f}  sd_within(MRbar/d2)={sd_within:.2f}")
print(f"  Ppk (overall spread)   = {ppk:.2f}")
print(f"  Cpk (within-subgroup)  = {cpk:.2f}")
print(f"  Shapiro-Wilk normality : W={W:.3f}  p={p_norm:.4f}  "
      f"-> {'approx. normal' if p_norm > 0.05 else 'NOT normal (index is descriptive only)'}")
worst = min(cpk, ppk)
if worst < 1.0:
    print("  --> index < 1.0: process is NOT capable of reliably meeting spec.")
elif worst < 1.33:
    print("  --> 1.0 <= index < 1.33: marginally capable — improvement recommended.")
else:
    print("  --> index >= 1.33: process is capable.")

print("\n15 lowest-yield batches (by harvested mass):\n", yield_kg.sort_values().head(15))
