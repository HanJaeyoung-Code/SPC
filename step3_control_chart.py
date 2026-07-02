"""
SPC CONTROL CHART — I-MR Chart on Batch-Average pH
===================================================

WHY THIS CHART TYPE,  I-MR (Individual & Moving Range)?
  After the root cause analysis we have one summary value per batch
  (the batch mean of a process variable). That is a single observation,
  not a subgroup , so the correct chart is the Individual (I) chart.
  The Moving Range (MR) chart always accompanies it to track variability.
    • I-chart  → detects shifts in the MEAN across batches
    • MR-chart → detects sudden jumps in VARIABILITY between batches

WHY pH?
  pH is actively controlled during fermentation. If the I-MR chart
  shows pH is in statistical control, it rules out pH excursions as a
  cause of yield variation, directly supporting the finding in
  Root_Cause.py that substrate management, not environment, is the driver.

WHY PHASE 1 vs PHASE 2?
  Standard SPC practice:
    Phase 1 --> a stable baseline window used only to ESTIMATE control limits
    Phase 2 —-> all data plotted against those limits to see what's in/out
  We use the first 70 batches as Phase 1 (~70% of the dataset).
"""

import numpy as np
import matplotlib.pyplot as plt
from step0_data_loader import df, batch_col


# ---------------------------------------------------------------------------
# Nelson run rules (issue #4)
# ---------------------------------------------------------------------------
# A single point beyond 3 sigma (Rule 1) catches only large, sudden shifts.
# Real SPC also watches for SMALL but sustained shifts and trends that never
# breach 3 sigma. The "zones" are bands of width 1 sigma either side of CL:
#   Zone A = 2-3 sigma, Zone B = 1-2 sigma, Zone C = 0-1 sigma.
# We implement the most common subset:
#   Rule 1: 1 point beyond 3 sigma            (gross shift)
#   Rule 2: 9 points in a row on one side     (sustained mean shift)
#   Rule 3: 6 points in a row increasing/decreasing (trend / drift)
#   Rule 5: 2 of 3 in a row beyond 2 sigma, same side
#   Rule 6: 4 of 5 in a row beyond 1 sigma, same side
def nelson_rules(x, cl, sigma):
    x = np.asarray(x, float)
    d = x - cl                      # signed distance from center line
    n = len(x)
    v = {1: [], 2: [], 3: [], 5: [], 6: []}
    v[1] = [i for i in range(n) if abs(d[i]) > 3 * sigma]
    for i in range(n - 8):                                  # Rule 2: 9 same side
        seg = d[i:i + 9]
        if np.all(seg > 0) or np.all(seg < 0):
            v[2] += range(i, i + 9)
    for i in range(n - 5):                                  # Rule 3: 6 trending
        seg = x[i:i + 6]
        if np.all(np.diff(seg) > 0) or np.all(np.diff(seg) < 0):
            v[3] += range(i, i + 6)
    for i in range(n - 2):                                  # Rule 5: 2/3 > 2 sigma
        seg = d[i:i + 3]
        if (seg > 2 * sigma).sum() >= 2 or (seg < -2 * sigma).sum() >= 2:
            v[5] += range(i, i + 3)
    for i in range(n - 4):                                  # Rule 6: 4/5 > 1 sigma
        seg = d[i:i + 5]
        if (seg > 1 * sigma).sum() >= 4 or (seg < -1 * sigma).sum() >= 4:
            v[6] += range(i, i + 5)
    return {r: sorted(set(idx)) for r, idx in v.items()}

# ---------------------------------------------------------------------------
# Step 1 — Choose the variable and collapse time-series → one value per batch
# ---------------------------------------------------------------------------
# Each batch has hundreds of rows (one per time-step). We reduce each batch
# to its mean pH — one number per batch. That single number is the
# "individual" observation the I-chart will track across batches.

cpv_col = "pH(pH:pH)"

batch_means = (df.groupby(batch_col)[cpv_col]
                 .mean()
                 .sort_index()          # ensure chronological batch order
                 .reset_index(drop=True))  # use position 0-99 as x-axis

n_batches = len(batch_means)
print(f"Variable : {cpv_col}")
print(f"Batches  : {n_batches}\n")

# ---------------------------------------------------------------------------
# Step 2 — Define Phase 1 (baseline) and estimate process parameters
# ---------------------------------------------------------------------------
PHASE1_N = 70          # batches 0-69 are used to build the limits
phase1 = batch_means.iloc[:PHASE1_N]

# Center Line = mean of Phase 1 batch averages.
# This is our best estimate of where the process "should" sit.
CL = phase1.mean()

# Moving Range = absolute difference between consecutive observations.
# MR_i = |x_i − x_{i-1}|
# We compute it only within Phase 1 so that limits are set on stable data.
# The first entry is NaN after .diff(), so we drop it.
MR_phase1 = phase1.diff().abs().dropna()
MR_bar = MR_phase1.mean()   # average moving range within Phase 1

# Convert MR_bar → estimated σ using the constant d2.
# d2 = 1.128 is a fixed value from statistical tables (the expected value
# of the range of 2 observations drawn from a standard normal distribution).
# It un-biases the moving range so it estimates the true process std dev.
d2 = 1.128
sigma_est = MR_bar / d2

print(f"Phase 1 ({PHASE1_N} batches):")
print(f"  Center line  CL  = {CL:.4f}")
print(f"  Avg MR       MR̄ = {MR_bar:.4f}")
print(f"  Estimated σ      = {sigma_est:.4f}")

# ---------------------------------------------------------------------------
# Step 3 — Calculate 3-sigma control limits
# ---------------------------------------------------------------------------
# ±3σ captures 99.73 % of variation in a stable normal process.
# A point outside these bounds is extremely unlikely by chance alone
# (probability ~0.27 %) — so we treat it as a real signal worth investigating.

UCL_I = CL + 3 * sigma_est    # Upper Control Limit — I-chart
LCL_I = CL - 3 * sigma_est    # Lower Control Limit — I-chart

print(f"  UCL (I-chart)    = {UCL_I:.4f}")
print(f"  LCL (I-chart)    = {LCL_I:.4f}\n")

# MR chart limits use different constants (D4, D3) because the moving range
# follows a different distribution than the raw measurements.
# For span = 2: D4 = 3.267, D3 = 0 (LCL is always 0 — range can't go negative).
D4 = 3.267
UCL_MR = D4 * MR_bar
LCL_MR = 0.0

# ---------------------------------------------------------------------------
# Step 4 — Compute moving range for ALL 100 batches (for the MR chart)
# ---------------------------------------------------------------------------
# We apply the Phase 1 limits to all 100 batches — that is the whole point
# of Phase 2: monitoring. The MR series for all batches is computed here.
MR_all = batch_means.diff().abs()   # first value is NaN — that is fine

# ---------------------------------------------------------------------------
# Step 5 — Identify out-of-control (OOC) points
# ---------------------------------------------------------------------------
# Any batch whose mean falls outside [LCL_I, UCL_I] is a signal.
# We flag them separately so they can be highlighted on the chart.
ooc_I  = batch_means[(batch_means > UCL_I) | (batch_means < LCL_I)]
ooc_MR = MR_all[MR_all > UCL_MR]

print(f"Out-of-control — I-chart  : {list(ooc_I.index)}  ({len(ooc_I)} points)")
print(f"Out-of-control — MR-chart : {list(ooc_MR.index)}  ({len(ooc_MR)} points)\n")

# Apply the full Nelson run-rule set, not just the single-point rule. This
# catches small sustained shifts and trends a 3-sigma test alone would miss.
rule_hits = nelson_rules(batch_means.values, CL, sigma_est)
print("Nelson run-rule signals (I-chart):")
for r, idx in rule_hits.items():
    note = {1: "1 pt beyond 3σ", 2: "9 in a row one side", 3: "6 trending",
            5: "2/3 beyond 2σ", 6: "4/5 beyond 1σ"}[r]
    print(f"  Rule {r} ({note:18}): {len(idx)} point(s)  {idx}")
print()

# ---------------------------------------------------------------------------
# Step 6 — Plot: I-chart on top, MR-chart on bottom
# ---------------------------------------------------------------------------
# Convention: always show both charts together.
# sharex=True links the x-axes so zooming/panning stays in sync.

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(13, 8), sharex=True)
fig.suptitle(
    "I-MR Control Chart — Batch-Average pH\n"
    f"(Phase 1: batches 1–{PHASE1_N}  →  limits applied to all {n_batches})",
    fontsize=13
)

x = batch_means.index   # positions 0 to 99

# --- I-chart (top panel) ---
ax1.plot(x, batch_means.values, "o-",
         color="steelblue", lw=1.2, markersize=4, label="Batch mean pH")

# The three horizontal reference lines
ax1.axhline(CL,    color="green", lw=1.8, linestyle="-",
            label=f"CL  = {CL:.3f}")
ax1.axhline(UCL_I, color="red",   lw=1.2, linestyle="--",
            label=f"UCL = {UCL_I:.3f}")
ax1.axhline(LCL_I, color="red",   lw=1.2, linestyle="--",
            label=f"LCL = {LCL_I:.3f}")

# Vertical dashed line separating Phase 1 from Phase 2
ax1.axvline(PHASE1_N - 0.5, color="black", linestyle=":", lw=1.5,
            label="Phase 1 | Phase 2")
ax1.text(PHASE1_N - 0.5 + 0.5, UCL_I, "Phase 2 →",
         fontsize=8, va="top", color="black")

# Highlight OOC points with red circles on top of the line
if not ooc_I.empty:
    ax1.plot(ooc_I.index, ooc_I.values, "ro", markersize=10,
             zorder=5, label=f"Out of control ({len(ooc_I)})")

ax1.set_ylabel("Batch-average pH")
ax1.legend(fontsize=8, loc="upper right")
ax1.set_title("Individual (I) Chart", fontsize=10)

# --- MR-chart (bottom panel) ---
ax2.plot(x, MR_all.values, "o-",
         color="darkorange", lw=1.2, markersize=4, label="Moving range")

ax2.axhline(MR_bar, color="green", lw=1.8, linestyle="-",
            label=f"MR̄  = {MR_bar:.4f}")
ax2.axhline(UCL_MR, color="red",   lw=1.2, linestyle="--",
            label=f"UCL = {UCL_MR:.4f}")
ax2.axhline(LCL_MR, color="red",   lw=1.2, linestyle="--",
            label=f"LCL = {LCL_MR:.1f}")

ax2.axvline(PHASE1_N - 0.5, color="black", linestyle=":", lw=1.5)

if not ooc_MR.empty:
    ax2.plot(ooc_MR.index, ooc_MR.values, "ro", markersize=10,
             zorder=5, label=f"Out of control ({len(ooc_MR)})")

ax2.set_xlabel("Batch number (0-indexed)")
ax2.set_ylabel("Moving range of pH")
ax2.legend(fontsize=8, loc="upper right")
ax2.set_title("Moving Range (MR) Chart", fontsize=10)

plt.tight_layout()
plt.show()

# ---------------------------------------------------------------------------
# Step 7 — Interpret and connect back to the root cause narrative
# ---------------------------------------------------------------------------
# The purpose of this chart is not to find problems in pH — it is to CONFIRM
# that pH is stable, which lets us rule it out as a root cause.
# If pH is in control, the yield variation must come from something else
# (and Root_Cause.py shows it is substrate concentration).

print("--- Interpretation ---")
n_ooc = len(ooc_I)
n_rule_signals = len(set().union(*rule_hits.values()))
print(f"(Run rules flag {n_rule_signals} batch-position(s) by any Nelson rule.)")
if n_ooc == 0 and n_rule_signals == 0:
    print("pH is IN statistical control across all 100 batches (no rule signals).")
    print("→ The pH control loop is functioning correctly in every batch.")
    print("→ pH excursions are NOT a driver of yield variation.")
    print("→ This supports the root cause finding: substrate management is the lever.")
elif n_ooc <= 3:
    print(f"pH shows {n_ooc} out-of-control point(s) — investigate those batches.")
    print("→ pH is largely stable; substrate management remains the primary driver.")
else:
    print(f"pH shows {n_ooc} OOC points — pH stability may be a contributing factor.")
    print("→ Consider adding pH to the root-cause comparison in Root_Cause.py.")
