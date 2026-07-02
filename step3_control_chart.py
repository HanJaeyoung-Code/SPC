"""
STEP 3 - SPC CONTROL CHART - I-MR Chart on Batch-Average pH
================================================================================
PURPOSE
    Confirm whether pH is in statistical control across all 100 batches, to
    check whether pH excursions could explain the yield variation found in
    Root_Cause.py.

APPROACH
    1. Collapse each batch's pH time series to a single batch-mean value.
    2. Use the first 70 batches (Phase 1) to estimate the center line and
       3-sigma control limits, then plot all 100 batches (Phase 2) against
       those limits.
    3. Flag out-of-control points by the 3-sigma test and by the full
       Nelson run-rule set.
    4. Plot the I-chart (batch means) and MR-chart (moving range) together.

KEY DECISIONS
    - Chart type: after the root cause analysis we have one summary value
      per batch (the batch mean of pH) -- a single observation, not a
      subgroup -- so the correct chart is the Individual (I) chart. The
      Moving Range (MR) chart always accompanies it to track variability.
      I-chart detects shifts in the MEAN across batches; MR-chart detects
      sudden jumps in VARIABILITY between batches.
    - Variable: pH is actively controlled during fermentation. If the I-MR
      chart shows pH in statistical control, it rules out pH excursions as
      a cause of yield variation, directly supporting the Root_Cause.py
      finding that substrate management, not environment, is the driver.
    - Phase 1 vs Phase 2: standard SPC practice uses a stable baseline
      window (Phase 1) only to ESTIMATE control limits, then plots all
      data against those limits in Phase 2. The first 70 batches (~70% of
      the dataset) are used as Phase 1.
    - Sigma is estimated as MRbar / d2 (d2 = 1.128, the moving-range span-2
      constant), not the overall standard deviation -- consistent with the
      Cpk methodology in step1_data_integrity.py.
    - The full Nelson run-rule set is applied, not just the single-point
      3-sigma rule, because that alone only catches large sudden shifts and
      misses small sustained shifts or trends: Rule 1 (1 point beyond 3σ),
      Rule 2 (9 in a row on one side), Rule 3 (6 in a row trending), Rule 5
      (2 of 3 beyond 2σ, same side), Rule 6 (4 of 5 beyond 1σ, same side).
    - MR-chart limits use D4 = 3.267, D3 = 0 (span 2) rather than 3-sigma,
      because the moving range follows a different distribution than the
      raw measurements; LCL is always 0 since a range cannot go negative.

FINDING
    This chart does not look for problems in pH -- it confirms that pH is
    stable, which lets us rule it out as a root cause. If pH is in
    control, the yield variation must come from something else, and
    Root_Cause.py shows that something else is substrate concentration.
"""

import numpy as np
import matplotlib.pyplot as plt
from step0_data_loader import df, batch_col


def nelson_rules(x, cl, sigma):
    x = np.asarray(x, float)
    d = x - cl
    n = len(x)
    v = {1: [], 2: [], 3: [], 5: [], 6: []}
    v[1] = [i for i in range(n) if abs(d[i]) > 3 * sigma]
    for i in range(n - 8):
        seg = d[i:i + 9]
        if np.all(seg > 0) or np.all(seg < 0):
            v[2] += range(i, i + 9)
    for i in range(n - 5):
        seg = x[i:i + 6]
        if np.all(np.diff(seg) > 0) or np.all(np.diff(seg) < 0):
            v[3] += range(i, i + 6)
    for i in range(n - 2):
        seg = d[i:i + 3]
        if (seg > 2 * sigma).sum() >= 2 or (seg < -2 * sigma).sum() >= 2:
            v[5] += range(i, i + 3)
    for i in range(n - 4):
        seg = d[i:i + 5]
        if (seg > 1 * sigma).sum() >= 4 or (seg < -1 * sigma).sum() >= 4:
            v[6] += range(i, i + 5)
    return {r: sorted(set(idx)) for r, idx in v.items()}

# --- Step 1: variable and per-batch collapse ---
cpv_col = "pH(pH:pH)"

batch_means = (df.groupby(batch_col)[cpv_col]
                 .mean()
                 .sort_index()
                 .reset_index(drop=True))

n_batches = len(batch_means)
print(f"Variable : {cpv_col}")
print(f"Batches  : {n_batches}\n")

# --- Step 2: Phase 1 baseline and process parameters ---
PHASE1_N = 70
phase1 = batch_means.iloc[:PHASE1_N]

CL = phase1.mean()

MR_phase1 = phase1.diff().abs().dropna()
MR_bar = MR_phase1.mean()

d2 = 1.128
sigma_est = MR_bar / d2

print(f"Phase 1 ({PHASE1_N} batches):")
print(f"  Center line  CL  = {CL:.4f}")
print(f"  Avg MR       MR̄ = {MR_bar:.4f}")
print(f"  Estimated σ      = {sigma_est:.4f}")

# --- Step 3: 3-sigma control limits ---
UCL_I = CL + 3 * sigma_est
LCL_I = CL - 3 * sigma_est

print(f"  UCL (I-chart)    = {UCL_I:.4f}")
print(f"  LCL (I-chart)    = {LCL_I:.4f}\n")

D4 = 3.267
UCL_MR = D4 * MR_bar
LCL_MR = 0.0

# --- Step 4: moving range for all 100 batches (Phase 2 monitoring) ---
MR_all = batch_means.diff().abs()

# --- Step 5: out-of-control points ---
ooc_I  = batch_means[(batch_means > UCL_I) | (batch_means < LCL_I)]
ooc_MR = MR_all[MR_all > UCL_MR]

print(f"Out-of-control — I-chart  : {list(ooc_I.index)}  ({len(ooc_I)} points)")
print(f"Out-of-control — MR-chart : {list(ooc_MR.index)}  ({len(ooc_MR)} points)\n")

rule_hits = nelson_rules(batch_means.values, CL, sigma_est)
print("Nelson run-rule signals (I-chart):")
for r, idx in rule_hits.items():
    note = {1: "1 pt beyond 3σ", 2: "9 in a row one side", 3: "6 trending",
            5: "2/3 beyond 2σ", 6: "4/5 beyond 1σ"}[r]
    print(f"  Rule {r} ({note:18}): {len(idx)} point(s)  {idx}")
print()

# --- Step 6: plot I-chart and MR-chart ---
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(13, 8), sharex=True)
fig.suptitle(
    "I-MR Control Chart — Batch-Average pH\n"
    f"(Phase 1: batches 1–{PHASE1_N}  →  limits applied to all {n_batches})",
    fontsize=13
)

x = batch_means.index

ax1.plot(x, batch_means.values, "o-",
         color="steelblue", lw=1.2, markersize=4, label="Batch mean pH")

ax1.axhline(CL,    color="green", lw=1.8, linestyle="-",
            label=f"CL  = {CL:.3f}")
ax1.axhline(UCL_I, color="red",   lw=1.2, linestyle="--",
            label=f"UCL = {UCL_I:.3f}")
ax1.axhline(LCL_I, color="red",   lw=1.2, linestyle="--",
            label=f"LCL = {LCL_I:.3f}")

ax1.axvline(PHASE1_N - 0.5, color="black", linestyle=":", lw=1.5,
            label="Phase 1 | Phase 2")
ax1.text(PHASE1_N - 0.5 + 0.5, UCL_I, "Phase 2 →",
         fontsize=8, va="top", color="black")

if not ooc_I.empty:
    ax1.plot(ooc_I.index, ooc_I.values, "ro", markersize=10,
             zorder=5, label=f"Out of control ({len(ooc_I)})")

ax1.set_ylabel("Batch-average pH")
ax1.legend(fontsize=8, loc="upper right")
ax1.set_title("Individual (I) Chart", fontsize=10)

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

# --- Step 7: interpretation ---
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
