"""
STAGE 3 - TRAJECTORY ANALYSIS: when do bad batches break away from the pack?
=============================================================================

Question:  Yield varies widely across batches. But *when* in the run do
           low-yield batches break away from the rest — and is it visible
           in the penicillin accumulation curve itself?

Approach:  Use the first 90 batches to set 3-sigma limits on final yield,
           then flag any batch that falls outside those limits. Reconstruct
           the full penicillin time series for every batch. Overlay flagged
           batches (red) against normal ones (grey), and draw a single
           "golden batch" curve — the average trajectory of all healthy
           batches — as the reference.

Finding:   Flagged batches are indistinguishable from normal ones in the early
           hours. The divergence becomes visible around mid-run (~hour 100),
           which pins the intervention window and motivates the deeper
           root-cause search in Root_Cause.py.
"""

import matplotlib.pyplot as plt
from step0_data_loader import df, batch_col, yield_kg   # batch_col already verified upstream

time_col = next(c for c in df.columns if "Time" in c)
pen_col  = "Penicillin concentration(P:g/L)"

# Flag batches on TRUE harvested yield (mass), consistent with Root_Cause.
# First 90 batches (by position) define the baseline for the 3-sigma limits;
# IndPenSim's faulted batches are 91-100, so this baseline is the normal set.
base = yield_kg.iloc[:90]
ucl, lcl = base.mean() + 3 * base.std(), base.mean() - 3 * base.std()
bad = yield_kg[(yield_kg > ucl) | (yield_kg < lcl)].index
print("Flagged batches:", list(bad))

# Overlay every batch's penicillin trajectory: red = flagged, grey = normal.
plt.figure(figsize=(11, 6))
for b, g in df.groupby(batch_col):
    g = g.sort_values(time_col)
    plt.plot(g[time_col], g[pen_col],
             color="red" if b in bad else "lightgrey",
             lw=1.5 if b in bad else 0.8,
             alpha=0.9 if b in bad else 0.5)

# Green "golden" curve = average of the healthy batches at each time point.
normal_ids = [b for b in yield_kg.index if b not in bad]
golden = df[df[batch_col].isin(normal_ids)].groupby(time_col)[pen_col].mean()
plt.plot(golden.index, golden.values, color="green", lw=2.5, label="golden (avg normal)")

plt.plot([], [], color="red", label="out-of-control")
plt.plot([], [], color="lightgrey", label="normal")
plt.xlabel("Time (h)")
plt.ylabel("Penicillin (g/L)")
plt.title("When do the bad batches break away from the pack?")
plt.legend()
plt.tight_layout()
plt.show()
