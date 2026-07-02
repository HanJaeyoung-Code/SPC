"""
STAGE 2 - SPC CONTROL CHART (RETIRED)
======================================

What we tried:  A standard Shewhart Individuals control chart on final yield,
                using batches 1-90 as the baseline to set the center line and
                3-sigma control limits.

Why it failed:  The "normal" batches range from ~6 to ~36 g/L, so the control
                limits came out at roughly 2-48 g/L — a net so wide it caught
                nothing, not even the worst batch.

What it taught: The wide limits are not a flaw in the method; they are a signal
                about the process. A single pair of control limits cannot describe
                a process this variable. More importantly, the spread between good
                and bad batches is *not* noise to filter out — it is the signal to
                explain. That reframe drives Stage 4 (Root_Cause.py).

Kept here as an honest record of an approach that did not fit the data.
"""

import numpy as np
import matplotlib.pyplot as plt
from step0_data_loader import final_pen

batches = final_pen.index.values
yields  = final_pen.values

# First 90 batches (by position) define "normal" variation.
baseline = final_pen.iloc[:90]
center = baseline.mean()
sigma  = baseline.std()
ucl, lcl = center + 3*sigma, center - 3*sigma

# Flag any batch whose final yield falls outside the limits.
out = (yields > ucl) | (yields < lcl)
print("Out-of-control batches:", batches[out])

# Control chart of final yield per batch.
plt.figure(figsize=(11, 5))
plt.plot(batches, yields, marker="o", zorder=1)
plt.scatter(batches[out], yields[out], color="red", zorder=2, label="out of control")
plt.axhline(center, color="green", label="center line")
plt.axhline(ucl, color="red", linestyle="--", label="control limits")
plt.axhline(lcl, color="red", linestyle="--")
plt.axhline(20, color="orange", linestyle=":", label="20 g/L target")
plt.xlabel("Batch number")
plt.ylabel("Final penicillin yield (g/L)")
plt.legend()
plt.show()