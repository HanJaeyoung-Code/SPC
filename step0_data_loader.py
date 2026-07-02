"""
STEP 0 - DATA LOADER (the spine)
================================
Loads the IndPenSim V3 time-series and the Statistics summary, and exposes the
shared objects every downstream step imports:

    df         full time-series (only the columns the pipeline needs)
    batch_col  the column that actually identifies batches (values 1..100)
    pen_col    penicillin-concentration column name
    final_pen  last logged concentration per batch  (proxy; trajectories only)
    yield_kg   total harvested penicillin mass per batch  (TRUE ranking target)
    fault      ground-truth fault label per batch (0 = no fault, 1 = fault)

PERFORMANCE (issue #7)
----------------------
The raw V3 file is ~2.5 GB because it carries thousands of Raman-spectrum columns
the analysis never touches. Loading the whole thing exhausts memory. So we:
  (a) read ONLY the columns the pipeline needs, via `usecols`, and
  (b) cache that slim frame to Parquet, so repeated runs load in a fraction of a
      second instead of re-parsing 2.5 GB of CSV.
Delete the .parquet cache if you ever change NEEDED_COLS.
"""

import pandas as pd
from pathlib import Path

HERE      = Path(__file__).parent
CSV_V3    = HERE / "100_Batches_IndPenSim_V3.csv"
CSV_STATS = HERE / "100_Batches_IndPenSim_Statistics.csv"
CACHE     = HERE / ".v3_subset.parquet"

pen_col = "Penicillin concentration(P:g/L)"

# Every column any step in the pipeline reads. Loading just these (instead of
# all ~2000+ columns) is what keeps the frame in memory.
NEEDED_COLS = [
    "Time (h)", pen_col, "Substrate concentration(S:g/L)",
    # batch-ID candidates (the real one is selected in _pick_batch_col)
    "Batch reference(Batch_ref:Batch ref)", " 1-Raman spec recorded",
    # process variables compared in step2_root_cause
    "Sugar feed rate(Fs:L/h)", "Aeration rate(Fg:L/h)", "Agitator RPM(RPM:RPM)",
    "Air head pressure(pressure:bar)", "Dissolved oxygen concentration(DO2:mg/L)",
    "pH(pH:pH)", "Temperature(T:K)", "carbon dioxide percent in off-gas(CO2outgas:%)",
    "PAA flow(Fpaa:PAA flow (L/h))", "Oil flow(Foil:L/hr)",
    "Oxygen Uptake Rate(OUR:(g min^{-1}))", "Carbon evolution rate(CER:g/h)",
    "Vessel Volume(V:L)",
]


def _load_v3():
    """Return the slim V3 frame, from the Parquet cache if available."""
    if CACHE.exists():
        return pd.read_parquet(CACHE)
    header = pd.read_csv(CSV_V3, nrows=0).columns.tolist()
    cols = [c for c in NEEDED_COLS if c in header]
    frame = pd.read_csv(CSV_V3, usecols=cols)
    try:
        frame.to_parquet(CACHE)        # best-effort speedup; needs pyarrow
    except Exception:
        pass                            # no parquet engine -> just skip caching
    return frame


df = _load_v3()


def _pick_batch_col(df):
    # NOTE: the column literally named "Batch reference(...)" holds only 2 unique
    # values in V3, so it is NOT the batch key. The column that increments 1..100
    # per batch is " 1-Raman spec recorded". The >50-unique test below selects the
    # right one regardless of its misleading name.
    for c in ["Batch reference(Batch_ref:Batch ref)", " 1-Raman spec recorded"]:
        if c in df.columns and df[c].nunique() > 50:
            return c
    raise KeyError("No column looks like a batch ID (expected ~100 unique values).")


batch_col = _pick_batch_col(df)

# Concentration proxy: last logged penicillin reading per batch (g/L). Kept for
# trajectory plots and the concentration-spec capability check only -- NOT the
# ranking target (correlates ~0.36 with mass; <peak in 42/100 batches).
final_pen = df.groupby(batch_col)[pen_col].last().sort_index()

# ---------------------------------------------------------------------------
# TRUE OUTPUT: harvested penicillin mass per batch, from the summary file.
# ---------------------------------------------------------------------------
_stats = pd.read_csv(CSV_STATS)
_stats.columns = [c.strip() for c in _stats.columns]
_stats = _stats.set_index("Batch ref").sort_index()

yield_kg = _stats["Penicllin_yield_total (kg)"]      # ranking target (mass)
fault    = _stats["Fault ref(0-NoFault 1-Fault)"]    # ground-truth labels

# Fail loudly (not with a strippable assert) if the V3 batch IDs and the
# Statistics "Batch ref" ever drift apart, so we never silently mis-join.
if list(final_pen.index) != list(yield_kg.index):
    raise ValueError(
        "V3 batch IDs do not match Statistics 'Batch ref' -- the yield join "
        "would be wrong. Check the batch column and the Statistics file."
    )
