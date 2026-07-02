"""
STEP 0 - DATA LOADER (the spine)
================================================================================
PURPOSE
    Load the IndPenSim V3 time-series and the Statistics summary, and expose
    the shared objects every downstream step imports:

        df         full time-series (only the columns the pipeline needs)
        batch_col  the column that actually identifies batches (values 1..100)
        pen_col    penicillin-concentration column name
        final_pen  last logged concentration per batch  (proxy; trajectories only)
        yield_kg   total harvested penicillin mass per batch  (TRUE ranking target)
        fault      ground-truth fault label per batch (0 = no fault, 1 = fault)

APPROACH
    1. Read only the columns the pipeline needs from the 2.5 GB V3 CSV
       (via `usecols`), then cache that slim frame to Parquet so repeated
       runs load in a fraction of a second. Delete .v3_subset.parquet if
       NEEDED_COLS ever changes.
    2. Pick the real batch-ID column (see KEY DECISIONS).
    3. Join the Statistics file for the true output target (yield_kg) and
       the fault labels.

KEY DECISIONS
    - Batch ID column: the column literally named "Batch reference(...)"
      holds only 2 unique values in V3, so it is NOT the batch key. The
      column that increments 1..100 per batch is " 1-Raman spec recorded".
      `_pick_batch_col` selects whichever candidate has >50 unique values,
      so it is correct regardless of misleading column names.
    - final_pen (last logged concentration) is kept only for trajectory
      plots and the concentration-spec capability check. It is NOT the
      ranking target: it correlates only ~0.36 with true harvested mass
      and misses the actual peak in 42/100 batches.
    - yield_kg (harvested penicillin mass, from the Statistics file) is the
      TRUE ranking target used everywhere else in the pipeline.
    - If the V3 batch IDs and the Statistics "Batch ref" ever drift apart,
      fail loudly (ValueError) rather than silently mis-joining the two
      data sources.
"""

import pandas as pd
from pathlib import Path

HERE      = Path(__file__).parent
CSV_V3    = HERE / "100_Batches_IndPenSim_V3.csv"
CSV_STATS = HERE / "100_Batches_IndPenSim_Statistics.csv"
CACHE     = HERE / ".v3_subset.parquet"

pen_col = "Penicillin concentration(P:g/L)"

NEEDED_COLS = [
    "Time (h)", pen_col, "Substrate concentration(S:g/L)",
    "Batch reference(Batch_ref:Batch ref)", " 1-Raman spec recorded",
    "Sugar feed rate(Fs:L/h)", "Aeration rate(Fg:L/h)", "Agitator RPM(RPM:RPM)",
    "Air head pressure(pressure:bar)", "Dissolved oxygen concentration(DO2:mg/L)",
    "pH(pH:pH)", "Temperature(T:K)", "carbon dioxide percent in off-gas(CO2outgas:%)",
    "PAA flow(Fpaa:PAA flow (L/h))", "Oil flow(Foil:L/hr)",
    "Oxygen Uptake Rate(OUR:(g min^{-1}))", "Carbon evolution rate(CER:g/h)",
    "Vessel Volume(V:L)",
]


def _load_v3():
    if CACHE.exists():
        return pd.read_parquet(CACHE)
    header = pd.read_csv(CSV_V3, nrows=0).columns.tolist()
    cols = [c for c in NEEDED_COLS if c in header]
    frame = pd.read_csv(CSV_V3, usecols=cols)
    try:
        frame.to_parquet(CACHE)
    except Exception:
        pass
    return frame


df = _load_v3()


def _pick_batch_col(df):
    for c in ["Batch reference(Batch_ref:Batch ref)", " 1-Raman spec recorded"]:
        if c in df.columns and df[c].nunique() > 50:
            return c
    raise KeyError("No column looks like a batch ID (expected ~100 unique values).")


batch_col = _pick_batch_col(df)

final_pen = df.groupby(batch_col)[pen_col].last().sort_index()

_stats = pd.read_csv(CSV_STATS)
_stats.columns = [c.strip() for c in _stats.columns]
_stats = _stats.set_index("Batch ref").sort_index()

yield_kg = _stats["Penicllin_yield_total (kg)"]
fault    = _stats["Fault ref(0-NoFault 1-Fault)"]

if list(final_pen.index) != list(yield_kg.index):
    raise ValueError(
        "V3 batch IDs do not match Statistics 'Batch ref' -- the yield join "
        "would be wrong. Check the batch column and the Statistics file."
    )
