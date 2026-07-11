# SPC & Root-Cause Analysis for Penicillin Fermentation

Takes 100 batches of raw fermentation data and works out which ones underperformed, when
they went wrong, and which process parameter is actually to blame. Built on IndPenSim V3,
a simulation of 100 industrial-scale (100,000 L) penicillin fed-batch runs.

`Statistical Process Control` · `Cpk/Ppk` · `Root Cause Analysis` · `Bioprocess Engineering`
· `Continued Process Verification (CPV)` · `Python` · `SciPy`

> IndPenSim V3 is a simulator (Goldrick et al.), which is the point — it ships with
> ground-truth fault labels, so I can check my conclusions against the real answer instead
> of just asserting them. Real plant data almost never gives you that.

## What I found

Short version: residual **glucose** is what separates the good batches from the bad ones.
It's the only parameter that survives multiple-comparison correction across 14 candidates
(Benjamini-Hochberg, q ≈ 0.003), and it lines up with the known biology — glucose
catabolite repression shuts down penicillin production when sugar is left to accumulate.

The timing matters as much as the driver. Good and bad batches look basically identical up
to about **hour 100**. After that the underperformers start letting residual sugar climb,
and that's where they lose yield. So there's a concrete window to act in, not just a vague
"manage your glucose."

I also spent time ruling things out, which felt as important as finding the driver. pH, for
instance, is rock-steady batch to batch — an I-MR chart puts it inside ±0.04 pH units — so
it's not a lever here even though it's the first thing people reach for. Same story for
temperature.

### The mistake I made (and caught)

Leaving this in because it's the most useful thing in the project. My first yield metric was
just the last logged penicillin concentration per batch. Seemed reasonable. It wasn't — it
only correlated 0.36 with the actual harvested mass from the statistics file. I'd been
ranking batches by the wrong number, and it had produced a confident-looking conclusion that
feed rate was the driver. Once I fixed the metric to true harvested mass, that whole
feed-rate story fell apart and glucose came through instead. If I hadn't cross-checked
against the harvest data I'd have shipped the wrong answer with a straight face.

### Being honest about the detector

The bad-batch detector catches 60% of the ground-truth faults (recall 0.60), and the alarm
threshold scores 68% on a held-out split. Those aren't dazzling numbers and I'm not going to
pretend they are. They're the *real* numbers — scored on batches the threshold wasn't tuned
on — not the inflated in-sample version that'd look better and mean nothing.

## Visuals

![winner vs loser gaps, FDR-flagged](assets/root_cause_gaps.png)
![residual substrate over time](assets/substrate_timing.png)
![I-MR control chart on pH](assets/control_chart_pH.png)

## How it's built

Everything hangs off one shared loader. It's the single place that decides which column is
the batch ID and what the yield target is, so the analysis stages physically can't disagree
about the basics. That sounds obvious but it's saved me from a specific bug — see the note
below on the batch column.

```
V3 time-series CSV  +  Statistics CSV (harvested mass + fault labels)
        |
        v
step0_data_loader.py     spine: memory-safe load, batch ID, yield_kg, fault labels
        |
        +-- step1_data_integrity.py    quality gate + capability (Cpk/Ppk)
        +-- step2_root_cause.py        what / when / validation against fault labels
        +-- step3_control_chart.py     I-MR chart on pH + Nelson rules
        +-- step4_trajectory.py        (optional) accumulation curves + golden batch
```

| Step | File | Job |
|---|---|---|
| 0 | `step0_data_loader.py` | Loads only the columns needed (Parquet-cached), resolves the batch ID, builds `yield_kg`, `final_pen`, `fault`. |
| 1 | `step1_data_integrity.py` | Data-quality gate: batch count, physical-range checks, capability + a normality test. |
| 2 | `step2_root_cause.py` | *What* separates winners from losers (Mann-Whitney U + FDR); *when* they diverge (substrate timing + holdout-validated alarm); *validation* against the fault labels. |
| 3 | `step3_control_chart.py` | I-MR control chart on batch-mean pH with the full Nelson run-rule set. |
| 4 | `step4_trajectory.py` | *(optional)* penicillin accumulation curves and a golden-batch reference. |
| 5 | `step5_archive_shewhart.py` | *(retired)* a Shewhart-on-yield chart that didn't work. Kept on purpose as a record of a wrong turn rather than deleted. |

## A few methodology choices worth explaining

**Why I-MR and not X-bar.** Each batch collapses to one summary value, and X-bar charts
need subgroups you don't have here. I-MR (Individuals & Moving Range) is the right tool for
one-value-per-unit data. Limits set in Phase 1, monitored in Phase 2, Nelson rules for the
run patterns.

**Cpk *and* Ppk, because people conflate them.** I report both — Ppk from the overall
spread, Cpk from the within-subgroup spread (MR̄/d₂) — with a Shapiro-Wilk normality caveat,
since capability numbers quietly assume normality and these two get mixed up constantly.

**FDR correction, so 14 tests don't manufacture a winner.** Running 14 parameters through
significance tests at once will hand you a false positive by chance. Benjamini-Hochberg
keeps that honest. Concretely: pH's raw p = 0.02 looks significant but correctly does *not*
survive correction. Glucose does.

**The alarm threshold is scored out-of-sample.** Derived on a train split, tested on
held-out batches. That's why the accuracy is a modest 68% and not a suspicious 95%.

## The batch-ID trap

Flagging this because it's a real gotcha in this dataset and it bit me before I added a
guard. The column that looks like the batch identifier isn't the one you want — the honest
per-batch axis is elsewhere and the obvious-looking column only has a couple of distinct
values. The loader now checks the resolved batch count is sane and fails loudly if it isn't,
instead of silently grouping everything wrong and producing plausible-but-fake results
downstream.

## Run it

```bash
pip install pandas numpy scipy matplotlib pyarrow

python step1_data_integrity.py     # data quality + capability
python step2_root_cause.py         # what drives yield, when it diverges, validation
python step3_control_chart.py      # I-MR control chart on pH
python step4_trajectory.py         # (optional) accumulation curves
```

Each script imports `step0_data_loader` on its own. Drop `100_Batches_IndPenSim_V3.csv` and
`100_Batches_IndPenSim_Statistics.csv` in the repo root first (the big V3 file is
git-ignored — grab it from the link below).

## Data

IndPenSim (Industrial-scale Penicillin Simulation), Goldrick et al. — a first-principles
model of a 100,000 L *Penicillium chrysogenum* fermentation, validated against historical
industrial data.

- Download: http://www.industrialpenicillinsimulation.com/ (~2.5 GB, process + Raman)
- Mirror / DOI: https://data.mendeley.com/datasets/pdnjz7zz5x/2
- Paper: Goldrick et al., *Modern day monitoring and control challenges outlined on an
  industrial-scale benchmark fermentation process*, Computers & Chemical Engineering.

One thing worth knowing when reading the results: the 100 batches aren't uniform. Batches
1–30 are recipe-driven, 31–60 operator-controlled, 61–90 run Advanced Process Control with
Raman, and 91–100 have injected faults. So some of the batch-to-batch spread reflects
different control strategies rather than pure noise — which is exactly why I validate
against the fault labels instead of trusting the spread on its own.

## What I'd actually tell an operator

1. **Feed to demand.** Match sugar feed to consumption so residual glucose stays near zero.
   A glucose-limited culture is the one that makes penicillin.
2. **Check at hour 100.** Before that, batches look the same. From there, watch for rising
   residual substrate and alarm on it.
3. **Use OUR and off-gas CO₂ as live health gauges** — but read them as symptoms.
4. **Act on the sugar, not the gauges.** If residual glucose climbs, trim the feed. Don't
   chase OUR/CO₂ directly; they're downstream of the real problem.

## Limitations

This is an association, not proof. It's an observational comparison across existing batches,
not a designed experiment — a confirmatory feed-strategy trial would be the honest next step
to actually close the loop.

PAA flow also comes up as significant, but I think it's coupled to the feed-control scheme
rather than an independent lever, so I flagged it for follow-up instead of acting on it.

And the alarm rule is the weakest part — a midpoint threshold that moves depending on the
split (~68% holdout). A learned cutoff (logistic regression, or Youden's J on the ROC) would
be the proper replacement.

