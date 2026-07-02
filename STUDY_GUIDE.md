# SPC Penicillin Project — Study Guide

*Everything you need to explain this project, and the theory behind it, to another person.*

This guide is written so you can (a) understand every decision in the code, (b) defend
every number, and (c) talk through the science without notes. Read it top to bottom
once; after that, the "Explain it in 90 seconds" and "Questions you might be asked"
sections are your revision sheet.

---

## 1. The elevator pitch

> We took 100 simulated industrial penicillin batches, ranked them by how much drug
> they actually produced, and asked *which process conditions separate the best batches
> from the worst, and when in the run the difference appears*. The answer: the winners
> hold **residual sugar near zero** throughout the run; the losers let sugar pile up
> after roughly **hour 100**, which switches off penicillin synthesis (glucose
> catabolite repression). Temperature and pH are well controlled and are not the
> driver. The actionable lever is **feed-to-demand sugar control**, watched live with
> oxygen-uptake and off-gas CO₂ as health gauges.

If you can say that paragraph and then back up each clause, you understand the project.

---

## 2. Background: the problem and the data

**What is being made.** Penicillin is produced by the mould *Penicillium chrysogenum*
in a **fed-batch** fermentation: you start with a batch of broth, then continuously
**feed** sugar (glucose/substrate) and a precursor while the cells grow and secrete
penicillin. The run lasts ~230–350 hours.

**The dataset — IndPenSim V3.** A high-fidelity *simulator* of a 100,000 L penicillin
fed-batch, released by Goldrick et al. It outputs ~40 process variables (plus thousands
of Raman-spectroscopy channels) logged on a time grid for each of 100 batches. Because
it is a simulator, we have something a real plant rarely gives us: **ground-truth
labels**. Batches **91–100 carry injected faults**; batches 1–90 run "normally."

**Two files we use.**

| File | What it holds | How we use it |
|---|---|---|
| `100_Batches_IndPenSim_V3.csv` (~2.5 GB) | Per-timestep time series for all 100 batches | Trajectories, batch-mean conditions, pH control chart |
| `100_Batches_IndPenSim_Statistics.csv` | One row per batch: harvested mass + fault label | **True yield target** and **validation labels** |

**A trap we hit and fixed:** the column literally named *"Batch reference"* holds only
**2** unique values — it is *not* the batch ID. The column that actually counts 1…100
per batch is `" 1-Raman spec recorded"`. The loader picks the batch column by the rule
"the one with >50 unique values," which lands on the right one despite the misleading
name.

---

## 3. The biology you must be able to explain

**Fed-batch and "feed to demand."** In a fed-batch you control the sugar feed rate. The
goal is a **glucose-limited** culture: feed sugar just fast enough that the cells
consume it immediately, so **residual** sugar in the broth stays near zero. A
glucose-limited culture is the state in which the cells make penicillin.

**Glucose catabolite repression — the mechanism.** If sugar accumulates (feed outruns
consumption), the high glucose concentration **represses the genes for penicillin
biosynthesis**: the cell preferentially burns the easy sugar for growth and stops
making the antibiotic. So *excess residual sugar = lost yield*. This is the single most
important sentence in the whole project, and it is established fermentation science, not
something we discovered — which is why our correlational finding is credible.

**Gauges vs levers — a crucial distinction.**
- **Levers** are things you can *set*: sugar feed rate, precursor (PAA) feed,
  temperature setpoint, aeration, agitation.
- **Gauges** are read-outs of culture health you *cannot dial directly*: **OUR**
  (oxygen uptake rate), **CER** (carbon evolution rate), **off-gas CO₂**. A healthy,
  active culture consumes more O₂ and makes more CO₂, so these rise with yield — but
  forcing CO₂ up does nothing; it's a symptom, not a cause. Acting on a gauge instead
  of the lever is a classic process-engineering mistake.

**PAA (phenylacetic acid).** The side-chain **precursor** fed to make penicillin G.
Our statistics flag PAA feed as a strong separator. Be careful interpreting it: more
precursor doesn't automatically mean more product, and PAA dosing is usually *coupled*
to the same feed-control logic as sugar. We report it honestly as significant but most
likely an *effect of the control scheme*, not an independent lever — flagged for
follow-up, not acted on blindly.

---

## 4. Architecture, tech stack, and how it works

### 4a. The tech stack (what each tool does for us, in plain terms)

This is a **data-analysis pipeline**, not an app — there's no database, server, or UI.
It's a small set of Python scripts that read CSV files, crunch numbers, and draw charts.

| Layer | Tool | What it actually does here |
|---|---|---|
| Language | **Python 3** | The glue that ties everything together. |
| Data handling | **pandas** | The workhorse. Loads the CSVs into a table (`DataFrame`), groups rows by batch, computes per-batch means/last values, aligns the two files. Think "Excel, but scripted." |
| Numerical math | **NumPy** | Fast array math under pandas; we also use it directly for the moving range, the Nelson-rule loops, and the train/test shuffle. |
| Statistics | **SciPy** (`scipy.stats`) | The Mann-Whitney U test, the Shapiro-Wilk normality test. |
| Plotting | **matplotlib** | All the figures: control charts, bar charts, trajectory plots. |
| Fast re-loading | **Parquet** (via `pyarrow`) | A compact binary cache of the slimmed-down data so repeat runs don't re-parse 2.5 GB of CSV. Optional — the code still works without it. |

There is **no machine-learning framework, no cloud, no pipeline orchestrator**. That's
deliberate: the problem is small enough that plain pandas + scipy is the right-sized
tool. Part of being a good engineer is *not* over-building.

### 4b. The architecture: one "spine" + independent "stages"

The design pattern is **a single shared loader (the spine) that every analysis script
imports.** Picture a backbone with ribs coming off it:

```
        100_Batches_IndPenSim_V3.csv          100_Batches_IndPenSim_Statistics.csv
        (2.5 GB time series)                  (1 row per batch: mass + fault label)
                 \                                   /
                  \                                 /
                   ▼                               ▼
            ┌───────────────────────────────────────────┐
            │           step0_data_loader.py            │   ◄── THE SPINE
            │  • loads only needed columns (memory-safe)│       (runs once,
            │  • finds the real batch-ID column         │        imported by all)
            │  • builds the shared objects:             │
            │      df, batch_col, final_pen,            │
            │      yield_kg (target), fault (labels)    │
            └───────────────────────────────────────────┘
                   │            │            │            │
        ┌──────────┘     ┌──────┘      ┌─────┘      ┌─────┘
        ▼                ▼             ▼            ▼
  step1_data_      step2_root_    step3_control_  step4_trajectory.py
  integrity.py     cause.py       chart.py        (optional)
  (gate: quality   (WHAT/WHEN/    (I-MR chart     step5_archive_shewhart.py
   + capability)    validation)    on pH)         (retired wrong turn)
```

**Why this shape?** Two ideas every engineer should recognize:

1. **Single source of truth.** The messy work of loading data and deciding "which column
   is the batch ID, what's the yield target" happens in **exactly one place**
   (`step0`). If that logic were copy-pasted into five scripts, a fix would have to be
   made five times and they'd drift apart. Centralizing it means every stage sees the
   *same* `yield_kg`, the *same* batch split — no contradictions.
2. **Separation of concerns.** Each stage does **one job** and can be run on its own:
   `step1` validates, `step2` finds the root cause, `step3` checks pH. They don't depend
   on each other — only on the spine. You can run, debug, or rewrite one without
   touching the others.

### 4c. The stages, one line each

| Step | File | Its single job |
|---|---|---|
| 0 | `step0_data_loader.py` | **Spine.** Load needed columns, pick batch ID, build `yield_kg` (true target), `final_pen` (proxy), `fault` (labels). |
| 1 | `step1_data_integrity.py` | **Gate.** Batch count, physical-range sanity, capability (Ppk + Cpk) vs a 20 g/L spec, normality test. |
| 2 | `step2_root_cause.py` | **Core.** Part A = *what* separates winners/losers (with FDR); Part B = *when* (substrate over time + validated alarm threshold); Part C = *validation* vs fault labels. |
| 3 | `step3_control_chart.py` | **I-MR control chart** on batch-mean pH with the full Nelson run-rule set. |
| 4 | `step4_trajectory.py` | *(optional)* penicillin accumulation curves; "golden batch" reference. |
| 5 | `step5_archive_shewhart.py` | *(retired)* the Shewhart-on-yield chart that failed — kept as an honest record. |

### 4d. How one run actually works (follow the data)

Say you type `python step2_root_cause.py`. Here is the chain of events:

1. **Import triggers the spine.** The first line imports from `step0_data_loader`, so
   Python runs that file top to bottom *first*.
2. **Load (memory-safe).** The loader checks for a Parquet cache; if none, it reads the
   big CSV but **only the ~18 columns the pipeline needs** (not the thousands of Raman
   columns), then saves that slim version to Parquet for next time. This is the step
   that turns an out-of-memory crash into a sub-second load.
3. **Reduce time series → one number per batch.** Each batch has hundreds of timestep
   rows. `pandas.groupby(batch_col)` collapses them: e.g. the *mean* of each process
   variable, or the *last* penicillin reading. Now we have one tidy row per batch.
4. **Join the truth.** The loader reads the Statistics file and attaches each batch's
   **harvested mass** (`yield_kg`) and **fault label**, checking the batch numbers line
   up before trusting the join.
5. **Hand control back to step2.** With the shared objects ready, `step2` ranks batches
   by `yield_kg`, splits top/bottom third, computes the normalized gaps, runs
   Mann-Whitney + FDR (Part A), tracks substrate over time with a train/test-validated
   threshold (Part B), and scores the split against the fault labels (Part C).
6. **Output.** Results print to the terminal; matplotlib opens the figures. Nothing is
   written back to the source data — the analysis is **read-only and reproducible**:
   run it again, get the same answer.

**The mental model:** *raw time series → (group by batch) → one row per batch → (join
true yield + labels) → compare groups → test → chart.* Every stage is a variation on
that same flow.

---

## 4e. The actual code results (what the scripts print)

These are the real console outputs from a clean run. Each block is followed by **how to
read it** so you can talk through any number.

### step1 — data integrity & capability

```
Grouping by: ' 1-Raman spec recorded'  ->  100 batches
Penicillin concentration(P:g/L)   min=0.00   max=36.18
pH(pH:pH)                         min=5.40   max=6.77
Temperature(T:K)                  min=296.84 max=302.18   (= 23.7–29.0 °C)
Dissolved oxygen (DO2:mg/L)       min=1.00   max=16.51

Final-concentration range : 3.16 to 36.16 g/L
Harvested-yield range     : 8.908e+05 to 4.448e+06 (true output target)

Process capability vs. LSL = 20.0 g/L (on final concentration)
  mean=24.01  sd_overall=8.61  sd_within(MRbar/d2)=7.48
  Ppk (overall spread)   = 0.16
  Cpk (within-subgroup)  = 0.18
  Shapiro-Wilk normality : W=0.898  p=0.0000  -> NOT normal (descriptive only)
  --> index < 1.0: process is NOT capable of reliably meeting spec.
```

**How to read it:** the batch column was correctly found (100 batches); physical ranges
are sane (pH ~5.4–6.8, temp ~24–29 °C), so the data is trustworthy. Capability is the
headline: both indices (~0.16–0.18) sit far below 1.0, meaning the spread is huge
relative to the 20 g/L spec — *not capable*. The Shapiro-Wilk p<0.0001 says the data
isn't normal, so treat the index as a rough description, not an exact defect rate.

### step2, Part A — what separates winners from losers (with FDR)

```
winners: 34 batches | losers: 33 batches   (ranked by harvested mass)

                                    gap   p_value   q_BH    significant
PAA flow(Fpaa:PAA flow)           -0.68   0.0011   0.0080      True
Substrate concentration(S:g/L)    -0.64   0.0002   0.0033      True
Oxygen Uptake Rate (OUR)           0.35   0.1900   0.4738      False
Carbon evolution rate (CER)        0.35   0.0739   0.2587      False
Temperature(T:K)                   0.34   0.6292   0.8809      False
Oil flow / Air pressure / Volume   0.3x   0.4–0.5  0.74        False
CO2 in off-gas                     0.28   0.2030   0.4738      False
pH(pH:pH)                          0.12   0.0214   0.0997      False  ← raw p<0.05 but q>0.05
Sugar feed rate(Fs:L/h)            0.08   0.9948   1.0000      False
Dissolved oxygen (DO2)            -0.03   0.7778   0.9899      False
```

**How to read it:** `gap` is the winner−loser difference in standard-deviation units
(negative = winners ran it lower). `p_value` is the raw Mann-Whitney result; `q_BH` is
that p-value after Benjamini-Hochberg correction for running 14 tests. Compare **q**, not
p, to 0.05. Only **PAA flow** and **substrate** survive. The pH row is the teaching
moment: its raw p=0.0214 *looks* significant, but after correction q=0.0997 — so we do
**not** claim pH as a driver. Feed *rate* (p=0.99) is flat, killing the old "winners feed
more" story.

### step2, Part B — alarm threshold, honestly validated

```
Substrate alarm threshold (TRAIN-derived, t>100h): 36.02 g/L
  Holdout validation on 34 unseen batches:
  losers caught 9/17, winners cleared 14/17, accuracy 68%
```

**How to read it:** we built the threshold on one half of the batches and tested it on
the other half it never saw. 68% is the *honest* performance. The earlier "perfect
separation" was fit and tested on the same data — overfitting. This is the difference
between a number you can trust and one you can't.

### step2, Part C — does "low yield" mean "faulted"?

```
faults (truth): [91..100]
confusion  TP=6  FP=27  FN=4  TN=63
precision = 0.18   recall = 0.60
faults that still yield WELL (winners): [93, 94, 96, 97]
```

**How to read it:** of the 10 truly-faulted batches we caught 6 (recall 0.60), but of the
33 we flagged only 6 were real faults (precision 0.18). Four faulted batches (93, 94, 96,
97) harvested *well*. Conclusion: a yield "loser" and an injected "fault" are different
things — yield is a useful but imperfect health proxy, and saying so is what makes the
analysis honest.

### step3 — I-MR control chart on pH

```
Phase 1 (70 batches):  CL = 6.4965   MR̄ = 0.0138   σ̂ = 0.0122
  UCL = 6.5332   LCL = 6.4598
Out-of-control — I-chart  : [6, 38, 66, 91, 97]   (5 points)
Nelson run-rule signals:
  Rule 1 (1 pt beyond 3σ)     : 5 points  [6, 38, 66, 91, 97]
  Rule 2 (9 in a row one side): 68 points
  Rule 3 / 5 / 6              : 0
```

**How to read it:** the control limits are CL ± 3σ̂ = 6.4965 ± 0.0367, i.e. a band only
**±0.037 pH units wide** because within-batch pH control is extremely tight (σ̂≈0.012).
Five batches breach it (two of them, 91 and 97, are fault batches), and Rule 2 fires for
long stretches. But the *entire* batch-to-batch pH spread is ~6.46–6.53 — trivially
small. So pH is **practically** well controlled even though it's **statistically**
flagged. This is the statistical-vs-practical-significance lesson in one chart.

---

## 5. SPC theory (the heart of the project)

**Common cause vs special cause.** Every process varies. **Common-cause** (chance)
variation is the inherent background noise of a stable process. **Special-cause**
(assignable) variation is a real, findable disturbance. SPC's entire purpose is to draw
a line — the **control limits** — that separates the two, so you only chase signals
worth chasing and don't "tamper" with noise.

**Shewhart control limits.** Center line (CL) = process mean; limits at **±3σ**. Why 3?
For a stable, roughly normal process, 99.73% of points fall within ±3σ, so a point
outside has ~0.27% chance of being noise — rare enough to treat as a real signal. 3σ is
the conventional balance between false alarms (too tight) and missed signals (too wide).

**Why the first SPC attempt failed (the instructive wrong turn).** We first put final
*yield* on a Shewhart chart, using batches 1–90 to set limits. The "normal" batches
themselves span ~6–36 g/L, so ±3σ landed around 2–48 g/L — limits so wide they caught
nothing, not even the worst batch. **Lesson:** a single control chart can't describe a
process whose normal output is that variable. More importantly, *the spread between good
and bad batches is not noise to filter out — it is the signal to explain.* That reframe
is what motivated the root-cause comparison.

**I-MR chart (Individuals & Moving Range).** Once each batch is reduced to one number
(its mean pH), you have **individual** observations, not subgroups — so the **X-bar**
chart (which needs subgroups of size ≥2) does not apply. The correct tool is the
**I-MR** pair:
- **I-chart** tracks the individual values → detects shifts in the **mean**.
- **MR-chart** tracks the **moving range** MRᵢ = |xᵢ − xᵢ₋₁| → detects shifts in
  **variability**. They are always read together.

**Estimating σ without subgroups.** With individuals you can't compute a within-subgroup
spread directly, so you use the average moving range:
σ̂ = MR̄ / d₂, with **d₂ = 1.128** for a moving range of span 2 (a tabulated constant =
the expected range of 2 standard-normal draws). MR-chart limits use **D₄ = 3.267**
(UCL = D₄·MR̄) and **D₃ = 0** (LCL = 0, since a range can't be negative).

**Phase 1 vs Phase 2.**
- **Phase 1** = a window you believe is stable, used *only to estimate* CL and limits.
- **Phase 2** = plot all data against those frozen limits to monitor.
We use the first 70 batches as Phase 1. (Honest caveat: Phase 1 should be chosen because
it is *verified* in control, not just "the first 70." Here it happens to exclude the
fault batches 91–100, which sit at the end.)

**Nelson run rules (issue #4).** A single point beyond 3σ (Rule 1) only catches large,
sudden shifts. Real SPC also watches for small but sustained shifts and trends using
"zones" (bands 1σ wide):
- Rule 1 — 1 point beyond 3σ (gross shift)
- Rule 2 — 9 points in a row on one side of CL (sustained mean shift)
- Rule 3 — 6 points in a row steadily increasing/decreasing (drift/trend)
- Rule 5 — 2 of 3 in a row beyond 2σ on the same side
- Rule 6 — 4 of 5 in a row beyond 1σ on the same side

**Statistical vs practical significance (the pH lesson).** Within-batch pH control is
*extremely* tight (σ̂ ≈ 0.012 pH units), so the I-chart limits are razor-thin and the
run rules flag several batches. But the entire batch-to-batch pH spread is only ≈6.46–
6.53 — utterly trivial in process terms. So pH is **practically** well controlled even
though it is **statistically** flagged. Detecting a difference is not the same as that
difference mattering. Always check the magnitude, not just the p-value or the chart.

---

## 6. Process capability — Cpk vs Ppk (issue #3)

Capability indices ask: *given the spec, how comfortably does the process fit inside it?*
With only a lower spec limit (LSL = 20 g/L on concentration, higher-is-better):

- **Ppk (process *performance*)** uses the **overall** standard deviation (total spread,
  all batches): Ppk = (mean − LSL) / (3·σ_overall).
- **Cpk (process *capability*)** uses the **within-subgroup** σ — here σ_within = MR̄/d₂,
  the short-term, "best the process can do" spread: Cpk = (mean − LSL) / (3·σ_within).

**The bug we fixed:** the original code computed the Ppk formula (overall std) but
*called it Cpk*. Both are now reported and correctly labelled.

**Reading the result:** both come out ≈0.16–0.18 — far below the usual 1.0 (capable) /
1.33 (good) thresholds, i.e. the process is *not* capable of reliably clearing 20 g/L.
**But** a Shapiro-Wilk test rejects normality (p < 0.0001), and capability indices
*assume* an approximately normal, in-control process. So the number is **descriptive
only** — it tells you "lots of spread relative to the spec," not a trustworthy
parts-per-million defect rate. (Note: harvested *mass* — our ranking target — has no
defined spec limit, so we can't compute a capability index on it without one. That's
why capability stays on concentration.)

---

## 7. The comparative statistics (issue #1, #2, #5)

**Yield target = harvested mass, not last concentration (issue #1).** The original code
ranked batches by the *last logged penicillin concentration*. Two problems we measured:
(a) it correlates only **~0.36** with the true harvested mass (total kg), because
mass = concentration × harvest volume and volumes differ; (b) in **42/100** batches the
last value sits >5% below the batch's peak (a logging tail-off). So we switched the
ranking target to `yield_kg` (harvested mass from the Statistics file). Concentration is
kept only for trajectory plots and the capability check.
> **Consequence that proves the point:** under the corrected target, the old secondary
> claim "winners feed *more* sugar" **disappears** (feed rate p ≈ 0.99). It was an
> artifact of ranking on the wrong variable.

**Winner/loser split.** Rank all 100 batches by `yield_kg`; top third = winners, bottom
third = losers; middle third dropped as ambiguous. We judge a batch on its conditions,
ignoring *when* it ran.

**Normalized gap.** For each variable, gap = (winner mean − loser mean) / (std across
batches). Dividing by the standard deviation puts every variable in the same "standard
deviation" units, so you can fairly compare pH (range ~6–7) against feed rate (0–100
L/h). Sign tells direction: negative = winners ran it lower.

**Mann-Whitney U test — and why non-parametric.** For each variable we test "do winners
and losers differ?" We use Mann-Whitney U (a rank-based test) instead of a t-test
because it does **not** assume normal distributions, which is safer for ~33 batches per
group of skewed process data. A small p-value means the gap is unlikely to be pure
sampling noise.

**Multiple comparisons + Benjamini-Hochberg FDR (issue #5).** We run one test per
variable — 14 tests. At α = 0.05 you expect ~0.7 "significant" results **by chance
alone**, so raw p < 0.05 over-claims. **Benjamini-Hochberg** controls the **False
Discovery Rate**: the expected fraction of false positives among the variables you call
significant. We compare the corrected **q-values** to 0.05.
> **Result after correction:** only **PAA flow** (gap ≈ −0.68) and **substrate
> concentration** (gap ≈ −0.64, raw p ≈ 0.0002) survive. **pH's raw p ≈ 0.02 does NOT
> survive** — a clean illustration of why the correction matters. Substrate is the
> mechanistically-supported, actionable lever; PAA is flagged but likely coupled to the
> feed scheme.

---

## 8. Validation thinking (issue #2 and #6)

**Use the ground-truth labels (issue #2).** The dataset labels batches 91–100 as
faulted. We had never checked our analysis against them. Treating "bottom-third by
yield" as a predicted-bad flag and scoring it against the fault labels gives a
**confusion matrix**:

|  | actually faulted | actually fine |
|---|---|---|
| flagged loser | TP = 6 | FP = 27 |
| not flagged | FN = 4 | TN = 63 |

→ **precision ≈ 0.18, recall ≈ 0.60.** Faults 93, 94, 96, 97 still **harvest well**.

**The key interpretation:** *a yield "loser" and an injected "fault" are different
concepts.* Most low-yield batches are normal variation (the very spread we're explaining),
and some faults don't depress total mass (e.g. a sensor fault). So yield is a **useful
but imperfect** proxy for batch health — and saying so out loud is what makes the
analysis trustworthy.

**Holdout validation of the alarm threshold (issue #6).** The original "alarm if
substrate > 10.3 g/L" was the *midpoint of the observed winner/loser peaks* — i.e. fit
and tested on the same batches, so its "clean separation" was guaranteed and
meaningless. We now **split winners and losers into train/test halves, derive the
threshold on train, and score it on the held-out batches it never saw**. Honest result:
**~68% accuracy**, and the threshold is split-dependent. The midpoint rule is a fragile
tripwire; a *learned* cutoff (logistic regression / Youden's J) validated on a holdout
should replace it before production. **Lesson:** never report performance of a rule on
the same data you used to build it — that's overfitting.

---

## 9. Corrected findings, in one place

| Question | Answer (corrected) |
|---|---|
| Ranking target | Harvested mass (`yield_kg`), not last concentration |
| Strongest separators (FDR q<0.05) | PAA flow (−0.68) and **residual substrate** (−0.64) |
| Mechanism | Glucose catabolite repression — excess residual sugar shuts off penicillin synthesis |
| Feed *rate* | Does **not** separate groups (p≈0.99) — old "feed more" claim dropped |
| pH | Practically well controlled; statistically flagged only because limits are razor-thin |
| Divergence time | ~hour 100 |
| Alarm threshold | Substrate rise after hour 100; midpoint rule ~68% holdout — replace with a validated learned cutoff |
| Yield vs fault | Different things: recall 0.60, precision 0.18 against ground-truth faults |
| Capability | Ppk ≈ Cpk ≈ 0.16–0.18 (not capable), but data non-normal → descriptive only |

---

## 10. Operating recommendations (what a process engineer would do)

1. **Feed to demand.** Match sugar feed to consumption so residual glucose stays near
   zero — keep the culture glucose-limited, the penicillin-producing state.
2. **Hour-100 checkpoint.** Batches look identical before ~hour 100; from there, watch
   residual substrate and act if it starts climbing. Use a *validated* cutoff, not the
   fragile midpoint rule.
3. **Watch live gauges (OUR, off-gas CO₂)** as early-warning health indicators.
4. **Act on the accumulation, not the gauges.** If sugar rises, trim the feed; don't try
   to force OUR/CO₂ up — they're symptoms.

---

## 11. Limitations / how to harden before acting

- **Association, not proof.** This is observational, not a designed experiment. The
  direction matches known science, but a confirmatory feed-strategy trial closes the loop.
- **Phase 1 chosen by position**, not verified stability; it happens to exclude the
  fault batches.
- **Split sensitivity.** Winner/loser at the 33rd/66th percentile — check other cutoffs.
- **PAA interpretation.** Significant but probably coupled to the feed scheme, not an
  independent cause.
- **Alarm rule is fragile.** Replace the midpoint with a learned, holdout-validated cutoff.
- **Unit label oddity.** The harvested-mass column is labelled "(kg)" but values are ~1e6;
  only relative ordering is used, so ranking is unaffected, but don't quote the absolute
  number as kilograms.

---

## 12. Explain it in 90 seconds (say this out loud)

> "We had 100 simulated penicillin batches. First I made sure I was ranking them by the
> right thing — total drug harvested, not just the last concentration reading, which only
> correlated 0.36 with real output. Then I split them into top-third and bottom-third
> producers and compared their average process conditions, testing each difference with
> a rank-based test and correcting for the fact that I ran 14 tests at once. Residual
> sugar came out as the key separator: winners hold it near zero, losers let it pile up
> after about hour 100, which is classic glucose catabolite repression — high sugar
> switches off penicillin synthesis. I confirmed pH wasn't the culprit with an I-MR
> control chart, and I sanity-checked my whole 'loser' definition against the dataset's
> real fault labels — it agreed about 60% of the time, which honestly tells you yield and
> 'fault' aren't the same thing. The recommendation is feed-to-demand sugar control with
> an hour-100 checkpoint, watching oxygen uptake and CO₂ as health gauges."

---

## 13. Questions you might be asked (and crisp answers)

- **Why I-MR not X-bar?** Each batch gives one summary number — individuals, no
  subgroups — so X-bar doesn't apply; I-MR is the chart for individuals.
- **Why ±3σ?** ~99.73% of a stable normal process sits inside ±3σ, so an outside point
  (~0.27% by chance) is worth investigating; it balances false alarms against misses.
- **What's d₂ = 1.128?** The tabulated factor converting the average moving range (span
  2) into an unbiased σ estimate.
- **Cpk vs Ppk?** Cpk uses short-term within-subgroup σ (best the process can do); Ppk
  uses overall σ (actual long-term spread). The original code mislabeled Ppk as Cpk.
- **Why Mann-Whitney over a t-test?** It doesn't assume normality — safer for small,
  skewed process samples.
- **Why FDR correction?** 14 tests at α=0.05 expect ~0.7 false hits; BH controls the
  false-discovery rate so the "significant" list is trustworthy. pH failed correction.
- **Why did "winners feed more" disappear?** It was an artifact of ranking on
  concentration; on true harvested mass, feed rate doesn't separate the groups.
- **Is substrate the *cause*?** It's the mechanistically-supported, actionable lever and
  matches known biology, but this is correlational — a controlled feed trial would prove it.
- **Why only 68% on the alarm threshold?** Because we finally tested it on data it wasn't
  built from. The earlier "perfect" separation was overfitting.
- **Why is the 2.5 GB file a problem?** It carries thousands of Raman columns we never
  use; we load only the needed columns (and cache to Parquet) so it fits in memory.

---

## 14. Glossary

- **Fed-batch:** fermentation where nutrients are fed continuously into a starting batch.
- **Substrate / residual sugar:** glucose left un-consumed in the broth; we want it ≈0.
- **Catabolite repression:** high glucose switches off penicillin-synthesis genes.
- **OUR / CER:** oxygen uptake rate / carbon evolution rate — culture-activity gauges.
- **PAA:** phenylacetic acid, the penicillin side-chain precursor (a fed input).
- **CL / UCL / LCL:** center line / upper / lower control limit.
- **Moving range (MR):** |xᵢ − xᵢ₋₁|, the step-to-step change used to estimate σ.
- **Phase 1 / Phase 2:** estimate limits / monitor against them.
- **Cpk / Ppk:** capability (within-σ) / performance (overall-σ) indices vs a spec.
- **Mann-Whitney U:** non-parametric test for a difference between two groups.
- **FDR / q-value:** false-discovery-rate-controlled significance for many tests.
- **Precision / recall:** of flagged faults, how many are real / of real faults, how many
  we caught.
- **Holdout / overfitting:** testing on unseen data / a rule that only looks good on the
  data it was built from.
