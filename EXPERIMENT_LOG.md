# transformer_v2 — Experiment & Change Log

Running log of code changes, jobs, and results for the `transformer_v2` line
(baseline `sw_d192` reproduction + variants `v2m1`, `v2m2`). All σ values are on the
test split of `cache_full10` (3.38 M events, strict cut). "robust" = IQR/1.349 core;
"raw-std" = `np.std` (tail-sensitive); "outlier" = frac `|ΔE/E|>20%`.

Last updated: 2026-06-25.

---

## 1. Results board (test split)

| run | dir / out_dir | rob ≤2 TeV | rob full | raw-std | binavg rob | binavg raw | outlier | med-bias |
|-----|---------------|-----------:|---------:|--------:|-----------:|-----------:|--------:|---------:|
| **baseline** sw_d192 | `transformer_v2/runs/v2_d192` | 1.63% | 1.75% | 8.60% | 1.69% | 7.90% | 2.05% | −0.73% |
| **wdx** (excl-1D WD) | `transformer_v2/runs/v2_d192_wdx` | **1.40%** | **1.50%** | 8.47% | **1.51%** | 7.69% | **1.47%** | **−0.05%** |
| ep100 (50→100) | `transformer_v2/runs/v2_d192_ep100` | 1.51% | 1.63% | 15.65% | 1.71% | 11.76% | 1.90% | +0.36% |
| **v2m1** (fit-weight) | `transformer_v2m1/runs/v2m1_d192` | 1.47% | 1.56% | **7.53%** | 1.59% | **7.10%** | 1.93% | −0.42% |
| v2m2 (heteroscedastic) | `transformer_v2m2/runs/v2m2_d192` | 7.40% | 8.13% | 19.43% | 8.31% | 15.62% | 10.76% | +2.32% |
| wdx+v2m1 (combined) | `transformer_v2m1/runs/v2m1_wdx_d192` | _running_ | | | | | | |
| v2m2b (hetero, fixed-w) | `transformer_v2m2b/runs/v2m2b_d192` | 1.46% | 1.54% | 19.36% | 1.56% | 16.69% | 1.96% | +0.59% |
| v2test (raw-sel, =sw_d192) | `transformer_v2test/runs/v2test_d192` | 1.65% | 1.83% | 6.61% | 1.61% | 6.11% | 1.68% | −1.59% |
| v2b best (composite) | `transformer_v2b/runs/v2b_d192` (metrics_best) | 1.54% | 1.69% | 12.14% | 1.63% | 10.47% | 1.81% | −0.87% |
| v2b best_robust | `…/metrics_best_robust` | 1.45% | 1.60% | 9.83% | 1.55% | 8.87% | 1.63% | −1.87% |
| v2b best_raw | `…/metrics_best_raw` | 1.64% | 1.82% | 8.10% | 1.60% | 7.38% | 1.60% | −1.16% |
| **v2cham best** (wdx+v2m1) | `transformer_v2cham/runs/v2cham_d192` (best) | 1.39% | 1.48% | 8.50% | 1.49% | 7.92% | 1.51% | −0.36% |
| **v2cham best_robust** | `…/metrics_best_robust` | **1.38%** | 1.47% | 9.15% | 1.49% | 8.06% | 1.47% | −0.50% |
| **v2cham best_raw** | `…/metrics_best_raw` | 1.39% | 1.49% | **7.05%** | 1.50% | 6.65% | 1.52% | −0.79% |
| cham ex0.3 (excess, tw0.3) | `…/v2cham_ex03_d192` (best) | 1.38% | 1.49% | 8.57% | 1.49% | 7.79% | 1.51% | −0.78% |
| cham ex0.5 (excess, tw0.5) | `…/v2cham_ex05_d192` (best) | 1.44% | 1.55% | **6.92%** | 1.55% | 6.46% | 1.64% | −0.56% |
| cham ex1.0 (excess, tw1.0) | `…/v2cham_ex10_d192` (best) | 1.45% | 1.58% | 7.16% | 1.53% | 6.59% | 1.56% | −0.27% |

Reference (historical, "indicative only" — re-scored): `sw_d192` robust ≤2 TeV ≈ 1.36%,
raw-std overall ≈ 6.52%, binavg-raw ≈ 5.97%.

**Headline:** `wdx` is a broad win (core, outlier, calibration) → promoted to default.
`v2m1` gives the best raw-std/tail (its design goal). `ep100` regressed the tail.
`v2m2` blew up (weighter runaway — see §4). Combined `wdx+v2m1` queued.

---

## 2. Job timetable (SLURM, account ihepai, RTX 5090, cache_full10, d192)

| job id | name | submitted/started | ended | elapsed | status | notes |
|-------:|------|-------------------|-------|--------:|--------|-------|
| 81716 | smk_v2m12 | 2026-06-23 23:10:55 | 2026-06-23 23:12:30 | 0:01:35 | DONE | combined smoke (v2m1+v2m2), both rc=0 |
| 81701 | tf_v2_d192 | 2026-06-24 09:20:20 | 2026-06-24 15:06:19 | 5:45:59 | DONE | Step 0 baseline |
| 81702 | tf_v2_ep100 | 2026-06-24 09:21:20 | 2026-06-24 18:39:59 | 9:18:39 | DONE | Step 1a; early-stopped ~ep 80 |
| 81703 | tf_v2_wdx | 2026-06-24 15:06:20 | 2026-06-24 20:51:14 | 5:44:54 | DONE | Step 1b (wd_exclude_1d) |
| 81812 | tf_v2m1 | 2026-06-24 18:40:00 | 2026-06-25 00:32:21 | 5:52:21 | DONE | fit-quality weighting |
| 81813 | tf_v2m2 | 2026-06-24 20:51:15 | 2026-06-25 01:29:18 | 4:38:03 | DONE | heteroscedastic; early-stop ep 30 (runaway) |
| 82010 | tf_v2m1wdx | 2026-06-25 | — | — | RUNNING | combined wdx + v2m1 |
| 82026 | tf_v2m2b | 2026-06-25 | — | — | RUNNING | v2m2 weighter-fixed retry |
| 82120 | tf_v2test | 2026-06-25 (submitted) | — | — | PENDING | exact sw_d192 (raw-std selection) |
| 82121 | tf_v2b | 2026-06-25 | — | — | COMPLETED | configurable metric + multi-best ckpts |
| 82316 | tf_v2cham | 2026-06-25 | — | — | COMPLETED | CHAMPION: wdx+v2m1 + multi-best (tail_weight=0) |
| 82347 | tf_chamx03 | 2026-06-25/26 | — | — | COMPLETED | v2cham + select tail_term=excess, tail_weight=0.3 |
| 82348 | tf_chamx05 | 2026-06-25/26 | — | — | COMPLETED | v2cham + excess, tail_weight=0.5 |
| 82349 | tf_chamx10 | 2026-06-25/26 | — | — | COMPLETED | v2cham + excess, tail_weight=1.0 |
| 82357 | tf_evalnb | 2026-06-26 | — | — | COMPLETED | re-bin v1/v2 on new high-E edges for the method-comparison fig |
| 81715 | smk_v2m1 | 2026-06-23 | — | — | CANCELLED | superseded by combined smoke 81716 |

QOS note: `gpunormal` caps submitted jobs at 4 per user; the smoke was packed into one
job (81716) to fit, and the two GPU nodes saw heavy external load so jobs queued before
running.

---

## 3. Code changes (chronological)

### 3.1 Cleanups to the shared baseline code (`transformer_v2/`, identical to `transformer/`)
- **train.py**
  - `torch.compile` kept ON but made non-fatal: wrapped the `torch.compile()` call in
    try/except AND guarded the training step so an inductor crash (first forward or
    mid-training recompile) falls back to eager (numerically identical), skipping one batch.
  - `validate()`: per-task val losses now **event-weighted** (was a size-biased mean over
    batches; the token-budget sampler gives wildly varying batch event-counts).
  - training loop: `train_total` / `train_<task>` columns also event-weighted.
  - optimizer: gated **`train.wd_exclude_1d`** param-group split — exclude 1-D params
    (LayerNorm / bias / AttnPool-query) from weight decay (standard transformer recipe).
- **evaluate.py**
  - `amp_dtype_of(cfg)` added → autocast honors `train.amp_dtype` (was hardcoded bf16).
  - **robust IQR/1.349 core σ is now the PRIMARY reported metric** (`overall_res`,
    `binwise_mean_res`, `..._le_cut`, `bias_aware_metric`), matching train-time selection;
    raw-std kept as `*_raw_std` diagnostics; added `outlier_frac`; plots overlay
    robust+raw; histogram title shows both.
- **probe.py**
  - `amp_dtype_of(cfg)` → both autocast blocks honor `train.amp_dtype` (was hardcoded bf16).
- **config/base.yaml**
  - added `train.wd_exclude_1d` knob (initially `false` = exact sw_d192 repro).

### 3.2 `transformer_v2m1` — fit-quality-weighted physics supervision (new dir)
- **data/dataset.py**: per-event `fit_resid = Σ|ehit−expe|/Σehit`, computed at load time
  over the CSR layout (no cache rebuild); returned per event + collated.
- **train.py**: `aux_weight()` = `clip((train_median/resid)^β, w_min, 1)` (β=1.0,
  w_min=0.2); applied to **concept + recon** losses only (energy never); train-median
  scale set in `main()`; validation logs the unweighted aux losses.
- **config/base.yaml**: `loss.fit_quality_weight: {enabled:true, apply_to:[concept,recon],
  beta:1.0, w_min:0.2}`.

### 3.3 `transformer_v2m2` — heteroscedastic concept head (new dir)
- **models/bottleneck.py**: concept head emits `2·n_concepts` (mean + per-event-per-concept
  log-variance) when `model.concept_heteroscedastic`; returns `concept_logvar`.
- **models/model.py**: passes `concept_logvar` into the output dict.
- **losses/objectives.py**: `concept_nll` = `0.5·exp(−s)·err² + 0.5·s`, s clamped ±6.
- **train.py**: uses `concept_nll` when heteroscedastic, else Huber.
- **probe.py**: new calibration section (mean σ and `corr(σ,|err|)` per concept).
- **config/base.yaml**: `model.concept_heteroscedastic: true`.
- README documents the parallel-head alternative (not coded) and the weighter caveat.

### 3.4 Promotions / 2026-06-25
- **`transformer_v2/config/base.yaml`**: `wd_exclude_1d` promoted **`false → true`**
  (proven win; set `false` to recover the exact sw_d192 reproduction).
- **`transformer_v2m1/job_v2m1_wdx.sub`** created → combined `wdx + v2m1` run (82010).

### 3.5 `transformer_v2m2b` — weighter-fixed retry of v2m2 (new dir, 2026-06-25)
Same heteroscedastic concept head as `v2m2`; only the multi-task combination is fixed.
- **train.py**: when `model.concept_heteroscedastic` and `loss.concept_fixed_weight` is set,
  the KG weighter is built over `[energy, recon]` only and the step does
  `total = weighter({energy,recon}) + concept_fixed_weight · concept_nll`. So the concept
  NLL bypasses BOTH the learnable `s` and the EMA-normalization (the two causes of the
  `w_concept→1e7` runaway). `w_concept` logged as the fixed coefficient.
- **config/base.yaml**: `loss.concept_fixed_weight: 1.0`; `wd_exclude_1d: false` (clean A/B
  vs baseline and vs v2m2); `model.concept_heteroscedastic: true` (inherited).
- Validated: py_compile + CPU wiring sanity (weighter tasks = [energy, recon], concept added
  as fixed term, total finite). Submitted as job 82026.

### 3.6 `transformer_v2test` — exact sw_d192 reproduction (new dir, 2026-06-25)
The one thing that made `v2` differ from `sw_d192` in the SAVED model was the selection
metric. This dir reverts it.
- **train.py::validate**: selection metric back to the original **raw-std bias-aware**
  `sqrt(np.std² + mean_bias²)` (was robust `sqrt((IQR/1.349)² + median²)`). Both still logged.
- **config/base.yaml**: `wd_exclude_1d: false` (the v2 promotion is NOT applied here);
  everything else already = sw_d192. Submitted as job 82120.

### 3.7 `transformer_v2b` — configurable selection metric + multi-best checkpoints (new dir)
- **(a)** `train.py::validate` builds a **configurable composite** metric from `config
  train.select_metric` (`core` robust|raw_std, `bias` median|mean|none, `tail_term`
  none|outlier|raw_std|excess, `tail_weight`): `sqrt(core²+bias²) + tail_weight·tail_term`.
  Defaults reproduce the v2_d192 method exactly (verified bit-identical at tail_weight=0).
- **(b)** `train.py::main` saves **three** best checkpoints per run: `best.pt` (composite,
  drives early stop), `best_robust.pt` (min robust core), `best_raw.pt` (min raw std =
  sw_d192-style) — compare selection strategies post-hoc, no retraining.
- **config/base.yaml**: `select_metric: {core:robust, bias:median, tail_term:none,
  tail_weight:0.0}`, `wd_exclude_1d: false`. `job_v2b.sub` evaluates all three checkpoints
  into `metrics_<ck>.json`. Submitted as job 82121.

---

## 4. Key findings & diagnoses

- **KG weighter is stable in all healthy runs.** `w_energy` peaks ~1.3 (start ~1.0,
  final ~1.05); `w_recon` ~1.1; `w_concept` ~1.0. The loss-normalized KG + log-var
  weight-decay fix works — no runaway (vs the historical `w_energy→450`).
- **The raw-std gap vs historical sw_d192 (8.60% vs 6.52%) is a selection artifact, not a
  regression.** The bias-aware robust metric `√(robust²+median_bias²)` is tail-insensitive
  and trades core width for calibration; the baseline's selected epoch (34) had good bias
  but a heavier tail than later epochs (44: val raw 6.35% ≈ historical, robust 1.55%, not
  selected). The robust *core* (~1.4–1.6%) is reproduced to within noise. The historical
  1.36% is itself "indicative only" (re-scored on a raw-std-selected checkpoint).
- **`wdx` win** — excluding 1-D params from WD improves core, outlier fraction, AND bias.
- **`v2m1` win on the tail** — fit-quality down-weighting gives the best raw-std (7.53%) of
  all runs; confirms poorly-fit (label-noisy) events were inflating the tail. Orthogonal to
  `wdx` → combined run queued.
- **`ep100`** — longer training nudged the core down but produced a few catastrophic
  outliers (raw-std 15.65% while outlier-count fell): not adopted.
- **`v2test` reproduces sw_d192** — raw-std selection recovers the historical RAW numbers
  (raw-std 6.61%≈6.52%, binavg-raw 6.11%≈5.97%, bias −1.59%≈−1.37%); its robust core is 1.65%,
  confirming the "1.36% robust" was the re-scored "indicative only" number. The raw-std gap
  was entirely the selection metric.
- **Selection > architecture for the TAIL.** `v2b` (one run) yields best_robust (robust 1.45%)
  vs best_raw (raw 8.10%) vs best/composite (1.54% / 12.14%) — selection alone moves raw-std
  ~4 pp at ~constant core. And raw-std is HIGH-VARIANCE run-to-run (6.6–19.4%) while robust
  core is stable (~1.4–1.65%): chase robust, treat raw-std as noisy.
- **`v2m2b`: weighter fix worked, idea still rejected.** No blow-up (robust core 1.46% vs the
  broken 7.40%) — confirms v2m2's failure was the weighter interaction. But the heteroscedastic
  head still makes extreme energy outliers (raw-std 19.36% at normal outlier count) → no win.
- **`v2m2` FAILED — weighter runaway.** The heteroscedastic concept NLL goes negative as
  concepts are learned (`val_concept` min −1.0); the homoscedastic KG weighter
  EMA-normalizes by loss magnitude and is then *rewarded* for driving `w_concept = exp(−s)`
  to **7×10⁷**, corrupting the shared encoder (energy core 7.40%). Same runaway class as
  the original `w_energy→450`, re-triggered by the sign flip. NOT a refutation of the idea —
  fix: take the concept task OUT of the learnable/EMA weighter (fixed weight) so only the
  per-event NLL provides adaptivity. (Retry = "v2m2b".)

---

## 5. Current state / next steps

1. ✅ Promoted `wd_exclude_1d=true` to the `transformer_v2` default.
2. ⏳ `wdx + v2m1` combined run (82010) RUNNING — expected best (wdx core + v2m1 tail).
3. ⏳ `v2m2b` (82026) PENDING — heteroscedastic concept head with the concept task on a
   FIXED weight (out of the learnable/EMA KG weighter); clean A/B vs baseline and v2m2.
4. ✅ `v2test` (82120) — reproduced sw_d192 RAW numbers (6.61%≈6.52%); raw-std gap = selection.
5. ✅ `v2b` (82121) — multi-best confirms selection moves raw-std ~4 pp at ~constant core.
6. ✅ `v2m2b` (82026) — weighter fix worked; heteroscedastic concept idea rejected (raw 19.4%).
7. ⏳ CHAMPION `v2cham` (82316, PENDING) — `transformer_v2cham` = wdx + v2m1 fit-weighting +
   v2b multi-best/configurable selection (tail_weight=0). One run → best.pt/best_robust.pt/
   best_raw.pt; pick best_robust for core (~1.38% target), best_raw for tail.
8. ⬜ Physics frontier (separate): saturation + lateral-leak concepts (kx_Eloss/ElossSat,
   kx_expehit_imag) targeting the high-E tail with available data.
