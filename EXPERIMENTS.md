# Concurrent direction experiments — 2026-08-06

All experiments use the exact same train/validation/test cache and event split as
the d192 baseline.  Each has a distinct seed, output directory, SLURM log, and
resolved `config.json`.  The held-out test set remains untouched during checkpoint
selection; validation p68 chooses `best.pt`.

Each variant requests two CPU workers.  Focus and wide use `gpunormal`, while view
uses the account's separate one-job `gpuintera` allowance; without another active
GPU run this permits all three experiments to overlap without exceeding site QoS.

First-sweep jobs `129152` (baseline) and `129998` (focus) completed but collapsed
to approximately `440 mrad` test p68 versus `6.78 mrad` for 3D fit.  Jobs `130114`
(view) and `130000` (wide) remain active; view was the first useful model, reaching
`35.45 mrad` validation p68 at epoch 63.

## angle_focus_d192

Keeps the proven d192 encoder for up to 100 epochs but replaces learned uncertainty weights with fixed,
EMA-normalized priorities.  Direction gets full weight while energy, reconstruction,
and 3D-fit-derived concepts are weak regularizers.  A deeper angle MLP and smaller
Huber transition emphasize precise core events.

## angle_view_d192

Uses independent attention pools for X-view and Y-view ECAL layers for up to 80 epochs, matching how
the detector measures the two projected slopes.  It directly minimizes robust
unit-vector chord distance, which is the small-angle equivalent of the reported
opening-angle error.

## angle_wide_d256

Tests for up to 60 epochs whether the resolution is capacity-limited: eight 256-wide blocks, a larger
physics/free bottleneck and view-aware direction head.  It combines standardized
slope Huber and direct chord losses.  The token budget is reduced to keep RTX 5090
memory usage bounded.

The primary comparison is held-out `model.p68_mrad` versus `fit3d.p68_mrad` in
each run's `metrics.json`.  Improvement is only claimed if the Transformer value
is smaller on the same test events.

## Follow-ups after the first results

`angle_centroid_residual_d192` computes an fp32 energy-weighted centroid in every
readout layer, fits the X and Y centroids versus physical z, and initializes a
view-aware Transformer as a zero correction to those two analytic slopes.  Its
validation/test output records the fixed centroid baseline separately from the
corrected model.

A preflight sweep selected layer-energy power `1.5`: with the corrected view
mapping its analytic p68 was `42.31 mrad` on 542 validation events (the original
mapping gave roughly `450 mrad`).

The full-cache convention check showed that the stored ECAL view denotes fibre
orientation: view 0 measures the `ky` projection and view 1 measures `kx`
(`corr=0.976/0.973` on the preflight validation events).  Both follow-ups
therefore set `component_views=[1,0]`; earlier running view jobs retain their
original mapping so their checkpoints remain internally consistent.

`angle_view_chord_noaux_d192` isolates the best-performing ingredients from the
first sweep—view-specific pooling and pure chord loss—while setting energy,
3D-fit reconstruction, and 3D-fit concept weights to zero.  Both follow-ups defer
early stopping until at least 40 epochs because the first view model only broke
the mean-direction symmetry after approximately 20 epochs.

Follow-up job `170817` (centroid residual) is running; job `170840` (corrected
view/chord with no auxiliary gradients) is valid and pending on the two-GPU
`gpunormal` per-user limit.

## Fixed-10-MeV cheap controls — 2026-08-11

Jobs `485937`--`485940` completed successfully.  They used the corrected physical
mapping (stored view 1 measures `kx`, stored view 0 measures `ky`), a d96/four-block
encoder, 400,000 training events, 100,000 validation events, zero auxiliary-loss
weights, and the full 507,602-event test split.  Training and evaluation both used
a fixed 10 MeV cell threshold.  The residual variants robustly normalized the
train-only target `MC slope - analytic centroid slope` per component; the fitted
residual center was approximately `(0.024, -0.011)` and its scale approximately
`(0.0251, 0.0258)`, versus the global slope scale of approximately `0.286`.

| Control | Slurm | Epoch | Median [mrad] | p68 [mrad] | p90 [mrad] | p95 [mrad] | >20 mrad |
|---|---:|---:|---:|---:|---:|---:|---:|
| Direct, no reflection | 485937 | 36 | 8.228 | 10.869 | 17.474 | 21.943 | 6.700% |
| Direct, corrected reflection | 485938 | 38 | 8.112 | 10.768 | 17.449 | 21.873 | 6.662% |
| Centroid residual, no reflection | 485939 | 39 | 4.825 | 7.060 | 13.823 | 18.233 | 3.819% |
| Centroid residual, corrected reflection | 485940 | 38 | 5.425 | 7.687 | 14.529 | 19.054 | 4.335% |
| Same-event 3D fit | -- | -- | 4.727 | 6.780 | 13.618 | 18.933 | 4.431% |

The no-reflection normalized residual is the winner.  It is 4.13% worse than
3D fit in p68 but already improves p95, mean, RMS, and the >20 mrad outlier rate.
Its test p68 is better than 3D fit above 500 GeV: `4.646 vs 4.844 mrad` at
500--1000 GeV, `4.078 vs 4.473 mrad` at 1000--2000 GeV, and `4.219 vs
4.712 mrad` at 2000--4000 GeV.  The remaining deficit is concentrated below
approximately 300 GeV.  Its best validation checkpoint was the final epoch, with
validation/test p68 `7.092/7.060 mrad`, so this cheap run showed no saturation or
validation-to-test degradation.

Corrected reflections no longer cause the old collapse and slightly help direct
regression, but they worsen the normalized residual p68 by approximately 8.9%.
At milliradian precision, cell-index reversal is not an exact detector symmetry
in the presence of layer-dependent alignment offsets and absolute channel
embeddings.  Reflection augmentation is therefore disabled for the next residual
experiments unless a geometry-exact transform is implemented.

## Full-data d192 residual search — submitted 2026-08-11

Four fixed-10-MeV, no-reflection, normalized-centroid-residual jobs were
submitted with the full training cache and a d192/six-block encoder. Each job
may train for at most 100 epochs; validation early stopping cannot trigger
before epoch 40 and then uses a patience of 25 epochs.

| Variant | Slurm | Main change |
|---|---:|---|
| `angle_residual_full_d192_t10` | 558488 | Full-data/capacity reference |
| `angle_residual_coreloss_d192_t10` | 558489 | Huber delta 0.5, chord weight 0.25 |
| `angle_residual_lowenergy_d192_t10` | 558490 | Train-only low-energy angle weighting |
| `angle_residual_physicsaux_d192_t10` | 558491 | Weak energy/reconstruction/concept auxiliary losses |

The superseded submissions `558354`--`558357`, which incorrectly disabled
early stopping through epoch 100, were cancelled and must not be used for
comparison.

### Completed full-data d192 results — 2026-08-13

Jobs `558488`--`558491` all completed with exit code `0:0`. They used the full
2,368,813-event training split, fixed 10 MeV runtime threshold, no reflection,
the corrected `[1,0]` view routing, normalized centroid residual, a 100-epoch
maximum, minimum epoch 40 before early stopping, and patience 25. All reached
epoch 99; validation selected epochs 91--95. Angular results below are degrees
(the evaluator's stored values were converted by `180/(pi*1000)`).

| Variant | Slurm | Best epoch | Median [deg] | p68 [deg] | p90 [deg] | p95 [deg] | >1.1459 deg |
|---|---:|---:|---:|---:|---:|---:|---:|
| Full baseline | 558488 | 91 | 0.1813 | 0.2769 | 0.5846 | 0.7882 | 1.574% |
| Core loss | 558489 | 95 | 0.1819 | 0.2769 | 0.5841 | 0.7872 | 1.597% |
| Low-energy weighting | 558490 | 95 | 0.1831 | 0.2793 | 0.5910 | 0.7961 | 1.635% |
| Physics auxiliary | 558491 | 95 | **0.1782** | **0.2719** | **0.5707** | **0.7666** | **1.419%** |
| Same-event 3D fit | -- | -- | 0.2708 | 0.3885 | 0.7803 | 1.0848 | 4.431% |

The physics-auxiliary residual is the new model of record. Relative to 3D fit it
improves p68 by 30.00%, median by 34.21%, p90 by 26.86%, p95 by 29.33%, RMS by
49.67%, and the large-angle fraction by 67.98%. It wins every stored energy bin
(12.5% at 20--50 GeV, increasing to 51.0% at 2--4 TeV) and all incidence bins.
Weak energy/reconstruction/concept weights `0.05/0.05/0.10` improve p68 by 1.79%
over the angle-only full baseline. The core-loss change is neutral, while the
proposed train-only low-energy weighting is slightly worse in every energy bin.

The physics-auxiliary model remains ECAL-only at inference. MC direction/energy
and 3D-fit concepts/expected deposits are supervised targets during training;
none is an input token or inference-time correction.
