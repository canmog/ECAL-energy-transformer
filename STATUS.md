# angleTransformer status — 2026-08-06

## Scope

Reconstruct the incoming MC direction from ECAL cell deposits only, using the
`ecalTransformer` d192 geometry/model/training settings. The primary output is
`(dx/dz,dy/dz)`; energy, expected-cell reconstruction, and 11 concepts remain
auxiliary training tasks.

## Dataset extension

The 50 `datasets/full/*-full.root` files are augmented from the paired `mcinfo`
files by `(run,event)`. New branches are:

```text
mcTheta mcPhi mcDirX mcDirY mcDirZ mcKX mcKY
```

Each rewrite is temporary-file-first and atomic. Before replacement it checks the
entry count and every original branch name/title. Independent verification checks:

- all original 3D-fit branches and leaf layouts;
- deterministic content samples across every original leaf;
- `mcEne` against `mcinfo.p` for every event;
- run/event agreement among full, MC, and 3D-fit for every event;
- angle branches against `mcinfo.theta/phi` and unit-vector normalization.

## Implementation

- Direction-aware preprocessing/cache with persistent run/event identity.
- Physically correct X/Y reflection of truth and 3D-fit slopes.
- Learned-query two-component angle head.
- MC-truth angle Huber loss, never weighted by 3D-fit quality.
- Checkpoint selection by validation 68% angular containment.
- fp32 evaluation with median/p68/p90/p95 in mrad versus energy/incidence.
- Same-event `kx_ShwrKX/KY` baseline, symmetry/ablation probes, and NPZ export.

## Validation completed

- All 50 full files were updated and independently verified: batch summary
  `ok=49 skip=1 fail=0 verify_fail=0` (file 0000 was updated during preflight).
- Aggregate audit: 50 full, 50 3D-fit, and 50 MC files with exactly 17,743,152
  entries in each chain; all required branches are present and no
  `.angle-tmp.root` files remain.
- The strengthened verifier was applied to every file. It checks 30,525
  deterministic original-leaf values per file exactly and exhaustively checks
  event keys, energy, and angle truth; maximum direction-component discrepancy
  was approximately `2.98e-08`.
- ROOT dry run on the largest representative file: 474,615/474,615 matched,
  26 existing branches preserved, 7 added.
- Committed first-file independent check: 33 branches, 30,525 original leaf
  values sampled exactly, maximum direction-component discrepancy `2.98e-08`.
- Original `ecalTransformer/data.inspect_root.py` passes on the rewritten file.
- Real 20k-event preprocess: 3,619 selected events, 1,060,676 tokens.
- The old and new preprocessors produce bit-exact shared cache fields (`off`,
  tokens, energy, and concepts) on those same 20k input events.
- CPU forward: correct energy/recon/concept/angle output shapes and four losses.
- SLURM smoke job `125367`: `COMPLETED`, exit `0:0`; preprocess, two epochs,
  evaluate, probes, plots, and prediction export all completed.

The two-epoch/2,535-event smoke model is only a runnability test; its angular
numbers are not a physics result.

## Production

The all-file ROOT update and independent verification are complete.
`job_angle_d192.sub` builds the full cache if needed and runs the 50-epoch d192
training, fp32 evaluation, probes, and prediction export.

Production job `129152` was submitted on 2026-08-06 and started on `aigpu004`.
Its CUDA check, direction unit tests, and 50-file ROOT schema inspection passed;
the full-cache build completed with 3,384,017 selected events and 992,930,301
tokens, and training started.

First results showed that global-pooling jobs `129152` and `129998` collapsed to
approximately `440 mrad` test p68 versus `6.78 mrad` for 3D fit.  View-aware job
`130114` reached `35.45 mrad` validation p68 at epoch 63; wide job `130000` is
still running.

A convention audit then found that fibre-orientation view labels measure the
orthogonal coordinate: direction components require `component_views=[1,0]`.
The corrected analytic centroid estimate reaches `42.31 mrad` p68 on the small
preflight validation set.  Follow-up `170817` learns a zero-initialized neural
correction to that baseline and is running; corrected view/chord/no-aux job
`170840` is pending on the normal-QoS two-GPU limit.  Success remains strictly
test p68 below the same-event 3D-fit p68 in each run's `metrics.json`.

## Latest result — 2026-08-11

Four fixed-10-MeV d96 controls (`485937`--`485940`) completed on the same held-out
507,602-event test set.  Propagating the corrected view convention through the
augmentation, probes, and tests removed the old mean-direction collapse.
Residual-specific robust normalization then reduced test p68 from the analytic
centroid's `43.244 mrad` to `7.060 mrad` without reflections, compared with
`6.780 mrad` for 3D fit.  This small 400k-event model already improves the 3D-fit
p95 (`18.233 vs 18.933 mrad`), RMS (`9.778 vs 13.294 mrad`), and >20 mrad rate
(`3.819% vs 4.431%`), and beats its p68 in every energy bin above 500 GeV.

Reflection augmentation worsened residual p68 to `7.687 mrad`, so the current
best configuration is the normalized centroid residual without reflections.
Its best checkpoint was the final epoch (validation/test p68
`7.092/7.060 mrad`); the next search should scale this configuration and target
the remaining low-energy core-resolution deficit.

Four full-data d192 follow-ups were submitted as jobs `558488`--`558491` with a
100-epoch maximum, minimum epoch 40 before early stopping, and patience 25.
They compare the scaled reference, a wider core Huber region, train-only
low-energy weighting, and weak physics auxiliary supervision. Jobs `558488`
and `558489` started immediately; `558490` and `558491` are valid submissions
waiting on the site's two-GPU per-user QoS limit. Earlier IDs
`558354`--`558357` were cancelled because their minimum early-stop epoch was
incorrectly set to 100.

## Full-data direction result — 2026-08-13

Jobs `558488`--`558491` completed successfully. The new result of record is job
`558491`, the full-data d192 normalized centroid residual with weak physics
auxiliaries. On the identical 507,602-event test sample it reaches
median/p68/p90/p95 `0.1782/0.2719/0.5707/0.7666 degree`, compared with
`0.2708/0.3885/0.7803/1.0848 degree` for 3D fit. Its p68 is 30.0% better, its RMS
49.7% better, and its fraction above 1.1459 degree falls from 4.431% to 1.419%.
It beats 3D fit in every stored energy and incidence bin.

The primary MC benchmark is therefore met at the fixed 10 MeV threshold. The next
priority is ISS-facing robustness at fixed 20 and 30 MeV (plus a 20--30 MeV
threshold-jitter control), repeated seeds, paired bootstrap intervals, and
detector-domain variations. No lower-threshold scan is recommended.
