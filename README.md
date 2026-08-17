# angleTransformer

Reconstruct the MC incidence direction from AMS-02 ECAL cells with the same
geometry, sparse-token Transformer, strict selection, and optimization settings
as `ecalTransformer`.

This workspace is a backup device. Do not run Git/GitHub commands here. Edit the
local tree, let Mutagen synchronize it to
`lsl:/aifs/user/data/lishanglin/chenhao`, and execute data/GPU work on `lsl`.

## Direction convention

The network regresses two slopes rather than `(theta,phi)`:

```text
mcKX = dx/dz = tan(mcTheta) cos(mcPhi)
mcKY = dy/dz = tan(mcTheta) sin(mcPhi)
```

For these downward-going events the incoming unit vector is
`(-mcKX,-mcKY,-1)/sqrt(1+mcKX^2+mcKY^2)`. This representation has no `phi=+/-pi`
discontinuity. X reflection flips `mcKX`; Y reflection flips `mcKY`.

## Augmenting datasets/full

`root/addMcAngleBatch.C` clones every existing branch into a same-directory
temporary ROOT file, adds these truth branches, verifies the result, and only then
atomically replaces the original pathname:

```text
mcTheta mcPhi mcDirX mcDirY mcDirZ mcKX mcKY
```

The updater matches MC and 3D-fit events on `(run,event)`, not entry position.
`root/verifyFullAngles.C` is an independent, entry-by-entry verifier against both
source files.

Dry-run one file without modifying it:

```bash
ssh lsl
cd /aifs/user/data/lishanglin/chenhao/angleTransformer
root -l -b -q 'root/addMcAngleBatch.C("/aifs/user/data/lishanglin/datasets/full/eBep0_254000.list_0000.emini08-full.root","/aifs/user/data/lishanglin/datasets/mcinfo/eBep0_254000.list_0000.emini08.root",false)'
```

Update and independently verify all 50 files:

```bash
bash root/update_full_angles.sh
bash root/verify_all_full_angles.sh
```

The scripts are idempotent: fully updated files are verified and skipped.

## Model and metrics

The primary output is standardized `(mcKX,mcKY)` from a learned-query attention
pool. Energy, expected-cell reconstruction, and the 11 concept targets remain
auxiliary tasks. The angle truth is never down-weighted by 3D-fit quality.

Checkpoint selection uses the validation 68% containment opening angle. Evaluation
is fp32 and reports median/p68/p90/p95 in degrees versus energy and incidence angle,
alongside the `kx_ShwrKX/KY` 3D-fit baseline on exactly the same events.

## Run on lsl

```bash
cd /aifs/user/data/lishanglin/chenhao/angleTransformer
bash smoke_test.sh

# Full cache + interactive training/evaluation
bash run.sh

# Or production training (builds the full cache when needed)
sbatch job_angle_d192.sub
```

`export_predictions.py` writes run/event IDs, slopes, unit vectors, theta/phi,
truth, and the 3D-fit baseline to `predictions_<split>.npz`.

The three concurrent direction-improvement variants, their physical motivation,
and current job IDs are recorded in `EXPERIMENTS.md`.  Submit the same controlled
set with `bash submit_angle_experiments.sh` when no copies are already queued.

## Post-J1 jobs: learned fit and joint reconstruction

Two full-data jobs extend the fixed-10-MeV J1 result without changing or removing
any Transformer outputs. Both use the J1 d192/six-block architecture, seed 73111,
no reflection, a 100-epoch maximum, minimum epoch 40 before early stopping, and
patience 25. Submit them together with:

```bash
bash submit_angle_joint2.sh
```

`angle_residual_robustfit_d192_t10` changes only J1's analytic starting axis. It
first performs the original `E_layer^1.5` centroid fit, predicts a bounded
per-layer reliability multiplier from ECAL-only layer energy, width, occupancy,
maximum-cell fraction, depth, and first-fit residual, and refits the two views.
The reliability output is initialized to one, so training starts exactly at the
fixed J1 fit. The residual head, normalization, loss, schedule, and zero auxiliary
weights remain J1.

Submitted as SLURM job `823429` (`an5_robust`).

`angle_energy_multitask_d192_t10` keeps the fixed J1 centroid and trains one shared
Transformer on all four established tasks: MC angle, MC energy, expected-cell
reconstruction, and all 11 concepts. It uses EMA-normalized Kendall--Gal weighting
with log-variance decay 0.01. Fit-quality weights apply only to reconstruction and
concept losses; MC energy and MC angle are never fit-quality weighted. It saves
separate best-angle and best-energy validation checkpoints, and evaluates energy
with the fp32 Gaussian-core estimator used by the promoted energy model.

Submitted as SLURM job `823430` (`an5_joint`).

At inference both jobs remain ECAL-only. MC truth, expected deposits, concepts, and
3D-fit slopes are targets or references, never input tokens.

### Completed result and repeated-seed replicas (2026-08-16)

Job `823429` completed all 100 epochs. Its learned two-pass coarse fit improved
the fixed-centroid p68 from `2.4777 deg` to `2.2127 deg`, but its final angle p68
was `0.2776 deg`, slightly worse than J1's `0.2769 deg`; this branch is not
promoted.

Job `823430` also completed all 100 epochs. Its best-angle checkpoint (epoch 99)
has angle p68/p90/p95 of `0.2770/0.5709/0.7606 deg`, RMS `0.3843 deg`, and
`1.332%` of events above `1.1459 deg`. Thus p68 matches J1 while angular tails
improve. Its energy Gaussian-core resolution is `1.0965%` below 2 TeV and
`1.1100%` over all energies; raw energy tails remain slightly worse than the
energy-only champion.

Two exact joint-model replicas were submitted with:

```bash
bash submit_angle_joint_seeds.sh
```

- job `974909` (`an5_js123`): seed 123,
  `runs/angle_energy_multitask_d192_t10_s123`;
- job `974910` (`an5_js777`): seed 777,
  `runs/angle_energy_multitask_d192_t10_s777`.

Both use the same data split, fixed 10 MeV threshold, architecture, four losses,
optimizer, 100-epoch maximum, epoch-40 early-stopping floor, patience 25, and
best-angle/best-energy evaluation as job `823430`. Only seed, job name, and output
directory differ.
