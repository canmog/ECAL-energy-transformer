# ECAL Transformer — energy and incidence-direction reconstruction

This repository now supports three explicit training modes with one sparse-token
Transformer implementation:

| profile | primary reconstruction | default auxiliaries | primary checkpoint |
|---|---|---|---|
| `config/base.yaml` | energy | expected cells, concepts | energy composite |
| `config/angle.yaml` | MC `(kx,ky)` | energy, expected cells, concepts | angular p68 |
| `config/joint.yaml` | energy and MC `(kx,ky)` | expected cells, concepts | configured selection, plus independent best-energy/best-angle files |

Training validates the complete configuration and both cache splits before the
first batch. Missing inputs, targets, normalization metadata, or enabled heads
raise a field-specific error; no zero-valued targets are synthesized. Existing
packed NPZ energy caches remain supported, and the same dataset interface also
opens split memory-mapped NPY caches.

The direction projection convention is physically fixed as stored view 1 to
X/`kx` and stored view 0 to Y/`ky` (`component_views: [1, 0]`). See
[`docs/ANGLE_RECONSTRUCTION.md`](docs/ANGLE_RECONSTRUCTION.md) for the selected
results, commands, inference contract, and provenance.

Typical IHEP commands:

```bash
python train.py --config config/angle.yaml
python evaluate_angle.py --config config/angle.yaml
python export_predictions.py --config config/angle.yaml   # no MC truth required

python train.py --config config/joint.yaml
```

`bash run.sh <config>` dispatches the matching end-to-end workflow. Production
SLURM templates are `job_angle_d192_integrated.sub` and
`job_joint_d192_integrated.sub`.

## Energy v3 — Gaussian-core selection and evaluation

Fork of `transformer_v2cham` (the champion: wdx + v2m1 fit-quality weighting on the
`sw_d192` model). The trained objective is UNCHANGED; v3 changes the **resolution
estimator** and adds the capacity re-check and two analysis tools.

## What's new vs v2cham

* `utils/stats.py::gauss_core` — the calorimetry-standard **Gaussian core sigma**:
  binned Gaussian fit restricted to `mu +/- 2 sigma`, iterated until the fitted sigma
  is stable (seeded from median + IQR/1.349; corrected-truncated-moments fallback,
  then robust fallback — never crashes on a weird epoch-0 distribution).
* `train.py` — validation logs `val_res_gauss` / `val_bias_gauss`; the selection
  metric supports `core: gauss`, `bias: gauss` (the v3 default); multi-best now also
  saves `best_gauss.pt` (min Gaussian core alone) next to best/best_robust/best_raw.
* `evaluate.py` — Gaussian core (overall / <=2 TeV / per-bin) is the PRIMARY metric
  (`overall_res_gauss`, `sigma_estimator: gauss_core_iter2sigma`); robust core stays
  as the cross-check; raw std + outlier fraction stay as tail diagnostics. The
  residual histogram now overlays the fitted Gaussian core.
* `rescore_gauss.py` — re-scores every saved v2-lineage checkpoint + the 3D-fit
  anchor (`cache_m2:anchor` = kx_EneL2Cor) with all three estimators on the SAME
  cache_full10 test split; writes `runs/rescore/rescore_gauss.{json,md}`.
* `outlier_study.py` — profiles the |dE/E|>20% (and >100%) events of a chosen
  checkpoint against per-event physics variables (E, tokens, deposited energy,
  max-cell fraction, fit residual, first/last layer, the 11 concepts, |slope|,
  edge distance); writes `runs/outliers/outliers.md` + rate plots + worst-100 CSV.

## Jobs

| job | what |
|---|---|
| `job_rescore.sub`  | rescore all checkpoints + champion outlier profile (no training) |
| `job_v3_d192.sub`  | champion objective + gauss selection, d192 (A/B vs v2cham: selection estimator only) |
| `job_v3_d256.sub`  | capacity re-check, d256, lr 2e-4 |
| `job_v3_d320.sub`  | capacity re-check, d320, lr 1.5e-4 |

Estimator definitions: gauss = iterative +/-2 sigma binned fit; robust = IQR/1.349;
raw = np.std. Outlier = fraction |dE/E| > 20%. All selection restricted to E <= 2 TeV.

## Evaluation precision (found 2026-07-02)

bf16 inference adds a median 0.38% per-event energy perturbation, inflating the
Gaussian core by 5-13% relative (measured A/B: 1.12 -> 1.26% at 1-2 TeV). Training
stays bf16, but EVALUATION is fp32: the v3 job files pass `train.amp_dtype=fp32` to
evaluate.py, and `rescore_gauss.py --fp32` re-scores historical checkpoints without
the penalty. Historical (v1/v2) numbers are bf16-evaluated.
