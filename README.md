# transformer/ — physics-grounded ECAL energy reconstruction (AMS-02), build "dual_v1"

A Transformer over ECAL cell deposits that is **forced to reason through physics**
instead of a black box. The 3D-fit shower parameters shape an intermediate layer (a
soft concept bottleneck); the energy estimate is split into a physics-grounded part and
a free correction, so the representation is inspectable and physically grounded.

## This build (`config/base.yaml` defaults)

```
input    :  kx_ehit           per-cell deposition, sparse tokens (E > 10 MeV)
energy   :  e_phys + e_free    e_phys from the concept-grounded pooled h_phys;
                               e_free a deep correction -> standardised log(mcEne)
recon    :  kx_expehit         per-cell fitted deposit (denoise)
concepts :  soft bottleneck -> 7 energy-relevant 3D-fit params (x0/y0/kx/ky dropped)
balance  :  NormalizedWeighter (EMA loss-norm + FIXED weights: energy 1.0, aux 0.3)
energy wt:  power-law rolloff above 2 TeV on the energy loss AND the val metric
```

* **Soft concept bottleneck** (`models/bottleneck.py`): at block `tap_block` (=3) the
  token rep splits into `h_phys` (supervised by the 3D-fit concepts) ‖ `h_free`
  (residual), recombined for the upper blocks. The additive `tokens +` residual bypass
  is **OFF** (`bottleneck_residual: false`), so the upper blocks read *only* the
  recombined sub-spaces — the physics sub-space can no longer be routed around.
* **Dual energy head** (`dual_energy_head: true`): `energy = e_phys + e_free`. `e_phys`
  is read off the *same pooled `h_phys`* that predicts the concepts (a physics baseline,
  gauge-fixed by its own auxiliary energy loss); `e_free` is the deep, full-depth
  correction. Additive in standardised log-E ⇒ multiplicative on E. Plus the per-cell
  `kx_expehit` recon head.
* **Loss balance** (`losses/uncertainty.py::NormalizedWeighter`): each task loss is
  EMA-normalised to O(1), then summed with FIXED manual weights (energy 1.0, aux 0.3).
  Replaces Kendall–Gal homoscedastic weighting, whose learnable log-variance ran away
  (w_energy ≈ 450) and destabilised training on the large dataset.
* **Energy-region weighting** (`losses.energy_rolloff`): unit weight up to `e_cut`
  = 2 TeV (the ECAL electron reliability limit), power-law (index 2.7) decay above, on
  both the training energy loss and the validation selection metric — so the
  leakage-dominated high-E tail no longer drives the score (no hard cut).
* **Concept trim** (`data.concept_use`): the bottleneck predicts the 7 energy-relevant
  concepts (`shwr_z0, shwr_a0, frac_lat, frac_rear, tmax, lat_width, long_width`);
  `x0/y0/kx/ky` are dropped — energy-irrelevant and not linearly decodable from a pooled
  head (R²≈0). The cache still stores all 11; the trim is a load-time subset, no rebuild.

Every item above is a config flag (`model.bottleneck_residual`, `model.dual_energy_head`,
`loss.weighting`, `loss.weights`, `loss.energy_rolloff`, `data.concept_use`) so each can
be toggled for ablation; defaults in code reproduce the pre-dual_v1 behaviour.

## Variants — controlled comparison (same events, same stack, only one thing differs)

* **`../transformer_m4`** — same dual head, but the physics/free split is at the encoder
  **output** (end-split), not mid-network. Isolates the split location.
* **`../transformer_m2`** — **anchored** energy: `kx_EneL2Cor` + a tight log-space
  residual (no dual head). Isolates the value of the in-detector leakage anchor.
* **`../transformer_m3`** — clean cell→energy, no scaffolding (control).

## Geometry — physical, never indices (`data/geometry.py`)

From `../geo.md` + the AMS-02 ECAL papers:
* 9 superlayers, **view alternates per superlayer**: `view = (ilayer//2) % 2`
  → X = superlayers {0,2,4,6,8} (5), Y = {1,3,5,7} (4). `OffSetMC[sl][ilayer%2]`.
* depth `z = Ecal_Z[ilayer]` (cm); pitch 9 mm; each layer ≈ 1 X₀.
* A layer measures one projection → token = `(z, t, view)`; attention fuses views.
* Augmentation = x/y **reflections only** (90° is invalid: 5 ≠ 4 superlayers); concept
  targets are sign-flipped accordingly.

## Run (IHEP GPU node, RTX 5090 / CUDA 13)

This cluster is SLURM. Submit the batch job (sources the CUDA-13 env, trains → evaluates
→ probes on `cache_full10`):

```bash
sbatch job_dual_v1.sub
# or interactively, after `source .../HREDML/loadCondaEnvCuda13.sh`:
python data/preprocess.py --config config/base.yaml      # ROOT -> cache (once)
python train.py    --config config/base.yaml
python evaluate.py --config config/base.yaml
python probe.py    --config config/base.yaml
```

Acceleration: bf16 autocast, Flash SDPA attention, `torch.compile`, TF32. Override
anything: `python train.py --set model.d_model=256 train.compile=false`.

## Did it learn physics? (`probe.py`)

1. **Linear probe** — R² of `h_phys` vs `h_free` to the concepts.
2. **Subspace ablation** — zero `h_phys` vs `h_free` and measure the energy shift (now
   meaningful: with the bypass off, killing `h_phys` actually moves the energy).
3. **Dual-head decomposition** — σ/E of `e_phys` alone vs the full `e_phys + e_free`, and
   the free head's contribution.
4. **Mirror stress test** — energy must be invariant under x-reflection.

## Files

| Path | Purpose |
|------|---------|
| `config/base.yaml` | all knobs; selection = contained + single-shower; 10 MeV cell cut |
| `data/geometry.py` | physical cell geometry (per-superlayer view) |
| `data/inspect_root.py` | verify branches/units — run first |
| `data/preprocess.py` | ROOT → tokenised CSR cache + concept/energy targets + meta |
| `data/dataset.py` | dataset, token features, reflection augmentation, concept-trim, collate |
| `models/embedding.py` | token + positional embedding |
| `models/encoder.py` | pre-LN SDPA Transformer blocks |
| `models/bottleneck.py` | soft concept bottleneck (residual-bypass flag + `e_phys` head) |
| `models/heads.py` | energy + recon heads (+ reserved registry) |
| `models/model.py` | assembly; `energy = e_phys + e_free`; forward returns a dict |
| `losses/objectives.py` | per-task losses with optional per-sample weights |
| `losses/uncertainty.py` | `NormalizedWeighter` / `UncertaintyWeighter` / `FixedWeighter` |
| `train.py` / `evaluate.py` / `probe.py` | train / metrics+plots (+ ≤2 TeV) / physics tests |
| `job_dual_v1.sub` | SLURM job (train→eval→probe on `cache_full10`, 50 epochs) |

See `../var.md` for the ECAL branch dictionary and `../transformerReport/report.tex` for
the running log of edits and results.
