# transformer_v2/ — clean `sw_d192` baseline (foundation for the next step)

A **clean reproduction of the `sw_d192` baseline** — the best ECAL energy-resolution
result obtained so far on the complete AMS-02 dataset. This directory exists to be a
tidy, known-good starting point: nothing from the regressed `dual_v1` experiment line
is active, so the next idea is built on top of the configuration that actually wins.

See `../transformerReport/report.tex` (and `report.pdf`) for the full study. The model,
physics rationale, geometry, and probes are documented in `../transformer/README.md`;
this file only covers what is specific to `transformer_v2`.

## What "the `sw_d192` baseline" is

`sw_d192` on the complete 50-file dataset (strict cache, `cache_full10`):

```
bin-averaged sigma/E :  5.97 %  (full range)     5.62 %  (E <= 2 TeV)
robust-core sigma    :  1.36 %  (IQR/1.349, E <= 2 TeV)
overall sigma/E      :  6.52 %      bias -1.37 %      d_model = 192
```

It matches the single-file baseline on the **complete** data and is markedly better at
TeV (3 TeV bin 8.8 % vs the broken 18.4 %). Concepts are learned at R² ≈ 0.95–1.00 for
the energy-relevant set. Per the report, **no variant has beaten it**: the later
`dual_v1` "new stack" (NormalizedWeighter + 2 TeV rolloff + 7-concept trim, dual head,
bypass-off) cost ~+1.66 % binned σ/E, outweighing every architecture/anchor gain. Hence
this reset.

## The config (`config/base.yaml`) = `sw_d192`, exactly

The code here is the same `dual_v1` build as `../transformer/`, but **every new-stack
knob is config-gated back to the pre-`dual_v1` behaviour**, reproducing `sw_d192`:

| knob | `sw_d192` (here) | `dual_v1` (off here) |
|------|------------------|----------------------|
| `model.dual_energy_head` | `false` — plain energy head | `true` — `e_phys + e_free` |
| `model.bottleneck_residual` | `true` — `tokens +` skip ON | `false` — bypass removed |
| `data.concept_use` | *(absent)* — all **11** concepts | 7-concept trim |
| `loss.weighting` | *(absent)* → **Kendall–Gal** (`uncertainty`) | `normalized` (fixed aux 0.3) |
| `loss.energy_rolloff` | *(absent)* — no high-E down-weight | enabled (2 TeV, index 2.7) |
| `model.dropout` | `0.05` | `0.1` |
| `train.epochs` | `50` | `120` |

Two `dual_v1`-era *improvements* that are not part of the regression are **kept**, since
they only make evaluation/selection more honest (they do not change the trained objective
for this config):

* `train.py::validate` selects on the **robust-core** metric √(σ_robust² + bias_median²),
  σ_robust = IQR/1.349, on E ≤ 2 TeV — tail-insensitive, so model selection is stable.
* `evaluate.py` reports an E ≤ 2 TeV summary alongside the full range
  (`e_cut` defaults to 2000 GeV when `loss.energy_rolloff` is absent).

Verified: `config/base.yaml` resolves to plain head / bypass ON / 11 concepts /
Kendall–Gal / no rolloff / dropout 0.05 / 50 epochs.

## Data / cache

By default `paths.cache_dir` points at the **shared strict cache**
`/aifs/.../transformer/cache_full10` (10 MeV cell cut, contained + single-shower,
3.38 M events, 11 concepts stored) — the exact cache `sw_d192` trained on. This avoids
re-preprocessing 3.38 M events. To build a private cache instead:

```bash
python -m data.preprocess --config config/base.yaml --set paths.cache_dir=cache_full10
```

## Run (IHEP GPU node, RTX 5090 / CUDA 13)

```bash
sbatch job_v2.sub          # train -> evaluate -> probe on cache_full10, d192, 50 epochs
```

Or interactively, after bringing up the node (`../aiGPU.md`) and
`source .../HREDML/loadCondaEnvCuda13.sh`:

```bash
python train.py    --config config/base.yaml
python evaluate.py --config config/base.yaml
python probe.py    --config config/base.yaml
# override anything: python train.py --config config/base.yaml --set model.d_model=256
```

`bash smoke_test.sh` runs a 2-epoch end-to-end check on a **private** 20 k-event throwaway
cache (`cache_smoke`) — it never touches the shared `cache_full10`.

## Files

| Path | Purpose |
|------|---------|
| `config/base.yaml` | all knobs; **set to the `sw_d192` baseline** |
| `data/geometry.py` | physical cell geometry (per-superlayer view) |
| `data/inspect_root.py` | verify ROOT branches/units — run first on a new cache |
| `data/preprocess.py` | ROOT → tokenised CSR cache + concept/energy targets + meta |
| `data/dataset.py` | dataset, token features, reflection augmentation, concept-trim, collate |
| `models/embedding.py` | token + positional embedding |
| `models/encoder.py` | pre-LN SDPA Transformer blocks |
| `models/bottleneck.py` | soft concept bottleneck (residual-bypass + `e_phys` flags) |
| `models/heads.py` | energy + recon heads (+ reserved registry) |
| `models/model.py` | assembly; forward returns a dict |
| `losses/objectives.py` | per-task Huber losses with optional per-sample weights |
| `losses/uncertainty.py` | `UncertaintyWeighter` (Kendall–Gal) / `NormalizedWeighter` / `FixedWeighter` |
| `train.py` / `evaluate.py` / `probe.py` | train / metrics+plots / physics probes |
| `job_v2.sub` | SLURM job (train→eval→probe on `cache_full10`, d192, 50 epochs) |
| `run.sh` / `smoke_test.sh` / `summarize_runs.py` | full pipeline / quick check / sweep summary |

See `../var.md` for the ECAL branch dictionary and `../geo.md` for the geometry.
