# Code Analysis: `transformer/` — AMS-02 ECAL Transformer

> Analysis based **on the code only** (comments/README deliberately not trusted, per request).
> Date: 2026-06-09. Scope: every `.py` / `.sh` / `.yaml` file plus the cached artifacts in `cache/` and `runs/base/`.

---

## 1. What this project does

It is a complete, self-contained pipeline that trains a Transformer to do **electromagnetic-calorimeter energy reconstruction** for the AMS-02 ECAL (18 layers × 72 cells, alternating X/Y readout views), with an **interpretability mechanism** ("soft concept bottleneck") bolted into the middle of the network, plus a verification suite.

Concretely, the code:

1. **Reads a ROOT file** (`tree3dfit` tree) containing per-event 18×72 cell-energy grids (`kx_ehit`, stored in ~MeV), a matching grid of fitted/expected cell energies (`kx_expehit`), MC truth energy (`mcEne`, GeV), and 8 scalar "3D-fit" shower parameters (axis position x0/y0/z0, direction slopes kx/ky, amplitude a0, lateral/rear leakage fractions).
2. **Tokenizes** each event into a sparse set of hit cells (every cell above 0.1 MeV becomes one token), stores them in a CSR-style ragged cache (`train/val/test.npz`) with normalization stats in `meta.json`.
3. **Trains a 6-block Transformer encoder** (d_model=192, 8 heads, ~3.2 M parameters — consistent with the 12.7 MB float32 checkpoint) on three simultaneous objectives:
   - **energy**: a scalar head (attention pooling + MLP) regressing standardized log(mcEne);
   - **recon**: a per-token head regressing log1p(`kx_expehit`) of each hit cell (i.e. reproduce the 3D fit's expected deposit per cell);
   - **concept**: at block 3 the token representation is linearly split into a 64-dim "physics" subspace and a 128-dim "free" subspace, recombined residually; the pooled physics subspace must predict the 8 standardized 3D-fit parameters.
   The three losses are auto-balanced with Kendall–Gal homoscedastic uncertainty weighting; model selection / early stopping uses a bias-aware metric `sqrt(res² + bias²)` of the relative energy residual on the validation split.
4. **Evaluates** on the test split: σ/E and bias binned in true energy, per-concept RMSE/R², four diagnostic plots.
5. **Probes** the trained model with three falsification tests: (a) closed-form ridge regression from frozen `h_phys` vs `h_free` to the concepts (is physics linearly concentrated in the supervised subspace?), (b) causal ablation — zero one subspace at the tap and measure the RMS shift of the energy output, (c) an x-mirror invariance stress test.

Physics-aware details that are genuinely implemented in code (not just claimed):

- **Geometry** (`data/geometry.py`): per-layer z positions, per-view common offsets, per-(superlayer, readout) micron-level alignment offsets, view assignment `view = (ilayer//2) % 2` (10 X-layers / 8 Y-layers). Each token carries `[log1p(E_MeV), t_norm, z_norm, depth_norm, view_x, view_y]` plus a learned positional embedding indexed by `layer*72 + cell`.
- **Reflection augmentation** (`data/dataset.py`): x- or y-mirror with 50% probability each, implemented as cell re-indexing `icell → 71−icell` restricted to the affected view's tokens, with sign flips of concept targets (x0, kx flip under x-mirror; y0, ky under y-mirror). 90° rotation is correctly *not* offered (5 vs 4 superlayers per view make it invalid).
- **Masking**: padded tokens are excluded as attention *keys* (SDPA boolean mask `(B,1,1,L)`), excluded from all pooling (softmax over `-inf`), and excluded from the recon loss. Zero-token events are dropped at preprocessing, which is what protects the all-`-inf` softmax from NaN.

---

## 2. File-by-file findings

### `data/geometry.py`
- Builds `[18,72]` lookup tables `t_cm / z_cm / view / depth`. Transverse coordinate = centered cell index × 0.9 cm + per-view common offset + per-(superlayer, readout) fine offset (µm→cm).
- Self-consistent; the `__main__` self-test checks the view pattern.
- **Latent bug**: `OFFSET` has keys `"MC"` and `"ISS"` only, but `COMMON_OFFSET` and the config enum allow `"TB"`. `build_geometry_table("TB")` raises `KeyError`. Harmless today (config uses `MC`) but the `TB` switch is broken.
- Normalization constants (`_T_HALF ≈ 33.4 cm`, z mid/half-range) are defined here so dataset and probe agree — good single-source design.

### `data/preprocess.py`
- Streams the tree in 20 k-event chunks via `uproot.iterate`, applies selection flags (only `require_truth: mcEne > 0` is on), tokenizes cells `> 0.1 MeV`, accumulates CSR offsets, drops zero-token events, makes a seeded random 70/15/15 split, saves npz + `meta.json` (normalization computed from **train only** — correct, no leakage).
- `_as_3d` assumes the `[18][72]` branch is layer-major row order when reshaping `(n, 18, 72)`. This is an *assumption about the ROOT layout* that nothing verifies programmatically; `inspect_root.py` only lets a human eyeball it. If the branch were cell-major, every geometric feature would be silently wrong. Worth one explicit assertion (e.g. compare per-layer occupancy against a known event).
- `max_events` is honored only at chunk granularity (overshoot up to one chunk) — fine.
- `gather()` is O(N) python loop over events — slow but correct.

### `data/dataset.py`
- `EcalTokens.__getitem__`: applies augmentation (train only), rebuilds geometry features from (layer, cell) after the mirror, recomputes `pos_id` after the mirror — internally consistent (the positional embedding and the `t_norm` feature always agree).
- **Augmentation subtlety**: a mirror maps `base → −base` but the alignment offsets are *not* mirrored, so a mirrored event's tokens sit ~2·(common+fine offset) ≈ 1.5–2.6 mm away from where a genuinely mirrored shower would be, while the concept target is exactly `−x0`. This injects mm-level noise into the augmented concept supervision — negligible vs the 9 mm pitch and the ~19 cm spread of x0, but the mirror is approximate, not exact.
- **Real issue (moderate)**: augmentation uses the **global** `np.random` state inside DataLoader workers. On Linux/fork, all 6 workers inherit an identical NumPy RNG state, so the per-worker coin-flip sequences are identical (correlated augmentation across workers). Standard fix: `worker_init_fn` or a per-dataset `np.random.Generator` seeded from `torch.initial_seed()`.
- `collate` zero-pads to the batch max length and returns `valid`; it also builds `pad_mask = ~valid` which **no consumer ever uses** (dead output; the model takes `valid` directly).

### `data/inspect_root.py`
- Pure diagnostics: branch presence, shapes, value ranges, tokens/event at the 0.1 and 5 MeV thresholds. No issues.

### `models/embedding.py`
- 2-layer MLP on the 6 input features + learned `nn.Embedding(1296, d_model)` positional table, LayerNorm, dropout. Sound.

### `models/encoder.py`
- Standard pre-LN blocks with fused QKV and `F.scaled_dot_product_attention`; key-side masking only. Padded *queries* still attend and produce garbage outputs, but every downstream consumer masks them — correct, and the standard efficient choice.

### `models/bottleneck.py`
- `h_phys = Linear(d_model→64)`, `h_free = Linear(d_model→128)`, recombined `Linear(192→d_model)` added residually with LayerNorm. The concept head reads `AttnPool(h_phys)`. The `phys_scale`/`free_scale` arguments multiply the subspaces **only on the recombination path** (the concept output itself is always unscaled) — exactly what the probe's causal ablation needs. Architecturally clean.
- Note: this is a *soft* bottleneck in the weakest sense — the residual connection `tokens + recombine(...)` means the upper blocks always see the full unfiltered token stream too. The concept supervision shapes `to_phys`, but nothing forces the energy path *through* the bottleneck; that is precisely what probe test 2 is designed to measure rather than assume. Internally consistent design.

### `models/heads.py`
- `AttnPool`: single learned-query attention pooling with `-inf` masking — relies on ≥1 valid token per event (guaranteed by preprocessing). `EnergyHead` = pool→MLP→scalar; `ReconHead` = per-token MLP→scalar. `angle/position/pid` are stub heads that raise if enabled. All fine.

### `models/model.py`
- Wires embed → lower 3 blocks → bottleneck → upper 3 blocks → heads; carries `log_mean/log_std` as buffers (so they persist in checkpoints — correct, evaluate/probe restore them via `load_state_dict`). `predict_energy_gev = exp(mean + std·ŷ)`. Sound.

### `losses/objectives.py`
- Huber (δ=0.1) on standardized log-E and standardized concepts; recon Huber masked by `valid` and normalized per event by token count. All support optional per-sample weights (unused so far). Correct.

### `losses/uncertainty.py`
- Kendall–Gal: `0.5·exp(−s)·L + 0.5·s` per regression task, learnable `s` initialized 0. Correct implementation.

### `train.py`
- bf16 autocast, optional `torch.compile(dynamic=True)`, AdamW over model+weighter params, linear warmup → cosine (epoch-stepped), grad-clip on model params, CSV logging, bias-aware best checkpoint + early stop (patience 20). Mostly solid. Issues:
  - **Misleading logging**: per-epoch task losses/weights are obtained by calling `weighter({k: tensor(0.0)})`, so the CSV's `loss_energy/loss_recon/loss_concept` columns are **always 0.0** (confirmed in `runs/base/metrics.csv`); only the `w_*` columns and the separate `val_*` columns are meaningful. Cosmetic but confusing.
  - **Weight decay (1e-2) is applied to the uncertainty log-variances** (weighter params are in the same AdamW group), shrinking `s → 0` and biasing task weights toward 1 — partially defeats the auto-balancing. Also decays LayerNorm/embedding params (common simplification, mild).
  - Grad clipping covers `raw.parameters()` but not the weighter's two scalars (trivial impact).
  - Train loss accumulator includes the `+0.5·s` regularizer terms, so `train_total` is not comparable to validation losses (and can go negative).
- No correctness bugs found in the training loop itself: loss masking, autocast usage, eval/train mode handling, checkpointing, normalization round-trip are all right.

### `evaluate.py`
- Computes relative residuals with float32 upcast before exp — correct; binned σ/bias skipping bins with <20 events; concepts de-standardized before RMSE/R²; writes `metrics.json` + 4 plots. Correct. Minor: if all bins are sparse the plots silently come out empty; `binned` returning a `map` is consumed once, fine.

### `probe.py`
- Test 1 (linear probe): closed-form ridge, 50/50 split, R² per concept for `pooled_phys` vs mean-pooled `h_free`. **Methodological caveat**: `h_phys` is pooled with the *trained* attention pool (optimized for concept prediction) while `h_free` gets a plain masked mean — the comparison is tilted in favor of `h_phys`, so "h_phys ≫ h_free" is partially built in by construction. A fair version would mean-pool both, or fit the ridge on token-level features.
- Test 2 (ablation): runs the model with `phys_scale=0` / `free_scale=0` and reports RMS relative energy shift. Correctly causal given the bottleneck wiring.
- Test 3 (mirror): flips `t_norm` (feature index 1, correct) for X-view tokens (`view_x` at index 4, correct) and remaps `pos_id` — consistent with the dataset's augmentation convention, including the same mm-level alignment approximation. The printed label says "mean |dE/E|" but the code computes the **RMS** — the number is right, the label isn't.

### `utils/config.py`
- YAML → recursive namespace with dotted CLI overrides and YAML-style scalar coercion (handles lowercase `false`/`null` before `ast.literal_eval` — a real pitfall correctly avoided). Sound.

### `config/base.yaml`
- Internally consistent with the code, except these keys are **read by nothing** (dead config): `data.max_tokens` (no truncation is implemented anywhere — harmless since the physical max is 1296), `train.optimizer`, `train.amp_dtype` (bf16 is hardcoded), `loss.uncertainty_weighting` (the weighter is always used). If someone sets `amp_dtype: fp16` or `uncertainty_weighting: false`, nothing changes — silent-ignore traps.
- All event-selection cuts except `require_truth` are off, i.e. training currently includes laterally/rear-leaking and multi-shower events.

### `run.sh` / `smoke_test.sh`
- Linear orchestration: env → inspect → preprocess → train → evaluate → probe. `run.sh` does `set -uo pipefail` first and `set -e` only after the env check; `smoke_test.sh` runs a 20 k-event preprocess and a 2-epoch training. Consistent with the code entry points.

---

## 3. State of the artifacts in this directory (important)

The `cache/` and `runs/base/` contents are **from the smoke test, not a real training run**:

- `runs/base/config.json` records `epochs: 2, batch_size: 256, compile: false, num_workers: 2` — exactly the `smoke_test.sh` overrides; `metrics.csv` has 2 epochs.
- `cache/` holds **6,921 events** (4,845 train / 1,038 val / 1,038 test), ~565 tokens/event (~3.9 M tokens total), built from the smoke test's `max_events=20000`. So of 20,000 raw events only ~35% survived `mcEne > 0` — worth understanding before scaling up (is `mcEne` unfilled for 2/3 of the tree, or is the branch semantics different from assumed?).
- Therefore `metrics.json` numbers (overall σ/E = 214%, bias = 61%, concept R² ≈ 0 for all 8 concepts) describe a model trained for **~40 optimizer steps on ~5 k events** and say nothing about the method. The per-bin trend after just 2 epochs (σ/E ≈ 5% at 3 TeV, ≈ 8% at 1.5 TeV, blowing up below 100 GeV; val concept loss 0.072 ≈ the predict-the-mean value 0.075) is exactly what an undertrained model looks like. The energy spectrum is strongly weighted to high energy (median ≈ exp(6.29) ≈ 540 GeV); the 0–100 GeV bins have only tens of test events.

---

## 4. Is it feasible?

**Yes — the core task and the architecture are sound and well within established practice.** Specifically:

1. **Energy regression from sparse calorimeter cells with a set/point-cloud Transformer** is a proven approach in HEP (particle-cloud transformers, image/graph calorimetry networks). A 3.2 M-parameter encoder over ≤1296 tokens with key-masked SDPA is computationally trivial for the stated hardware; at ~565 tokens/event and batch 512 the attention cost is modest.
2. **The targets are learnable in principle.** mcEne is MC truth, so the supervision is clean. The 8 concept parameters come from a 3D fit that is itself (approximately) a deterministic function of the same hit grid the network sees — so high concept R² is achievable, and the recon target (`kx_expehit`, the fit's smooth expectation per cell) is likewise a function of the inputs. There is no information-theoretic obstruction anywhere in the design.
3. **The interpretability scheme is coherent.** The soft bottleneck costs nothing in capacity (residual bypass), the supervision can only shape — not constrain — the energy path, and the probe suite is the right set of falsification tests for exactly that weakness (the ablation test directly measures whether energy actually routes through `h_phys`). One probe (linear R² comparison) is mildly biased in favor of the expected conclusion (pooling asymmetry, §2 probe.py) and should be fixed before being quoted.
4. **The physics constraints are handled honestly**: view-aware reflection augmentation with concept sign flips, no invalid 90° rotation, normalization from train only, bias-aware model selection that a constant predictor cannot win, zero-token-event handling that prevents NaN attention.
5. **The real risks are data, not method**:
   - **Statistics**: the current cache (6.9 k events) is 1–2 orders of magnitude too small for the ambition (multi-task, TeV-range, percent-level resolution). The full file must be processed (and the 65% loss to `mcEne > 0` understood) before any conclusion.
   - **Physics of the setup**: with all containment/single-shower cuts off, the energy head is also trained on leaking and multi-shower events; at 17 X₀ and multi-TeV energies the resolution floor will be set by shower-leakage fluctuations, not by the model. A NN can learn the *average* leakage correction from shape (that is a known strength of this approach), but the config's off-by-default cuts mean the first full run will mix regimes. The selection switches already exist in code, so this is a one-line config decision, not a refactor.
   - **One unverified assumption**: the `(18, 72)` layer-major reshape of the ROOT branches. If wrong, geometry features are silently scrambled. Verify once with a known event display before trusting any physics output.

**Verdict.** The codebase is coherent, carefully masked, free of show-stopper bugs, and the scientific design (multi-task energy + micro-reconstruction + soft concept bottleneck + falsification probes) is realistic. What exists in `runs/` so far is only a 2-epoch smoke test and must not be read as a result. Before the real run: fix the worker-RNG augmentation duplication, exclude the uncertainty log-variances from weight decay, decide the containment/single-shower cuts, verify the branch memory layout, and process the full dataset.

---

## 5. Issue list (consolidated)

| # | Severity | Where | Issue |
|---|----------|-------|-------|
| 1 | Note | `runs/`, `cache/` | All stored artifacts are from the 2-epoch / 20 k-event smoke test; metrics.json is not a physics result. |
| 2 | Moderate | `data/dataset.py` + `train.py` | Global `np.random` + forked DataLoader workers ⇒ identical augmentation RNG streams in all 6 workers. |
| 3 | Moderate | `train.py` | AdamW weight decay (1e-2) applied to Kendall–Gal log-variances, biasing task weights toward 1. |
| 4 | Moderate | `data/preprocess.py` | `(n,18,72)` reshape assumes layer-major branch layout; no programmatic verification. Only ~35% of raw events pass `mcEne > 0` — semantics unverified. |
| 5 | Minor | `train.py` | CSV `loss_*` columns always 0 (weighter called with zero losses just to log weights). |
| 6 | Minor | `probe.py` | Linear-probe comparison pools `h_phys` with the trained attention pool but `h_free` with a plain mean — tilted comparison. Mirror metric is RMS but printed as "mean". |
| 7 | Minor | `data/geometry.py` | `data_type: TB` raises `KeyError` (no TB entry in `OFFSET`). |
| 8 | Minor | `config/base.yaml` | Dead keys silently ignored: `max_tokens`, `optimizer`, `amp_dtype`, `loss.uncertainty_weighting`. |
| 9 | Minor | `data/dataset.py` | Mirror augmentation/probe is approximate: alignment offsets are not mirrored (~2 mm systematic on augmented events). `pad_mask` produced but never consumed. |
| 10 | Info | `train.py` | `train_total` includes the `+0.5·s` regularizer (not comparable to val losses; can be negative). Grad clip excludes weighter params. |
