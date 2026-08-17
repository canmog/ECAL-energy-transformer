# Integrated angle reconstruction

## Promoted result

The selected direction model is the full-data d192 normalized-centroid-residual
run with weak physics auxiliaries (historical job 558491, seed 73111). On the
507,602-event held-out selected test split it achieved:

| metric | Transformer | same-event 3D fit |
|---|---:|---:|
| median opening angle | 0.1782 deg | 0.2708 deg |
| p68 | 0.2719 deg | 0.3885 deg |
| p90 | 0.5707 deg | 0.7803 deg |
| p95 | 0.7666 deg | 1.0848 deg |
| fraction above 1.1459 deg | 1.419% | 4.431% |

The p68 improvement is 30.0%. The model wins in every stored energy and
incidence bin. These rounded values are navigation aids; the authoritative
resolved configuration and metric JSON are under
[`provenance/angle/results`](../provenance/angle/results).

The seed-73111 joint model's best-angle checkpoint reached 0.2770 deg angle p68
and 1.0965% Gaussian-core energy resolution below 2 TeV. Exact joint replicas
reached 0.2791 deg (seed 123) and 0.2776 deg (seed 777). The repeated result is
therefore stable at the few-millidegree level.

## Direction and detector convention

The network represents direction with slopes:

```text
kx = dx/dz = tan(theta) cos(phi)
ky = dy/dz = tan(theta) sin(phi)
incoming unit vector = (-kx, -ky, -1) / sqrt(1 + kx^2 + ky^2)
```

The corrected ECAL projection mapping is:

```text
stored view 1 -> X projection -> kx
stored view 0 -> Y projection -> ky
component_views = [1, 0]
```

This convention is shared by geometry, reflections, centroid fitting, the
view-aware head, probes, tests, and configs. The former `[0,1]` routing was a bug
and is not retained as a compatibility option for new training.

## Task and cache contracts

`task.mode` chooses `energy`, `angle`, or `joint`. `task.auxiliary` then lists
additional trained outputs. The supplied profiles resolve to:

| profile | active losses | required cache targets |
|---|---|---|
| energy | energy, reconstruction, concepts | `energy`, `tok_expe`, `concepts` |
| angle | angle, energy, reconstruction, concepts | `angle`, `energy`, `tok_expe`, `concepts` |
| joint | energy, angle, reconstruction, concepts | `energy`, `angle`, `tok_expe`, `concepts` |

All modes additionally require `off`, `tok_layer`, `tok_cell`, and `tok_ehit` as
model inputs. Optional loss features declare their own dependencies: enabling
angle-energy weighting requires `energy`, and enabling fit-quality weighting for
reconstruction/concepts requires `tok_expe`.

Before training, the code validates the task/head/loss configuration, then checks
both train and validation splits and all required normalization metadata. An
absent dependency stops immediately with the operation, cache path, split,
missing fields, and available fields. `collate()` includes only fields actually
loaded; it never manufactures targets.

Legacy `<split>.npz` and memory-mapped `<split>/<field>.npy` layouts expose the
same logical dataset interface. Schema v2 metadata records the storage layout,
available fields, geometry, and corrected component mapping. Historical energy
caches continue to work unchanged.

## MC evaluation versus ISS prediction

These are intentionally different contracts:

- `evaluate_angle.py` is an MC benchmark. It requires MC angle truth, MC energy
  for resolution-versus-energy, stable `run,event`, and—under the production
  configs—the same-event `fit_angle` reference. Active auxiliary targets are
  required only when their metrics are reported.
- `export_predictions.py` is deployment inference. By default it requires only
  hit-token inputs and stable `run,event`; normalization comes from the training
  checkpoint. It can therefore run on ISS caches with no MC truth, expected-hit
  target, concepts, or 3D-fit direction.
- MC columns are opt-in during export with `--include-angle-truth`,
  `--include-energy-truth`, and `--include-fit-angle`. Requesting one makes that
  field strictly required rather than filling it with zeros.

Examples:

```bash
# Train and score the promoted angle objective.
python train.py --config config/angle.yaml
python evaluate_angle.py --config config/angle.yaml

# Target-free ISS-style export.
python export_predictions.py --config config/angle.yaml \
  --set paths.cache_dir=/path/to/iss_token_cache

# MC export with explicit references.
python export_predictions.py --config config/angle.yaml \
  --include-angle-truth --include-energy-truth --include-fit-angle

# Joint training saves best.pt, best_angle.pt, best_energy.pt, and the
# energy core/tail selection checkpoints.
python train.py --config config/joint.yaml
```

## Data preparation and provenance

`data.inspect_root` now checks every configured ROOT file and fails before cache
creation if any tree or branch is absent. `data.preprocess` writes packed NPZ;
`data.preprocess_memmap` provides the two-pass full/uncut NPY layout. The latter
is deliberately restricted to angle/joint profiles because it stores direction
truth, the 3D-fit baseline, event identity, and quality grouping fields.

The ROOT augmentation and independent verifier are in [`root`](../root). They
match MC and reconstruction entries by `(run,event)`, preserve and audit the
original tree, write through a temporary file, and verify all added direction
branches before replacement.

The full pre-integration source is preserved on branch
`angle-reconstruction-snapshot`. Curated resolved configs, selected and uncut
cache metadata, direction metrics, and all three joint-seed metrics are also
tracked in this branch under [`provenance/angle`](../provenance/angle).
