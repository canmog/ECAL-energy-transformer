#!/usr/bin/env bash
# Four full-data, 100-epoch searches around the winning fixed-10-MeV normalized
# centroid residual.  No test-set information is used for checkpoint selection.
set -euo pipefail
cd "$(dirname "$0")"

COMMON=(
  seed=73111
  data.runtime_threshold_mev=10.0
  model.d_model=192
  model.n_blocks=6
  model.n_heads=8
  model.tap_block=3
  model.d_phys=64
  model.d_free=128
  model.dropout=0.03
  heads.angle.type=view_residual
  'heads.angle.component_views=[1,0]'
  heads.angle.centroid_weight_power=1.5
  heads.angle.residual_normalization=true
  heads.angle.residual_norm_max_events=300000
  heads.angle.residual_scale=1.0
  'heads.angle.hidden=[256,128]'
  loss.weighting=normalized
  loss.weights.angle=1.0
  loss.weights.energy=0.0
  loss.weights.recon=0.0
  loss.weights.concept=0.0
  loss.angle.mode=hybrid
  loss.angle.delta=0.1
  loss.angle.epsilon=0.001
  loss.angle.chord_weight=1.0
  train.epochs=100
  train.lr=2.0e-4
  train.warmup_epochs=5
  # Train for at most 100 epochs.  Validation early stopping is disabled only
  # through epoch 40, then uses the standard 25-epoch patience window.
  train.early_stop_min_epochs=40
  train.early_stop_patience=25
  train.batch_size=512
  train.max_tokens_per_batch=65536
  train.augment.reflect_x=false
  train.augment.reflect_y=false
)

submit_variant() {
  local job_name=$1
  local variant=$2
  shift 2
  sbatch --job-name="$job_name" job_angle_next4.sub \
    "$variant" "${COMMON[@]}" "$@"
}

# J1: scaled reference -- isolate the gain from full data/capacity/training.
submit_variant an4_base angle_residual_full_d192_t10 \
  seed=73111

# J2: keep the p68 region quadratic and retain a smaller physical tail penalty.
submit_variant an4_core angle_residual_coreloss_d192_t10 \
  seed=73111 loss.angle.delta=0.5 loss.angle.chord_weight=0.25

# J3: emphasize the remaining sub-300-GeV deficit using MC energy in training only.
submit_variant an4_lowE angle_residual_lowenergy_d192_t10 \
  seed=73111 loss.angle_energy_weight.enabled=true \
  loss.angle_energy_weight.pivot_gev=300.0 \
  loss.angle_energy_weight.power=0.5 \
  loss.angle_energy_weight.w_min=0.5 \
  loss.angle_energy_weight.w_max=3.0 \
  loss.angle_energy_weight.normalize_batch=true

# J4: weak 3D-fit/MC auxiliary supervision; none of these targets is an input.
submit_variant an4_phys angle_residual_physicsaux_d192_t10 \
  seed=73111 loss.weights.energy=0.05 loss.weights.recon=0.05 \
  loss.weights.concept=0.10
