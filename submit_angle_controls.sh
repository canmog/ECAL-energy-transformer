#!/usr/bin/env bash
# Four cheap controls: direct vs normalized centroid residual, each without and
# with the corrected physical X/Y reflection augmentation.
set -euo pipefail
cd "$(dirname "$0")"

COMMON=(
  data.runtime_threshold_mev=10.0
  data.train_max_events=400000
  data.val_max_events=100000
  model.d_model=96
  model.n_blocks=4
  model.n_heads=4
  model.tap_block=2
  model.d_phys=32
  model.d_free=64
  model.dropout=0.03
  'heads.angle.component_views=[1,0]'
  'heads.angle.hidden=[128,64]'
  loss.weighting=normalized
  loss.weights.angle=1.0
  loss.weights.energy=0.0
  loss.weights.recon=0.0
  loss.weights.concept=0.0
  loss.angle.mode=hybrid
  loss.angle.delta=0.1
  loss.angle.epsilon=0.001
  loss.angle.chord_weight=1.0
  train.epochs=40
  train.lr=2.0e-4
  train.warmup_epochs=3
  train.early_stop_min_epochs=20
  train.early_stop_patience=12
  train.batch_size=512
  train.max_tokens_per_batch=65536
)

submit_control() {
  local job_name=$1
  local variant=$2
  shift 2
  sbatch --job-name="$job_name" job_angle_controls.sub \
    "$variant" "${COMMON[@]}" "$@"
}

# C1: direct view-aware slopes, no physical reflections.
submit_control act10_d0 angle_ctrl_direct_noaugment_t10 \
  seed=61001 heads.angle.type=view \
  train.augment.reflect_x=false train.augment.reflect_y=false

# C2: same direct model with corrected view-1/X and view-0/Y reflections.
submit_control act10_d1 angle_ctrl_direct_augment_t10 \
  seed=61001 heads.angle.type=view \
  train.augment.reflect_x=true train.augment.reflect_y=true

# C3: centroid plus MC-minus-centroid normalized residual, no reflections.
submit_control act10_r0 angle_ctrl_residual_noaugment_t10 \
  seed=62001 heads.angle.type=view_residual \
  heads.angle.centroid_weight_power=1.5 \
  heads.angle.residual_normalization=true \
  heads.angle.residual_norm_max_events=200000 \
  heads.angle.residual_scale=1.0 \
  train.augment.reflect_x=false train.augment.reflect_y=false

# C4: identical normalized residual with the corrected physical reflections.
submit_control act10_r1 angle_ctrl_residual_augment_t10 \
  seed=62001 heads.angle.type=view_residual \
  heads.angle.centroid_weight_power=1.5 \
  heads.angle.residual_normalization=true \
  heads.angle.residual_norm_max_events=200000 \
  heads.angle.residual_scale=1.0 \
  train.augment.reflect_x=true train.augment.reflect_y=true
