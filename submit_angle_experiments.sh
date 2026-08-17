#!/usr/bin/env bash
# Submit three complementary searches around the production d192 baseline.
set -euo pipefail
cd "$(dirname "$0")"

submit_variant() {
  local job_name=$1
  local variant=$2
  local qos=$3
  shift 3
  sbatch --qos="$qos" --job-name="$job_name" job_angle_variant.sub "$variant" "$@"
}

# Strongest low-risk lever from the energy study: longer d192 training.  Here the
# loss balance is deliberately angle-first and auxiliary tasks remain regularizers.
submit_variant ang_focus angle_focus_d192 gpunormal \
  seed=31415 \
  loss.weighting=normalized \
  loss.weights.angle=1.0 \
  loss.weights.energy=0.15 \
  loss.weights.recon=0.10 \
  loss.weights.concept=0.15 \
  loss.angle.mode=huber \
  loss.angle.delta=0.03 \
  'heads.angle.hidden=[256,128]' \
  model.dropout=0.03 \
  train.epochs=100 \
  train.lr=2.0e-4 \
  train.warmup_epochs=6 \
  train.early_stop_patience=25

# Detector-inductive-bias experiment: independent X/Y view pooling and a direct,
# robust physical direction objective instead of component-space Huber.
submit_variant ang_view angle_view_d192 gpuintera \
  seed=27182 \
  heads.angle.type=view \
  'heads.angle.hidden=[256,128]' \
  loss.weighting=normalized \
  loss.weights.angle=1.0 \
  loss.weights.energy=0.10 \
  loss.weights.recon=0.10 \
  loss.weights.concept=0.10 \
  loss.angle.mode=chord \
  loss.angle.epsilon=0.001 \
  model.dropout=0.03 \
  train.epochs=80 \
  train.lr=2.0e-4 \
  train.warmup_epochs=6 \
  train.early_stop_patience=22

# Capacity experiment: larger/deeper shared encoder plus the view-aware head and
# a hybrid component/direction loss.  Smaller token batches bound memory usage.
submit_variant ang_wide angle_wide_d256 gpunormal \
  seed=16180 \
  model.d_model=256 \
  model.n_blocks=8 \
  model.n_heads=8 \
  model.tap_block=4 \
  model.d_phys=96 \
  model.d_free=160 \
  heads.angle.type=view \
  'heads.angle.hidden=[256,128]' \
  'heads.energy.hidden=[192]' \
  'heads.recon.hidden=[192]' \
  loss.weighting=normalized \
  loss.weights.angle=1.0 \
  loss.weights.energy=0.15 \
  loss.weights.recon=0.10 \
  loss.weights.concept=0.15 \
  loss.angle.mode=hybrid \
  loss.angle.delta=0.05 \
  loss.angle.epsilon=0.001 \
  loss.angle.chord_weight=0.2 \
  train.epochs=60 \
  train.lr=1.5e-4 \
  train.warmup_epochs=6 \
  train.early_stop_patience=20 \
  train.batch_size=384 \
  train.max_tokens_per_batch=49152
