#!/usr/bin/env bash
# Two approved full-data controls: an isolated learned robust layer fit on J1,
# and one shared Transformer trained on angle+energy+recon+all concepts.
set -euo pipefail
cd "$(dirname "$0")"

COMMON=(
  seed=73111
  data.runtime_threshold_mev=10.0
  data.concept_use=null
  model.d_model=192
  model.n_blocks=6
  model.n_heads=8
  model.tap_block=3
  model.d_phys=64
  model.d_free=128
  model.dropout=0.03
  model.bottleneck_residual=true
  model.dual_energy_head=false
  heads.energy.enabled=true
  heads.recon.enabled=true
  heads.angle.enabled=true
  heads.angle.type=view_residual
  'heads.angle.component_views=[1,0]'
  heads.angle.centroid_weight_power=1.5
  heads.angle.residual_normalization=true
  heads.angle.residual_norm_max_events=300000
  heads.angle.residual_scale=1.0
  'heads.angle.hidden=[256,128]'
  loss.angle.mode=hybrid
  loss.angle.delta=0.1
  loss.angle.epsilon=0.001
  loss.angle.chord_weight=1.0
  loss.fit_quality_weight.enabled=true
  'loss.fit_quality_weight.apply_to=[concept,recon]'
  loss.fit_quality_weight.beta=1.0
  loss.fit_quality_weight.w_min=0.2
  train.epochs=100
  train.lr=2.0e-4
  train.warmup_epochs=5
  train.early_stop_min_epochs=40
  train.early_stop_patience=25
  train.batch_size=512
  train.max_tokens_per_batch=65536
  train.wd_exclude_1d=true
  train.augment.reflect_x=false
  train.augment.reflect_y=false
)

submit_variant() {
  local job_name=$1
  local mode=$2
  local variant=$3
  shift 3
  sbatch --job-name="$job_name" job_angle_joint2.sub \
    "$mode" "$variant" "${COMMON[@]}" "$@"
}

# Task 1: J1 unchanged except for the zero-initialized learned two-pass fit.
submit_variant an5_robust robust angle_residual_robustfit_d192_t10 \
  heads.angle.learned_robust_fit=true \
  heads.angle.robust_fit_hidden=16 \
  heads.angle.robust_fit_max_multiplier=4.0 \
  loss.weighting=normalized \
  loss.weights.angle=1.0 \
  loss.weights.energy=0.0 \
  loss.weights.recon=0.0 \
  loss.weights.concept=0.0

# Task 2: the established energy scaffolding plus the J1 direction objective.
# EMA-normalized Kendall--Gal balances all four active tasks.
submit_variant an5_joint joint angle_energy_multitask_d192_t10 \
  heads.angle.learned_robust_fit=false \
  train.joint_energy_angle=true \
  train.energy_select_e_cut_gev=2000.0 \
  loss.weighting=uncertainty \
  loss.uncertainty_weighting=true \
  loss.logvar_weight_decay=0.01
