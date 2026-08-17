#!/usr/bin/env bash
# Two repeated-seed replicas of the completed joint angle+energy model.
# The only changes from job 823430 are seed, output directory, and job name.
set -euo pipefail
cd "$(dirname "$0")"

COMMON=(
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
  heads.angle.learned_robust_fit=false
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
  train.joint_energy_angle=true
  train.energy_select_e_cut_gev=2000.0
  loss.weighting=uncertainty
  loss.uncertainty_weighting=true
  loss.logvar_weight_decay=0.01
)

submit_seed() {
  local seed=$1
  local job_name=$2
  local variant=$3
  sbatch --job-name="$job_name" job_angle_joint2.sub \
    joint "$variant" "${COMMON[@]}" "seed=$seed"
}

submit_seed 123 an5_js123 angle_energy_multitask_d192_t10_s123
submit_seed 777 an5_js777 angle_energy_multitask_d192_t10_s777
