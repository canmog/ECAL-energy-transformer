#!/usr/bin/env bash
# Two follow-ups selected after the first direction results.
set -euo pipefail
cd "$(dirname "$0")"

# Highest-priority run: analytic X/Y centroid-axis estimate plus a zero-initialized
# view Transformer correction.  It uses the free gpunormal slot alongside d256.
sbatch --qos=gpunormal --job-name=ang_resid job_angle_variant.sub \
  angle_centroid_residual_d192 \
  seed=42424 \
  heads.angle.type=view_residual \
  'heads.angle.component_views=[1,0]' \
  heads.angle.centroid_weight_power=1.5 \
  heads.angle.residual_scale=1.0 \
  'heads.angle.hidden=[256,128]' \
  loss.weighting=normalized \
  loss.weights.angle=1.0 \
  loss.weights.energy=0.0 \
  loss.weights.recon=0.0 \
  loss.weights.concept=0.0 \
  loss.angle.mode=chord \
  loss.angle.epsilon=0.001 \
  model.dropout=0.03 \
  train.epochs=100 \
  train.lr=2.0e-4 \
  train.warmup_epochs=3 \
  train.early_stop_min_epochs=40 \
  train.early_stop_patience=35

# Controlled isolation of the successful view+chord ingredients: identical raw
# direction head, but no energy/reconstruction/3D-fit concept gradients.  It uses
# gpunormal and queues automatically when both per-user GPU slots are occupied.
sbatch --qos=gpunormal --job-name=ang_noaux job_angle_variant.sub \
  angle_view_chord_noaux_d192 \
  seed=51515 \
  heads.angle.type=view \
  'heads.angle.component_views=[1,0]' \
  'heads.angle.hidden=[256,128]' \
  loss.weighting=normalized \
  loss.weights.angle=1.0 \
  loss.weights.energy=0.0 \
  loss.weights.recon=0.0 \
  loss.weights.concept=0.0 \
  loss.angle.mode=chord \
  loss.angle.epsilon=0.001 \
  model.dropout=0.03 \
  train.epochs=120 \
  train.lr=2.0e-4 \
  train.warmup_epochs=4 \
  train.early_stop_min_epochs=40 \
  train.early_stop_patience=40
