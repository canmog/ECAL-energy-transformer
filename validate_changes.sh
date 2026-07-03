#!/usr/bin/env bash
# Validation battery for the ANALYSIS.md-driven changes. Run on the GPU node.
source /aifs/user/home/lishanglin/HREDML/loadCondaEnvCuda13.sh >/dev/null 2>&1
set -uo pipefail
cd /aifs/user/data/lishanglin/chenhao/transformer
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
C=/aifs/user/data/lishanglin/chenhao/transformer/cache_sel

echo "===== 1. geometry TB switch (was KeyError) ====="
python - <<'EOF'
import sys; sys.path.insert(0, ".")
from data.geometry import build_geometry_table
for dt in ("MC", "ISS", "TB"):
    t = build_geometry_table(dt)
    print(f"  {dt}: ok, t range [{t['t_cm'].min():.2f}, {t['t_cm'].max():.2f}] cm")
EOF

echo "===== 2. train 1ep FixedWeighter + fp32 + epochs==warmup (was ZeroDivision) ====="
python train.py --config config/base.yaml --set paths.cache_dir=$C paths.out_dir=runs/_v2 \
  train.epochs=1 train.warmup_epochs=1 train.compile=false \
  loss.uncertainty_weighting=false train.amp_dtype=fp32 2>&1 | tail -2

echo "===== 3. probe with fair mean-pooling on s7 winner ====="
python probe.py --config config/base.yaml --set paths.cache_dir=$C \
  paths.out_dir=runs/s7_d05long model.dropout=0.05 2>&1 | tail -22

rm -rf runs/_v2
echo "VALIDATION_DONE"
