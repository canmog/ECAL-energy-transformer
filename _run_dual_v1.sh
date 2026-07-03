#!/bin/bash
set -e
source /aifs/user/home/lishanglin/HREDML/loadCondaEnvCuda13.sh
cd /aifs/user/data/lishanglin/chenhao/transformer
C=/aifs/user/data/lishanglin/chenhao/transformer/cache_full10
O=/aifs/user/data/lishanglin/chenhao/transformer/runs/dual_v1
echo "START $(date) on $(hostname)"
python train.py    --config config/base.yaml --set paths.cache_dir=$C paths.out_dir=$O train.epochs=50
echo "TRAIN_DONE $(date)"
python evaluate.py --config config/base.yaml --set paths.cache_dir=$C paths.out_dir=$O
python probe.py    --config config/base.yaml --set paths.cache_dir=$C paths.out_dir=$O
echo "DUAL_V1_DONE $(date)"
