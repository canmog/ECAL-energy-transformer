#!/bin/bash
# A/B: strict single-shower 50-file data, 11 concepts vs 8 concepts (controlled).
# Run detached on the GPU node; monitor via runs/ab.log + per-run metrics.csv.
cd /aifs/user/data/lishanglin/chenhao/transformer || exit 1
source /aifs/user/home/lishanglin/HREDML/loadCondaEnvCuda13.sh >/dev/null 2>&1
export PYTORCH_ALLOC_CONF=expandable_segments:True
set -o pipefail   # NOT set -u: the conda env script trips an unbound-var exit under -u
CFG=config/base.yaml
ME=2000000          # events READ; ~19% pass contained+single-shower -> ~385k kept

echo "===== [$(date '+%H:%M:%S')] PREPROCESS cache_s11 (11 concepts, strict) ====="
python -m data.preprocess --config $CFG --set paths.cache_dir=cache_s11 data.max_events=$ME || { echo "PREPROCESS_FAIL"; exit 2; }
echo "===== [$(date '+%H:%M:%S')] SLICE -> cache_s08 (8 concepts) ====="
python _make8.py || { echo "SLICE_FAIL"; exit 3; }

echo "===== [$(date '+%H:%M:%S')] TRAIN arm11 (11 concepts) ====="
python train.py    --config $CFG --set paths.cache_dir=cache_s11 paths.out_dir=runs/strict11 model.dropout=0.05 train.compile=false || { echo "ARM11_TRAIN_FAIL"; exit 4; }
python evaluate.py --config $CFG --set paths.cache_dir=cache_s11 paths.out_dir=runs/strict11 || true
python probe.py    --config $CFG --set paths.cache_dir=cache_s11 paths.out_dir=runs/strict11 || true
echo "===== [$(date '+%H:%M:%S')] ARM11_DONE ====="

echo "===== [$(date '+%H:%M:%S')] TRAIN arm08 (8 concepts) ====="
python train.py    --config $CFG --set paths.cache_dir=cache_s08 paths.out_dir=runs/strict08 model.dropout=0.05 train.compile=false || { echo "ARM08_TRAIN_FAIL"; exit 5; }
python evaluate.py --config $CFG --set paths.cache_dir=cache_s08 paths.out_dir=runs/strict08 || true
echo "===== [$(date '+%H:%M:%S')] ARM08_DONE ====="
echo "===== AB_DONE $(date '+%H:%M:%S') ====="
