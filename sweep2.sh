#!/usr/bin/env bash
# Sweep round 2 — refine round-1 winner (base size + dropout 0.05, still improving
# at epoch 120). Axes: longer schedule, dropout 0.05 -> 0.0, capacity DOWN.
source /aifs/user/home/lishanglin/HREDML/loadCondaEnvCuda13.sh
set -uo pipefail
cd /aifs/user/data/lishanglin/chenhao/transformer
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
CACHE=/aifs/user/data/lishanglin/chenhao/transformer/cache_sel
RUNS=/aifs/user/data/lishanglin/chenhao/transformer/runs
COMPILE=${COMPILE:-false}
LONG="train.epochs=240 train.early_stop_patience=30"

run_one() {
  local name=$1; shift
  if [ -f "$RUNS/$name/DONE" ]; then echo "== skip $name (DONE)"; return 0; fi
  echo "===== $(date '+%H:%M:%S') RUN $name : $* ====="
  python train.py --config config/base.yaml \
      --set paths.cache_dir=$CACHE paths.out_dir=$RUNS/$name train.compile=$COMPILE "$@" \
    && python evaluate.py --config config/base.yaml \
      --set paths.cache_dir=$CACHE paths.out_dir=$RUNS/$name "$@" \
    && touch "$RUNS/$name/DONE" \
    || echo "!!!!! RUN $name FAILED"
}

run_one s7_d05long  model.dropout=0.05 $LONG
run_one s8_d00long  model.dropout=0.0  $LONG
run_one s9_smalld05 model.d_model=128 model.d_free=64 model.dropout=0.05 $LONG

echo "SWEEP_ROUND2_COMPLETE $(date '+%H:%M:%S')"
python summarize_runs.py
