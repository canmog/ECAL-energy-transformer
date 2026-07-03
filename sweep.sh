#!/usr/bin/env bash
# Hyperparameter sweep round 1 on the selected (contained, single-shower) FULL sample.
# Run detached on the GPU node:  setsid nohup bash sweep.sh > runs/sweep.log 2>&1 &
# Each variant trains + evaluates into its own runs/<name>; a DONE marker makes the
# script resumable if the allocation dies mid-sweep.
source /aifs/user/home/lishanglin/HREDML/loadCondaEnvCuda13.sh
set -uo pipefail
cd /aifs/user/data/lishanglin/chenhao/transformer
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
CACHE=/aifs/user/data/lishanglin/chenhao/transformer/cache_sel
RUNS=/aifs/user/data/lishanglin/chenhao/transformer/runs
COMPILE=${COMPILE:-false}

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

run_one s1_base
run_one s2_d256    model.d_model=256 model.d_free=192
run_one s3_deep    model.n_blocks=8 model.tap_block=4
run_one s4_lr1e3   train.lr=1.0e-3
run_one s5_drop05  model.dropout=0.05
run_one s6_tok128k train.max_tokens_per_batch=131072 train.lr=6.0e-4

echo "SWEEP_ROUND1_COMPLETE $(date '+%H:%M:%S')"
python summarize_runs.py
