#!/bin/bash
# Smoke for the stabilised weighter + 10 MeV cut + bucketing/compile. Detached on node.
cd /aifs/user/data/lishanglin/chenhao/transformer || exit 1
source /aifs/user/home/lishanglin/HREDML/loadCondaEnvCuda13.sh >/dev/null 2>&1
export PYTORCH_ALLOC_CONF=expandable_segments:True
set -o pipefail
CFG=config/base.yaml
ME=200000

echo "##### [$(date +%T)] PREPROCESS cache_sm11 (10 MeV, single-shower, 11 concepts) #####"
python -m data.preprocess --config $CFG --set paths.cache_dir=cache_sm11 data.max_events=$ME 2>&1 \
  | grep -E "input file|kept|tokens|dropped|wrote meta"

echo "##### [$(date +%T)] SLICE -> cache_sm08 (8 concepts) #####"
python - <<'PY'
import json, os, numpy as np
SRC, DST = "cache_sm11", "cache_sm08"; os.makedirs(DST, exist_ok=True)
for sp in ("train", "val", "test"):
    z = dict(np.load(f"{SRC}/{sp}.npz")); z["concepts"] = z["concepts"][:, :8]
    np.savez(f"{DST}/{sp}.npz", **z)
m = json.load(open(f"{SRC}/meta.json")); m["n_concepts"] = 8
for k in ("concept_names","concept_reflect_x","concept_reflect_y","concept_coord","concept_mean","concept_std"):
    m[k] = m[k][:8]
json.dump(m, open(f"{DST}/meta.json", "w"), indent=2); print("sliced 11 -> 8:", DST)
PY

runit() {            # name cache <extra --set args>
  local name=$1 cache=$2; shift 2
  echo "##### [$(date +%T)] RUN $name  cache=$cache  $* #####"
  # -u: unbuffered so each epoch line flushes; awk stamps unix time -> epoch deltas
  python -u train.py --config $CFG --set paths.cache_dir=$cache paths.out_dir=runs/$name \
     train.epochs=5 train.warmup_epochs=1 model.dropout=0.05 train.num_workers=4 "$@" 2>&1 \
     | awk '{print systime()"\t"$0; fflush()}' \
     | grep -E "\] train=|InductorError|Traceback|recompil"
  echo "  -> w_* + val from metrics.csv:"
  tail -n+2 runs/$name/metrics.csv 2>/dev/null \
     | awk -F, '{printf "    ep%d w_energy=%.3f w_concept=%.3f w_recon=%.3f val_res=%.1f%%\n",$1,$11,$10,$12,$5*100}'
}

runit sm_eager      cache_sm11 train.compile=false
runit sm_cmp_bucket cache_sm11 train.compile=true train.token_bucket=128
runit sm_cmp_nobkt  cache_sm11 train.compile=true
runit sm_cmp08_bkt  cache_sm08 train.compile=true train.token_bucket=128
echo "##### [$(date +%T)] SMOKE2_DONE #####"
