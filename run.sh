#!/usr/bin/env bash
# End-to-end IHEP workflow for config/base.yaml, angle.yaml, or joint.yaml.
set -euo pipefail

source /aifs/user/home/lishanglin/HREDML/loadCondaEnvCuda13.sh
cd /aifs/user/data/lishanglin/chenhao/ecalTransformer
CFG=${1:-config/base.yaml}

readarray -t SETTINGS < <(python - "$CFG" <<'PY'
import sys
from utils.config import load_config
from utils.tasks import task_mode
cfg, _ = load_config(sys.argv[1])
print(task_mode(cfg))
print(cfg.paths.cache_dir)
print(cfg.paths.out_dir)
PY
)
MODE=${SETTINGS[0]}
CACHE=${SETTINGS[1]}
OUTPUT=${SETTINGS[2]}

python -m unittest discover -s tests -v
python -m data.geometry

cache_complete=true
for split in train val test; do
  if [[ ! -f "$CACHE/$split.npz" && ! -d "$CACHE/$split" ]]; then
    cache_complete=false
  fi
done
if [[ ! -f "$CACHE/meta.json" ]]; then
  cache_complete=false
fi
if [[ "$cache_complete" != true ]]; then
  python -m data.inspect_root --config "$CFG"
  python -m data.preprocess --config "$CFG"
fi

python train.py --config "$CFG"
case "$MODE" in
  energy)
    python evaluate.py --config "$CFG"
    python probe.py --config "$CFG"
    ;;
  angle)
    python evaluate_angle.py --config "$CFG"
    python evaluate_centroid.py --config "$CFG" --weight-power 1.5
    python probe_angle.py --config "$CFG"
    ;;
  joint)
    mkdir -p "$OUTPUT/eval_best_angle" "$OUTPUT/eval_best_energy"
    python evaluate_angle.py --config "$CFG" --ckpt "$OUTPUT/best_angle.pt" \
      --set "paths.out_dir=$OUTPUT/eval_best_angle"
    python evaluate_angle.py --config "$CFG" --ckpt "$OUTPUT/best_energy.pt" \
      --set "paths.out_dir=$OUTPUT/eval_best_energy"
    ;;
  *)
    echo "unsupported training mode: $MODE" >&2
    exit 2
    ;;
esac

if [[ "$MODE" != energy ]]; then
  # This deployment export intentionally does not request MC truth or 3D fit.
  python export_predictions.py --config "$CFG"
fi
echo "ECAL_TASK_OK mode=$MODE output=$OUTPUT"
