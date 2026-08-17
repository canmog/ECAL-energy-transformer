#!/usr/bin/env bash
# End-to-end runnability test on lsl/IHEP. Uses private throwaway outputs.
source /aifs/user/home/lishanglin/HREDML/loadCondaEnvCuda13.sh
set -uo pipefail
cd /aifs/user/data/lishanglin/chenhao/angleTransformer || exit 1

C=cache_smoke
O=runs/smoke
ROOT_INPUT=${1:-/aifs/user/data/lishanglin/datasets/full/eBep0_254000.list_0000.emini08-full.root}
SETS=("paths.cache_dir=$C" "paths.out_dir=$O" "paths.root_file=$ROOT_INPUT" \
      "train.num_workers=2")

python -c "import torch; print('torch',torch.__version__,'cuda',torch.cuda.is_available(),torch.cuda.get_device_name(0))" || exit 11
python -m unittest discover -s tests -v || exit 12
python -m data.geometry || exit 13
python -m data.inspect_root --config config/base.yaml --set "${SETS[@]}" || exit 14
python -m data.preprocess --config config/base.yaml --set "${SETS[@]}" data.max_events=20000 || exit 15
python train.py --config config/base.yaml --set "${SETS[@]}" train.epochs=2 train.warmup_epochs=1 \
  train.compile=false train.batch_size=256 || exit 16
python evaluate.py --config config/base.yaml --set "${SETS[@]}" || exit 17
python probe.py --config config/base.yaml --set "${SETS[@]}" || exit 18
python export_predictions.py --config config/base.yaml --set "${SETS[@]}" \
  --output "$O/predictions_test.npz" || exit 19
echo "ANGLE_SMOKE_OK"
