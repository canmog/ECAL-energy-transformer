#!/usr/bin/env bash
# Full local-to-IHEP workflow after datasets/full has been angle-augmented.
source /aifs/user/home/lishanglin/HREDML/loadCondaEnvCuda13.sh
set -euo pipefail
cd /aifs/user/data/lishanglin/chenhao/angleTransformer
CFG=${1:-config/base.yaml}

python -m unittest discover -s tests -v
python -m data.geometry
python -m data.inspect_root --config "$CFG"
python -m data.preprocess --config "$CFG"
python train.py --config "$CFG"
python evaluate.py --config "$CFG"
python probe.py --config "$CFG"
python export_predictions.py --config "$CFG"

