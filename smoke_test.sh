#!/usr/bin/env bash
# Quick runnability smoke test (NOT for results): tiny preprocess + 2 short epochs.
source /aifs/user/home/lishanglin/HREDML/loadCondaEnvCuda13.sh
set -uo pipefail
cd /aifs/user/data/lishanglin/chenhao/transformer
step(){ echo; echo "########## $1 ##########"; }

step "torch + CUDA"
python -c "import torch;print('torch',torch.__version__,'cuda_ok',torch.cuda.is_available(),torch.cuda.get_device_name(0))" || exit 11

step "geometry self-test"
python -m data.geometry || exit 12

step "inspect_root"
python -m data.inspect_root --config config/base.yaml || exit 13

step "preprocess (20k events)"
python -m data.preprocess --config config/base.yaml --set data.max_events=20000 || exit 14

step "train (2 epochs, compile off, small)"
python train.py --config config/base.yaml \
  --set train.epochs=2 train.warmup_epochs=1 train.compile=false train.num_workers=2 train.batch_size=256 || exit 15

step "evaluate"
python evaluate.py --config config/base.yaml || exit 16

step "probe"
python probe.py --config config/base.yaml || exit 17

echo; echo "########## ALL SMOKE STEPS PASSED ##########"
