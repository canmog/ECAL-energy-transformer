#!/usr/bin/env bash
# End-to-end on the IHEP GPU node (RTX 5090 / CUDA 13).
# Bring up the node first (see ../aiGPU.md), then from this dir:  bash run.sh
set -uo pipefail

# 1. CUDA-13 conda env (torch 2.9 + cu130, sm_120; uproot/awkward/numpy/sklearn).
source /aifs/user/home/lishanglin/HREDML/loadCondaEnvCuda13.sh
python -c "import torch; print('torch', torch.__version__, 'cuda', torch.cuda.is_available())" || exit 1
set -e

CFG=${1:-config/base.yaml}

# 2. Sanity-check the ROOT branches / units (run once).
python -m data.inspect_root --config "$CFG"

# 3. ROOT -> tokenised cache + concept/energy targets + meta.json.
python -m data.preprocess --config "$CFG"

# 4. Train (energy + recon + concept, uncertainty-weighted; bias-aware early stop).
python train.py --config "$CFG"

# 5. Evaluate (sigma/E & bias vs E, concept R^2, plots).
python evaluate.py --config "$CFG"

# 6. Physics-probe battery (linear probe / ablation / mirror).
python probe.py --config "$CFG"
