#!/usr/bin/env bash
# Safely augment every datasets/full/*-full.root file with MC direction truth.
# Run on lsl after Mutagen has synchronized ecalTransformer/root/.
set -uo pipefail

BASE=/aifs/user/data/lishanglin/datasets
CODE=/aifs/user/data/lishanglin/chenhao/ecalTransformer/root
FULL_DIR="$BASE/full"
MC_DIR="$BASE/mcinfo"
FIT_DIR="$BASE/3dfit"
LOG=${1:-/aifs/user/data/lishanglin/chenhao/ecalTransformer/update_full_angles.log}
exec > >(tee -a "$LOG") 2>&1

ok=0
skip=0
fail=0
verify_fail=0

run_one() {
  local full=$1
  local base mc fit out
  base=$(basename "$full" -full.root)
  mc="$MC_DIR/$base.root"
  fit="$FIT_DIR/$base-3dfit.root"
  if [[ ! -f "$mc" || ! -f "$fit" ]]; then
    echo "ANGLE_BATCH_FAILED missing pair base=$base mc=$mc fit=$fit"
    fail=$((fail + 1))
    return
  fi

  echo "ANGLE_BATCH_START base=$base time=$(date '+%F %T')"
  out=$(root -l -b -q "$CODE/addMcAngleBatch.C(\"$full\",\"$mc\")" 2>&1)
  echo "$out" | grep -E 'ANGLE_(UPDATE|ALREADY)' || true
  if echo "$out" | grep -q 'ANGLE_UPDATE_OK'; then
    ok=$((ok + 1))
  elif echo "$out" | grep -q 'ANGLE_ALREADY_PRESENT'; then
    skip=$((skip + 1))
  else
    echo "$out" | tail -30
    echo "ANGLE_BATCH_FAILED update base=$base"
    fail=$((fail + 1))
    return
  fi

  out=$(root -l -b -q "$CODE/verifyFullAngles.C(\"$full\",\"$mc\",\"$fit\")" 2>&1)
  echo "$out" | grep -E 'ANGLE_VERIFY_(OK|FAILED)' || true
  if ! echo "$out" | grep -q 'ANGLE_VERIFY_OK'; then
    echo "$out" | tail -30
    verify_fail=$((verify_fail + 1))
  fi
}

echo "ANGLE_BATCH_BEGIN time=$(date '+%F %T') host=$(hostname)"
shopt -s nullglob
files=("$FULL_DIR"/*-full.root)
echo "ANGLE_BATCH_FILES count=${#files[@]}"
for full in "${files[@]}"; do
  run_one "$full"
done
echo "ANGLE_BATCH_DONE ok=$ok skip=$skip fail=$fail verify_fail=$verify_fail time=$(date '+%F %T')"

(( fail == 0 && verify_fail == 0 ))
