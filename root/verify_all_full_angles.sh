#!/usr/bin/env bash
# Independent resumable verification of every updated full ROOT file.
set -uo pipefail

BASE=/aifs/user/data/lishanglin/datasets
CODE=/aifs/user/data/lishanglin/chenhao/angleTransformer/root
ok=0
fail=0

shopt -s nullglob
files=("$BASE/full"/*-full.root)
for full in "${files[@]}"; do
  base=$(basename "$full" -full.root)
  mc="$BASE/mcinfo/$base.root"
  fit="$BASE/3dfit/$base-3dfit.root"
  out=$(root -l -b -q "$CODE/verifyFullAngles.C(\"$full\",\"$mc\",\"$fit\")" 2>&1)
  if echo "$out" | grep -q 'ANGLE_VERIFY_OK'; then
    echo "$out" | grep 'ANGLE_VERIFY_OK'
    ok=$((ok + 1))
  else
    echo "$out" | tail -30
    fail=$((fail + 1))
  fi
done
echo "ANGLE_VERIFY_ALL_DONE files=${#files[@]} ok=$ok fail=$fail"
(( fail == 0 && ok == ${#files[@]} ))

