#!/bin/bash
set -euo pipefail
cd "$(dirname "$0")"

PREP_JOB=$(sbatch --parsable --job-name=ang_all10_cache job_cache_all10.sub)
EVAL_JOB=$(sbatch --parsable --dependency="afterok:${PREP_JOB}" \
  --job-name=ang_all10_eval job_eval_all10_seed73111.sub)
echo "submitted cache job $PREP_JOB"
echo "submitted dependent evaluation job $EVAL_JOB"
