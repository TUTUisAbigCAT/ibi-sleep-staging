#!/bin/bash
#SBATCH --account=def-plago
#SBATCH --gpus-per-node=1
#SBATCH --time=04:00:00
#SBATCH --job-name=ibi-folds
#SBATCH --output=/home/yxtao/links/scratch/ibi-sleep-staging/logs/run_folds_%j.txt
# One model, one arm, one seed on the 5 fixed folds (GPU). Submit from the repository root:
#   sbatch experiments/submit_run_folds.sh MODEL FOLDS_JSON SETS RESULTS ARM SEED [more run_folds.py options]
# MODEL: watchsleepnet | insightsleepnet | sleepconvnet | sleepppgnet | mambasleepnet (needs mamba_ssm).
# The job activates ./venv (as the repository's other scripts do); e.g. every arm with seeds 0-2:
#   D=/home/yxtao/links/scratch/ibi-sleep-staging
#   for A in real gen_Deep gen_REM gen_Deep-REM gen_random20 shuffled_Deep shuffled_REM shuffled_random20; do
#     for S in 0 1 2; do sbatch experiments/submit_run_folds.sh mambasleepnet $D/folds.json $D/sets $D/results $A $S; done
#   done
# A finished model/arm/seed is skipped; RESULTS/<model>/summary.csv is rewritten over what is finished.
# Arguments, not environment variables: Trillium starts jobs with --export=NONE.

set -euo pipefail
cd "${SLURM_SUBMIT_DIR}"
if [ "$#" -lt 6 ]; then
  echo "Usage: sbatch experiments/submit_run_folds.sh MODEL FOLDS_JSON SETS RESULTS ARM SEED [more options]" >&2
  exit 1
fi
MODEL="$1"; FOLDS="$2"; SETS="$3"; RESULTS="$4"; ARM="$5"; SEED="$6"; shift 6
if [ ! -f ./venv/bin/activate ]; then
  echo "no ./venv here: create one, or link an existing one (ln -s /path/to/venv venv)" >&2
  exit 1
fi
source ./venv/bin/activate
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2
python -c "import torch, sklearn, tqdm; print('torch', torch.__version__, 'cuda', torch.cuda.is_available())"
if [ "$MODEL" = "mambasleepnet" ]; then
  python -c "import mamba_ssm" || { echo "mambasleepnet needs mamba_ssm in ./venv" >&2; exit 1; }
fi
python -B -u experiments/run_folds.py --model "$MODEL" --folds "$FOLDS" --sets "$SETS" --output-dir "$RESULTS" \
  --arms "$ARM" --seeds "$SEED" --num-workers 4 "$@"
