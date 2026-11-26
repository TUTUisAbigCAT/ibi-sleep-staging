#!/bin/bash
#SBATCH --account=def-plago
#SBATCH --gpus-per-node=1
#SBATCH --time=04:00:00
#SBATCH --job-name=ibi-folds
#SBATCH --output=/home/yxtao/links/scratch/ibi-sleep-staging/logs/run_folds_%j.txt
# One model, one arm, one seed on the 5 fixed folds (GPU). Submit from the repository root:
#   sbatch experiments/submit_run_folds.sh MODEL FOLDS_JSON SETS RESULTS ARM SEED [more run_folds.py options]
# MODEL: watchsleepnet | insightsleepnet | sleepconvnet | sleepppgnet | mambasleepnet (needs mamba_ssm).
# Environment: the same as WatchSleepNet_public's MambaSleepNet jobs on Trillium (train_transfer_mamba.sh):
# these modules, then that project's venv (torch 2.5.1, mamba_ssm 2.2.4). Every arm with seeds 0-2:
#   D=/home/yxtao/links/scratch/ibi-sleep-staging
#   for A in real gen_Deep gen_REM gen_Deep-REM gen_random20 shuffled_Deep shuffled_REM shuffled_random20; do
#     for S in 0 1 2; do sbatch experiments/submit_run_folds.sh mambasleepnet $D/folds.json $D/sets $D/results $A $S; done
#   done
# A finished model/arm/seed is skipped; RESULTS/<model>/summary.csv is rewritten over what is finished.
# Class-weighted cross-entropy: add --class-weights sqrt (results in RESULTS/<model>_cw-sqrt/, kept apart).
# Arguments, not environment variables: Trillium starts jobs with --export=NONE.

set -euo pipefail
cd "${SLURM_SUBMIT_DIR}"
if [ "$#" -lt 6 ]; then
  echo "Usage: sbatch experiments/submit_run_folds.sh MODEL FOLDS_JSON SETS RESULTS ARM SEED [more options]" >&2
  exit 1
fi
MODEL="$1"; FOLDS="$2"; SETS="$3"; RESULTS="$4"; ARM="$5"; SEED="$6"; shift 6
VENV=/home/yxtao/links/projects/def-plago/yxtao/WatchSleepNet_public/venv
if [ ! -f "$VENV/bin/activate" ]; then
  echo "no venv at $VENV (the WatchSleepNet_public environment); edit VENV in this script" >&2
  exit 1
fi
module load StdEnv/2023 gcc/12.3
module load opencv/4.9.0
module load python/3.11
source "$VENV/bin/activate"
# $HOME and $PROJECT are read-only inside Trillium jobs: keep caches on the job's local disk.
export XDG_CACHE_HOME="${SLURM_TMPDIR:-/tmp}/cache" MPLCONFIGDIR="${SLURM_TMPDIR:-/tmp}/matplotlib"
export TRITON_CACHE_DIR="${SLURM_TMPDIR:-/tmp}/triton"
mkdir -p "$XDG_CACHE_HOME" "$MPLCONFIGDIR" "$TRITON_CACHE_DIR"
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2
python -c "import torch, sklearn, tqdm; print('torch', torch.__version__, 'cuda', torch.cuda.is_available())"
if [ "$MODEL" = "mambasleepnet" ]; then
  python -c "import mamba_ssm" || { echo "mambasleepnet needs mamba_ssm in $VENV" >&2; exit 1; }
fi
python -B -u experiments/run_folds.py --model "$MODEL" --folds "$FOLDS" --sets "$SETS" --output-dir "$RESULTS" \
  --arms "$ARM" --seeds "$SEED" --num-workers 4 "$@"
