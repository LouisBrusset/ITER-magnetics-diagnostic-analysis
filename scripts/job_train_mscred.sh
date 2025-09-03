#!/bin/bash
set -euo pipefail

#SBATCH --job-name=mscred-training
#SBATCH --mail-user=louis.brusset@iter.org
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --partition=titan
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --mem=16G
#SBATCH --time=1-12:00:00
#SBATCH --output=/home/ITER/brussel/Documents/ITER-magnetics-diagnostic-analysis/scripts/files/result_%j.out
#SBATCH --error=/home/ITER/brussel/Documents/ITER-magnetics-diagnostic-analysis/scripts/files/error_%j.err


echo "Beginning execution at: $(date)"
echo "-------------------------------"
echo "Hostname: $(hostname)"
echo "Job ID: ${SLURM_JOB_ID}"

module load Python/3.11.5-GCCcore-13.2.0

cd /home/ITER/brussel/Documents/ITER-magnetics-diagnostic-analysis
source .venv/bin/activate
cd src/magnetics_diagnostic_analysis/project_mscred

echo "Current directory: $(pwd)"
echo "Python version: $(python --version)"
echo "UV version: $(uv --version)"

echo "Checking GPU availability:"
python -c "import torch; print(f'PyTorch version: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}'); print(f'GPU count: {torch.cuda.device_count()}')"

if ! uv run train_mscred.py; then
    echo "Training failed!" >&2
    exit 1
fi

echo "----------------------------"
echo "Ending execution at: $(date)"
echo "Job completed successfully !"