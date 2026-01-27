#!/bin/bash
#SBATCH --nodes=1 
#SBATCH --gres=gpu:4
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=12
#SBATCH --account=training2560
#SBATCH --partition=dc-gpu-devel
#SBATCH --time=00:45:00
#SBATCH --output=./slurm_report/%j.out
#SBATCH --error=./slurm_report/%j.err

# Export the necessary environment variables for huggingface offline mode
export HF_DATASETS_OFFLINE=1

# Get number of cpu per task
export SRUN_CPUS_PER_TASK="$SLURM_CPUS_PER_TASK"

export CUDA_VISIBLE_DEVICES=0,1,2,3

# We activate our environemnt
source ./sc_venv_template_HPC_supporter_course/activate.sh
# The above path is a virtual environment that was created and tested previously.
# If you create a new virtual environment, you should update the path above and activate yours.

export PYTHONPATH="${PYTHONPATH}:$(pwd)"

srun --cpu_bind=none python train/Lit_training.py
