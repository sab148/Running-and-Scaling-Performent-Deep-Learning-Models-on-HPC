#!/bin/bash
#SBATCH --nodes=2 
#SBATCH --gres=gpu:4
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=32
#SBATCH --account=training2560
#SBATCH --partition=dc-gpu
#SBATCH --time=00:45:00
#SBATCH --output=./slurm_report/%j.out
#SBATCH --error=./slurm_report/%j.err

# Export the necessary environment variables for huggingface offline mode
export HF_DATASETS_OFFLINE=1

# Get number of cpu per task
export SRUN_CPUS_PER_TASK="$SLURM_CPUS_PER_TASK"

export CUDA_VISIBLE_DEVICES=0,1,2,3

# Extracts the first hostname from the list of allocated nodes to use as the master address.
export MASTER_ADDR="$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)"

# Modifies the master address to allow communication over InfiniBand cells.
if [ "$SYSTEMNAME" = juwelsbooster ] \
       || [ "$SYSTEMNAME" = juwels ] \
       || [ "$SYSTEMNAME" = jurecadc ] \
       || [ "$SYSTEMNAME" = jusuf ]; then
    # Allow communication over InfiniBand cells on JSC machines.
    MASTER_ADDR="$MASTER_ADDR"i
fi

export MASTER_PORT=7010

# We activate our environemnt
source ./sc_venv_template_HPC_supporter_course/activate.sh
# The above path is a virtual environment that was created and tested previously.
# If you create a new virtual environment, you should update the path above and activate yours.

export PYTHONPATH="${PYTHONPATH}:$(pwd)"

srun --cpus-per-task=32 python train/Lit_training.py
