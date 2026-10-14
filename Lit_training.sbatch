#!/bin/bash
#SBATCH --nodes=1
#SBATCH --gres=gpu:1  ## TODO 2: Set the number of GPUs to use for training.
#SBATCH --ntasks-per-node=1 ## TODO 3: Set the number of GPUs to use for training.
#SBATCH --cpus-per-task=32
#SBATCH --account=training2657
#SBATCH --partition=dc-gpu
#SBATCH --time=00:45:00
#SBATCH --output=./slurm_report/%j.out
#SBATCH --error=./slurm_report/%j.err

##SBATCH --reservation=inm_res # For today only

# Export the necessary environment variables for huggingface offline mode
export HF_DATASETS_OFFLINE=1

# Get number of cpu per task
export SRUN_CPUS_PER_TASK="$SLURM_CPUS_PER_TASK"

## TODO 4: Set the CUDA_VISIBLE_DEVICES environment variable to 0,1,2,3 to specify the GPUs to use for training.


## TODO 5: 
# 1. Extracts the first hostname
# 2. Allow communication over InfiniBand cells.
# 3. Setup MASTER_ADDR and MASTER_PORT

module load Stages/2025
module load GCC uv

# We activate our environemnt
source ~/course/.venv/bin/activate
# The above path is a virtual environment that was created and tested previously.
# If you create a new virtual environment, you should update the path above and activate yours.

export PYTHONPATH="${PYTHONPATH}:$(pwd)"

srun --cpus-per-task=32 python train/Lit_training.py
