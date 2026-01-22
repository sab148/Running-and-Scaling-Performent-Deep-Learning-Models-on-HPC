#!/bin/bash
#SBATCH --nodes=1 
#SBATCH --gres=gpu:1 
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --account=training2560
#SBATCH --partition=dc-gpu-devel
#SBATCH --time=02:00:00
#SBATCH --output=./slurm_report/%j.out
#SBATCH --error=./slurm_report/%j.err

# Adds the current directory to PYTHONPATH so Python can import local modules
export PYTHONPATH="${PYTHONPATH}:$(pwd)"

# Export the necessary environment variables for huggingface offline mode
export HF_DATASETS_OFFLINE=1

# Get number of cpu per task
export SRUN_CPUS_PER_TASK="$SLURM_CPUS_PER_TASK"

# We activate our environemnt
# The above path is a virtual environment that was created and tested previously.
# If you create a new virtual environment, you should update the path above and activate yours.
source ./sc_venv_template_HPC_supporter_course/activate.sh


PROFILE=false

# Parse args
for arg in "$@"; do
    case $arg in
        --profile)
            PROFILE=true
            shift
            ;;
    esac
done


echo "Running with NSYS profiling..."

mkdir -p nsys_logs
srun ./run_profile.sh train/single_gpu_training_profiling.py --profile

