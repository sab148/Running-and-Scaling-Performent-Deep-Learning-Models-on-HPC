#!/bin/bash
#SBATCH --nodes=1
#SBATCH --gres=gpu:4  
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=128
#SBATCH --account=atmlaml
#SBATCH --partition=dc-gpu-devel
#SBATCH --time=00:10:00
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
export MASTER_PORT=7010

# Prevent NCCL not figuring out how to initialize.
export NCCL_SOCKET_IFNAME=ib0
# Prevent Gloo not being able to communicate.
export GLOO_SOCKET_IFNAME=ib0

# We activate our environemnt
source /p/scratch/atmlaml/HPC-Supporters-Course/sc_venv_template_HPC_supporter_course/activate.sh
# The above path is a virtual environment that was created and tested previously.
# If you create a new virtual environment, you should update the path above and activate yours.
export PYTHONPATH="${PYTHONPATH}:$(pwd)"

# Launch a distributed training job across multiple nodes and GPUs
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

if [ "$PROFILE" = true ]; then
    echo "Running with NSYS profiling..."
    
    mkdir -p nsys_logs
    srun torchrun_jsc \
        --nnodes=$SLURM_NNODES \
        --rdzv_backend c10d \
        --nproc_per_node=gpu \
        --rdzv_id $RANDOM \
        --rdzv_endpoint=$MASTER_ADDR:$MASTER_PORT \
        --no-python ./run_profile.sh train/ddp_training.py --profile

else
    echo "Running without profiling..."
    srun torchrun_jsc \
        --nnodes=$SLURM_NNODES \
        --rdzv_backend c10d \
        --nproc_per_node=gpu \
        --rdzv_id $RANDOM \
        --rdzv_endpoint=$MASTER_ADDR:$MASTER_PORT \
        train/ddp_training.py
fi