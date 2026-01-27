#!/bin/bash
#SBATCH --nodes=1
#SBATCH --gres=gpu:4
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=128
#SBATCH --account=training2560
#SBATCH --partition=dc-gpu
#SBATCH --time=00:20:00
#SBATCH --output=./slurm_report/%j.out
#SBATCH --error=./slurm_report/%j.err

# Export the necessary environment variables for huggingface offline mode
export HF_DATASETS_OFFLINE=1

# Get number of cpu per task
export SRUN_CPUS_PER_TASK="$SLURM_CPUS_PER_TASK"

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

# Prevent NCCL not figuring out how to initialize.
export NCCL_SOCKET_IFNAME=ib0
# Prevent Gloo not being able to communicate.
export GLOO_SOCKET_IFNAME=ib0

# We activate our environemnt
source ./sc_venv_template_HPC_supporter_course/activate.sh
# The above path is a virtual environment that was created and tested previously.
# If you create a new virtual environment, you should update the path above and activate yours.
export PYTHONPATH="${PYTHONPATH}:$(pwd)"
module load tensorboard
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
    srun env -u CUDA_VISIBLE_DEVICES bash -c 'torchrun \
       --nproc-per-node=gpu \
       --nnodes="$SLURM_JOB_NUM_NODES" \
       --rdzv-id="$SLURM_JOB_ID" \
       --rdzv-endpoint="$MASTER_ADDR":"$MASTER_PORT" \
       --rdzv-backend=c10d \
       --rdzv-conf=is_host="$(if ((SLURM_NODEID)); then echo 0; else echo 1; fi)" \
       --local-addr="$(if ((SLURM_NODEID)); then echo $MASTER_ADDR; else hostname; fi)" \
       --no-python ./run_profile.sh train/fsdp_training.py --profile'

else
    echo "Running without profiling..."
    srun env -u CUDA_VISIBLE_DEVICES bash -c 'torchrun \
       --nproc-per-node=gpu \
       --nnodes="$SLURM_JOB_NUM_NODES" \
       --rdzv-id="$SLURM_JOB_ID" \
       --rdzv-endpoint="$MASTER_ADDR":"$MASTER_PORT" \
       --rdzv-backend=c10d \
       --rdzv-conf=is_host="$(if ((SLURM_NODEID)); then echo 0; else echo 1; fi)" \
       --local-addr="$(if ((SLURM_NODEID)); then echo $MASTER_ADDR; else hostname; fi)" \
        train/fsdp_training.py'
fi