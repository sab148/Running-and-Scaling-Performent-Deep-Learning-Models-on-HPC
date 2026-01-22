#!/usr/bin/env bash
#SBATCH --nodes=1
#SBATCH --gres=gpu:4
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=12
#SBATCH --account=training2556
#SBATCH --partition=dc-gpu-devel
#SBATCH --time=00:30:00
#SBATCH --output=./slurm_report/%j.out
#SBATCH --error=./slurm_report/%j.err

set -xe

# We activate our environemnt
source ./sc_venv_template_HPC_supporter_course/activate.sh
# The above path is a virtual environment that was created and tested previously.
# If you create a new virtual environment, you should update the path above and activate yours.

export SRUN_CPUS_PER_TASK="$SLURM_CPUS_PER_TASK"

export MASTER_ADDR="$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)"
if [ "$SYSTEMNAME" = juwelsbooster ] \
       || [ "$SYSTEMNAME" = juwels ] \
       || [ "$SYSTEMNAME" = jurecadc ] \
       || [ "$SYSTEMNAME" = jusuf ]; then
    # Allow communication over InfiniBand cells on JSC machines.
    MASTER_ADDR="$MASTER_ADDR"i
fi
export MASTER_PORT=54123

# Prevent NCCL not figuring out how to initialize.
export NCCL_SOCKET_IFNAME=ib0
# Prevent Gloo not being able to communicate.
export GLOO_SOCKET_IFNAME=ib0

# Export the necessary environment variables for huggingface offline mode
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export WANDB_DISABLED=true

srun torchrun_jsc \
     --nproc_per_node=gpu \
     --nnodes="$SLURM_JOB_NUM_NODES" \
     --rdzv_id="$SLURM_JOB_ID" \
     --rdzv_endpoint="$MASTER_ADDR":"$MASTER_PORT" \
     --rdzv_backend=c10d \
     "train/hf_training.py" -t ddp

srun torchrun_jsc \
     --nproc_per_node=gpu \
     --nnodes="$SLURM_JOB_NUM_NODES" \
     --rdzv_id="$SLURM_JOB_ID" \
     --rdzv_endpoint="$MASTER_ADDR":"$MASTER_PORT" \
     --rdzv_backend=c10d \
     "train/hf_training.py" -t fsdp
