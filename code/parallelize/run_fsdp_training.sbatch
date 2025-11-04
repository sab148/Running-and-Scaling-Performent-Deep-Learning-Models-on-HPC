#!/bin/bash
#SBATCH --nodes=2
#SBATCH --gres=gpu:4  
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=128
#SBATCH --account=training2543
#SBATCH --partition=dc-gpu
#SBATCH --time=03:00:00
#SBATCH --output=%j.out
#SBATCH --error=%j.err

#SBATCH --reservation=ai-course-day2 # For today only

# Export the necessary environment variables for huggingface offline mode
export HF_DATASETS_OFFLINE=1
# Get number of cpu per task
export SRUN_CPUS_PER_TASK="$SLURM_CPUS_PER_TASK"
export CUDA_VISIBLE_DEVICES=0,1,2,3

# Extracts the first hostname from the list of allocated nodes to use as the master address.
MASTER_ADDR="$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)"
# Modifies the master address to allow communication over InfiniBand cells.
MASTER_ADDR="${MASTER_ADDR}i"
# Get IP for hostname.
export MASTER_ADDR="$(nslookup "$MASTER_ADDR" | grep -oP '(?<=Address: ).*')"
export MASTER_PORT=7010

# We activate our environemnt
source /p/scratch/atmlaml/HPC-Supporters-Course/sc_venv_template_HPC_supporter_course/activate.sh
# The above path is a virtual environment that was created and tested previously.
# If you create a new virtual environment, you should update the path above and activate yours.

# Launch a distributed training job across multiple nodes and GPUs
srun --cpu_bind=none bash -c "torchrun_jsc \
    --nnodes=$SLURM_NNODES \
    --rdzv_backend c10d \
    --nproc_per_node=gpu \
    --rdzv_id $RANDOM \
    --rdzv_endpoint=$MASTER_ADDR:$MASTER_PORT \
    --rdzv_conf=is_host=\$(if ((SLURM_NODEID)); then echo 0; else echo 1; fi) \
    fsdp_training.py "