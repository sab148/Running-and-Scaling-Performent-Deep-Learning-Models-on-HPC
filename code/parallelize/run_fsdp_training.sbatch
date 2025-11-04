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
source $HOME/course/sc_venv_template/activate.sh

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
    srun --cpu_bind=none bash -c "
       
    # Create rank-specific directory
    mkdir -p nsys_logs
    
    echo \"Rank \${SLURM_PROCID} starting profiling...\"

    nsys profile \
        --duration=60 \
        --delay=30 \
        --gpu-metrics-device=all \
        --nic-metrics=true \
        --stop-on-exit=false \
        --trace=nvtx,cuda,osrt \
        --python-sampling=true \
        --python-sampling-frequency=1 \
        --output=nsys_logs/nsys_logs_rank\${SLURM_PROCID} \
        --cuda-memory-usage=true \
        --force-overwrite=true \
        --python-functions-trace=config/profiling.json \
        python fsdp_training.py"
else
    echo "Running without profiling..."
    srun --cpu_bind=none python fsdp_training.py
fi