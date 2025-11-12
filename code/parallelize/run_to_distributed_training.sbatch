#!/bin/bash
#SBATCH --nodes=1 ## TODO 17: Increase the number of nodes (e.g., 2) to run the training on multiple nodes.
#SBATCH --gres=gpu:1 ## TODO 14: Set the number of GPUs to use for training.
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --account=atmlaml
#SBATCH --partition=dc-gpu
#SBATCH --time=03:00:00
#SBATCH --output=%j.out
#SBATCH --error=%j.err



# Export the necessary environment variables for huggingface offline mode
export HF_DATASETS_OFFLINE=1

# Get number of cpu per task
export SRUN_CPUS_PER_TASK="$SLURM_CPUS_PER_TASK"
## TODO 14: Set the CUDA_VISIBLE_DEVICES environment variable to 0,1,2,3 to specify the GPUs to use for training.


## TODO 15: 
# 1. Extracts the first hostname
# 2. Allow communication over InfiniBand cells.
# 3. Setup MASTER_ADDR and MASTER_PORT

# We activate our environemnt
source /p/scratch/atmlaml/HPC-Supporters-Course/sc_venv_template_HPC_supporter_course/activate.sh
# The above path is a virtual environment that was created and tested previously.
# If you create a new virtual environment, you should update the path above and activate yours.

## TODO 16: Replace this line with the distributed training launch script that uses torchrun_jsc and pass the following arguments:
# --nnodes=$SLURM_NNODES that specifies the total number of nodes to use
# --rdzv_backend c10d that sets the backend for rendezvous (process coordination) to PyTorch’s c10d
# --nproc_per_node=gpu that indicates the number of processes per node, matching the number of GPUs
# --rdzv_id $RANDOM that assigns a unique identifier for the rendezvous session using a random value
# --rdzv_endpoint=$MASTER_ADDR:$MASTER_PORT that specifies the address and port for the rendezvous server
# --rdzv_conf=is_host=$(if ((SLURM_NODEID)); then echo 0; else echo 1; fi) that configures whether the node is the rendezvous host based on its SLURM ID
# train.py that runs the training script using the specified distributed setup

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
    srun ./run_profile.sh to_distributed_training.py --profile

else
    echo "Running without profiling..."
    srun --cpu_bind=none python to_distributed_training.py
fi