#!/bin/bash
#SBATCH --nodes=1 ## TODO 18: Increase the number of nodes (e.g., 2) to run the training on multiple nodes.
#SBATCH --gres=gpu:1 ## TODO 15: Set the number of GPUs to use for training.
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --account=atmlaml
#SBATCH --partition=dc-gpu-devel
#SBATCH --time=00:10:00
#SBATCH --output=./slurm_report/%j.out
#SBATCH --error=./slurm_report/%j.err



# Export the necessary environment variables for huggingface offline mode
export HF_DATASETS_OFFLINE=1

# Get number of cpu per task
export SRUN_CPUS_PER_TASK="$SLURM_CPUS_PER_TASK"
## TODO 15: Set the CUDA_VISIBLE_DEVICES environment variable to 0,1,2,3 to specify the GPUs to use for training.


## TODO 16: 
# 1. Extracts the first hostname
# 2. Allow communication over InfiniBand cells.
# 3. Setup MASTER_ADDR and MASTER_PORT

# We activate our environemnt
source ./sc_venv_template_HPC_supporter_course/activate.sh
# The above path is a virtual environment that was created and tested previously.
# If you create a new virtual environment, you should update the path above and activate yours.

## TODO 17: Replace this line with the distributed training launch script that uses torchrun_jsc and pass the following arguments:
# --nnodes=$SLURM_NNODES that specifies the total number of nodes to use
# --rdzv_backend c10d that sets the backend for rendezvous (process coordination) to PyTorch’s c10d
# --nproc_per_node=gpu that indicates the number of processes per node, matching the number of GPUs
# --rdzv_id $RANDOM that assigns a unique identifier for the rendezvous session using a random value
# --rdzv_endpoint=$MASTER_ADDR:$MASTER_PORT that specifies the address and port for the rendezvous server
# --rdzv_conf=is_host=$(if ((SLURM_NODEID)); then echo 0; else echo 1; fi) that configures whether the node is the rendezvous host based on its SLURM ID
# train.py that runs the training script using the specified distributed setup

#export PYTHONPATH="/p/home/jusers/kasravi1/shared/course/Running-and-Scaling-Performent-Deep-Learning-Models-on-HPC:${PYTHONPATH}"
export PYTHONPATH="${PYTHONPATH}:$(pwd)"

srun --cpu_bind=none python train/single_gpu_training.py