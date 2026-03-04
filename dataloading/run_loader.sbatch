#!/bin/bash -x

# SLURM SUBMIT SCRIPT
#SBATCH --nodes=1                    
#SBATCH --gres=gpu:4                  
#SBATCH --ntasks-per-node=4            
#SBATCH --cpus-per-task=24            
#SBATCH --time=00:10:00
#SBATCH --partition=dc-gpu
#SBATCH --account=training2609
#SBATCH --output=%j.out
#SBATCH --error=%j.err

#SBATCH --reservation=Run_Scal_day2 # For today only

export CUDA_VISIBLE_DEVICES=0,1,2,3    # Very important to make the GPUs visible
export SRUN_CPUS_PER_TASK="$SLURM_CPUS_PER_TASK"

# We activate our environemnt
source ./sc_venv_template_HPC_supporter_course/activate.sh
# The above path is a virtual environment that was created and tested previously.
# If you create a new virtual environment, you should update the path above and activate yours.
export PYTHONPATH="${PYTHONPATH}:$(pwd)"

time srun python3 imagenet_loaders.py --dset_type "fs" --data_root "/p/scratch/training2609/"