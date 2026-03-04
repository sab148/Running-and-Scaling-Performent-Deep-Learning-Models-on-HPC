#!/bin/bash -x

#SBATCH --nodes=1          
#SBATCH --gres=gpu:1     
#SBATCH --ntasks-per-node=1  
#SBATCH --cpus-per-task=96
#SBATCH --time=10:00:00
#SBATCH --partition=dc-gpu
#SBATCH --account=training2609
#SBATCH --output=%j.out
#SBATCH --error=%j.err

#SBATCH --reservation=Run_Scal_day2 # For today only

export SRUN_CPUS_PER_TASK="$SLURM_CPUS_PER_TASK"
# We activate our environemnt
source ./sc_venv_template_HPC_supporter_course/activate.sh
# The above path is a virtual environment that was created and tested previously.
# If you create a new virtual environment, you should update the path above and activate yours.
export PYTHONPATH="${PYTHONPATH}:$(pwd)"

mkdir -p "/p/scratch/training2609/$USER"

time srun python save_imagenet_files.py  --dset_type "h5" --target_folder "/p/scratch/training2609/$USER"