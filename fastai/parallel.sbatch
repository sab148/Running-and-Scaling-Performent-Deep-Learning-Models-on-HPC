#!/bin/bash
#SBATCH --account=training2543
#SBATCH --mail-user=MYUSER@fz-juelich.de
#SBATCH --mail-type=ALL
#SBATCH --nodes=1
#SBATCH --job-name=cats-parallel
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=64
#SBATCH --output=output-parallel.%j
#SBATCH --error=error-parallel.%j
#SBATCH --time=00:10:00
#SBATCH --partition=dc-gpu
##SBATCH --reservation=ai-course-day1 # For today only

export NCCL_DEBUG=INFO
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export CUDA_VISIBLE_DEVICES="0,1,2,3"
export MASTER_PORT=12802
echo "NODELIST="${SLURM_NODELIST}
master_addr=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
export MASTER_ADDR=$master_addr"i"

cd $HOME/course/2025-06-course-Bringing-Deep-Learning-Workloads-to-JSC-supercomputers/code/fastai
source sc_venv_template/activate.sh # Now we finally use the fastai module

srun python cats-parallel.py
