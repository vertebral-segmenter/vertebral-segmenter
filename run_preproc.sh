#!/bin/bash
#SBATCH --nodes=1       # Request GPU "generic resources"
#SBATCH --gpus-per-node=1       # Request GPU "generic resources"
#SBATCH --cpus-per-task=6  # Cores proportional to GPUs: 6 on Cedar, 16 on Graham.
#SBATCH --time=00:15:00
#SBATCH --output=preproc-%j.out

# Graham setup
# module load python/3.9.6
# source venv/bin/activate
# Mist setup
module load anaconda3/2021.05 cuda/11.4.4 gcc/10.3.0
source activate vertebral

python data_preprocessing/intensity_scaling.py
