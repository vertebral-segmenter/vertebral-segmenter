#!/bin/bash
#SBATCH --nodes=1       # Request GPU "generic resources"
#SBATCH --gpus-per-node=1       # Request GPU "generic resources"
#SBATCH --cpus-per-task=6  # Cores proportional to GPUs: 6 on Cedar, 16 on Graham.
#SBATCH --mem=32000M
#SBATCH --time=00:15:00
#SBATCH --output=preproc-%j.out

# Graham setup
# module load python/3.9.6
# source venv/bin/activate
# Mist setup
module load cuda/11.4 python/3.9.6 qt/5.12.8 geos llvm/8.0.1
source venv/bin/activate

python data_preprocessing/intensity_scaling.py
