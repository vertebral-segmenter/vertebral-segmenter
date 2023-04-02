#!/bin/bash
#SBATCH --gres=gpu:1       # Request GPU "generic resources"
#SBATCH --cpus-per-task=6  # Cores proportional to GPUs: 6 on Cedar, 16 on Graham.
#SBATCH --mem=32000M       # Memory proportional to GPUs: 32000 Cedar, 64000 Graham.
#SBATCH --time=00:02:00
#SBATCH --output=pretrain-%j.out

module load python/3.9.6
source venv/bin/activate

python -c "import torch; print(\"GPUs\", torch.cuda.device_count());"

python data_preprocessing/intensity_scaling.py
