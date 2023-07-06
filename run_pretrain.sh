#!/bin/bash
#SBATCH --nodes=1       # Request GPU "generic resources"
#SBATCH --gpus-per-node=1       # Request GPU "generic resources"
#SBATCH --cpus-per-task=6  # Cores proportional to GPUs: 6 on Cedar, 16 on Graham.
#SBATCH --time=18:00:00
#SBATCH --output=pretrain-%j.out
#SBATCH --mail-user=154757929sherry@gmail.com
#SBATCH --mail-type=ALL

# Graham setup
# module load python/3.9.6
# source venv/bin/activate
# Mist setup
module load anaconda3/2021.05 cuda/11.4.4 gcc/10.3.0
source activate vertebral

python -c "import torch; print(\"GPUs\", torch.cuda.device_count());"

# Expected args:
# 1. no dilation: --logdir="no_dilation_0" --lr=6e-7 --num_steps=6000 --use_ssl_pretrained=pretrain/pretrained_models/model_swinvit.pt
# 2. dilation: --use_dilated_swin --logdir="dilation_0" --lr=3e-6 --num_steps=6000 --use_ssl_pretrained=pretrain/pretrained_models/model_swinvit.pt
# Resume (add flag)
# 1. --resume=pretrain/runs/no_dilation_0/model_final_epoch.pt
# 2. --resume=pretrain/runs/dilation_0/model_final_epoch.pt
set -euox
# Single process version (no dilation)
python pretrain.py --use_checkpoint  --batch_size=1 --lrdecay --eval_num=200  $@
# Single process version (with dilation)
# python pretrain.py --use_checkpoint --use_dilated_swin --logdir="dilation_0" --batch_size=1 --num_steps=6000 --lrdecay --eval_num=200 --lr=3e-6 --use_ssl_pretrained

