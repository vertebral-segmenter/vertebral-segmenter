#!/bin/bash
#SBATCH --gres=gpu:1       # Request GPU "generic resources"
#SBATCH --cpus-per-task=6  # Cores proportional to GPUs: 6 on Cedar, 16 on Graham.
#SBATCH --mem=32000M       # Memory proportional to GPUs: 32000 Cedar, 64000 Graham.
#SBATCH --time=24:00:00
#SBATCH --output=pretrain-%j.out

module load python/3.9.6
source venv/bin/activate

python -c "import torch; print(\"GPUs\", torch.cuda.device_count());"

# Single process version
python pretrain.py --use_checkpoint --logdir="single_proc" --batch_size=1 --num_steps=1100 --lrdecay --eval_num=200 --logdir=0 --lr=6e-7 --use_ssl_pretrained
# Distributed version
# python -m torch.distributed.launch --logdir="distributed" --nproc_per_node=2 --master_port=11223 pretrain.py --batch_size=1 --num_steps=1100 --lrdecay --eval_num=200 --lr=6e-7 --decay=0.1 --use_ssl_pretrained
