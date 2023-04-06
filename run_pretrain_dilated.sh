#!/bin/bash
#SBATCH --nodes=1       # Request GPU "generic resources"
#SBATCH --gpus-per-node=1       # Request GPU "generic resources"
#SBATCH --cpus-per-task=6  # Cores proportional to GPUs: 6 on Cedar, 16 on Graham.
#SBATCH --time=06:00:00
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

# Single process version (with dilation)
python pretrain.py --use_checkpoint --use_dilated_swin --logdir="dilation_0" --batch_size=1 --num_steps=6000 --lrdecay --eval_num=200 --lr=3e-6 --use_ssl_pretrained=pretrain/pretrained_models/model_swinvit.pt
# python pretrain.py --use_checkpoint --use_dilated_swin --logdir="dilation_1" --batch_size=1 --num_steps=3000 --lrdecay --eval_num=200 --lr=6e-7 --use_ssl_pretrained=pretrain/pretrained_models/model_swinvit_dilation_5147718.pt

# Single process resume training (with dilation)
# python pretrain.py --use_checkpoint --use_dilated_swin --logdir="dilation_2" --batch_size=1 --num_steps=2200 --lrdecay --eval_num=200 --lr=6e-7 --resume pretrain/runs/dilation_1/model_final_epoch.pt

# Distributed version
# python -m torch.distributed.launch --logdir="distributed" --nproc_per_node=2 --master_port=11223 pretrain.py --batch_size=1 --num_steps=1100 --lrdecay --eval_num=200 --lr=6e-7 --decay=0.1 --use_ssl_pretrained
