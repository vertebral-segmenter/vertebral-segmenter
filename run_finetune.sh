#!/bin/bash
#SBATCH --nodes=1       # Request GPU "generic resources"
#SBATCH --gpus-per-node=1       # Request GPU "generic resources"
#SBATCH --cpus-per-task=6  # Cores proportional to GPUs: 6 on Cedar, 16 on Graham.
#SBATCH --time=23:15:00
#SBATCH --output=finetune-%j.out
#SBATCH --mail-user=154757929sherry@gmail.com
#SBATCH --mail-type=ALL


# Graham setup
# module load python/3.9.6
# source venv/bin/activate
# Mist setup
module load anaconda3/2021.05 cuda/11.4.4 gcc/10.3.0
source activate vertebral

set -euox
python -c "import torch; print(\"GPUs\", torch.cuda.device_count());"

# Single process version (with dilation)
# Expected arguments $@
# 1.1. dilation, regular loss, no pretrain: --use_dilated_swin --logdir="dilation_regloss_nopretrain" --optim_lr=1e-4 --regular_dice --use_ssl_pretrained=pretrain/pretrained_models/model_swinvit.pt
# 1.2. dilation, regular loss, with pretrain: --use_dilated_swin --logdir="dilation_regloss_pretrain" --optim_lr=1e-4 --regular_dice --use_ssl_pretrained=pretrain/pretrained_models/model_swinvit_dilation_370768.pt
# 1.3. dilation, custom loss, no pretrain: --use_dilated_swin --logdir="dilation_customloss_nopretrain" --optim_lr=1e-4 --use_ssl_pretrained=pretrain/pretrained_models/model_swinvit.pt
# 1.4. dilation, custom loss, with pretrain: --use_dilated_swin --logdir="dilation_customloss_pretrain" --optim_lr=1e-4 --use_ssl_pretrained=pretrain/pretrained_models/model_swinvit_dilation_370768.pt
# 1.1. no dilation, regular loss, no pretrain: --logdir="nodilation_regloss_nopretrain" --optim_lr=1e-4 --regular_dice --use_ssl_pretrained=pretrain/pretrained_models/model_swinvit.pt
# 1.2. no dilation, regular loss, with pretrain: --logdir="nodilation_regloss_pretrain" --optim_lr=1e-4 --regular_dice --use_ssl_pretrained=pretrain/pretrained_models/model_swinvit_no_dilation_370767.pt
# 1.3. no dilation, custom loss, no pretrain: --logdir="nodilation_customloss_nopretrain" --optim_lr=1e-4 --use_ssl_pretrained=pretrain/pretrained_models/model_swinvit.pt
# 1.4. no dilation, custom loss, with pretrain: --logdir="nodilation_customloss_pretrain" --optim_lr=1e-4 --use_ssl_pretrained=pretrain/pretrained_models/model_swinvit_no_dilation_370767.pt

python finetune.py --batch_size=1 --lrschedule=warmup_cosine --infer_overlap=0.5 --feature_size=48 --roi_x=96 --roi_y=96 --roi_z=96 --save_checkpoint --data_dir=./ --json_list=finetune/jsons/dataset.json $@
