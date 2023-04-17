#!/bin/bash
#SBATCH --nodes=1       # Request GPU "generic resources"
#SBATCH --gpus-per-node=1       # Request GPU "generic resources"
#SBATCH --cpus-per-task=6  # Cores proportional to GPUs: 6 on Cedar, 16 on Graham.
#SBATCH --time=00:20:00
#SBATCH --output=inference-%j.out
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
# 1.1. dilation, regular loss, no pretrain: --use_dilated_swin --exp_name="inf_dilation_regloss_nopretrain" --model=finetune/runs/dilation_customloss_nopretrain/model.pt
# 1.2. dilation, regular loss, with pretrain: --use_dilated_swin --exp_name="inf_dilation_regloss_pretrain" --model=finetune/runs/dilation_regloss_pretrain/model.pt
# 1.3. dilation, custom loss, no pretrain: --use_dilated_swin --exp_name="inf_dilation_customloss_nopretrain" --model=finetune/runs/dilation_customloss_nopretrain/model.pt
# 1.4. dilation, custom loss, with pretrain: --use_dilated_swin --exp_name="inf_dilation_customloss_pretrain" --model=finetune/runs/dilation_customloss_pretrain/model.pt
# 1.1. no dilation, regular loss, no pretrain: --exp_name="inf_nodilation_regloss_nopretrain" --model=finetune/runs/nodilation_regloss_nopretrain/model.pt
# 1.2. no dilation, regular loss, with pretrain: --exp_name="inf_nodilation_regloss_pretrain" --model=finetune/runs/nodilation_regloss_pretrain/model.pt
# 1.3. no dilation, custom loss, no pretrain: --exp_name="inf_nodilation_customloss_nopretrain" --model=finetune/runs/nodilation_customloss_nopretrain/model.pt
# 1.4. no dilation, custom loss, with pretrain: --exp_name="inf_nodilation_customloss_pretrain" --model=finetune/runs/nodilation_customloss_pretrain/model.pt

python inference.py --infer_overlap=0.5 --feature_size=48 --roi_x=96 --roi_y=96 --roi_z=96 --data_dir=./ --json_list=finetune/jsons/dataset.json $@
