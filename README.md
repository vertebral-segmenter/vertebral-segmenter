
## Env Setup

```
conda create -n vertebral python=3.9
conda activate vertebral

pip install -r requirements.txt # for linux
pip install -r requirements-win.txt # for windows

# Validate cuda installation
python -c "import torch; print(torch.cuda.device_count());"
# Should see output 2
```

## Prepare data

```sh
# Copy rat data into prtraining data folder
cp -r /home/smsmt/Rat_mCT_new/. ./pretrain/data/
# Prepare rat data json
python scripts/create_pretrain_dataset_json.py
```

Modify `pretrain/utils/data_utils.py` to load the json and data from the right path.

Get previous pretrained weights `model_swinvit.pt` from [here](https://github.com/Project-MONAI/research-contributions/tree/6ca48250bcffc455482caf8328d6c8b149145257/SwinUNETR/Pretrain). Store it in `pretrain/pretrained_models/model_swinvit.pt`

## Pretrain

### Single GPU Training From Scratch

The ROI x y z specifies the patch size

#### Training on top of previous pretrained weight
```
python pretrain.py --use_checkpoint --batch_size=1 --num_steps=100000 --lrdecay --eval_num=500 --logdir=0 --lr=6e-7 --use_ssl_pretrained

# Distributed version
python -m torch.distributed.launch --nproc_per_node=2 --master_port=11223 pretrain.py --batch_size=1 --num_steps=100000 --lrdecay --eval_num=500 --lr=6e-7 --decay=0.1 --use_ssl_pretrained
```

#### Resume training

```
python pretrain.py --use_checkpoint --batch_size=1 --num_steps=100000 --lrdecay --eval_num=500 --logdir=0 --lr=6e-6 --resume pretrain\pretrained_models\model_swinvit.pt
```

#### Pretrain from scratch

```
python pretrain.py --use_checkpoint --batch_size=1 --num_steps=100000 --lrdecay --eval_num=500 --logdir=0 --lr=6e-6 --roi_x=<Roi_x> --roi_y=<Roi_y> --roi_z=<Roi_z>
```

### Multi GPU Distributed Training

```bash
python -m torch.distributed.launch --nproc_per_node=<Num-GPUs> --master_port=11223 pretrain.py --batch_size=<Batch-Size> --num_steps=<Num-Steps> --lrdecay --eval_num=<Eval-Num> --logdir=<Exp-Num> --lr=<Lr>

# The config used in paper
python -m torch.distributed.launch --nproc_per_node=8 --master_port=11223 pretrain.py --batch_size=1 --num_steps=100000 --lrdecay --eval_num=500 --lr=6e-6 --decay=0.1
```



## Finetune

### Data preparation

Get data by running `scripts/load_synapse_data.py`

unzip `dataset/Abdomen/rawdata.zip` into `finetune/dataset`, so that this path (`finetune/dataset`) contains `Testing` and `Training` folder.

Prepare dataset json by running `scripts/create_finetune_dataset_json.py`

Reference: https://github.com/Project-MONAI/research-contributions/tree/main/SwinUNETR/BTCV#data-preparation

Default parameters

```
python finetune.py --batch_size=1 --logdir=unetr_test --optim_lr=1e-4 --lrschedule=warmup_cosine --infer_overlap=0.5 --feature_size=48 --use_ssl_pretrained --roi_x=96 --roi_y=96 --roi_z=96 --save_checkpoint --data_dir=./ --json_list=finetune/jsons/dataset.json --use_ssl_pretrained 
```

With self-supervised encoder weights

```
python finetune.py --json_list=<json-path> --data_dir=<data-path> --feature_size=48 --use_ssl_pretrained --roi_x=96 --roi_y=96 --roi_z=96  --use_checkpoint --batch_size=<batch-size> --max_epochs=<total-num-epochs> --save_checkpoint
```


## Compute Canada
Note: add ssh key to bitbucket before cloning.

### Environment on internal node (without GPUs)
```
python -m venv environment # no conda on compute canada
source environment/bin/activate

pip install -r requirements.txt # should take a while
```

### Prepare data

1. Add bender's ssh public key to CC
2. SSH into bender
3. Run the following command to copy source data

```sh
# Where data reside on bender
LOCAL_DIR=/home/smsmt/Rat_mCT_new
# Where data reside on compute canada (CC), make sure this folder exist on CC
CC_DIR=/scratch/yuanshe5/vertebral-segmentation-rat-l2/pretrain/data/

scp ${LOCAL_DIR}/*.nii yuanshe5@graham.computecanada.ca:${CC_DIR}
```

### Submitting jobs and GPU test

Create pytorch-test.sh:
```
#!/bin/bash
#SBATCH --gres=gpu:3       # Request GPU "generic resources"
#SBATCH --cpus-per-task=6  # Cores proportional to GPUs: 6 on Cedar, 16 on Graham.
#SBATCH --mem=32000M       # Memory proportional to GPUs: 32000 Cedar, 64000 Graham.
#SBATCH --time=0-03:00
#SBATCH --output=%N-%j.out

module load python/3.7.7
virtualenv --no-download $SLURM_TMPDIR/env
source $SLURM_TMPDIR/env/bin/activate
# pip install -r requirements.txt # takes too long
pip install --no-index torch

python -c "import torch; print(\"GPUs\", torch.cuda.device_count());"
```

Submit job: 
```
sbatch pytorch-test.sh 
```

Should get "Submitted batch job {idx}". By default the output is placed in a file named "slurm-", suffixed with the job ID number and ".out", e.g. slurm-{idx}.out, in the directory from which the job was submitted.


### Useful links

https://docs.alliancecan.ca/wiki/PyTorch#PyTorch_with_Multiple_CPUs
https://docs.alliancecan.ca/wiki/Running_jobs#Memory