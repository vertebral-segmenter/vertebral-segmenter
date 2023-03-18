
## Env Setup

```
conda create -n vertebral python=3.9
pip install -r requirements.txt # for linux
pip install -r requirements-win.txt # for windows
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