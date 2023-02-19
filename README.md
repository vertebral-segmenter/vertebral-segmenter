
## Env Setup

```
conda create -n vertebral-seg python=3.9
pip install -r requirements.txt # for linux
pip install -r requirements-win.txt # for windows
```

## Prepare data

After data and corresponding json is prepared using the script in: `scripts/create_pretrain_dataset_json.py`, modify `pretrain/utils/data_utils.py` to load the json and data from the right path.

## Pretrain

### Single GPU Training From Scratch

The ROI x y z specifies the patch size

#### Pretrain from scratch

```
python pretrain.py --use_checkpoint --batch_size=1 --num_steps=100000 --lrdecay --eval_num=500 --logdir=0 --lr=6e-6 --roi_x=<Roi_x> --roi_y=<Roi_y> --roi_z=<Roi_z>
```

#### Resume training

```
python pretrain.py --use_checkpoint --batch_size=1 --num_steps=100000 --lrdecay --eval_num=500 --logdir=0 --lr=6e-6 --resume pretrain\pretrained_models\model_swinvit.pt
```

#### Training on top of previous pretrained weight
```
python pretrain.py --use_checkpoint --batch_size=1 --num_steps=100000 --lrdecay --eval_num=500 --logdir=0 --lr=6e-6 --use_ssl_pretrained
```

### Multi GPU Distributed Training

```bash
python -m torch.distributed.launch --nproc_per_node=<Num-GPUs> --master_port=11223 pretrain.py --batch_size=<Batch-Size> --num_steps=<Num-Steps> --lrdecay --eval_num=<Eval-Num> --logdir=<Exp-Num> --lr=<Lr>

# The config used in paper
python -m torch.distributed.launch --nproc_per_node=8 --master_port=11223 pretrain.py --batch_size=1 --num_steps=100000 --lrdecay --eval_num=500 --lr=6e-6 --decay=0.1
```



## Finetune

Default parameters

```
python finetune/main.py --feature_size=32 --batch_size=1 --logdir=unetr_test --fold=0 --optim_lr=1e-4 --lrschedule=warmup_cosine --infer_overlap=0.5 --save_checkpoint --data_dir=/dataset/dataset0/
```

With self-supervised encoder weights

```
python main.py --json_list=<json-path> --data_dir=<data-path> --feature_size=48 --use_ssl_pretrained --roi_x=96 --roi_y=96 --roi_z=96  --use_checkpoint --batch_size=<batch-size> --max_epochs=<total-num-epochs> --save_checkpoint
```