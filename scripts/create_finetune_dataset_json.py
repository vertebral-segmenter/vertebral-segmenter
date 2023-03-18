import os
import numpy as np
import json


def get_dataset_id(filename):
    """Get dataset id from the training image file name"""
    return filename.split('.')[0].lstrip("img")

def get_label_filename(dataset_id):
    return f"label{dataset_id}.nii.gz"

dataset_base_path = 'finetune/dataset/'
# dataset_base_path = 'finetune/dataset'
train_img_path = dataset_base_path+'Training/img/'
train_label_path = dataset_base_path+'Training/label/'
test_img_path = dataset_base_path+'Testing/img/'

dst_path = 'finetune/jsons/dataset.json'

base_json =  {
    "description": "btcv yucheng",
    "labels": {
        "0": "background",
        "1": "spleen",
        "2": "rkid",
        "3": "lkid",
        "4": "gall",
        "5": "eso",
        "6": "liver",
        "7": "sto",
        "8": "aorta",
        "9": "IVC",
        "10": "veins",
        "11": "pancreas",
        "12": "rad",
        "13": "lad"
    },
    "licence": "yt",
    "modality": {
        "0": "CT"
    },
    "name": "btcv",
    "numTest": 20,
    "numTraining": 80,
    "reference": "Vanderbilt University",
    "release": "1.0 06/08/2015",
    "tensorImageSize": "3D",
    "test": [],
    "training": [],
    "validation": []
}


validation_prob = 0.2
training_fns = [fn for fn in os.listdir(train_img_path) if fn.endswith('.nii.gz')]
test_fns = [fn for fn in os.listdir(test_img_path) if fn.endswith('.nii.gz')]

choice = np.random.choice(range(len(training_fns)), size=(int(validation_prob * len(training_fns)),), replace=False)    
train_fn_idx = np.zeros(len(training_fns), dtype=bool)
train_fn_idx[choice] = True

for image_fn_idx, is_train in enumerate(train_fn_idx):
    image_fn = training_fns[image_fn_idx]
    add_to_set = base_json["training"] if is_train else base_json["validation"]
    did = get_dataset_id(image_fn)
    label_fn = get_label_filename(did)
    image_path = train_img_path + image_fn
    label_path = train_label_path + label_fn
    assert os.path.exists(image_path), f"path [{image_path}] does not exist"
    assert os.path.exists(label_path), f"path [{label_path}] does not exist"
    add_to_set.append({
        "image": image_path,
        "label": label_path
    })

for image_fn in test_fns:
    image_path = test_img_path + image_fn
    base_json["test"].append(image_path)

with open(dst_path, 'w') as fp:
    json.dump(base_json, fp, indent=2)