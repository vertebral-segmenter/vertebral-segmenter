import os
import numpy as np
import json

dataset_base_path = 'finetune/data/'
img_path = dataset_base_path+'scans/'
label_path = dataset_base_path+'labels/'

dst_path = 'finetune/jsons/dataset.json'

base_json =  {
    "description": "murine mCT",
    "labels": {
        "0": "background",
        "1": "l2",
    },
    "licence": "yt",
    "modality": {
        "0": "CT"
    },
    "name": "btcv",
    "numTest": 2,
    "numTraining": 10,
    "reference": "University of Toronto",
    "release": "1.0 04/04/2023",
    "tensorImageSize": "3D",
    "test": [],
    "training": [],
    "validation": []
}

def get_image_id(filename):
    """Get dataset id from the training image file name"""
    return [num for num in filename.split('_') if num.isnumeric()][0]

def get_label_filename(dataset_id):
    return f"{dataset_id}_segmentation.nii"

def img_has_label(label_path, img_fn):
    did = get_image_id(img_fn)
    label_fn = get_label_filename(did)
    label_path = os.path.join(label_path, label_fn)
    return os.path.exists(label_path)

def add_img_to_path(add_to_set, img_path, label_path, img_fn):
    did = get_image_id(fn)
    label_fn = get_label_filename(did)
    image_path = os.path.join(img_path, img_fn)
    label_path = os.path.join(label_path, label_fn)
    if not os.path.exists(label_path): return
    add_to_set.append({
        "image": image_path,
        "label": label_path
    })

validation_prob = 0.2
fns = [fn for fn in os.listdir(img_path) if fn.endswith('.nii')]
train_fns = [fn for fn in fns if img_has_label(label_path, fn)]
test_fns = [fn for fn in fns if not img_has_label(label_path, fn)]
train_fns, val_fns = np.split(train_fns, (np.array([0.8])*len(train_fns)).astype(int))

for fn in train_fns:
    add_to_set = base_json["training"]
    add_img_to_path(add_to_set, img_path, label_path, fn)

for fn in val_fns:
    add_to_set = base_json["validation"]
    add_img_to_path(add_to_set, img_path, label_path, fn)

for fn in test_fns:
    base_json["test"].append(os.path.join(img_path, fn))

with open(dst_path, 'w') as fp:
    json.dump(base_json, fp, indent=2)