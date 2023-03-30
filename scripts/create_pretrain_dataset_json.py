"""Given a directory containing all pretraining images, create the json for train/val split."""

import os
import numpy as np

data_dir = "pretrain/data_scaled" # Modify this

files = os.listdir(data_dir)
training = []
validation = []

for f in files:
    p = np.random.random(1)[0]
    if p < 0.2:
        validation.append(f)
    else:
        training.append(f)

prefix = f"{data_dir}/"

result = {
    "training": [{"image": prefix+f} for f in training],
    "validation": [{"image": prefix+f} for f in validation],
}

import json
with open('pretrain/jsons/dataset.json', 'w') as fp:
    json.dump(result, fp, indent=2)