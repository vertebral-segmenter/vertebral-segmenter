"""Given a directory containing all pretraining images, create the json for train/val split."""

import os
import numpy as np

data_dir = "data" # Modify this

files = os.listdir(data_dir)
training = []
validation = []

for f in files:
    p = np.random.random(1)[0]
    if p < 0.3:
        validation.append(f)
    else: training.append(f)

prefix = f"{data_dir}/"

result = {
    "training": [{"image": prefix+f} for f in training],
    "validation": [{"image": prefix+f} for f in training],
}

import json
with open('dataset.json', 'w') as fp:
    json.dump(result, fp, indent=2)