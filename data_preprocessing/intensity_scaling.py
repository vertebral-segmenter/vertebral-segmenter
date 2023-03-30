"""Scale the intensity of voxels to between -1000 to 1000"""

import os
import numpy as np
import nibabel as nib

data_dir = "pretrain/data" # Modify this
target_dir = "pretrain/data_scaled"

# target scale range
b_min = -1000
b_max = 1000

if not os.path.exists(target_dir):
    os.makedirs(target_dir)

files = os.listdir(data_dir)
training = []
validation = []

err_files = []

for f in files:
    try:
        nii = nib.load(os.path.join(data_dir, f))
        data = nii.get_fdata()
        a_min = np.min(data)
        a_max = np.max(data)
        data = ((data - a_min) / (a_max - a_min)) * (b_max - b_min) + b_min
        scaled_nii = nib.Nifti1Image(data, nii.affine, nii.header)
        nib.save(scaled_nii, os.path.join(target_dir, f))
    except:
        err_files.append(f)
        continue

print("Image with invalid format:", err_files)
