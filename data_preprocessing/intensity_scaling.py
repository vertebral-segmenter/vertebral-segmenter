"""Scale the intensity of voxels to between -1000 to 1000"""

import os
import numpy as np
import nibabel as nib
import shutil


def scale_image(src_file, dst_file, b_min, b_max):
    nii = nib.load(src_file)
    data = nii.get_fdata()
    a_min = np.min(data)
    a_max = np.max(data)
    data = ((data - a_min) / (a_max - a_min)) * (b_max - b_min) + b_min
    scaled_nii = nib.Nifti1Image(data, nii.affine, nii.header)
    nib.save(scaled_nii, dst_file)

def classify_label(src_file, dst_file):
    nii = nib.load(src_file)
    data = nii.get_fdata()
    data[data > 0.5] = 1
    data[data <= 0.5] = 0
    scaled_nii = nib.Nifti1Image(data, nii.affine, nii.header)
    nib.save(scaled_nii, dst_file)

def pretrain_main():
    data_dir = "pretrain/data" # Modify this
    target_dir = "pretrain/data_scaled"

    # target scale range
    b_min = -1000
    b_max = 1000

    if not os.path.exists(target_dir):
        os.makedirs(target_dir)

    files = os.listdir(data_dir)
    err_files = []

    for f in files:
        try:
            src_file = os.path.join(data_dir, f)
            dst_file = os.path.join(target_dir, f)
            scale_image(src_file, dst_file, b_min, b_max)
        except:
            err_files.append(f)
            continue
    print("Image with invalid format:", err_files)

def finetune_main():
    data_dir = "finetune/data/scans" # Modify this
    target_dir = "finetune/data_scaled/scans"

    label_dir = "finetune/data/labels"
    target_label_dir = "finetune/data_scaled/labels"

    # target scale range
    b_min = -1000
    b_max = 1000

    if not os.path.exists(target_dir):
        os.makedirs(target_dir)
    if not os.path.exists(target_label_dir):
        os.makedirs(target_label_dir)

    err_files = []

    for f in os.listdir(data_dir):
        try:
            src_file = os.path.join(data_dir, f)
            dst_file = os.path.join(target_dir, f)
            scale_image(src_file, dst_file, b_min, b_max)
        except:
            err_files.append(f)
            continue
    for f in os.listdir(label_dir):
        try:
            src_file = os.path.join(label_dir, f)
            classify_label(src_file, dst_file)
            dst_file = os.path.join(target_label_dir, f)
        except:
            err_files.append(f)
            continue
    print("Image with invalid format:", err_files)

if __name__ == "__main__":
    # pretrain_main()
    finetune_main()