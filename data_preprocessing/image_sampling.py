# Check image sampling rate
import os
import nibabel as nib
import numpy as np

src_dir = '/home/sherryyuan/rat_dataset'
img_dir = os.path.join(src_dir, 'images')
label_dir = os.path.join(src_dir, 'labels')

err_files = []
for fn in os.listdir(img_dir):
    f = os.path.join(img_dir, fn)
    try:
        nifti_img = nib.load(f)
    except:
        err_files.append(f)
        continue
    image_spacing = nifti_img.header.get_zooms()
    if np.abs(np.array(image_spacing) - 0.035).sum() > 0.01:
        print(image_spacing)
print(f)