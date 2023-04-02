# Check opening errors and image sampling rate
import os
import nibabel as nib
import numpy as np
from data_preprocessing.image_analysis.nifti_processing import convert_nii_gz_to_nii, convert_nifti_to_dtype, resample_nifti_img

src_path = r"T:\S@leh\Rat_mCT_new"
data_type = 'int16'
spacing = 0.035

err_files = []
err_spacing = {}
err_dtype = {}
err_size = {}
for file in os.listdir(src_path):
    if file.lower().endswith('.nii'):
        file_path = os.path.join(src_path, file)
        try:
            nifti_img = nib.load(file_path)
        except:
            err_files.append(file)
            os.remove(file_path)
            print(f'{file} removed from dataset.')
            continue

        image_spacing = nifti_img.header.get_zooms()
        if np.abs(np.array(image_spacing) - spacing).sum() > 0.01:
            err_spacing[file] = image_spacing

        if nifti_img.get_data_dtype() != data_type:
            err_dtype[file] = nifti_img.get_data_dtype()
            converted_img = convert_nifti_to_dtype(nifti_img, output_dtype=data_type)
            nib.save(converted_img, file_path)
            print(f'{file} dtype converted to {data_type}.')

        file_size_mb = round(os.stat(file_path).st_size / (1024 * 1024))
        if 150 < file_size_mb:
            err_size[file] = f'{file_size_mb}mb'


print(f'File opening errors: {err_files}')
print(f'File spacing errors: {err_spacing}')
print(f'File header dtype errors: {err_dtype}')
print(f'File size bigger than 150mb: {err_size}')