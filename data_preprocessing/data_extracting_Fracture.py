# import required module
import math
import os
import shutil
import nibabel as nib
from data_preprocessing.image_analysis.nifti_processing import convert_nii_gz_to_nii, change_dtype, \
    resample_nifti_img, resize_and_resample_nifti
from scipy.ndimage import zoom

# assign directories
src_path = r"T:\AlliT\Code\fracture_prediction\pix2pixtrain\images"
dst_path = r"T:\S@leh\Rat_mCT_new"


# iterate over files in the source directory (fractured serie)
for root, dirs, files in os.walk(src_path):
    if 'L2' in root and 'Fractured' in root and files:
        for file in files:
            if (file.lower().endswith('.nii.gz') and not any([list_file.lower().endswith('.nii') for list_file in os.listdir(root)]))\
                    or file.lower().endswith('.nii'):
                src_file = os.path.join(root, file)
                if file.lower().endswith('.nii.gz'):
                    convert_nii_gz_to_nii(src_file, src_file[:-3])
                    print(f"{file} data converted from .gz file.")
                    src_file = src_file[:-3]
                scan_number = [num for dir in src_file.split('\\') for num in dir.split('_') if num.isnumeric()][0]
                dst_file_name = f"frac_{scan_number}_scan_cropped.nii"
                if dst_file_name in os.listdir(dst_path):
                    print(f"# {dst_file_name} already exist in the dataset.")
                    continue
                dst_file = os.path.join(dst_path, dst_file_name)

                try:
                    # Save the resampled image as a new NIfTI file
                    input_img = nib.load(src_file)
                    desired_spacing = (0.035, 0.035, 0.035)
                    order = 5
                    file_size_mb = os.stat(src_file).st_size / (1024 * 1024)
                    if 150.0 < file_size_mb:
                        scale_factor = round(1/math.sqrt(file_size_mb/80), 2)
                        resampled_img = resize_and_resample_nifti(input_img, scale_factor, desired_spacing, order)
                    else:
                        resampled_img = resample_nifti_img(input_img, desired_spacing, order)
                    converted_img = change_dtype(resampled_img, output_dtype='int16')
                    nib.save(converted_img, dst_file)
                    print(f"{dst_file_name} data resampled and copied...")
                except:
                    print(f"## Error - Can't resample {src_file}.")



# if __name__ == '__main__':
#     input_img = nib.load(r"T:\AlliT\Code\fracture_prediction\pix2pixtrain\images\1003_L1,L2,L3_Healthy_ZA_Microloading\Fractured\AT_CIHR_Microloading_1003_Fractured_resampled.nii")
#     scale_factor = 0.7
#     desired_spacing = (0.035, 0.035, 0.035)
#     converted_img = convert_nifti_to_dtype(resize_and_resample_nifti(input_img, scale_factor, desired_spacing))
#     nib.save(converted_img, r'D:\\vertebral-segmentation-rat-l2\\data_preprocessing\\AT_CIHR_Microloading_1003_Fractured.nii')
#
#     print(f"data converted...")