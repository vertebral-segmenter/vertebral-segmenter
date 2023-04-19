# Check opening errors and image sampling rate
import os
import nibabel as nib
import numpy as np
import re

from data_preprocessing.image_analysis.data_extraction import setup_logger, scan_number
from data_preprocessing.image_analysis.nifti_processing import convert_nii_gz_to_nii, change_dtype, resample_nifti_img, \
    read_nifti_info, compute_volume, compute_scale_factor, resize_nifti_image

scans_preprocessing_folder = "D:\Rat_mCT_v1\pre-processing"
scans_finetuning_folder = "D:\Rat_mCT_v1\finetuning\scans"
labels_finetuning_folder = "D:\Rat_mCT_v1\finetuning\labels"

scans_folder = r"D:\Rat_mCT_v1\scans"
labels_folder = r"D:\Rat_mCT_v1\labels"
desired_scan_dtype = 'int16'
desired_label_dtype = 'uint8'
desired_spacing = np.array([0.035, 0.035, 0.035])
scan_data_lb = -1000
scan_data_ub = 10000
flag = False

logger = process_logger = setup_logger('process', 'logs/process.log')

# List all files in the scans and labels folders
scan_files = os.listdir(scans_folder)
label_files = os.listdir(labels_folder)

# Initialize an empty list to store label volumes
label_volumes = []


def get_serie(scan_num):
    return int(str(scan_num)[0])


# Iterate through the label files
for label_file in label_files:
    # Check if the file has a .nii extension (case-insensitive)
    if label_file.lower().endswith('.nii'):
        label_path = os.path.join(labels_folder, label_file)
        label_num = scan_number(label_file)

        scan_file = next((f for f in scan_files if f.startswith(f"{label_num}_")), None)
        if not scan_file:
            logger.error(f"There is a segmentation {label_num} without scan: {label_file}")
            # os.remove(label_path)
            logger.info(f"{label_file} is removed.")
            continue

        # Load the label file as a NIfTI image
        try:
            label_img = nib.load(label_path)
        except Exception as e:
            logger.error(f"Error loading label file {label_num}: {label_file} - {str(e)}")
            # If there's an error loading the file, remove it and log the error
            # os.remove(label_path)
            continue

        # Compute the volume of the label file
        volume = compute_volume(label_img)
        # Add the computed volume to the list of label volumes
        label_volumes.append(volume)

        # Log the computed volume for the current label file
        # logger.info(
        #     f"Volume {label_num}: {compute_volume(label_img)} {compute_scale_factor(label_img, desired_volume=90, volume_epsilon=3)}")

# Find the maximum and minimum label volumes
max_label_volume = max(label_volumes)
min_label_volume = min(label_volumes)

# Log the maximum and minimum label volumes
logger.info(f"Max label volume: {max_label_volume}")
logger.info(f"Min label volume: {min_label_volume}")

for scan_file in scan_files:
    # Check if the file has a .nii extension (case-insensitive)
    if scan_file.lower().endswith('.nii'):
        scan_path = os.path.join(scans_folder, scan_file)
        scan_num = scan_number(scan_file)
        # Find the corresponding label file
        label_file = next((f for f in label_files if f.startswith(f"{scan_num}_")), None)
        if label_file:
            flag = True
            label_path = os.path.join(labels_folder, label_file)
            label_img = nib.load(label_path)

        try:
            # Load the scan file as a NIfTI image
            scan_img = nib.load(scan_path)
        except:
            # If there's an error loading the file, remove it and log the error
            # os.remove(scan_path)
            if flag:
                # os.remove(label_path)
                logger.error(f"Error loading scan file {scan_num}: {label_file} removed.")
                flag = False
            logger.error(
                f"Error loading scan file {scan_num}: {scan_file}{f' and {label_file} ' if flag else ' '}removed.")
            continue

        if flag:
            print(scan_img.shape, label_img.shape)
            scale_factor = compute_scale_factor(label_img, desired_volume=105 if get_serie(scan_num) == 2 else 90)
            if scale_factor:
                scan_img = resize_nifti_image(scan_img, factor_size=scale_factor)
                label_img = resize_nifti_image(label_img, factor_size=scale_factor, no_range_change=False, order=1)
                logger.info(f"Resized scan and segmentation {scan_num} images with scale factor: {scale_factor} to "
                            f"new shape: {scan_img.shape}")
                print(scan_img.shape, label_img.shape)
        # Read NIfTI image information
        scan_data = read_nifti_info(scan_img)

        # Check if the image spacing is as desired
        if np.abs(np.array(scan_data['image_spacing']) - desired_spacing).sum() > 0.01:
            logger.error(f"Undesired image spacing for scan {scan_num} file: {scan_data['image_spacing']}")
            continue

        # Check if the data type of the scan is as desired
        if scan_data['scalar_dtype'] != desired_scan_dtype:
            logger.warning(
                f"Undesired data type for scan {scan_num} file: {scan_data['scalar_dtype']} | Data type changed to {desired_scan_dtype}")
            scan_img = change_dtype(scan_img, new_dtype=desired_scan_dtype)

        # Check if the file size is larger than 150 MB
        scan_size_mb = round(os.stat(scan_path).st_size / (1024 * 1024))
        if 150.0 < scan_size_mb:
            logger.warning(f"Scan {scan_num} file size larger than 150 MB: {scan_size_mb}")

        # Check the scan scalar data range
        if scan_data['scalar_range'][0] != -1000 or scan_data['scalar_range'][1] != 10000:
            logger.warning(f"Scan {scan_num} scalar data range: {scan_data['scalar_range']}")

        # save data
        scan_dest_path = os.path.join(scans_preprocessing_folder, scan_file)
        nib.save(label_img, scan_dest_path)
        logger.info(f"Scan {scan_file} copied to folder: {scan_dest_path}")

        # Check if the label file exists
        if flag:
            label_data = read_nifti_info(label_img)

            # Check if the image dimensions are the same for both files
            if np.abs(np.array(scan_data['image_dimensions']) - np.array(
                    label_data['image_dimensions'])).sum() > 0.01:
                logger.error(f"Image dimensions not the same for {scan_num} files. "
                             f"Scan: {scan_data['image_dimensions']} - Segmentation: {label_data['image_dimensions']}")
                continue

            # Check if the image spacing for the label file is as desired
            if np.abs(np.array(label_data['image_spacing']) - desired_spacing).sum() > 0.01:
                logger.error(f"Undesired image spacing for segmentation {scan_num} file: {label_data['image_spacing']}")
                continue

            # Check if the data type of the label file is as desired
            if label_data['scalar_dtype'] != desired_label_dtype:
                logger.warning(
                    f"Undesired data type for segmentation {scan_num} file: {label_data['scalar_dtype']} | Data type changed to {desired_label_dtype}")
                scan_img = change_dtype(scan_img, new_dtype=desired_label_dtype)

            # Check if the label file size is larger than 150 MB
            label_size_mb = round(os.stat(label_path).st_size / (1024 * 1024))
            if 150.0 < label_size_mb:
                logger.warning(f"Segmentation {scan_num} file size larger than 150 MB: {label_size_mb}")

            # Check if the image origins are the same for both files
            if np.abs(np.array(scan_data['image_origin']) - np.array(label_data['image_origin'])).sum() > 0.01:
                logger.warning(f"Image origins not the same for {scan_num} files. "
                               f"Scan: {scan_data['image_origin']} - Segmentation: {label_data['image_origin']}")

                # Update the image origin if one of them is (0, 0, 0)
                if (scan_data['image_origin'] == 0).all():
                    new_affine = scan_img.affine
                    new_affine[:3, 3] = label_data['image_origin']
                    scan_img = nib.Nifti1Image(scan_img.get_fdata(), new_affine)
                elif (label_data['image_origin'] == 0).all():
                    new_affine = label_img.affine
                    new_affine[:3, 3] = scan_data['image_origin']
                    label_img = nib.Nifti1Image(label_img.get_fdata(), new_affine)
                else:
                    logger.error(f"Cannot make the image_origin the same for both scan and segmentation")
                    # continue

            if (label_data['ijk_to_ras_matrix'] != scan_data['ijk_to_ras_matrix']).any():
                logger.warning(f"Not the same ijk_to_ras matrix for {scan_num} files. "
                               f"scan: {scan_data['ijk_to_ras_matrix']} - segmentation: {label_data['ijk_to_ras_matrix']}")

            # save data
            label_dest_path = os.path.join(labels_finetuning_folder, label_file)
            nib.save(label_img, label_dest_path)
            logger.info(f"Segmentation {label_file} copied to folder: {label_dest_path}")

            scan_dest_path = os.path.join(scans_finetuning_folder, scan_file)
            nib.save(label_img, scan_dest_path)
            logger.info(f"Scan {scan_file} copied to folder: {scan_dest_path}")


