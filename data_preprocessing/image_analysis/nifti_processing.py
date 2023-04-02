import os
import numpy as np
import nibabel as nib
from scipy.ndimage import zoom
from nibabel.processing import resample_to_output


def rescale_data(data, new_min, new_max):
    old_min, old_max = np.min(data), np.max(data)
    normalized_data = (data.astype(np.float32) - old_min) / (old_max - old_min)
    new_data = normalized_data * (new_max - new_min) + new_min
    return new_data

def zoom_image(data, affine, new_spacing):
    old_spacing = np.diag(affine)[:3]
    zoom_factors = 10 * (old_spacing / new_spacing)
    resampled_data = zoom(data, zoom_factors, order=5)  # Linear interpolation (order=1)
    new_affine = np.copy(affine)
    np.fill_diagonal(new_affine, np.append(new_spacing, 1))
    return resampled_data, new_affine


def convert_nii_gz_to_nii(gz_file_path, nii_file_path):
    # Check if the loaded file has the '.nii.gz' extension
    if not gz_file_path.lower().endswith('.nii.gz'):
        print(f"{gz_file_path} is not a .nii.gz file.")
        return

    nii_gz_file = nib.load(gz_file_path)
    nib.save(nii_gz_file, nii_file_path)

def resample_nifti_img(input_img, new_spacing=(0.035, 0.035, 0.035), order=3):
    # Resample the image to the desired spacing
    resampled_img = resample_to_output(input_img, new_spacing, order)

    # Return the new resampled NIfTI image
    return resampled_img


def convert_nifti_to_dtype(input_img, output_dtype='int16'):
    # Read the image data
    input_data = input_img.get_fdata()

    # Check if the data type is the same as output_dtype
    if input_img.get_data_dtype() == output_dtype:
        return input_img

    # Convert the input data to the desired output data type
    converted_data = input_data.astype(output_dtype)

    # Update the header to reflect the new data type
    new_header = input_img.header.copy()
    new_header.set_data_dtype(output_dtype)

    # Create a new NIfTI image with the converted data and the same affine transformation as the input image
    converted_img = nib.Nifti1Image(converted_data, input_img.affine, new_header)

    # Return the new dtype NIfTI image
    return converted_img


def resize_and_resample_nifti(input_img, scale_factor=0.5, desired_spacing=(0.035, 0.035, 0.035), order=3):
    # Get the image data, affine transformation, and spacing
    input_data = input_img.get_fdata()
    input_affine = input_img.affine

    # Compute the zoom factors for resizing and resampling
    resize_factors = [scale_factor] * 3

    # Resize the image data
    resized_data = zoom(input_data, resize_factors, order=order)  # Linear interpolation (order=1)
    resized_img = nib.Nifti1Image(resized_data, input_affine)

    # Resample the resized image data to the desired spacing
    resampled_img = resample_nifti_img(resized_img, desired_spacing, order)  # Linear interpolation (order=1)

    # Return the new scaled and resampled NIfTI image
    return resampled_img

if __name__ == '__main__':
    pass
