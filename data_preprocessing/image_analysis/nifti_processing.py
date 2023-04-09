from concurrent.futures import ThreadPoolExecutor
import numpy as np
import nibabel as nib
from scipy import ndimage
from scipy.ndimage import zoom
from nibabel.processing import resample_to_output
import pydicom
import os
from tqdm import tqdm


def load_nifti(file_path):
    img = nib.load(file_path)
    data = img.get_fdata()
    return img, data


def save_nifti(data, img, file_path):
    new_img = nib.Nifti1Image(data, img.affine, img.header)
    nib.save(new_img, file_path)


def process_segmentation_image(segmentation_image):
    """
    Process a segmentation image by checking if it contains only 0s and 1s. If not,
    set all non-zero values to 1 and change the data type to binary (boolean).

    Args:
        segmentation_image (nibabel.nifti1.Nifti1Image): A 3D segmentation image.

    Returns:
        nibabel.nifti1.Nifti1Image: Processed segmentation image with binary data type.
    """
    # Convert the segmentation image to a NumPy array
    segmentation_array = segmentation_image.get_fdata()

    # Check if the image contains only 0s and 1s
    unique_values = np.unique(segmentation_array)
    if not np.array_equal(unique_values, [0, 1]):
        # Set all non-zero values to 1
        segmentation_array[segmentation_array != 0] = 1

    # Convert the data type to binary (boolean)
    binary_array = segmentation_array.astype(np.bool)

    # Update the header with the new data type
    new_header = segmentation_image.header.copy()
    new_header.set_data_dtype(np.uint8)

    # Create a new NIfTI image with the binary data, the original image's affine transformation, and the updated header
    binary_image = nib.Nifti1Image(binary_array, segmentation_image.affine, header=new_header)

    return binary_image


def clip_nifti_image(img, lower_bound=-1000, upper_bound=10000):
    """
    Clip the scalar data of a NIfTI image between specified lower and upper bounds.

    Args:
        img (nibabel.nifti1.Nifti1Image): Input NIfTI image.
        lower_bound (int, optional): Lower bound for clipping. Default is -1000.
        upper_bound (int, optional): Upper bound for clipping. Default is 10000.

    Returns:
        nibabel.nifti1.Nifti1Image: Clipped NIfTI image.
    """
    # Extract the image data array
    img_data = img.get_fdata()

    # Clip the image data between the specified lower and upper bounds
    clipped_data = img_data.clip(lower_bound, upper_bound)

    # Create a new NIfTI image with the clipped data and the original image's affine transformation
    clipped_img = nib.Nifti1Image(clipped_data, img.affine)

    return clipped_img


def compute_volume(segmentation_image):
    """
    Compute the volume of the L2 vertebrae segmentation from a 3D binary image.

    Args:
        segmentation_image (nibabel.nifti1.Nifti1Image): A 3D binary image with L2 vertebrae segmentation.

    Returns:
        float: The volume of the L2 vertebrae in cubic millimeters.
    """
    # Get the image's spacing (voxel size) in millimeters
    spacing = segmentation_image.header.get_zooms()

    # Calculate the volume of a single voxel in cubic millimeters
    voxel_volume = spacing[0] * spacing[1] * spacing[2]

    # Convert the segmentation image to a NumPy array
    segmentation_array = segmentation_image.get_fdata()

    # Count the number of voxels with a value of 1 (L2 vertebrae segmentation)
    segmented_voxels = (segmentation_array != 0).sum()

    # Calculate the volume of the L2 vertebrae in cubic millimeters
    volume = segmented_voxels * voxel_volume

    return volume


def crop_roi(data, centroid, size):
    z_min = int(max(0, centroid[0] - size[0] // 2))
    z_max = int(min(data.shape[0], centroid[0] + size[0] // 2))
    y_min = int(max(0, centroid[1] - size[1] // 2))
    y_max = int(min(data.shape[1], centroid[1] + size[1] // 2))
    x_min = int(max(0, centroid[2] - size[2] // 2))
    x_max = int(min(data.shape[2], centroid[2] + size[2] // 2))
    return data[z_min:z_max, y_min:y_max, x_min:x_max]


def rescale_data(data, new_min, new_max):
    old_min, old_max = np.min(data), np.max(data)
    normalized_data = (data.astype(np.float32) - old_min) / (old_max - old_min)
    new_data = normalized_data * (new_max - new_min) + new_min
    return new_data


def zoom_image(data, affine, new_spacing):
    old_spacing = np.diag(affine)[:3]
    zoom_factors = 10 * (old_spacing / new_spacing)
    resampled_data = zoom(data, zoom_factors, order=5)
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


def change_dtype(input_img, output_dtype='int16'):
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


def resize_and_resample_nifti(input_img, scale_factor=0.5, desired_spacing=None, order=3):
    # Get the image data, affine transformation, and spacing
    input_data = input_img.get_fdata()
    input_affine = input_img.affine
    image_header = input_img.header

    if scale_factor != -1:
        # Compute the zoom factors for resizing and resampling
        resize_factors = [scale_factor] * 3

        # Resize the image data
        resized_data = zoom(input_data, resize_factors, order=order)  # Linear interpolation (order=1)
        resized_img = nib.Nifti1Image(resized_data, input_affine)
    else:
        resized_img = input_img

    if desired_spacing is None:
        desired_spacing = image_header.get_zooms()

    # Resample the resized image data to the desired spacing
    resampled_img = resample_nifti_img(resized_img, desired_spacing, order)  # Linear interpolation (order=1)

    # Return the new scaled and resampled NIfTI image
    return resampled_img


def load_dicom_file(filepath):
    return pydicom.dcmread(filepath)


def load_dicom_files(folder):
    dicom_files = []
    for file in os.listdir(folder):
        if file.endswith(".DCM;1"):
            dicom_files.append(pydicom.dcmread(os.path.join(folder, file)))
    dicom_files.sort(key=lambda x: float(x.ImagePositionPatient[2]))
    return dicom_files


def load_dicom_files_parallel(folder):
    filepaths = [os.path.join(folder, file) for file in os.listdir(folder) if file.endswith(".DCM;1")]

    with ThreadPoolExecutor() as executor:
        dicom_files = list(
            tqdm(executor.map(load_dicom_file, filepaths), total=len(filepaths), desc="Loading DICOM files"))

    dicom_files.sort(key=lambda x: float(x.ImagePositionPatient[2]))
    return dicom_files


def dicom_to_nifti(dicom_files):
    dimensions = (int(dicom_files[0].Rows), int(dicom_files[0].Columns), len(dicom_files))
    voxel_sizes = (float(dicom_files[0].PixelSpacing[0]), float(dicom_files[0].PixelSpacing[1]),
                   float(dicom_files[1].ImagePositionPatient[2]) - float(dicom_files[0].ImagePositionPatient[2]))

    nifti_data = np.zeros(dimensions, dtype=np.int16)
    for i, dicom_file in enumerate(tqdm(dicom_files, desc="Converting to NIfTI")):
        nifti_data[:, :, i] = dicom_file.pixel_array

    nifti_affine = np.diag(voxel_sizes + (1,))
    nifti_image = nib.Nifti1Image(nifti_data, nifti_affine)

    return nifti_image


def main():
    input_folder = r"T:\CIHR Data\3) MicroCT\800-series\850\850_T13,L1,L2,L3,L4_ZR75_Untreated_MicroCT_2836"
    output_filename = os.path.normpath(
        r"D:\vertebral-segmentation-rat-l2\data_preprocessing\\" + os.path.basename(input_folder) + ".nii").replace(
        '\\', '/')
    print(output_filename)

    # dicom_files = load_dicom_files(input_folder)
    # print('DICOM files loaded...')

    dicom_files_parr = load_dicom_files_parallel(input_folder)
    print('DICOM files loaded...')

    nifti_image = dicom_to_nifti(dicom_files_parr)
    scaled_image = resize_and_resample_nifti(nifti_image, scale_factor=0.2)
    nib.save(scaled_image, output_filename)
    print('DICOM to Nifti converted...')


if __name__ == "__main__":
    main()

# if __name__ == '__main__':
#     nifti_file = r"T:\S@leh\Rat_mCT_new\frac_917_scan_cropped.nii"
#     input_img = nib.load(nifti_file)
#     input_data = input_img.get_fdata()
#     input_affine = input_img.affine
#     scale_factor = 0.9
#     resize_factors = [scale_factor] * 3
#     resized_data = zoom(input_data, resize_factors, order=5)  # Linear interpolation (order=1)
#     resized_img = nib.Nifti1Image(resized_data, input_affine)
#     nib.save(convert_nifti_to_dtype(resized_img, output_dtype='int16'), nifti_file)
#
#     print(f"{nifti_file} resized...")
