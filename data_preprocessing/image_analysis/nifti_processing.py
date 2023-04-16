from concurrent.futures import ThreadPoolExecutor
import numpy as np
import nibabel as nib
from scipy import ndimage
from scipy.ndimage import zoom
from nibabel.processing import resample_to_output
import pydicom
import os
from tqdm import tqdm
import nrrd


def nrrd_header_to_nifti_affine(header):
    """
    Convert an NRRD header's information to a NIfTI affine matrix.

    Args:
        header (dict): The header of an NRRD file.

    Returns:
        numpy.ndarray: A 4x4 affine matrix for a NIfTI image.
    """
    spacings = header.get('space directions', np.eye(3))

    # Ensure spacings has the correct shape (3, 3) and remove first nan row
    if spacings.shape != (3, 3):
        spacings = spacings[1:]

    spacings = np.vstack([spacings, np.zeros(3)])
    origin = header.get('space origin', np.zeros(3))
    origin = np.append(origin, 1)
    affine = np.column_stack((spacings, origin))

    return affine


def load_nrrd_as_nifti(src_file):
    """
    Load an NRRD file and return it as a Nifti1Image object.

    Args:
        src_file (str): The path to the NRRD file.

    Returns:
        nibabel.nifti1.Nifti1Image: A 3D image loaded from the NRRD file.
    """
    # Load the NRRD file
    data, header = nrrd.read(src_file)

    # Create a new NIfTI image with the data and the affine transformation derived from the NRRD header
    nifti_img = nib.Nifti1Image(data, nrrd_header_to_nifti_affine(header))

    return nifti_img


def load_as_nifti(file_path):
    if file_path.endswith('.nii'):
        nifti_img = nib.load(file_path)
    elif file_path.endswith('.nrrd'):
        nifti_img = load_nrrd_as_nifti(file_path)
    else:
        raise Exception("Error! data can't be loaded...")
    return nifti_img


def save_nifti(data, img, file_path):
    new_img = nib.Nifti1Image(data, img.affine, img.header)
    nib.save(new_img, file_path)


def process_scan_image(scan_image, new_dtype='int16', clip_min=-1000, clip_max=10000):
    """
    Process a scanned image by converting its data type and clipping its intensity values.

    Args:
        scan_image (nibabel.nifti1.Nifti1Image): Input scanned image to be processed.
        new_dtype (str, optional): The target data type for the converted image, default is 'int16'.
        clip_min (int, optional): The minimum intensity value for clipping, default is -1000.
        clip_max (int, optional): The maximum intensity value for clipping, default is 10000.

    Returns:
        nibabel.nifti1.Nifti1Image: The processed image with the specified data type and clipped intensity values.
    """

    # Clip the intensity values of the converted image within the specified bounds
    clipped_image = clip_nifti_image(scan_image, lower_bound=clip_min, upper_bound=clip_max)

    # Convert the input image to the specified data type
    converted_image = change_dtype(clipped_image, new_dtype)

    return converted_image


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

    if len(segmentation_array.shape) == 3:
        # Set all non-zero values to 1
        mask = (segmentation_array != 0)
    elif len(segmentation_array.shape) == 4:
        # Create a boolean mask for segmentation_array[0] and segmentation_array[1] where the value is 1
        mask = (segmentation_array[0] == 1) | (segmentation_array[1] == 1)
    else:
        raise Exception(
            f"Error! The loaded segmentation doesn't have appropriate shape! shape: {segmentation_array.shape}")

    # Create the binary_array with the desired shape (331, 429, 256)
    binary_array = np.zeros(mask.shape, dtype=np.uint8)

    # Apply the mask to update the binary_array with 1s where the mask is True
    binary_array[mask] = 1

    # Update the header with the new data type
    new_header = segmentation_image.header.copy()
    new_header.set_data_dtype(np.uint8)

    # Create a new NIfTI image with the binary data, the original image's affine transformation, and the updated header
    binary_image = nib.Nifti1Image(binary_array, segmentation_image.affine, header=new_header)

    return binary_image


def clip_nifti_image(img, lower_bound, upper_bound):
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


def rescale_array(data, new_min, new_max, dtype=None):
    data = data.astype(np.float64)  # Convert to float64 to avoid overflow issues
    data_min = np.min(data)
    data_max = np.max(data)

    rescaled_data = new_min + (data - data_min) * (new_max - new_min) / (data_max - data_min)

    if dtype is not None:
        rescaled_data = rescaled_data.astype(dtype)  # Convert back to the desired data type

    return rescaled_data


def zoom_image(data, affine, new_spacing):
    old_spacing = np.diag(affine)[:3]
    zoom_factors = old_spacing / new_spacing
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


def change_dtype(input_img, new_dtype):
    # Read the image data
    input_data = input_img.get_fdata()

    # Check if the data type is the same as output_dtype
    if input_img.get_data_dtype() == new_dtype:
        return input_img

    # Convert the input data to the desired output data type
    converted_data = input_data.astype(new_dtype)

    # Update the header to reflect the new data type
    new_header = input_img.header.copy()
    new_header.set_data_dtype(new_dtype)

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


def rescale_nifti_image(data, affine, new_dims):
    # Calculate rescale factors
    old_dims = np.array(data.shape)
    rescale_factors = new_dims / old_dims

    # Rescale the image data
    rescaled_data = zoom(data, rescale_factors)

    # Calculate the new affine matrix
    new_affine = affine.copy()
    new_affine[:3, :3] = affine[:3, :3] * (1 / rescale_factors)

    return rescaled_data, new_affine


def main():
    # input_folder = r"T:\CIHR Data\3) MicroCT\800-series\850\850_T13,L1,L2,L3,L4_ZR75_Untreated_MicroCT_2836"
    # output_filename = os.path.normpath(
    #     r"D:\vertebral-segmentation-rat-l2\data_preprocessing\\" + os.path.basename(input_folder) + ".nii").replace(
    #     '\\', '/')
    # print(output_filename)
    #
    # # dicom_files = load_dicom_files(input_folder)
    # # print('DICOM files loaded...')
    #
    # dicom_files_parr = load_dicom_files_parallel(input_folder)
    # print('DICOM files loaded...')
    #
    # nifti_image = dicom_to_nifti(dicom_files_parr)
    # scaled_image = resize_and_resample_nifti(nifti_image, scale_factor=0.2)
    # nib.save(scaled_image, output_filename)
    # print('DICOM to Nifti converted...')
    nifti_img = load_as_nifti(
        r"T:\CIHR Data\16) Stereology\1100-Series\1123_L2_Healthy_Untreated_Stereology\1123_L2_Trabecular_Segmentation.seg.nrrd")
    processed_img = process_segmentation_image(nifti_img)
    nib.save(processed_img, r'D:\vertebral-segmentation-rat-l2\data_preprocessing\file.nii')


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
