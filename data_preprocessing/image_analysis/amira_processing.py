import os
import numpy as np
import re
import nibabel as nib

from data_preprocessing.image_analysis.amira_binary_processing import read_amira, parse_binary_amira_header
from data_preprocessing.image_analysis.nifti_processing import rescale_array, zoom_image, rescale_nifti_image, \
    resize_and_resample_nifti


def read_amira_header(amira_file) -> str:
    with open(amira_file, 'rb') as f:
        header = b""
        while True:
            line = f.readline()
            if line.startswith(b"@1"):
                break
            header += line
    header = header.decode('utf-8', errors='ignore')

    return header


def parse_scan_amira_header(header: str):
    pattern = r"Lattice\s+(\d+)\s+(\d+)\s+(\d+)"
    match = re.search(pattern, header)
    dims = np.array([int(match.group(i)) for i in range(1, 4)])

    pattern = r"BoundingBox\s+([\d\.\-e]+)\s+([\d\.\-e]+)\s+([\d\.\-e]+)\s+([\d\.\-e]+)\s+([\d\.\-e]+)\s+([\d\.\-e]+)"
    match = re.search(pattern, header)
    bbox = np.array([float(match.group(i)) for i in range(1, 7)])

    pattern = r'Content\s+"(?:\d+x\d+x\d+\s+)?(\w+)'
    match = re.search(pattern, header)

    if match is None:
        print("Header:")
        print(header)
        print("Pattern:")
        print(pattern)
        raise ValueError("Data type not found in header. Please check the input file.")

    data_type = match.group(1)

    return dims, bbox, data_type


def numpy_dtype_from_amira_dtype(amira_dtype):
    amira_dtype_to_numpy_dtype = {
        "byte": np.int8,
        "short": np.int16,
        "ushort": np.uint16,
        "float": np.float32,
        "double": np.float64,
    }
    return amira_dtype_to_numpy_dtype.get(amira_dtype.lower(), None)


def find_data_start_position(amira_file):
    with open(amira_file, 'rb') as f:
        while True:
            line = f.readline()
            if line.startswith(b"@1"):
                data_start_position = f.tell()
                break
    return data_start_position


def read_amira_data(amira_file, dims, data_type):
    data_start_position = find_data_start_position(amira_file)
    data_points = dims[0] * dims[1] * dims[2]
    np_dtype = numpy_dtype_from_amira_dtype(data_type)
    if np_dtype is None:
        raise ValueError(f"Unsupported data type: {data_type}. Please check the input file.")

    data = np.fromfile(amira_file, dtype=np_dtype, offset=data_start_position, count=data_points)

    if data.size != data_points:
        raise ValueError(f"Expected {data_points} data points but read {data.size}. Please check the input file.")

    data = data.reshape(dims[::-1]).T
    return data


def create_affine_matrix(bbox, dims) -> np.ndarray:
    # Calculate the size of the bounding box
    bbox_size = bbox[1::2] - bbox[::2]

    # Calculate the spacing
    spacing = 10 * (bbox_size / (dims - 1))

    # Create the affine matrix
    affine = np.eye(4)
    affine[:3, :3] = np.diag(spacing)
    affine[:3, 3] = bbox[::2]

    return affine


def convert_amira_to_nifti(amira_file, lower_bound=-1000, upper_bound=10000):
    header = read_amira_header(amira_file)
    dims, bbox, data_type = parse_scan_amira_header(header)
    data = read_amira_data(amira_file, dims, data_type)

    # Create the affine matrix
    affine = create_affine_matrix(np.array(bbox), np.array(dims))

    # BoundingBox 0.000100017 0.7806 0.00103998 0.78854 0.00097996 1.40798,

    fix_point = 256
    rescaled_data = np.where(data >= 0, data % fix_point, data % -fix_point)
    rescaled_data = np.where(rescaled_data >= fix_point / 2, rescaled_data - fix_point, rescaled_data)
    rescaled_data = np.where(rescaled_data <= -fix_point / 2, rescaled_data + fix_point, rescaled_data)

    rescaled_data = rescale_array(rescaled_data, lower_bound, upper_bound, dtype='int16')

    # Create the NIfTI image
    nifti_img = nib.Nifti1Image(rescaled_data, affine)

    return nifti_img


def convert_binary_amira_to_nifti(amira_file, new_dims=np.array([224, 226, 403]),
                                  desired_spacing=(0.035, 0.035, 0.035)):
    # Calculate affine matrix using the function created earlier
    header = read_amira_header(amira_file)
    dims, bbox, data_type = parse_binary_amira_header(header)
    affine = create_affine_matrix(bbox, dims)
    data = read_amira(amira_file)

    dlist = data['data']
    merged = {}
    for row in dlist:
        merged.update(row)
    if 'data' not in merged:
        raise ValueError(f'Only binary .am files are supported')
    arr = merged['data']

    print(arr.shape)
    print(affine)
    resampled_array, new_affine = rescale_nifti_image(arr, affine, new_dims)
    # Create and save the NIfTI image
    print(resampled_array.shape)
    print(new_affine)
    nifti_img = nib.Nifti1Image(resampled_array, new_affine)
    nifti_img = resize_and_resample_nifti(nifti_img, scale_factor=-1, desired_spacing=desired_spacing)

    return nifti_img


def main():
    # file_type = 'scan'
    amira_path = r"T:\CIHR Data\16) Stereology\700-Series\701_L2_HELA_Untreated_Stereo\Mik_Rat701_L1-L3_volume.am"
    nifti_path = 'D:\\vertebral-segmentation-rat-l2\\data_preprocessing\\' + \
                 os.path.splitext(os.path.basename(amira_path))[0] + '.nii'
    nifti_img = convert_amira_to_nifti(amira_path)
    # save as nifti image
    nib.save(nifti_img, nifti_path)
    print(f"{nifti_path} converted...")

    # file_type = 'label'
    amira_path_label = r"T:\CIHR Data\16) Stereology\700-Series\708_L2_ACE1_Untreated_Stereo\VBCortAfterLevelSet1SegL2 (resampled and cropped).am"
    nifti_path_label = 'D:\\vertebral-segmentation-rat-l2\\data_preprocessing\\' + \
                       os.path.splitext(os.path.basename(amira_path_label))[0] + '.nii'
    nifti_img_label = convert_binary_amira_to_nifti(amira_path_label)
    # save as nifti image
    nib.save(nifti_img_label, nifti_path_label)
    print(f"{nifti_path_label} converted...")


if __name__ == '__main__':
    main()
