import os
import numpy as np
import re
import nibabel as nib
from scipy.ndimage import zoom

# def parse_seg_amira_header(amira_file):
#     with open(amira_file, 'rb') as f:
#         header = b""
#         while True:
#             line = f.readline()
#             if line.startswith(b"@1"):
#                 break
#             header += line
#     header = header.decode('utf-8', errors='ignore')
#
#     pattern = r"Lattice\s+(\d+)\s+(\d+)\s+(\d+)"
#     match = re.search(pattern, header)
#     dims = [int(match.group(i)) for i in range(1, 4)]
#
#     pattern = r"BoundingBox\s+([\d\.\-e]+)\s+([\d\.\-e]+)\s+([\d\.\-e]+)\s+([\d\.\-e]+)\s+([\d\.\-e]+)\s+([\d\.\-e]+)"
#     match = re.search(pattern, header)
#     bbox = [float(match.group(i)) for i in range(1, 7)]
#
#     pattern = r'Content\s+"(?:\d+x\d+x\d+\s+)?(\w+).*'
#     match = re.search(pattern, header)
#
#     if match is None:
#         print("Header:")
#         print(header)
#         print("Pattern:")
#         print(pattern)
#         raise ValueError("Data type not found in header. Please check the input file.")
#
#     data_type = match.group(1)
#
#     return dims, bbox, data_type


def parse_scan_amira_header(amira_file):
    with open(amira_file, 'rb') as f:
        header = b""
        while True:
            line = f.readline()
            if line.startswith(b"@1"):
                break
            header += line
    header = header.decode('utf-8', errors='ignore')

    pattern = r"Lattice\s+(\d+)\s+(\d+)\s+(\d+)"
    match = re.search(pattern, header)
    dims = [int(match.group(i)) for i in range(1, 4)]

    pattern = r"BoundingBox\s+([\d\.\-e]+)\s+([\d\.\-e]+)\s+([\d\.\-e]+)\s+([\d\.\-e]+)\s+([\d\.\-e]+)\s+([\d\.\-e]+)"
    match = re.search(pattern, header)
    bbox = [float(match.group(i)) for i in range(1, 7)]

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
    print(amira_dtype)
    return amira_dtype_to_numpy_dtype.get(amira_dtype.lower(), None)


def find_data_start_position(amira_file):
    with open(amira_file, 'rb') as f:
        while True:
            line = f.readline()
            if line.startswith(b"@1"):
                data_start_position = f.tell()
                break
    return data_start_position


# def read_amira_data_binary_labels(amira_file, dims):
#     data_points = np.prod(dims)
#     with open(amira_file, 'rb') as f:
#         # Read until the data section
#         while True:
#             line = f.readline()
#             if line.startswith(b"@1"):
#                 break
#
#         # Read the data and convert each byte to a sequence of binary values
#         byte_data = np.frombuffer(f.read(), dtype=np.uint8)
#         bit_data = np.unpackbits(byte_data)
#         data = bit_data[:data_points].reshape(dims)
#
#     return data


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


def rescale_data(data, new_min, new_max):
    old_min, old_max = np.min(data), np.max(data)
    normalized_data = (data.astype(np.float32) - old_min) / (old_max - old_min)
    new_data = normalized_data * (new_max - new_min) + new_min
    return new_data


def resample_image(data, affine, new_spacing):
    old_spacing = np.diag(affine)[:3]
    zoom_factors = 10 * (old_spacing / new_spacing)
    print(zoom_factors)
    resampled_data = zoom(data, zoom_factors, order=5)  # Linear interpolation (order=1)
    new_affine = np.copy(affine)
    np.fill_diagonal(new_affine, np.append(new_spacing, 1))
    return resampled_data, new_affine


def convert_amira_to_nifti(amira_file, nifti_file, desired_spacing=(0.035, 0.035, 0.035)):
    dims, bbox, data_type = parse_scan_amira_header(amira_file)
    data = read_amira_data(amira_file, dims, data_type)

    # Calculate the spacing and create the affine transformation matrix
    spacing = [abs((bbox[i + 3] - bbox[i]) / (dims[i] - 1)) for i in range(3)]
    affine = np.diag(spacing + [1])
    affine[:-1, -1] = bbox[:3]

    # Create the NIfTI image
    rescaled_data = rescale_data(data, -512, 512).astype(np.int16)

    # Resample the image data to the desired spacing
    resampled_data, new_affine = resample_image(rescaled_data, affine, desired_spacing)

    nifti_img = nib.Nifti1Image(resampled_data, new_affine)
    nifti_img = nib.as_closest_canonical(nifti_img)

    # Save the NIfTI image
    nib.save(nifti_img, nifti_file)


if __name__ == '__main__':
    # file_type = 'scan'
    amira_file = r"T:\\S@leh\\CIHR data (Rat_mCT)\\700-Series\\714_L2_Healthy_Untreated_Stereo\\Mik_Rat714_L1-L3 (Saved in 4 - Resampled).am"
    # amira_file = "T:\\S@leh\\CIHR data (Rat_mCT)\\700-Series\\714_L2_Healthy_Untreated_Stereo\\Rat 714 Verte segmentation - Geoff.am"
    nifti_file = 'D:\\vertebral-segmentation-rat-l2\\data_preprocessing\\' + os.path.splitext(os.path.basename(amira_file))[0] + '.nii'
    convert_amira_to_nifti(amira_file, nifti_file)

    nifti_img = nib.load(nifti_file)
    nifti_data = nifti_img.get_fdata()
    print(f"NIfTI data min: {nifti_data.min()}, NIfTI data max: {nifti_data.max()}")
    print("converted...")
