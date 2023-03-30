import os
import numpy as np
import re
import nibabel as nib

def parse_seg_amira_header(amira_file):
    pass

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


def convert_amira_to_nifti(amira_file, nifti_file, file_type='scan'):
    if file_type == 'scan':
        dims, bbox, data_type = parse_scan_amira_header(amira_file)
    elif file_type == 'segmentation':
        dims, bbox, data_type = parse_seg_amira_header(amira_file)
    data = read_amira_data(amira_file, dims, data_type)

    print(f"Data type: {data_type}")
    print(f"Data min: {data.min()}, Data max: {data.max()}")

    # Calculate the spacing and create the affine transformation matrix
    spacing = [(bbox[i + 3] - bbox[i]) / (dims[i] - 1) for i in range(3)]
    affine = np.diag(spacing + [1])
    affine[:-1, -1] = bbox[:3]

    # Create the NIfTI image
    rescaled_data = rescale_data(data, -512, 512).astype(np.int16)
    clipped_data = np.clip(-rescaled_data, -2000, None)
    nifti_img = nib.Nifti1Image(clipped_data, affine)
    nifti_img = nib.as_closest_canonical(nifti_img)

    # Save the NIfTI image
    nib.save(nifti_img, nifti_file)


if __name__ == '__main__':
    amira_file = "T:\\S@leh\\CIHR data (Rat_mCT)\\700-Series\\718_L2_Healthy_Untreated_Stereo\\Mik_Rat718_L1-L3 (saved in 4 - Resampled).am"
    nifti_file = 'D:\\vertebral-segmentation-rat-l2\\data_preprocessing\\' + os.path.splitext(os.path.basename(amira_file))[0] + '.nii'
    convert_amira_to_nifti(amira_file, nifti_file)

    nifti_img = nib.load(nifti_file)
    nifti_data = nifti_img.get_fdata()
    print(f"NIfTI data min: {nifti_data.min()}, NIfTI data max: {nifti_data.max()}")
    print("converted...")