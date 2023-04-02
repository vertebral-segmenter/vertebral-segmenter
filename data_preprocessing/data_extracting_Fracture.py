# import required module
import os
import shutil
import nibabel as nib
from nibabel.processing import resample_to_output

# assign directories
src_path = r"T:\AlliT\Code\fracture_prediction\pix2pixtrain\images"
dst_path = r"T:\S@leh\Rat_mCT_new"


def convert_nii_gz_to_nii(gz_file_path, nii_file_path):
    # Check if the loaded file has the '.nii.gz' extension
    if not gz_file_path.lower().endswith('.nii.gz'):
        print(f"{gz_file_path} is not a .nii.gz file.")
        return

    nii_gz_file = nib.load(gz_file_path)
    nib.save(nii_gz_file, nii_file_path)

def resample_nifti_file(input_img, new_spacing=(0.035, 0.035, 0.035), order=3):
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
                    print(f"# {dst_file_name} exist in the dataset.")
                    continue
                dst_file = os.path.join(dst_path, dst_file_name)
                # if src_file
                #     os.stat(src_file).st_size / (1024 * 1024)
                try:
                    # Save the resampled image as a new NIfTI file
                    input_img = nib.load(src_file)
                    converted_img = convert_nifti_to_dtype(input_img, output_dtype='int16')
                    resampled_img = resample_nifti_file(converted_img, new_spacing=(0.035, 0.035, 0.035), order=5)
                    nib.save(resampled_img, dst_file)
                    print(f"{dst_file_name} data resampled and copied...")
                except:
                    print(f"## Error - Can't resample {src_file}.")


# if __name__ == '__main__':
#     input_img = nib.load(r"T:\AlliT\Code\fracture_prediction\pix2pixtrain\images\1003_L1,L2,L3_Healthy_ZA_Microloading\Fractured\AT_CIHR_Microloading_1003_Fractured_resampled.nii")
#     converted_img = convert_nifti_to_dtype(input_img, output_dtype='int16')
#     # resampled_img = resample_nifti_file(converted_img, new_spacing=(0.035, 0.035, 0.035), order=5)
#     nib.save(converted_img, r'D:\\vertebral-segmentation-rat-l2\\data_preprocessing\\AT_CIHR_Microloading_1003_Fractured_int16.nii')
#     print(f"data converted...")