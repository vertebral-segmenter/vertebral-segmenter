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

def resample_nifti_file(input_file, output_file, new_spacing=(0.035, 0.035, 0.035)):
    # Load the input NIfTI file
    input_img = nib.load(input_file)

    # Resample the image to the desired spacing
    resampled_img = resample_to_output(input_img, new_spacing, order=3)

    # Save the resampled image as a new NIfTI file
    nib.save(resampled_img, output_file)


# iterate over files in the source directory (fractured serie)
for root, dirs, files in os.walk(src_path):
    if 'L2' in root and 'Fractured' in root and files:
        for file in files:
            src_file = os.path.join(root, file)
            if not file.lower().endswith('.nii') or file.lower().endswith('.nii.gz') and any([list_file.lower().endswith('.nii') for list_file in os.listdir(root)]):
                break
            elif file.lower().endswith('.nii.gz'):
                convert_nii_gz_to_nii(src_file, src_file[:-3])
                print(f"{file} data converted from .gz file.")
                src_file = src_file[:-3]
            scan_number = [num for dir in src_file.split('\\') for num in dir.split('_') if num.isnumeric()][0]
            dst_file_name = f"frac_{scan_number}_scan_cropped.nii"
            dst_file = os.path.join(dst_path, dst_file_name)
            try:
                resample_nifti_file(src_file, dst_file, new_spacing=(0.035, 0.035, 0.035))
                print(f"{dst_file_name} data resampled and copied...")
            except:
                print(f"## Error - Can't resample {src_file}.")



