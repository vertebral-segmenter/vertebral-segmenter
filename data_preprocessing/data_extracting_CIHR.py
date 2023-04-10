# import required module
import os
import shutil
import nibabel as nib
from data_preprocessing.image_analysis.amira_processing import convert_amira_to_nifti

# assign directories
src_path = r"T:\S@leh\CIHR data (Rat_mCT)"
dst_scan_path = r"D:\Rat_mCT_v0\scans"
dst_label_path = r"D:\Rat_mCT_v0\labels"


def copy_file(src_file, dst_path, file_name=None, file_extension=None):
    if file_name is None:
        file_name = os.path.basename(src_file)
    if file_extension is None:
        file_extension = os.path.splitext(os.path.basename(src_file))[1]
    file = file_name + file_extension
    if file in os.listdir(dst_path):
        print(f"# {file} already exist in the dataset.")
        return
    shutil.copy(src_file, os.path.join(dst_path, file))
    print(f"{file} copied...")


def scan_number(root_path):
    for dir in root_path.split('\\'):
        if dir.isnumeric():
            return dir
        else:
            for num in dir.split('_'):
                if num.isnumeric():
                    return num
                else:
                    return 'None'


# iterate over amira files in the source directory (series 700)
serie_path = os.path.join(src_path, "700-Series")
for root, dirs, files in os.walk(serie_path):
    if 'L2' in root and files:
        for file in files:
            if os.path.splitext(file.lower())[1] == '.am':
                if ('l1-l3' in file.lower() or 'l2' in file.lower()) and not 'upsampled' in file.lower() and not 'mask' in file.lower() and not 'vbcort' in file.lower():
                    src_file = os.path.join(root, file)
                    dst_file_name = f"{scan_number(src_file)}_scan_cropped.nii"
                    dst_file = os.path.join(dst_path, dst_file_name)
                    try:
                        nifti_img = convert_amira_to_nifti(src_file)
                        # Save the NIfTI image
                        nib.save(nifti_img, dst_file)
                        print(f"{dst_file_name} created...")
                    except:
                        print(f"## Error - Can't convert {file}.")


# iterate over files in the source directory (series 800)
serie_path = os.path.join(src_path, "800-Series")
for root, dirs, files in os.walk(serie_path):
    if 'L2' in root and files:
        for file in files:
            if file.lower().endswith('nii'):
                if ('cropped' in file.lower() and 'resampled' in file.lower() or 'scan' in file.lower()) and not 'segmentation' in file.lower():
                    src_file = os.path.join(root, file)
                    # dst_path = dst_image_path if not 'segmentation' in file.lower() else dst_label_path
                    file_name = f"{scan_number(src_file)}_scan_cropped"
                    copy_file(src_file, dst_path, file_name=file_name, file_extension='.nii')

                # if ('cropped' in file.lower() and 'resampled' in file.lower() or 'scan' in file.lower()):
                #     dst_path = dst_image_path if not 'segmentation' in file.lower() else dst_label_path
                #     src_file = os.path.join(root, file)
                #
                #     scan_number = [num for num in src_file.split('/') if num.isnumeric()][0]
                #     dst_file_name = f"{scan_number}_scan_cropped{'' if not 'segmentation' in file.lower() else '_total'}.nii"
                #     dst_file = os.path.join(dst_path, dst_file_name)
                #     if not os.path.exists(dst_file_name): shutil.copy(src_file, dst_file)
                #     print(f"Data Copied: {src_file} -> {dst_file}")

# iterate over files in the source directory (series 900)
serie_path = os.path.join(src_path, "900-Series")
for root, dirs, files in os.walk(serie_path):
    if 'L2' in root and files:
        for file in files:
            if file.lower().endswith('nii'):
                if 'resampled' in file.lower() and not 'segmentation' in file.lower():
                    src_file = os.path.join(root, file)
                    file_name = f"{scan_number(src_file)}_scan_cropped"
                    copy_file(src_file, dst_path, file_name=file_name, file_extension='.nii')


# iterate over files in the source directory (series 1000)
serie_path = os.path.join(src_path, "1000-Series")
for root, dirs, files in os.walk(serie_path):
    if 'L2' in root and files:
        for file in files:
            if file.lower().endswith('nii'):
                if 'scan' in file.lower() and not 'segmentation' in file.lower():
                    src_file = os.path.join(root, file)
                    file_name = f"{scan_number(src_file)}_scan_cropped"
                    copy_file(src_file, dst_path, file_name=file_name, file_extension='.nii')


# iterate over files in the source directory (series 1100)
# serie_path = os.path.join(src_path, "1100-Series")
# for root, dirs, files in os.walk(serie_path):
#     if 'L2' in root and files:
#         for file in files:
#             if file.lower().endswith('nii'):
#                 if 'scan' in file.lower() and not 'segmentation' in file.lower():
#                     src_file = os.path.join(root, file)
#                     scan_number = [num for num in src_file.split('\\') if num.isnumeric()][0]
#                     dst_file_name = f"{scan_number}_scan_cropped.nii"
#                     if dst_file_name in os.listdir(dst_path):
#                         dst_file = os.path.join(dst_path, f"{scan_number}_scan_cropped_ERROR.nii")
#                         print(f"ERROR => scan {scan_number} => '{file}'")
#                     else:
#                         shutil.copy(src_file, dst_path)
#                         dst_file = os.path.join(dst_path, dst_file_name)
#                         os.rename(os.path.join(dst_path, file), dst_file)
#                         print(f"{dst_file_name} data copied...")


# # Rename manually copied files in destination folder
# for root, dirs, files in os.walk(dst_path):
#         for file in files:
#             if file.lower().endswith('nii') and not '_scan_cropped' in file.lower():
#                     src_file = os.path.join(root, file)
#                     scan_number = [num for num in file.split('_') if num.isnumeric()][0]
#                     dst_file_name = f"{scan_number}_scan_cropped.nii"
#                     dst_file = os.path.join(dst_path, dst_file_name)
#                     os.rename(src_file, dst_file)
#                     print(f"{dst_file_name} data renamed...")
