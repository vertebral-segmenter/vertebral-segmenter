# import required module
import os, sys
import shutil
from amira_to_nifti import convert_amira_to_nifti

# assign directories
src_path = r"T:\S@leh\CIHR data (Rat_mCT)"
dst_path = r"T:\S@leh\Rat_mCT_new"

# iterate over amira files in the source directory (series 700)
serie_path = os.path.join(src_path, "700-Series")
for root, dirs, files in os.walk(serie_path):
    if 'L2' in root and files:
        for file in files:
            if os.path.splitext(file.lower())[1] == '.am':
                if ('l1-l3' in file.lower() or 'l2' in file.lower()) and not 'upsampled' in file.lower() and not 'mask' in file.lower() and not 'vbcort' in file.lower():
                    src_file = os.path.join(root, file)
                    scan_number = root.split('\\')[-1].split('_')[0] if root.split('\\')[-1].split('_')[0].isnumeric() else '7xx'
                    dst_file_name = f"{scan_number}_scan_cropped.nii"
                    if dst_file_name in os.listdir(dst_path):
                        print(f"## Error - scan {scan_number} exist - {file}")
                        dst_file_name = os.path.splitext(file)[0] + '.nii'
                    dst_file = os.path.join(dst_path, dst_file_name)
                    try:
                        convert_amira_to_nifti(src_file, dst_file)
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
                    shutil.copy(src_file, dst_path)

                    scan_number = [num for num in src_file.split('\\') if num.isnumeric()][0]
                    dst_file_name = f"{scan_number}_scan_cropped.nii"
                    dst_file = os.path.join(dst_path, dst_file_name)
                    os.rename(os.path.join(dst_path, file), dst_file)

                    print(f"{dst_file_name} data copied...")


# iterate over files in the source directory (series 900)
serie_path = os.path.join(src_path, "900-Series")
for root, dirs, files in os.walk(serie_path):
    if 'L2' in root and files:
        for file in files:
            if file.lower().endswith('nii'):
                if 'resampled' in file.lower() and not 'segmentation' in file.lower():
                    src_file = os.path.join(root, file)
                    shutil.copy(src_file, dst_path)

                    scan_number = [num for num in src_file.split('\\') if num.isnumeric()][0]
                    dst_file_name = f"{scan_number}_scan_cropped.nii"
                    dst_file = os.path.join(dst_path, dst_file_name)
                    os.rename(os.path.join(dst_path, file), dst_file)

                    print(f"{dst_file_name} data copied...")


# iterate over files in the source directory (series 1000)
serie_path = os.path.join(src_path, "1000-Series")
for root, dirs, files in os.walk(serie_path):
    if 'L2' in root and files:
        for file in files:
            if file.lower().endswith('nii'):
                if 'scan' in file.lower() and not 'segmentation' in file.lower():
                    src_file = os.path.join(root, file)
                    scan_number = [num for num in src_file.split('\\') if num.isnumeric()][0]
                    dst_file_name = f"{scan_number}_scan_cropped.nii"
                    if dst_file_name in os.listdir(dst_path):
                        dst_file = os.path.join(dst_path, f"{scan_number}_scan_cropped_ERROR.nii")
                        print(f"ERROR => scan {scan_number} => '{file}'")
                    else:
                        shutil.copy(src_file, dst_path)
                        dst_file = os.path.join(dst_path, dst_file_name)
                        os.rename(os.path.join(dst_path, file), dst_file)
                        print(f"{dst_file_name} data copied...")


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

