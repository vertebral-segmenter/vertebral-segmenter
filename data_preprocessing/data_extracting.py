# import required module
import os, sys
import shutil

# assign directories
src_path = "T:\S@leh\CIHR data (Rat_mCT)"
dst_path = "T:\S@leh\Rat_mCT_new"

# # iterate over files in the source directory (series 800)
# serie_path = os.path.join(src_path, "800-Series")
# for root, dirs, files in os.walk(serie_path):
#     if 'L2' in root and files:
#         for file in files:
#             if file.lower().endswith('nii'):
#                 if ('cropped' in file.lower() and 'resampled' in file.lower() or 'scan' in file.lower()) and not 'segmentation' in file.lower():
#                     src_file = os.path.join(root, file)
#                     shutil.copy(src_file, dst_path)
#
#                     scan_number = [num for num in src_file.split('\\') if num.isnumeric()][0]
#                     dst_file_name = f"{scan_number}_scan_cropped.nii"
#                     dst_file = os.path.join(dst_path, dst_file_name)
#                     os.rename(os.path.join(dst_path, file), dst_file)
#
#                     print(f"{dst_file_name} data copied...")
#
#
# # iterate over files in the source directory (series 900)
# serie_path = os.path.join(src_path, "900-Series")
# for root, dirs, files in os.walk(serie_path):
#     if 'L2' in root and files:
#         for file in files:
#             if file.lower().endswith('nii'):
#                 if 'resampled' in file.lower() and not 'segmentation' in file.lower():
#                     src_file = os.path.join(root, file)
#                     shutil.copy(src_file, dst_path)
#
#                     scan_number = [num for num in src_file.split('\\') if num.isnumeric()][0]
#                     dst_file_name = f"{scan_number}_scan_cropped.nii"
#                     dst_file = os.path.join(dst_path, dst_file_name)
#                     os.rename(os.path.join(dst_path, file), dst_file)
#
#                     print(f"{dst_file_name} data copied...")
#
#
# # iterate over files in the source directory (series 1000)
# serie_path = os.path.join(src_path, "1000-Series")
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

sys.path.append(psoas_segmentor_module_repo)
convert_file( fname, csv_fname, nrrd_fname)