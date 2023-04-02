# import required module
import os
import shutil

# assign directories
src_path = r"T:\Osteosarcopenia\Osteosarcopenia_Stereology"
dst_path = r"T:\S@leh\Rat_mCT_new"

# iterate over files in the source directory (series 2000)
for root, dirs, files in os.walk(src_path):
    if 'Stereology' in root and files:
        for file in files:
            if file.lower().endswith('nii'):
                if 'scan' in file.lower() and 'resampled' in file.lower():
                    src_file = os.path.join(root, file)
                    shutil.copy(src_file, dst_path)

                    scan_number = [num for num in src_file.split('\\') if num.isnumeric()][0]
                    dst_file_name = f"{scan_number}_scan_cropped.nii"
                    dst_file = os.path.join(dst_path, dst_file_name)
                    os.rename(os.path.join(dst_path, file), dst_file)

                    print(f"{dst_file_name} data copied...")