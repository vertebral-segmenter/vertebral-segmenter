import os
from data_preprocessing.image_analysis.data_extraction import convert_and_copy_image, scan_number, setup_logger


# iterate over amira files in the source directory (series 700)
def process_700_series(src_path, dst_scan_path=r"D:\Rat_mCT_v1\scans", dst_label_path=r"D:\Rat_mCT_v1\labels",
                       logger=None):
    series700_path = os.path.join(src_path, "700-Series")
    for root, dirs, files in os.walk(series700_path):
        if 'L2' in root and files:
            for file in files:
                if os.path.splitext(file.lower())[1] == '.am':
                    src_file = os.path.join(root, file)
                    if (
                            'l1-l3' in file.lower() or 'l2' in file.lower()) and not 'upsampled' in file.lower() and not 'mask' in file.lower() and not 'vbcort' in file.lower():
                        is_label = False
                        file_name = f"{scan_number(src_file)}_scan_cropped"
                        convert_and_copy_image(src_file, dst_scan_path, new_file_name=file_name,
                                               new_file_extension='.nii',
                                               is_label=is_label, logger=logger)


# iterate over files in the source directory (series 800)
def process_800_series(src_path, dst_scan_path=r"D:\Rat_mCT_v1\scans", dst_label_path=r"D:\Rat_mCT_v1\labels",
                       logger=None):
    series800_path = os.path.join(src_path, "800-Series")
    for root, dirs, files in os.walk(series800_path):
        if 'L2' in root and files:
            for file in files:
                if file.lower().endswith('nii') or file.lower().endswith('nrrd'):
                    if 'resampled' in file.lower() or 'scan' in file.lower():
                        src_file = os.path.join(root, file)
                        if '_segmentation' in file.lower():
                            is_label = True
                            file_name = f"{scan_number(src_file)}_segmentation"
                            convert_and_copy_image(src_file, dst_label_path, new_file_name=file_name,
                                                   new_file_extension='.nii',
                                                   is_label=is_label, logger=logger)
                        elif 'segmentation' not in file.lower():
                            is_label = False
                            file_name = f"{scan_number(src_file)}_scan_cropped"
                            convert_and_copy_image(src_file, dst_scan_path, new_file_name=file_name,
                                                   is_label=is_label, logger=logger)


# iterate over files in the source directory (series 900)
def process_900_series(src_path, dst_scan_path=r"D:\Rat_mCT_v1\scans", dst_label_path=r"D:\Rat_mCT_v1\labels",
                       logger=None):
    series900_path = os.path.join(src_path, "900-Series")
    for root, dirs, files in os.walk(series900_path):
        if 'L2' in root and files:
            for file in files:
                if file.lower().endswith('nii') or file.lower().endswith('nrrd'):
                    if 'resampled' in file.lower():
                        src_file = os.path.join(root, file)
                        is_label = True if 'segmentation' in file.lower() else False
                        dst_path = dst_label_path if is_label else dst_scan_path
                        file_name = f"{scan_number(src_file)}{'_segmentation' if is_label else '_scan_cropped'}"
                        convert_and_copy_image(src_file, dst_path, new_file_name=file_name, new_file_extension='.nii',
                                               is_label=is_label, logger=logger)


# iterate over files in the source directory (series 1000)
def process_1000_series(src_path, dst_scan_path=r"D:\Rat_mCT_v1\scans", dst_label_path=r"D:\Rat_mCT_v1\labels",
                        logger=None):
    series1000_path = os.path.join(src_path, "1000-Series")
    for root, dirs, files in os.walk(series1000_path):
        if 'L2' in root and files and 'old' not in root:
            for file in files:
                src_file = os.path.join(root, file)
                if file.lower().endswith('nii'):
                    if 'scan' in file.lower():
                        if '_segmentation' in file.lower():
                            is_label = True
                            file_name = f"{scan_number(src_file)}_segmentation"
                            convert_and_copy_image(src_file, dst_label_path, new_file_name=file_name,
                                                   new_file_extension='.nii',
                                                   is_label=is_label, logger=logger)
                        elif 'segmentation' not in file.lower():
                            is_label = False
                            file_name = f"{scan_number(src_file)}_scan_cropped"
                            convert_and_copy_image(src_file, dst_scan_path, new_file_name=file_name,
                                                   is_label=is_label, logger=logger)
                    elif 'CIHR' in file and (
                            'L1-L3' in file or 'L2' in file and (
                            'resampled' in file.lower() or 'cropped' in file.lower())):
                        file_name = f"{scan_number(src_file)}_scan_cropped"
                        convert_and_copy_image(src_file, dst_scan_path, new_file_name=file_name,
                                               new_file_extension='.nii', is_label=False, logger=logger)
                elif file.lower().endswith('.nrrd') and 'trabecular_segmentation' in file.lower():
                    file_name = f"{scan_number(src_file)}_segmentation"
                    convert_and_copy_image(src_file, dst_label_path, new_file_name=file_name, new_file_extension='.nii',
                                           is_label=True, logger=logger)
                # 1039 label is wrong!


# iterate over files in the source directory (series 1100)
def process_1100_series(src_path, dst_scan_path=r"D:\Rat_mCT_v1\scans", dst_label_path=r"D:\Rat_mCT_v1\labels",
                        logger=None):
    series1100_path = os.path.join(src_path, "1100-Series")
    for root, dirs, files in os.walk(series1100_path):
        if 'L2' in root and files:
            for file in files:
                src_file = os.path.join(root, file)
                if int(scan_number(src_file)) in [1104, 1105, 1106]:
                    # should extract data manually for CT 1104, 1105, 1106
                    continue
                elif file.lower().endswith('nii') and 'CIHR' in file and (
                        'L4_cropped_resampled' in file or 'L4_resampled_cropped' in file or 'L2_resampled' in file):
                    file_name = f"{scan_number(src_file)}_scan_cropped"
                    convert_and_copy_image(src_file, dst_scan_path, new_file_name=file_name, new_file_extension='.nii',
                                           is_label=False, logger=logger)
                elif file.lower().endswith('nrrd') and 'l2_trabecular_segmentation' in file.lower():
                    file_name = f"{scan_number(src_file)}_segmentation"
                    convert_and_copy_image(src_file, dst_label_path, new_file_name=file_name, new_file_extension='.nii',
                                           is_label=True, logger=logger)
                elif int(scan_number(src_file)) in range(1107, 1111) and file.endswith('L2.nii'):
                    file_name = f"{scan_number(src_file)}_scan_cropped"
                    convert_and_copy_image(src_file, dst_scan_path, new_file_name=file_name, new_file_extension='.nii',
                                           is_label=False, logger=logger)
                elif int(scan_number(src_file)) in range(1107, 1111) and file.endswith('L2_SlicerSegmentation.nii'):
                    file_name = f"{scan_number(src_file)}_segmentation"
                    convert_and_copy_image(src_file, dst_label_path, new_file_name=file_name, new_file_extension='.nii',
                                           is_label=True, logger=logger)


def main():
    # assign directories
    src_path = r"T:\CIHR Data\16) Stereology"
    dst_scan_path = r"D:\Rat_mCT_v1\scans"
    dst_label_path = r"D:\Rat_mCT_v1\labels"

    # Create a separate logger for each function and configure the log file
    process_700_logger = setup_logger('process_700', 'logs/CIHR_process_700.log')
    process_800_logger = setup_logger('process_800', 'logs/CIHR_process_800.log')
    process_900_logger = setup_logger('process_900', 'logs/CIHR_process_900.log')
    process_1000_logger = setup_logger('process_1000', 'logs/CIHR_process_1000.log')
    process_1100_logger = setup_logger('process_1100', 'logs/CIHR_process_1100.log')

    process_700_series(src_path, dst_scan_path, dst_label_path, logger=process_700_logger)
    process_800_series(src_path, dst_scan_path, dst_label_path, logger=process_800_logger)
    process_900_series(src_path, dst_scan_path, dst_label_path, logger=process_900_logger)
    process_1000_series(src_path, dst_scan_path, dst_label_path, logger=process_1000_logger)
    process_1100_series(src_path, dst_scan_path, dst_label_path, logger=process_1100_logger)


if __name__ == "__main__":
    main()

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
