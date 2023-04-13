# import required module
import os
import shutil
import nibabel as nib
from data_preprocessing.image_analysis.amira_processing import convert_amira_to_nifti
from data_preprocessing.image_analysis.nifti_processing import process_segmentation_image, process_scan_image
import logging


def setup_logger(name, log_file, level=logging.DEBUG):
    """Function to set up a logger with the given name and file."""
    logger = logging.getLogger(name)
    logger.setLevel(level)

    # Create a file handler for outputting log messages to a file
    file_handler = logging.FileHandler(log_file, mode='w')  # Set mode to 'w' to overwrite the file
    file_handler.setLevel(level)

    # Create a formatter and add it to the file handler
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s', datefmt='%m/%d/%Y %I:%M:%S %p')
    file_handler.setFormatter(formatter)

    # Add the file handler to the logger
    logger.addHandler(file_handler)

    return logger


def convert_and_copy_image(src_file, dst_path, new_file_name=None, new_file_extension=None, is_label=False,
                           logger=None):
    """
    Convert and copy an image file to a new location, optionally changing the file name and extension.

    Args:
        src_file (str): Path to the source image file.
        dst_path (str): Path to the destination directory.
        new_file_name (str, optional): New file name. Defaults to None.
        new_file_extension (str, optional): New file extension. Defaults to None.
        is_label (bool, optional): If True, processes the input image as a segmentation image. Defaults to False.
    """
    # Split the source file name and extension
    file_name, file_extension = os.path.splitext(os.path.basename(src_file))
    # Set the new file name and extension if provided
    new_file_name = new_file_name or file_name
    new_file_extension = new_file_extension or file_extension
    file = new_file_name + new_file_extension

    # Check if the file already exists in the destination path
    if file in os.listdir(dst_path):
        logger.warning(f"File exists in {dst_path}: {file}") if logger else print(f"File exists in {dst_path}: {file}")
        return

    try:
        # Load the source image using nibabel
        input_img = nib.load(src_file)
    except:
        logger.error(f"Error in loading file: {src_file}.") if logger else print(f"Error in loading file: {src_file}.")
        return

    # Convert Amira image to Nifti image if necessary
    if file_extension == '.am' and new_file_extension == '.nii':
        try:
            nifti_img = input_img
            logger.info(f">Amira to Nifti converted: {src_file}") if logger else print(f">Amira to Nifti converted: {src_file}")
        except:
            logger.error(f"Error in converting Amira file: {src_file}") if logger else print(f"Error in converting Amira file: {src_file}")
            return
    else:
        nifti_img = input_img

    # Process the image if it's a segmentation image
    try:
        if is_label:
            processed_img = process_segmentation_image(nifti_img)
        else:
            processed_img = process_scan_image(nifti_img)
    except:
        logger.error(f"Error in processing file: {src_file}.") if logger else print(f"Error in processing file: {src_file}.")
        return

    # Save the processed image in the destination path
    nib.save(processed_img, os.path.join(dst_path, file))
    logger.info(f"{'Segmentation' if is_label else 'Scan'} Copied: {src_file} -> {file}") if logger else print(f"{'Segmentation' if is_label else 'Scan'} Copied: {src_file} -> {file}")


def scan_number(root_path):
    for dir in root_path.split('\\'):
        if dir.isnumeric():
            return str(dir)
        else:
            for num in dir.split('_'):
                if num.isnumeric():
                    return str(num)
    return None


# iterate over amira files in the source directory (series 700)
# def process_700_series(src_path, dst_scan_path=r"D:\Rat_mCT_v1\scans", dst_label_path=r"D:\Rat_mCT_v1\labels",
#                        logger):
#     series700_path = os.path.join(src_path, "700-Series")
#     for root, dirs, files in os.walk(series700_path):
#         if 'L2' in root and files:
#             for file in files:
#                 if os.path.splitext(file.lower())[1] == '.am':
#                     if (
#                             'l1-l3' in file.lower() or 'l2' in file.lower()) and not 'upsampled' in file.lower() and not 'mask' in file.lower() and not 'vbcort' in file.lower():
#                         src_file = os.path.join(root, file)
#                         dst_file_name = f"{scan_number(src_file)}_scan_cropped.nii"
#                         dst_file = os.path.join(dst_scan_path, dst_file_name)
#                         try:
#                             nifti_img = convert_amira_to_nifti(src_file)
#                             # Save the NIfTI image
#                             nib.save(nifti_img, dst_file)
#                             print(f"{dst_file_name} created...")
#                         except:
#                             print(f"## Error - Can't convert {file}.")


# iterate over files in the source directory (series 800)
def process_800_series(src_path, dst_scan_path=r"D:\Rat_mCT_v1\scans", dst_label_path=r"D:\Rat_mCT_v1\labels",
                       logger=None):
    series800_path = os.path.join(src_path, "800-Series")
    for root, dirs, files in os.walk(series800_path):
        if 'L2' in root and files:
            for file in files:
                if file.lower().endswith('nii'):
                    if 'resampled' in file.lower() or 'scan' in file.lower():
                        src_file = os.path.join(root, file)
                        is_label = True if 'segmentation' in file.lower() else False
                        dst_path = dst_label_path if is_label else dst_scan_path
                        file_name = f"{scan_number(src_file)}{'_segmentation' if is_label else '_scan_cropped'}"
                        convert_and_copy_image(src_file, dst_path, new_file_name=file_name, new_file_extension='.nii',
                                               is_label=is_label, logger=logger)


# iterate over files in the source directory (series 900)
def process_900_series(src_path, dst_scan_path=r"D:\Rat_mCT_v1\scans", dst_label_path=r"D:\Rat_mCT_v1\labels",
                       logger=None):
    series900_path = os.path.join(src_path, "900-Series")
    for root, dirs, files in os.walk(series900_path):
        if 'L2' in root and files:
            for file in files:
                if file.lower().endswith('nii'):
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
        if 'L2' in root and files:
            for file in files:
                src_file = os.path.join(root, file)
                if file.lower().endswith('nii'):
                    if 'scan' in file.lower():
                        is_label = True if 'segmentation' in file.lower() else False
                        dst_path = dst_label_path if is_label else dst_scan_path
                        file_name = f"{scan_number(src_file)}{'_segmentation' if is_label else '_scan_cropped'}"
                        convert_and_copy_image(src_file, dst_path, new_file_name=file_name, new_file_extension='.nii',
                                               is_label=is_label, logger=logger)
                    elif 'CIHR' in file and (
                            'L1-L3' in file or 'L2' in file and (
                            'resampled' in file.lower() or 'cropped' in file.lower())):
                        file_name = f"{scan_number(src_file)}_scan_cropped"
                        convert_and_copy_image(src_file, dst_scan_path, new_file_name=file_name,
                                               new_file_extension='.nii', is_label=False, logger=logger)
                elif file.lower().endswith('nrrd') and 'trabecular_segmentation' in file.lower():
                    file_name = f"{scan_number(src_file)}_segmentation"
                    convert_and_copy_image(src_file, dst_label_path, new_file_name=file_name, new_file_extension='.nii',
                                           is_label=True, logger=logger)
                # 1039 label is wrong!


# iterate over files in the source directory (series 1100)
def process_1100_series(src_path, dst_scan_path=r"D:\Rat_mCT_v1\scans", dst_label_path=r"D:\Rat_mCT_v1\labels", logger=None):
    series1100_path = os.path.join(src_path, "1100-Series")
    for root, dirs, files in os.walk(series1100_path):
        if 'L2' in root and files:
            for file in files:
                src_file = os.path.join(root, file)
                if file.lower().endswith('nii') and 'CIHR' in file and (
                        'L4_cropped_resampled' in file or 'L2__resampled' in file):
                    file_name = f"{scan_number(src_file)}_scan_cropped"
                    convert_and_copy_image(src_file, dst_scan_path, new_file_name=file_name, new_file_extension='.nii',
                                           is_label=False, logger=logger)
                elif file.lower().endswith('nrrd') and 'l2_trabecular_segmentation' in file.lower():
                    file_name = f"{scan_number(src_file)}_segmentation"
                    convert_and_copy_image(src_file, dst_label_path, new_file_name=file_name, new_file_extension='.nii',
                                           is_label=True, logger=logger)
                elif scan_number(src_file) in range(1107, 1111) and file.endswith('L2.nii'):
                    file_name = f"{scan_number(src_file)}_scan_cropped"
                    convert_and_copy_image(src_file, dst_scan_path, new_file_name=file_name, new_file_extension='.nii',
                                           is_label=False, logger=logger)
                elif scan_number(src_file) in range(1107, 1111) and file.endswith('L2_SlicerSegmentation.nii'):
                    file_name = f"{scan_number(src_file)}_segmentation"
                    convert_and_copy_image(src_file, dst_label_path, new_file_name=file_name, new_file_extension='.nii',
                                           is_label=True, logger=logger)
                elif scan_number(src_file) in range(1104, 1117):
                    # should extract data manually
                    pass


if __name__ == "__main__":

    # assign directories
    src_path = r"T:\CIHR Data\16) Stereology"
    dst_scan_path = r"D:\Rat_mCT_v1\scans"
    dst_label_path = r"D:\Rat_mCT_v1\labels"

    # Create a separate logger for each function and configure the log file
    # process_700_logger = setup_logger('process_700', 'logs/CIHR_process_700.log')
    process_800_logger = setup_logger('process_800', 'logs/CIHR_process_800.log')
    process_900_logger = setup_logger('process_900', 'logs/CIHR_process_900.log')
    process_1000_logger = setup_logger('process_1000', 'logs/CIHR_process_1000.log')
    process_1100_logger = setup_logger('process_1100', 'logs/CIHR_process_1100.log')

    # process_700_series(src_path, dst_scan_path, dst_label_path, process_700_logger)
    process_800_series(src_path, dst_scan_path, dst_label_path, logger=process_800_logger)
    process_900_series(src_path, dst_scan_path, dst_label_path, logger=process_900_logger)
    process_1000_series(src_path, dst_scan_path, dst_label_path, logger=process_1000_logger)
    process_1100_series(src_path, dst_scan_path, dst_label_path, logger=process_1100_logger)

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
