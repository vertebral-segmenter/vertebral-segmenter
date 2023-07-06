import os
import nibabel as nib
from data_preprocessing.image_analysis.amira_processing import convert_amira_to_nifti
from data_preprocessing.image_analysis.nifti_processing import process_segmentation_image, process_scan_image, \
    load_as_nifti
import logging


def scan_number(root_path):
    for dir in root_path.split('\\'):
        if dir.isnumeric():
            return str(dir)
        else:
            for num in dir.split('_'):
                if num.isnumeric():
                    return str(num)
    return None


def setup_logger(name, log_file, level=logging.DEBUG, print_log=True):
    """Function to set up a logger with the given name and file."""
    logger = logging.getLogger(name)
    logger.setLevel(level)

    # Create a file handler for outputting log messages to a file
    file_handler = logging.FileHandler(log_file, mode='w')  # Set mode to 'w' to overwrite the file
    file_handler.setLevel(level)

    # Create a stream handler to print to the terminal
    if print_log:
        stream_handler = logging.StreamHandler()
        stream_handler.setLevel(logging.DEBUG)

    # Create a formatter and add it to the file handler
    file_handler_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s',
                                               datefmt='%m/%d/%Y %I:%M:%S %p')
    file_handler.setFormatter(file_handler_formatter)
    if print_log:
        stream_handler_formatter = logging.Formatter('%(levelname)s - %(message)s')
        stream_handler.setFormatter(stream_handler_formatter)

    # Add the file handler to the logger
    logger.addHandler(file_handler)
    if print_log:
        logger.addHandler(stream_handler)

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

    if file_extension != '.am':
        # Load the source image using nibabel if it is '.nii' or '.nii.gz' or '.nrrd'
        try:
            nifti_img = load_as_nifti(src_file)
        except:
            logger.error(f"Error in loading file: {src_file}.") if logger else print(
                f"Error in loading file: {src_file}.")
            return
    elif file_extension == '.am' and new_file_extension == '.nii':
        # Convert Amira image to Nifti image if necessary
        try:
            nifti_img = convert_amira_to_nifti(src_file)
            logger.info(f">Amira to Nifti converted: {src_file}") if logger else print(
                f">Amira to Nifti converted: {src_file}")
        except:
            logger.error(f"Error in converting Amira file: {src_file}") if logger else print(
                f"Error in converting Amira file: {src_file}")
            return

    # Process the image if it's a segmentation image
    try:
        if is_label:
            processed_img = process_segmentation_image(nifti_img)
        else:
            processed_img = process_scan_image(nifti_img)
    except:
        logger.error(f"Error in processing file: {src_file}.") if logger else print(
            f"Error in processing file: {src_file}.")
        return

    # Save the processed image in the destination path
    nib.save(processed_img, os.path.join(dst_path, file))
    logger.info(f"{'Segmentation' if is_label else 'Scan'} Copied: {src_file} -> {file}") if logger else print(
        f"{'Segmentation' if is_label else 'Scan'} Copied: {src_file} -> {file}")
