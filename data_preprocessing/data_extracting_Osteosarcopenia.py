# import required module
import os
from data_preprocessing.image_analysis.data_extraction import convert_and_copy_image, scan_number, setup_logger


# iterate over files in the source directory (series 2000)
def process_2000_series(src_path, dst_scan_path=r"D:\Rat_mCT_v1\scans", dst_label_path=r"D:\Rat_mCT_v1\labels",
                        logger=None):
    # iterate over files in the source directory (series 2000)
    for root, dirs, files in os.walk(src_path):
        if 'Stereology' in root and files:
            for file in files:
                if file.lower().endswith('nii'):
                    if ('scan' in file.lower() or 'seg' in file.lower()) and 'resampled' in file.lower():
                        src_file = os.path.join(root, file)
                        is_label = True if 'seg' in file.lower() else False
                        dst_path = dst_label_path if is_label else dst_scan_path
                        file_name = f"{scan_number(src_file)}{'_segmentation' if is_label else '_scan_cropped'}"
                        convert_and_copy_image(src_file, dst_path, new_file_name=file_name, new_file_extension='.nii',
                                               is_label=is_label, logger=logger)


def main():
    # assign directories
    src_path = r"T:\Osteosarcopenia\Osteosarcopenia_Stereology"
    dst_scan_path = r"D:\Rat_mCT_v1\scans"
    dst_label_path = r"D:\Rat_mCT_v1\labels"

    process_2000_logger = setup_logger('process_1100', 'logs/Osteosarcopenia_process_2000.log')
    process_2000_series(src_path, dst_scan_path, dst_label_path, logger=process_2000_logger)


if __name__ == "__main__":
    main()