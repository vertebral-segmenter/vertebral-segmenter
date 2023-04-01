# import required module
import os, sys
import shutil
import nibabel as nib

# assign directories
src_path = "/home/smsmt/Stereology data"
dst_image_path = "rat_dataset/images"
dst_label_path = "rat_dataset/labels"

# iterate over files in the source directory (series 800)
serie_path = os.path.join(src_path, "800-Series")
for root, dirs, files in os.walk(serie_path):
    if 'L2' in root and files:
        for file in files:
            if file.lower().endswith('nii'):
                if ('cropped' in file.lower() and 'resampled' in file.lower() or 'scan' in file.lower()):
                    dst_path = dst_image_path if not 'segmentation' in file.lower() else dst_label_path
                    src_file = os.path.join(root, file)

                    scan_number = [num for num in src_file.split('/') if num.isnumeric()][0]
                    dst_file_name = f"{scan_number}_scan_cropped{'' if not 'segmentation' in file.lower() else '_total'}.nii"
                    dst_file = os.path.join(dst_path, dst_file_name)
                    if not os.path.exists(dst_file_name): shutil.copy(src_file, dst_file)
                    print(f"Data Copied: {src_file} -> {dst_file}")


# iterate over files in the source directory (series 900)
serie_path = os.path.join(src_path, "900-Series")
for root, dirs, files in os.walk(serie_path):
    if 'L2' in root and files:
        for file in files:
            if file.lower().endswith('nii'):
                if 'resampled' in file.lower():
                    dst_path = dst_image_path if not 'segmentation' in file.lower() else dst_label_path
                    src_file = os.path.join(root, file)

                    scan_number = [num for num in src_file.split('/') if num.isnumeric()][0]
                    dst_file_name = f"{scan_number}_scan_cropped{'' if not 'segmentation' in file.lower() else '_total'}.nii"
                    dst_file = os.path.join(dst_path, dst_file_name)
                    if not os.path.exists(dst_file_name): shutil.copy(src_file, dst_file)
                    print(f"Data Copied: {src_file} -> {dst_file}")


# iterate over files in the source directory (series 1000)
serie_path = os.path.join(src_path, "1000-Series")
processed_scan_nums = {}
processed_label_nums = {}
for root, dirs, files in os.walk(serie_path):
    if 'L2' in root and len([f for f in files if f.lower().endswith('nii')]) > 0:
        
        image_files = [f for f in files if f.lower().endswith('nii') and 'scan' in f.lower() and "segmentation" not in f.lower()]
        if not image_files:
            image_files = [f for f in files if f.lower().endswith('nii') and ('resampled' in f.lower() or 'cropped' in f.lower()) and "segmentation" not in f.lower()]

        label_files = [f for f in files if f.lower().endswith('nii') and 'scan' in f.lower() and "segmentation" in f.lower()]
        if not label_files:
            label_files = [f for f in files if f.lower().endswith('nii') and "segmentation" in f.lower()]
        print("root:", root)
        image_file, label_file = None, None
        # load segmentation
        total_volume_labels = [f for f in label_files if "centrum" not in f.lower() and "shell" not in f.lower()]
        centrum_volume_labels = [f for f in label_files if "centrum" in f.lower()]
        if not image_files:
            print(root)
            print([f for f in files if f.lower().endswith('nii')])
            print("------------------------")
            continue
        for label_file in total_volume_labels + centrum_volume_labels:
            seg_src_file = os.path.join(root, label_file)
            seg_scan_number = [num for num in seg_src_file.split('/') if num.isnumeric() or (num.split('_')[0].isnumeric() and num.startswith('10'))][-1]
            seg_scan_number = seg_scan_number.split("_")[0]
            dst_file_name = f"{seg_scan_number}_scan_cropped_{'centrum' if 'centrum' in label_file.lower() else 'total'}.nii"
            seg_dst_file = os.path.join(dst_label_path, dst_file_name)
            if not os.path.exists(seg_dst_file): shutil.copy(seg_src_file, seg_dst_file)
            print(f"Seg src file: {seg_src_file}")
            print(f"Seg dst file: {seg_dst_file}")

        ## Load image file
        image_file = image_files[0]
        img_src_file = os.path.join(root, image_file)
        img_scan_number = [num for num in img_src_file.split('/') if num.isnumeric() or (num.split('_')[0].isnumeric() and num.startswith('10'))][-1]
        img_scan_number = img_scan_number.split("_")[0]
        assert not len(total_volume_labels + centrum_volume_labels) > 0 or img_scan_number == seg_scan_number, f"{img_scan_number} != {seg_scan_number}"
        dst_file_name = f"{img_scan_number}_scan_cropped.nii"
        img_dst_file = os.path.join(dst_image_path, dst_file_name)
        if not os.path.exists(img_dst_file): shutil.copy(img_src_file, img_dst_file)
        print(f"Img src file: {img_src_file}")
        print(f"Img dst file: {img_dst_file}")
        print("------------------------")


# iterate over files in the source directory (series 1100)
serie_path = os.path.join(src_path, "1100-Series")
for root, dirs, files in os.walk(serie_path):
    if 'L2' in root and files:
        image_files = [f for f in files if f.lower().endswith('nii') and 'l2' in f.lower() and "wholebone" not in f.lower() and "segmentation" not in f.lower()]
        label_files = [f for f in files if f.lower().endswith('nii') and 'l2' in f.lower() and "wholebone" not in f.lower()  and "segmentation" in f.lower()]
        image_file, label_file = None, None
        # load segmentation
        total_volume_labels = [f for f in label_files if "centrum" not in f.lower() and "shell" not in f.lower()]
        centrum_volume_labels = [f for f in label_files if "centrum" in f.lower()]
        for label_file in total_volume_labels + centrum_volume_labels:
            seg_src_file = os.path.join(root, label_file)
            seg_scan_number = [num for num in seg_src_file.split('/') if num.isnumeric() or (num.split('_')[0].isnumeric() and num.startswith('11'))][-1]
            seg_scan_number = seg_scan_number.split("_")[0]
            dst_file_name = f"{seg_scan_number}_scan_cropped_{'centrum' if 'centrum' in label_file.lower() else 'total'}.nii"
            seg_dst_file = os.path.join(dst_label_path, dst_file_name)
            if not os.path.exists(seg_dst_file): shutil.copy(seg_src_file, seg_dst_file)
            print(f"Seg src file: {seg_src_file}")
            print(f"Seg dst file: {seg_dst_file}")

        ## Load image file
        resampled_image = [f for f in image_files if 'resampled' in f.lower()]
        if resampled_image:
            image_file = resampled_image[0]
        else:
            image_file = image_files[0]
        img_src_file = os.path.join(root, image_file)
        img_scan_number = [num for num in img_src_file.split('/') if num.isnumeric() or (num.split('_')[0].isnumeric() and num.startswith('11'))][-1]
        img_scan_number = img_scan_number.split("_")[0]
        assert img_scan_number == seg_scan_number, f"{img_scan_number} != {seg_scan_number}"
        dst_file_name = f"{img_scan_number}_scan_cropped.nii"
        img_dst_file = os.path.join(dst_image_path, dst_file_name)
        if not os.path.exists(img_dst_file): shutil.copy(img_src_file, img_dst_file)
        print(f"Img src file: {img_src_file}")
        print(f"Img dst file: {img_dst_file}")
        print("------------------------")

