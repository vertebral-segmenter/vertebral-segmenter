import argparse
import os

import nibabel as nib
import numpy as np
import torch
from finetune.utils.data_utils import get_loader
from finetune.utils.utils import dice, resample_3d, R2Metric, iou

from monai.inferers import sliding_window_inference
from monai.networks.nets import SwinUNETR
from monai.networks.nets.swin_unetr_dilated import DilSwinUNETR

os.environ['PYTORCH_CUDA_ALLOC_CONF'] = "max_split_size_mb:512"

parser = argparse.ArgumentParser(description="Swin UNETR segmentation pipeline")
parser.add_argument(
    "--model", default="finetune/runs/dilation_customloss_nopretrain/model.pt", type=str, help="trained model checkpoint"
)
parser.add_argument("--data_dir", default="/dataset/dataset0/", type=str, help="dataset directory")
parser.add_argument("--exp_name", default="test1", type=str, help="experiment name")
parser.add_argument("--json_list", default="dataset_0.json", type=str, help="dataset json file")
parser.add_argument("--feature_size", default=48, type=int, help="feature size")
parser.add_argument("--infer_overlap", default=0.5, type=float, help="sliding window inference overlap")
parser.add_argument("--in_channels", default=1, type=int, help="number of input channels")
parser.add_argument("--out_channels", default=2, type=int, help="number of output channels")
parser.add_argument("--a_min", default=-1000.0, type=float, help="a_min in ScaleIntensityRanged")
parser.add_argument("--a_max", default=8000.0, type=float, help="a_max in ScaleIntensityRanged")
parser.add_argument("--b_min", default=0.0, type=float, help="b_min in ScaleIntensityRanged")
parser.add_argument("--b_max", default=1.0, type=float, help="b_max in ScaleIntensityRanged")
parser.add_argument("--space_x", default=0.035, type=float, help="spacing in x direction")
parser.add_argument("--space_y", default=0.035, type=float, help="spacing in y direction")
parser.add_argument("--space_z", default=0.035, type=float, help="spacing in z direction")
parser.add_argument("--roi_x", default=96, type=int, help="roi size in x direction")
parser.add_argument("--roi_y", default=96, type=int, help="roi size in y direction")
parser.add_argument("--roi_z", default=96, type=int, help="roi size in z direction")
parser.add_argument("--dropout_rate", default=0.0, type=float, help="dropout rate")
parser.add_argument("--distributed", action="store_true", help="start distributed training")
parser.add_argument("--workers", default=8, type=int, help="number of workers")
parser.add_argument("--RandFlipd_prob", default=0.2, type=float, help="RandFlipd aug probability")
parser.add_argument("--RandRotate90d_prob", default=0.2, type=float, help="RandRotate90d aug probability")
parser.add_argument("--RandScaleIntensityd_prob", default=0.1, type=float, help="RandScaleIntensityd aug probability")
parser.add_argument("--RandShiftIntensityd_prob", default=0.1, type=float, help="RandShiftIntensityd aug probability")
parser.add_argument("--spatial_dims", default=3, type=int, help="spatial dimension of input data")
parser.add_argument("--use_checkpoint", action="store_true", help="use gradient checkpointing to save memory")
parser.add_argument("--use_dilated_swin", action="store_true", help="use dilated swin unetr architecture instead")

# AR: modification for quicker inference
parser.add_argument("--to_save", default=True, type=bool, help="save seg results")


def main():
    args = parser.parse_args()
    args.test_mode = False
    args.val_mode = True
    output_directory = "finetune/outputs_val/" + args.exp_name
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)
    val_loader = get_loader(args)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    pretrained_pth = args.model
    if args.use_dilated_swin:
        model = DilSwinUNETR(
            img_size=(args.roi_x, args.roi_y, args.roi_z),
            in_channels=args.in_channels,
            out_channels=args.out_channels,
            feature_size=args.feature_size,
            drop_rate=0.0,
            attn_drop_rate=0.0,
            dropout_path_rate=0.0,
            use_checkpoint=args.use_checkpoint,
        )
    else:
        model = SwinUNETR(
            img_size=(args.roi_x, args.roi_y, args.roi_z),
            in_channels=args.in_channels,
            out_channels=args.out_channels,
            feature_size=args.feature_size,
            drop_rate=0.0,
            attn_drop_rate=0.0,
            dropout_path_rate=0.0,
            use_checkpoint=args.use_checkpoint,
        )

    model_dict = torch.load(pretrained_pth)["state_dict"]
    model.load_state_dict(model_dict)
    model.eval()
    model.to(device)

    with torch.no_grad():
        dice_list_case = []
        iou_list_case = []
        bv_R2 = R2Metric()
        tv_R2 = R2Metric()
        bvtv_R2 = R2Metric()

        for i, batch in enumerate(val_loader):
            val_inputs, val_labels = (batch["image"].cuda(), batch["label"].cuda())
            original_affine = batch["label_meta_dict"]["affine"][0].numpy()
            _, _, h, w, d = val_labels.shape
            target_shape = (h, w, d)
            img_name = batch["image_meta_dict"]["filename_or_obj"][0].split("/")[-1]
            print("Inference on case {}".format(img_name))
            val_outputs = sliding_window_inference(
                val_inputs, (args.roi_x, args.roi_y, args.roi_z), 4, model, overlap=args.infer_overlap, #mode="gaussian" #TODO remove mode
            )
            val_outputs = torch.softmax(val_outputs, 1).cpu().numpy()
            val_outputs = np.argmax(val_outputs, axis=1).astype(np.uint8)[0]
            val_labels = val_labels.cpu().numpy()[0, 0, :, :, :]
            val_outputs = resample_3d(val_outputs, target_shape)

            # Dice metric
            mean_dice = dice(val_outputs==1, val_labels==1)
            print("Mean Dice: {}".format(mean_dice))
            dice_list_case.append(mean_dice)

            # IoU metric
            mean_iou = iou(val_outputs==1, val_labels==1)
            print("Mean IoU: {}".format(mean_iou))
            iou_list_case.append(mean_iou)

            # R2 metric
            bv_R2.update((val_labels == 1).sum(), (val_outputs == 1).sum())
            tv_R2.update((val_labels == 0).sum(), (val_outputs == 0).sum())
            bvtv_R2.update((val_labels == 1).sum() / val_labels.size,
                           (val_outputs == 1).sum() / val_outputs.size)

            print("BV R2-value (running): {}".format(bv_R2.get_result()))
            print("TV R2-value (running): {}".format(tv_R2.get_result()))
            print("BV/TV R2-value (running): {}".format(bvtv_R2.get_result()))

            if args.to_save:
                nib.save(
                    nib.Nifti1Image(val_outputs.astype(np.uint8), original_affine), os.path.join(output_directory, img_name)
                )

        print("Overall Mean Dice: {}".format(np.mean(dice_list_case)))
        print("Overall Mean IoU: {}".format(np.mean(iou_list_case)))
        print("Overall BV R2-value: {}".format(bv_R2.get_result()))
        print("Overall TV R2-value: {}".format(tv_R2.get_result()))
        print("Overall BV/TV R2-value: {}".format(bvtv_R2.get_result()))

# Run inference on test images
def test():
    args = parser.parse_args()
    args.test_mode = True
    args.val_mode = False
    output_directory = "finetune/outputs_test/" + args.exp_name
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)
    val_loader = get_loader(args)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    pretrained_pth = args.model
    if args.use_dilated_swin:
        model = DilSwinUNETR(
            img_size=(args.roi_x, args.roi_y, args.roi_z),
            in_channels=args.in_channels,
            out_channels=args.out_channels,
            feature_size=args.feature_size,
            drop_rate=0.0,
            attn_drop_rate=0.0,
            dropout_path_rate=0.0,
            use_checkpoint=args.use_checkpoint,
        )
    else:
        model = SwinUNETR(
            img_size=(args.roi_x, args.roi_y, args.roi_z),
            in_channels=args.in_channels,
            out_channels=args.out_channels,
            feature_size=args.feature_size,
            drop_rate=0.0,
            attn_drop_rate=0.0,
            dropout_path_rate=0.0,
            use_checkpoint=args.use_checkpoint,
        )

    model_dict = torch.load(pretrained_pth)["state_dict"]
    model.load_state_dict(model_dict)
    model.eval()
    model.to(device)

    with torch.no_grad():
        dice_list_case = []
        bv_R2 = R2Metric()
        bvtv_R2 = R2Metric()

        for i, batch in enumerate(val_loader):
            val_inputs = batch["image"].cuda()
            original_affine = batch["image_meta_dict"]["original_affine"][0].numpy()
            _,h, w, d, _, _, _, _ = batch["image_meta_dict"]['dim'][0].numpy()
            target_shape = (h, w, d)
            img_name = batch["image_meta_dict"]["filename_or_obj"][0].split("/")[-1]
            print("Inference on case {}".format(img_name))
            val_outputs = sliding_window_inference(
                val_inputs, (args.roi_x, args.roi_y, args.roi_z), 4, model, overlap=args.infer_overlap, mode="gaussian" #TODO remove mode
            )
            val_outputs = torch.softmax(val_outputs, 1).cpu().numpy()
            val_outputs = np.argmax(val_outputs, axis=1).astype(np.uint8)[0]
            val_outputs = resample_3d(val_outputs, target_shape)

            if args.to_save:
                nib.save(
                    nib.Nifti1Image(val_outputs.astype(np.uint8), original_affine), os.path.join(output_directory, img_name)
                )



if __name__ == "__main__":
    main() # validation
    test() # test inference
