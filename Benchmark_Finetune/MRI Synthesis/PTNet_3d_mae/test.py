"""
Evaluation and visualization script for PTGAN validation.
Processes paired T1w-T2w MRI scans, predicts using the trained model,
computes PSNR/SSIM/MSE, and generates HTML visual summaries.
"""

# Evaluation script adapted from the pix2pixHD PyTorch implementation:
# https://github.com/NVIDIA/pix2pixHD

import os
import pandas as pd

from options.test_options import TestOptions
from models.models import create_model
import nibabel as nib
from nibabel.orientations import axcodes2ornt, ornt_transform, aff2axcodes, apply_orientation
import numpy as np
import torch
from data.data_util import norm_img, patch_slicer
from tqdm import tqdm
from nilearn.image import resample_to_img
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import mean_squared_error as mse
from util.visualizer import Visualizer

from options.test_options import TestOptions
from models.models import create_model

def mask_background(pred, reference):
    # Wherever reference (e.g. T1 or T2 GT) is zero, set pred to zero
    mask = reference > 1e-6
    pred[~mask] = 0
    return pred

def robust_norm(img, clip_percentile=99.9):
    upper = np.percentile(img, clip_percentile)
    img = np.clip(img, 0, upper)
    return (img - img.min()) / (img.max() - img.min() + 1e-8)


def construct_path(row):
    return os.path.join(
        base_dir, row["dataset"], "Skull-stripped", row["sub_id"], row["ses_id"], "anat"
    )

def find_scan(path, keyword):
    if not os.path.exists(path):
        return None
    for f in os.listdir(path):
        if f.endswith(".nii.gz") and keyword in f and not f.endswith("_mask.nii.gz"):
            return os.path.join(path, f)
    return None

def crop_nonzero_volume(volume, margin=0):
    """Crop the volume to the bounding box of non-zero values, with optional margin."""
    nonzero = np.argwhere(volume > 0)
    if nonzero.size == 0:
        return volume, (slice(0, volume.shape[0]), slice(0, volume.shape[1]), slice(0, volume.shape[2]))  # Return original if all zero
    min_coords = np.maximum(nonzero.min(axis=0) - margin, 0)
    max_coords = np.minimum(nonzero.max(axis=0) + 1 + margin, volume.shape)
    slices = tuple(slice(min_c, max_c) for min_c, max_c in zip(min_coords, max_coords))
    return volume[slices], slices

def preprocess_volume(path, target_spacing=(1.0, 1.0, 1.0)):
    nii = nib.load(path)
    volume = np.squeeze(nii.get_fdata())
    orig_ornt = axcodes2ornt(aff2axcodes(nii.affine))
    ras_ornt = axcodes2ornt(('R', 'A', 'S'))
    transform = ornt_transform(orig_ornt, ras_ornt)
    volume = apply_orientation(volume, transform)

  
    ori = volume.copy()
    volume = robust_norm(volume, 99.90)

    return volume, nii.affine, nii.header, ori

def predict_volume(model, volume, crop_margin=4):
    cropped_vol = volume
    pred = np.full(volume.shape, 0, dtype=np.float32)
    tmp_norm = np.zeros(volume.shape)

    scan_patches, _, tmp_idx = patch_slicer(
        cropped_vol, cropped_vol,
        patch_size=[64, 64, 64], 
        stride=[32, 32, 32],
        remove_bg=True, test=True, ori_path=None
    )

    for idx, patch in enumerate(scan_patches):
        ipt = torch.from_numpy(patch).to(dtype=torch.float).cuda()
        tmp_pred = model(ipt.reshape((1, 1,) + ipt.shape))
        patch_idx = tmp_idx[idx]
        patch_idx = (
            slice(patch_idx[0], patch_idx[1]), slice(patch_idx[2], patch_idx[3]),
            slice(patch_idx[4], patch_idx[5]))
        pred[patch_idx] += torch.squeeze(tmp_pred).detach().cpu().numpy()
        tmp_norm[patch_idx] += 1

    pred[tmp_norm > 0] /= tmp_norm[tmp_norm > 0]
    return pred

def evaluate_and_visualize(pred, target, input_image, scan_name, des_path, vis, i, age=None, age_group=None):
    ext = '.nii.gz'

    # Load and orient target volume to RAS
    tgt_data = np.squeeze(target.get_fdata())
    tgt_ornt = axcodes2ornt(aff2axcodes(target.affine))
    ras_ornt = axcodes2ornt(('R', 'A', 'S'))
    tgt_transform = ornt_transform(tgt_ornt, ras_ornt)
    tgt_data = apply_orientation(tgt_data, tgt_transform)

    # Step 1: Create NIfTI images
    pred_img = nib.Nifti1Image(pred, target.affine, header=target.header)
    tgt_img = nib.Nifti1Image(tgt_data, target.affine, header=target.header)

    # Step 2: Resample target to prediction space (for masking)
    tgt_resampled = resample_to_img(tgt_img, pred_img, interpolation='continuous', force_resample=True)
    tgt_data_resampled = tgt_resampled.get_fdata()

    # Step 3: Mask prediction using resampled target
    pred_masked = mask_background(pred_img.get_fdata(), tgt_data_resampled)

    # Step 4: Save masked prediction and resampled target
    nib.save(nib.Nifti1Image(pred_masked, pred_img.affine, header=pred_img.header),
             os.path.join(des_path, scan_name.split(ext)[0] + '_converted' + ext))
    nib.save(tgt_resampled, os.path.join(des_path, scan_name.split(ext)[0] + '_ori' + ext))

    # Step 5: Normalize after resampling and masking
    gt_norm = robust_norm(tgt_data_resampled, 99.90)
    pred_norm = robust_norm(pred_masked, 99.90)
    

    # Step 6: Optional resampling to match shapes
    pred_img_resampled = resample_to_img(
        nib.Nifti1Image(pred_norm, target.affine),
        nib.Nifti1Image(gt_norm, target.affine),
        interpolation='continuous',
        force_resample=True
    )
    pred_norm = pred_img_resampled.get_fdata()

    # Step 4: Save masked prediction and resampled target
    nib.save(nib.Nifti1Image(pred_masked, pred_img.affine, header=pred_img.header),
             os.path.join(des_path, scan_name.split(ext)[0] + '_converted' + ext))
    nib.save(tgt_resampled, os.path.join(des_path, scan_name.split(ext)[0] + '_ori' + ext))

    # Normalize input for visualization
    input_vis = input_image.copy()
    input_vis[input_vis < 0] = 0
    input_vis = (input_vis - input_vis.min()) / (input_vis.max() - input_vis.min() + 1e-6)


    # Ensure shape match before metric computation
    assert pred_norm.shape == gt_norm.shape, f"Shape mismatch: pred {pred_norm.shape}, gt {gt_norm.shape}"

    # Compute metrics
    psnr_val = psnr(gt_norm, pred_norm, data_range=1.0)
    ssim_val = ssim(gt_norm, pred_norm, data_range=1.0)
    mse_val = mse(gt_norm, pred_norm)

    metrics = {
        "psnr": psnr_val,
        "ssim": ssim_val,
        "mse": mse_val,
        "age": age,
        "age_group": age_group,
    }

    # Visualization
    vis.display_current_results({
        'prediction': pred_norm,
        'ground_truth': gt_norm,
        'input_modality': input_vis
    }, epoch=0, step=i)

    return metrics


def main():
    global base_dir
    opt = TestOptions().parse(save=False)
    opt.nThreads = 0  # test code only supports nThreads = 1
    opt.batchSize = 1  # test code only supports batchSize = 1
    opt.serial_batches = True  # no shuffle
    opt.no_flip = True  # no flip
    # opt.dataroot = '../datasets/'
    opt.checkpoints_dir = '../ckpt'
    

    opt.name = 'mae'
    opt.model = 'local3D_trans'

    test_csv = '../datasets/test+fetal.csv'
    base_dir = "../../datasets/"

    df_test = pd.read_csv(test_csv)
    test_paths = []
    torch.backends.cudnn.benchmark = True
    vis = Visualizer(opt)

    for _, row in df_test.iterrows():
        
            anat_path = construct_path(row)
            scan_path = find_scan(anat_path, "T1w")
            tgt_path = find_scan(anat_path, "T2w")
            if scan_path and tgt_path:
                test_paths.append({
                    "scan_path": scan_path,
                    "tgt_path": tgt_path,
                    "scan_name": os.path.basename(scan_path),
                    "age": row.get("age", None),
                    "age_group": row.get("age_group", None)
                })

    G = create_model(opt)
    assert torch.cuda.is_available(), "CUDA is required for model inference"
    G.cuda()
    G.eval()
    model_dir = os.path.join(opt.checkpoints_dir, opt.name)
    result_dir = model_dir.replace('ckpt', 'results')
    # res = os.path.join(result_dir, opt.name)
    ckpt_lst = ['latest.pth']
    print(ckpt_lst)

    metrics_list = []

    with torch.no_grad():
        for ckpts in ckpt_lst:
            print(ckpts)
            ckpt_path = os.path.join(model_dir, ckpts)
            des_path = os.path.join(result_dir, ckpts.split('.')[0] + '_outputs')  # e.g., t12t2_3d_outputs
            if not os.path.isdir(des_path):
                os.mkdir(des_path)

            G.load_state_dict(torch.load(ckpt_path), strict=True)
            G.eval()
            for i, entry in enumerate(tqdm(test_paths, desc=f"Testing {ckpts}")):
                scan_path = entry["scan_path"]
                tgt_path = entry["tgt_path"]
                scan_name = entry["scan_name"]
                age = entry["age"]
                age_group = entry["age_group"]

                input_volume, _, _, ori = preprocess_volume(scan_path)
                pred = predict_volume(G, input_volume)
                # pred = (pred + 1) / 2

                if os.path.exists(tgt_path):
                    tgt_scan = nib.load(tgt_path)
                    metrics = evaluate_and_visualize(pred, tgt_scan, ori, scan_name, des_path, vis, i, age, age_group)
                    metrics_list.append({
                        "image_index": i,
                        "image_name": scan_name,
                        **metrics
                    })
                    metrics_df = pd.DataFrame(metrics_list)
                    metrics_df.to_csv(os.path.join(result_dir, 'metrics_summary.csv'), index=False)
                    # print(f"[INFO] Processed {scan_name}: PSNR={metrics['psnr']:.4f}, SSIM={metrics['ssim']:.4f}, MSE={metrics['mse']:.4f}")
                else:
                    print(f"[WARN] Target scan not found for {scan_name}")
                    continue

if __name__ == "__main__":
    main()
