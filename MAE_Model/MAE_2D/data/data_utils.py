import pandas as pd
import numpy as np
import os
import nibabel as nib
import torch
import torchio as tio
import glob
from collections import defaultdict
import nibabel as nib
from nibabel.orientations import io_orientation, axcodes2ornt, ornt_transform, apply_orientation
import csv


def list_mae_domains(path_to_mae_root):
    assert os.path.isdir(
        path_to_mae_root), '%s is not a directory' % path_to_mae_root
    candidate_dir = [os.path.join(path_to_mae_root, i) for i in os.listdir(
        path_to_mae_root) if i.endswith('_train')]

    assert len(
        candidate_dir) > 0, 'no folder ends with _train found in %s' % path_to_mae_root
    return candidate_dir

def load_exclusion_set(csv_path):
    exclusion = set()
    with open(csv_path, newline='') as f:
        reader = csv.DictReader(f)
        for row in reader:
            key = (row['dataset'], row['sub_id'], row['ses_id'])
            exclusion.add(key)
    return exclusion

def collect_all_scans_neurips(base_dir, ext=".nii.gz"):
    """
    Collects all relevant scans from the NeurIPS-Clean-T1T2 project, supporting:
    - ADHD, ABIDE: project/<mode>/*/sub-*/ses-*/anat/*.nii.gz
    - Infant_nonlabel: project/*.nii.gz
    - MOMMA_fetal: project/sub-*/ses-*/anat/*.nii.gz
    - Others: project/(Preprocessed|Skull-stripped)/sub-*/ses-*/anat/*.nii.gz
    """
    scan_dict = defaultdict(list)
    exclusion = load_exclusion_set(os.path.join(base_dir, "subj_list", "sub_list_final.csv"))


    for name in os.listdir(base_dir):
        project_path = os.path.join(base_dir, name)
        if not os.path.isdir(project_path):
            continue

        if name in ["ADHD", "ABIDE"]:
            for mode in ["Preprocessed", "Skull-stripped"]:
                pattern = os.path.join(project_path, mode, "*", "sub-*", "ses-*", "anat", f"*{ext}")
                for path in glob.glob(pattern):
                    if path.endswith("_mask.nii.gz"):
                        continue
                    parts = path.split(os.sep)
                    sub = next((p for p in parts if p.startswith("sub-")), "sub-UNKNOWN")
                    ses = next((p for p in parts if p.startswith("ses-")), "ses-UNKNOWN")
                    if (name, sub, ses) in exclusion:
                        continue
                    scan_dict[(name, sub, ses)].append(path)
                    # print(sub, ses, path)

        elif name == "Infant_nonlabel":
            pattern = os.path.join(project_path, f"*{ext}")
            for path in glob.glob(pattern):
                if path.endswith("_mask.nii.gz"):
                    continue
                if (name, sub, ses) in exclusion:
                    continue
                scan_dict[(name, "flat", "flat")].append(path)

        elif name == "MOMMA_fetal":
            pattern = os.path.join(project_path, "sub-*", "ses-*", "anat", f"*{ext}")
            for path in glob.glob(pattern):
                if path.endswith("_mask.nii.gz"):
                    continue
                parts = path.split(os.sep)
                sub = next((p for p in parts if p.startswith("sub-")), "sub-UNKNOWN")
                ses = next((p for p in parts if p.startswith("ses-")), "ses-UNKNOWN")
                if (name, sub, ses) in exclusion:
                    continue
                scan_dict[(name, sub, ses)].append(path)

        else:
            for mode in ["Preprocessed", "Skull-stripped"]:
                pattern = os.path.join(project_path, mode, "sub-*", "ses-*", "anat", f"*{ext}")
                for path in glob.glob(pattern):
                    if path.endswith("_mask.nii.gz"):
                        continue
                    parts = path.split(os.sep)
                    sub = next((p for p in parts if p.startswith("sub-")), "sub-UNKNOWN")
                    ses = next((p for p in parts if p.startswith("ses-")), "ses-UNKNOWN")
                    if (name, sub, ses) in exclusion:
                        continue
                    scan_dict[(name, sub, ses)].append(path)

    return scan_dict

def list_finetune_domains(tgt_path, src_path):
    assert os.path.isdir(
        tgt_path), '%s is not a directory' % tgt_path
    candidate_dir = [os.path.join(tgt_path, i) for i in os.listdir(
        tgt_path) if i.endswith('_train')]
    assert len(
        candidate_dir) > 0, 'no folder ends with _train found in %s' % tgt_path

    assert os.path.isdir(
        src_path), '%s is not a directory' % src_path
    candidate_dir2 = [os.path.join(src_path, i) for i in os.listdir(
        src_path) if i.endswith('_img')]
    assert len(
        candidate_dir2) > 0, 'no folder ends with _img found in %s' % src_path
    return candidate_dir, candidate_dir2


def list_scans(path_to_fld, ext):

    assert os.path.isdir(path_to_fld), '%s is not a directory' % path_to_fld
    scans = []
    for root, _, fnames in sorted(os.walk(path_to_fld)):
        for fname in fnames:
            if fname.endswith(ext) and not fname.startswith('.'):
                scan_path = os.path.join(root, fname)
                scans.append(scan_path)

    return scans


def random_flip(img):
    # img: numpy ndarray

    tmp_odd1 = np.random.random_sample()
    tmp_odd2 = np.random.random_sample()
    # tmp_odd3 = np.random.random_sample()

    # flip at 50% chance
    if tmp_odd1 <= 0.5:
        img = np.flip(img, axis=0)

    if tmp_odd2 <= 0.5:
        img = np.flip(img, axis=1)

    # if tmp_odd3 <= 0.5:
    #     img = np.flip(img, axis=2)

    return img


# def norm_img(img, percentile=100):
#     # img = (img - np.min(img)) / (np.percentile(img, percentile) - np.min(img))
#     denom = np.percentile(img, percentile) - np.min(img)
#     if denom < 1e-5:
#         img = np.zeros_like(img)  # or skip normalization
#     else:
#         img = (img - np.min(img)) / denom
#     return np.clip(img, 0, 1)
def norm_img(img, percentile=100):
    if np.isnan(img).any() or np.isinf(img).any():
        return None
    min_val = np.min(img)
    max_val = np.percentile(img, percentile)
    denom = max_val - min_val
    epsilon = 1e-8  # small constant to avoid divide-by-zero
    if denom < epsilon:
        return None
    img = (img - min_val) / denom
    return np.clip(img, 0, 1)

def get_bounds(slice_3d: torch.Tensor):
    """
    Compute bounding box of non-zero region across all Z slices.
    Input: slice_3d (Z, H, W)
    Output: [x_min, x_max, y_min, y_max]
    """
    arr = slice_3d.cpu().numpy()
    mask = arr > 1e-5
    coords = np.argwhere(mask)  # shape: [N, 3] — [z, x, y]

    if coords.size == 0:
        # fallback: full field of view
        _, h, w = arr.shape
        return [0, h, 0, w]

    x_min = coords[:, 1].min()
    x_max = coords[:, 1].max() + 1
    y_min = coords[:, 2].min()
    y_max = coords[:, 2].max() + 1
    return [x_min, x_max, y_min, y_max]
    


def load_axial_aligned(path):
    img = nib.load(path)
    data = img.get_fdata()
    affine = img.affine

    # Get orientation of current image
    current_ornt = io_orientation(affine)
    # Desired orientation is RAS (Right-Anterior-Superior), where axial is last axis (Z)
    target_ornt = axcodes2ornt(('R', 'A', 'S'))
    
    # Compute transformation and apply to data
    transform = ornt_transform(current_ornt, target_ornt)
    data_aligned = apply_orientation(data, transform)

    return data_aligned

# check affine matrix ----- wrong header 