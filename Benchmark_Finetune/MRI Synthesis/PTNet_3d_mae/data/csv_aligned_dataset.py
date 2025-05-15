# data/csv_aligned_dataset.py
# Direction: T1w -> T2w
import random

import torch
import numpy as np
import pandas as pd
import nibabel as nib
import os
from data.base_dataset import BaseDataset
from nilearn.image import resample_to_img


def robust_norm(img, clip_percentile=99.9):
    upper = np.percentile(img, clip_percentile)
    img = np.clip(img, 0, upper)
    return (img - img.min()) / (img.max() - img.min() + 1e-8)


class CSVAlignedDataset(BaseDataset):
    def initialize(self, opt):
        self.opt = opt
        self.patch_size = (64, 64, 64)
        self.df = pd.read_csv(opt.train_csv if opt.phase == 'train' else opt.valid_csv)
        self.base_dir = opt.base_dir
        self.sample_list = self.build_path_list()

    def build_path_list(self):
        sample_list = []
        for _, row in self.df.iterrows():
            anat_path = os.path.join(
                self.base_dir, row["dataset"], "Skull-stripped", row["sub_id"], row["ses_id"], "anat"
            )
            input_path = self.find_scan(anat_path, "T1w")
            target_path = self.find_scan(anat_path, "T2w")
            if input_path and target_path:
                sample_list.append((input_path, target_path, row))
        return sample_list

    def find_scan(self, path, keyword):
        if not os.path.exists(path):
            return None
        for f in os.listdir(path):
            if f.endswith(".nii.gz") and keyword in f and not f.endswith("_mask.nii.gz"):
                return os.path.join(path, f)
        return None

    def __getitem__(self, index):
        import nibabel as nib
        input_path, target_path, _ = self.sample_list[index]


        t1_img = nib.load(input_path)
        t2_img = nib.load(target_path)

        # Reorient both T1 and T2 to RAS+ before resampling
        # Align both T1 and T2 volumes to RAS+ canonical orientation
        t1_img = nib.as_closest_canonical(t1_img)
        t2_img = nib.as_closest_canonical(t2_img)

        # Check voxel spacing consistency
        t1_zooms = t1_img.header.get_zooms()
        t2_zooms = t2_img.header.get_zooms()
        if not np.allclose(t1_zooms, t2_zooms, atol=0.1):
            print(f"[WARN] Zoom mismatch: T1={t1_zooms}, T2={t2_zooms} — consider resolution-standardizing step.")

        # Resample T2 to match T1 grid and affine if shapes differ
        if t1_img.shape != t2_img.shape:
            # print(f"[INFO] Resampling T2 to match T1 for {input_path}")
            t2_img = resample_to_img(t2_img, t1_img, interpolation='continuous', force_resample=True, copy_header=True)

        # Ensure both image shapes and affine matrices are now aligned
        assert t1_img.shape == t2_img.shape, f"Shape mismatch after resampling: {t1_img.shape} vs {t2_img.shape}"
        # if not np.allclose(t1_img.affine, t2_img.affine, atol=1e-2):
            # print(f"[WARN] Affine mismatch after resampling:\nT1 affine:\n{t1_img.affine}\nT2 affine:\n{t2_img.affine}")

        input_vol = np.squeeze(t1_img.get_fdata())
        target_vol = np.squeeze(t2_img.get_fdata())
        
        input_vol[input_vol < 0] = 0
        target_vol[target_vol < 0] = 0

        input_vol = input_vol.astype(np.float32)
        target_vol = target_vol.astype(np.float32)

        input_vol = robust_norm(input_vol)
        target_vol = robust_norm(target_vol)

        x, y, z = self.patch_size
        shape = input_vol.shape
        assert shape == target_vol.shape

        x_idx = random.randint(0, max(0, shape[0] - x))
        y_idx = random.randint(0, max(0, shape[1] - y))
        z_idx = random.randint(0, max(0, shape[2] - z))

        input_patch = input_vol[x_idx:x_idx + x, y_idx:y_idx + y, z_idx:z_idx + z]
        target_patch = target_vol[x_idx:x_idx + x, y_idx:y_idx + y, z_idx:z_idx + z]

        if not self.opt.no_flip and np.random.rand() > 0.5:
            axis = np.random.choice([0, 1, 2])
            input_patch = np.flip(input_patch, axis).copy()
            target_patch = np.flip(target_patch, axis).copy()

        input_tensor = torch.from_numpy(input_patch).float().unsqueeze(0)
        target_tensor = torch.from_numpy(target_patch).float().unsqueeze(0)

        return {

            'label': input_tensor,
            'image': target_tensor,
            'path': input_path,
        }

    def __len__(self):
        return len(self.sample_list)

    def name(self):
        return 'CSVAlignedDataset'
