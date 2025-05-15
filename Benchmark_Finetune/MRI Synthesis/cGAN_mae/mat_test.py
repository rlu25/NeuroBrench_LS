# -*- coding: utf-8 -*-
import os
import numpy as np
import nibabel as nib
import h5py
import pandas as pd
from skimage.transform import resize
from scipy.ndimage import zoom
from tqdm import tqdm  
import ants

# ------------------------------
# CONFIG
# ------------------------------
base_dir = "../../datasets"
csv_test = "../../datasets/test.csv"
output_h5 = "./datasets/nips/test/data.mat"
output_csv = "./datasets/nips/test/slice_metadata.csv"
target_shape = (256, 256)

# ------------------------------
# Load Subject Paths
# ------------------------------
def construct_path(row):
    return os.path.join(base_dir, row["dataset"], "Skull-stripped", row["sub_id"], row["ses_id"], "anat")

def find_scan(path, keyword):
    if not os.path.exists(path):
        return None
    for f in os.listdir(path):
        if f.endswith(".nii.gz") and keyword in f and not f.endswith("_mask.nii.gz"):
            return os.path.join(path, f)
    return None

def build_pairs(csv_path):
    df = pd.read_csv(csv_path)
    t1_paths, t2_paths, subj_ids = [], [], []
    for _, row in df.iterrows():
        anat_path = construct_path(row)
        t1 = find_scan(anat_path, "T1w")
        t2 = find_scan(anat_path, "T2w")
        if t1 and t2:
            t1_paths.append(t1)
            t2_paths.append(t2)
            subj_ids.append(row["sub_id"])
    return list(zip(t1_paths, t2_paths, subj_ids))

def build_all_pairs(csv_path, folder_base):
    """
    Combine scan pairs from both a CSV-driven dataset and a folder-driven dataset.

    Parameters:
    - csv_path (str): Path to CSV file with sub_id/ses_id/etc. for dHCP-style data
    - folder_base (str): Path to base folder containing BCP_T1w/img and BCP_T2w/img for BCP-style data

    Returns:
    - List of (t1_path, t2_path) tuples from both datasets
    """
    subj_ids = []
    # === dHCP-style from CSV ===
    df = pd.read_csv(csv_path)
    dhcp_pairs = []
    for _, row in df.iterrows():
        anat_path = construct_path(row)
        t1 = find_scan(anat_path, "T1w")
        t2 = find_scan(anat_path, "T2w")
        
        if t1 and t2 and row["sub_id"] == "sub-HCA6089375":
            dhcp_pairs.append((t1, t2))
            subj_ids.append(row["sub_id"])


    # === BCP-style from folders ===
    bcp_pairs = []
    t1_root = os.path.join(folder_base, "BCP_T1w", "img")
    t2_root = os.path.join(folder_base, "BCP_T2w", "img")
    subjects = sorted(os.listdir(t1_root))
    # print(subjects)
    subjects = [subjects[0]]
    for subj in subjects:
        t1_path = os.path.join(t1_root, subj)
        t2_path = os.path.join(t2_root, subj)
        if os.path.exists(t1_path) and os.path.exists(t2_path):
            bcp_pairs.append((t1_path, t2_path))
            subj_ids.append(subj.split(".")[0])
    print(list(zip(dhcp_pairs + bcp_pairs, subj_ids)))
    return list(zip(dhcp_pairs + bcp_pairs, subj_ids))

# ------------------------------
# Main Processing
# ------------------------------

def resize_to_match(source, target):
    factors = np.array(target.shape) / np.array(source.shape)
    return zoom(source, zoom=factors, order=1)  # linear interpolation
 
def normalize(data):
    data = np.clip(data, 0, None)
    return (data - data.min()) / (data.max() - data.min() + 1e-8)

def crop_nonzero_volume(volume, margin=4):
    nonzero = np.argwhere(volume > 0)
    if nonzero.size == 0:
        return volume, tuple(slice(0, s) for s in volume.shape)
    min_coords = np.maximum(nonzero.min(axis=0) - margin, 0)
    max_coords = np.minimum(nonzero.max(axis=0) + 1 + margin, volume.shape)
    slices = tuple(slice(min_c, max_c) for min_c, max_c in zip(min_coords, max_coords))
    return volume[slices], slices

def extract_axial_slices(volume, shape=(256, 256)):
    slices = []
    for z in range(volume.shape[2]):
        img = resize(volume.get_fdata()[:, :, z], shape, preserve_range=True, anti_aliasing=True)
        slices.append(img)
    return np.stack(slices, axis=0)  # (N, H, W)

def extract_center_slices(img):
    data = img.get_fdata()
    data = np.squeeze(data)
    data, _ = crop_nonzero_volume(data)
    z_center = data.shape[2] // 2
    slice_indices = list(range(z_center - 25, z_center + 26))
    target_shape = (256, 256)
    slices = np.stack([
        resize(data[:, :, idx], target_shape, preserve_range=True, anti_aliasing=True)
        for idx in slice_indices
    ], axis=0)  # shape (256, 256, 3)
    slices = (slices - slices.min()) / (slices.max() - slices.min() + 1e-8)
    return slices

def process_and_save(pairs, h5_path, csv_path):
    all_t1, all_t2, meta = [], [], []

    for [t1_path, t2_path], subj_id in tqdm(pairs, desc="Processing subjects"):
        
        t1_nii = nib.load(t1_path)
        t1 = ants.image_read(t1_path)
        t2 = ants.image_read(t2_path)
        reg = ants.registration(fixed=t1, moving=t2, type_of_transform='Affine')
        reg['warpedmovout'].to_filename("temp.nii.gz")
        reg_img = reg['warpedmovout']

        # Convert to numpy
        reg_np = reg_img.numpy()

        # Construct affine matrix (ANTs uses RAS+ orientation)
        spacing = np.array(reg_img.spacing)
        direction = np.array(reg_img.direction).reshape(3, 3)
        origin = np.array(reg_img.origin)

        affine = direction @ np.diag(spacing)
        affine = np.vstack([affine, origin])
        affine = np.column_stack([affine, [0, 0, 0, 1]])

        # Make NIfTI image
        t2_nii = nib.Nifti1Image(reg_np, affine)
        
        t1_slices = extract_axial_slices(t1_nii, shape=target_shape)
        t2_slices = extract_axial_slices(t2_nii, shape=target_shape)

        
        min_slices = min(t1_slices.shape[0], t2_slices.shape[0])
        for idx in range(min_slices):
            all_t1.append(t1_slices[idx])
            all_t2.append(t2_slices[idx])
            meta.append({
                "subject_id": subj_id,
                "slice_index": idx
            })

    all_t1 = np.stack(all_t1, axis=0)[:, np.newaxis]  # (N, 1, H, W)
    all_t2 = np.stack(all_t2, axis=0)[:, np.newaxis]

    os.makedirs(os.path.dirname(h5_path), exist_ok=True)
    with h5py.File(h5_path, 'w') as f:
        f.create_dataset('data_x', data=all_t1)
        f.create_dataset('data_y', data=all_t2)

    pd.DataFrame(meta).to_csv(csv_path, index=False)
    print(f"Saved slices to: {h5_path}")
    print(f"Saved metadata to: {csv_path}")

# ------------------------------
# Run Script
# ------------------------------
if __name__ == "__main__":
    test_pairs = build_all_pairs(csv_test, '../../test_data')
    process_and_save(test_pairs, output_h5, output_csv)
