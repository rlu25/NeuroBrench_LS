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
# Load CSV-based Subject Lists
# ------------------------------
base_dir = "../../datasets"
csv_train = "../../datasets/train_split.csv"
csv_val = "../../datasets/valid_split.csv"

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

def build_pairs(csv_path):
    df = pd.read_csv(csv_path)
    t1_paths = []
    t2_paths = []
    for _, row in df.iterrows():
        anat_path = construct_path(row)
        # print(row)
        # if row["dataset"] == "HBCD":
        t1 = find_scan(anat_path, "T1w")
        t2 = find_scan(anat_path, "T2w")
        if t1 and t2:
            t1_paths.append(t1)
            t2_paths.append(t2)
    return list(zip(t1_paths, t2_paths))

training_pairs = build_pairs(csv_train)
val_pairs = build_pairs(csv_val)

# ------------------------------
# Processing Function with Co-registration
# ------------------------------
def resize_to_match(source, target):
    factors = np.array(target.shape) / np.array(source.shape)
    return zoom(source, zoom=factors, order=1)  # linear interpolation

def crop_nonzero_volume(volume, margin=4):
    nonzero = np.argwhere(volume > 0)
    if nonzero.size == 0:
        return volume, tuple(slice(0, s) for s in volume.shape)
    min_coords = np.maximum(nonzero.min(axis=0) - margin, 0)
    max_coords = np.minimum(nonzero.max(axis=0) + 1 + margin, volume.shape)
    slices = tuple(slice(min_c, max_c) for min_c, max_c in zip(min_coords, max_coords))
    return volume[slices], slices

def extract_center_slices(img):
    data = img.get_fdata()
    data = np.squeeze(data)
    data, _ = crop_nonzero_volume(data)
    z_center = data.shape[2] // 2
    slice_indices = list(range(z_center - 1, z_center + 2))
    target_shape = (256, 256)
    slices = np.stack([
        resize(data[:, :, idx], target_shape, preserve_range=True, anti_aliasing=True)
        for idx in slice_indices
    ], axis=-1)  # shape (256, 256, 3)
    slices = (slices - slices.min()) / (slices.max() - slices.min() + 1e-8)
    return slices

def process_pair(t1_path, t2_path):
    nib.Nifti1Header.quaternion_threshold = -1e-06

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


    processed_t1 = extract_center_slices(t1_nii)
    processed_t2 = extract_center_slices(t2_nii)
    # print(f"\nProcessed {t1_path} and {t2_path}")

    return processed_t1, processed_t2

# ------------------------------
# Process and Save Function
# ------------------------------
def process_and_save(pairs, output_file, csv_save_path='./datasets/test+fetal_sub.csv'):
    all_data_x, all_data_y = [], []
    metadata = []

    for t1_file, t2_file in tqdm(pairs, desc="Processing subjects"):
        subj_id = os.path.basename(t1_file).split("_")[0]
        t1_slices, t2_slices = process_pair(t1_file, t2_file)

        all_data_x.append(t1_slices)
        all_data_y.append(t2_slices)
        metadata.append({
            "subject_id": subj_id,
            "slice_index": 0,  # center slice index placeholder
            "t1_path": t1_file,
            "t2_path": t2_file
        })

    all_data_x = np.stack(all_data_x, axis=2)  # shape: (256, 256, N, 3)
    all_data_y = np.stack(all_data_y, axis=2)

    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    with h5py.File(output_file, 'w') as f:
        f.create_dataset('data_x', data=all_data_x)
        f.create_dataset('data_y', data=all_data_y)
    print("Saved to:", output_file)

    df = pd.DataFrame(metadata)
    df.to_csv(csv_save_path, index=False)
    print("Saved metadata to:", csv_save_path)

# ------------------------------
# Run All
# ------------------------------
print("\nProcessing Training Data:")
process_and_save(training_pairs, './datasets/nips/train/data.mat', './datasets/train_sub.csv')

print("\nProcessing Validation Data:")
process_and_save(val_pairs, './datasets/nips/val/data.mat', './datasets/valid_sub.csv')
