"""
Convert to h5 utility.
Sample command to create new dataset
- python3 utils/convert_h5.py -dd /home/masterthesis/shayan/nas_drive/Data_Neuro/OASISchallenge/FS -ld /home/masterthesis/shayan/nas_drive/Data_Neuro/OASISchallenge -trv datasets/train_volumes.txt -tev datasets/test_volumes.txt -id MALC -rc Neo -o COR -df datasets/MALC/coronal
- python utils/convert_h5.py -dd /home/masterthesis/shayan/nas_drive/Data_Neuro/IXI/IXI_FS -ld /home/masterthesis/shayan/nas_drive/Data_Neuro/IXI/IXI_FS -ds 98,2 -rc FS -o COR -df datasets/IXI/coronal
"""

import argparse
import os

import h5py
import numpy as np

import common_utils
import data_utils as du
import preprocessor as preprocessor
import glob

def find_all_img_label_pairs(base_dir):
    img_dirs = sorted([d for d in os.listdir(base_dir) if d.endswith('_img')])
    pairs = []
    for img_dir in img_dirs:
        label_dir = img_dir.replace('_img', '_label')
        img_path_full = os.path.join(base_dir, img_dir)
        label_path_full = os.path.join(base_dir, label_dir)
        if not os.path.exists(label_path_full):
            print(f"[WARNING] Skipping: label folder {label_path_full} does not exist.")
            continue
        imgs = sorted(glob.glob(os.path.join(img_path_full, "*.nii.gz")))
        for img in imgs:
            fname = os.path.basename(img)
            label = os.path.join(label_path_full, fname.replace('.nii.gz', '_seg.nii.gz'))
            if os.path.exists(label):
                pairs.append((img, label))
            else:
                print(f"[WARNING] Label not found for: {img}")
    return pairs

def load_file_paths(data_dir):
    pairs = find_all_img_label_pairs(data_dir)
    return pairs

    
def pad_or_crop(vol, target_shape=(256, 256)):
    """
    vol: np.ndarray of shape (num_slices, H, W)
    target_shape: desired (H, W)
    Returns: np.ndarray of shape (num_slices, target_H, target_W)
    """
    padded = []
    for slice_2d in vol:
        h, w = slice_2d.shape
        target_h, target_w = target_shape

        # Pad if needed
        pad_h = max(0, target_h - h)
        pad_w = max(0, target_w - w)
        if pad_h > 0 or pad_w > 0:
            slice_2d = np.pad(
                slice_2d,
                ((pad_h // 2, pad_h - pad_h // 2), (pad_w // 2, pad_w - pad_w // 2)),
                mode="constant", constant_values=0
            )

        # Crop if needed
        h, w = slice_2d.shape
        crop_x = max(0, (h - target_h) // 2)
        crop_y = max(0, (w - target_w) // 2)
        slice_2d = slice_2d[crop_x:crop_x + target_h, crop_y:crop_y + target_w]

        padded.append(slice_2d)
    return np.stack(padded)




def _write_h5(data, label, class_weights, weights, f, mode):
    target_shape = (256, 256)  # or set dynamically from max shape
    data = [pad_or_crop(d, target_shape) for d in data]
    label = [pad_or_crop(l, target_shape) for l in label]
    class_weights = [pad_or_crop(cw, target_shape) for cw in class_weights]
    no_slices, H, W = data[0].shape
    with h5py.File(f[mode]['data'], "w") as data_handle:
        data_handle.create_dataset("data", data=np.concatenate(data).reshape((-1, H, W)))
    with h5py.File(f[mode]['label'], "w") as label_handle:
        label_handle.create_dataset("label", data=np.concatenate(label).reshape((-1, H, W)))
    with h5py.File(f[mode]['weights'], "w") as weights_handle:
        weights_handle.create_dataset("weights", data=np.concatenate(weights))
    with h5py.File(f[mode]['class_weights'], "w") as class_weights_handle:
        class_weights_handle.create_dataset("class_weights", data=np.concatenate(
            class_weights).reshape((-1, H, W)))


def convert_h5(train_data, test_data, f, orientation=preprocessor.ORIENTATION['coronal']):
    # Data splitting
    train_file_paths = load_file_paths(train_data)
    test_file_paths = load_file_paths(test_data)
  
    print("Train dataset size: %d, Test dataset size: %d" % (len(train_file_paths), len(test_file_paths)))
    # loading,pre-processing and writing train data
    print("===Train data===")
    data_train, label_train, class_weights_train, weights_train, _ = du.load_dataset(train_file_paths,
                                                                                     orientation,
                                                                                     
                                                                                     return_weights=True,
                                                                                     reduce_slices=True,
                                                                                     remove_black=True)

    _write_h5(data_train, label_train, class_weights_train, weights_train, f, mode='train')

    # loading,pre-processing and writing test data
    print("===Test data===")
    data_test, label_test, class_weights_test, weights_test, _ = du.load_dataset(test_file_paths,
                                                                                 orientation,
                                                                                 
                                                                                 return_weights=True,
                                                                                 reduce_slices=True,
                                                                                 remove_black=True)

    _write_h5(data_test, label_test, class_weights_test, weights_test, f, mode='test')


if __name__ == "__main__":
    import sys
    print("* Start *")
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_data", required=True, help="Path to train_data folder (with *_img and *_label)")
    parser.add_argument("--test_data", required=True, help="Path to test_data folder (with *_img and *_label)")
    parser.add_argument('--destination_folder', '-df', required=True, help='Path where to generate the h5 files')
    parser.add_argument('--orientation', '-o', required=True, help='Valid options are COR, AXI, SAG')
    

    sys.argv = [
        "convert_h5_neurips.py",
        "--train_data", "/opt/localdata/data/usr-envs/ruiying/Code/NeurIPS/QuickNet/quickNAT_MAE/train_data",
        "--test_data", "/opt/localdata/data/usr-envs/ruiying/Code/NeurIPS/QuickNet/quickNAT_MAE/test_data",
        "-o", "AXI",
        "-df", "/opt/localdata/data/usr-envs/ruiying/Code/NeurIPS/QuickNet/quickNAT_MAE/train_h5_new/axial"
    ]

    args = parser.parse_args()

    common_utils.create_if_not(args.destination_folder)

    f = {
        'train': {
            "data": os.path.join(args.destination_folder, "Data_train.h5"),
            "label": os.path.join(args.destination_folder, "Label_train.h5"),
            "weights": os.path.join(args.destination_folder, "Weight_train.h5"),
            "class_weights": os.path.join(args.destination_folder, "Class_Weight_train.h5"),
        },
        'test': {
            "data": os.path.join(args.destination_folder, "Data_test.h5"),
            "label": os.path.join(args.destination_folder, "Label_test.h5"),
            "weights": os.path.join(args.destination_folder, "Weight_test.h5"),
            "class_weights": os.path.join(args.destination_folder, "Class_Weight_test.h5")
        }
    }

    convert_h5(args.train_data, args.test_data, f,
               args.orientation)
    print("* Finish *")
