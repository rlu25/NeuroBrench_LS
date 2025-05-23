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


def apply_split(data_split, data_dir, label_dir):
    file_paths = du.load_file_paths(data_dir, label_dir)
    print("Total no of volumes to process : %d" % len(file_paths))
    train_ratio, test_ratio = data_split.split(",")
    train_len = int((int(train_ratio) / 100) * len(file_paths))
    train_idx = np.random.choice(len(file_paths), train_len, replace=False)
    test_idx = np.array([i for i in range(len(file_paths)) if i not in train_idx])
    train_file_paths = [file_paths[i] for i in train_idx]
    test_file_paths = [file_paths[i] for i in test_idx]
    return train_file_paths, test_file_paths


def _write_h5(data, label, class_weights, weights, f, mode):
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


def convert_h5(data_dir, label_dir, data_split, train_volumes, test_volumes, f, data_id, remap_config='Neo',
               orientation=preprocessor.ORIENTATION['coronal']):
    # Data splitting
    if data_split:
        train_file_paths, test_file_paths = apply_split(data_split, data_dir, label_dir)
    elif train_volumes and test_volumes:
        train_file_paths = du.load_file_paths(data_dir, label_dir, data_id, train_volumes)
        test_file_paths = du.load_file_paths(data_dir, label_dir, data_id, test_volumes)
    else:
        raise ValueError('You must either provide the split ratio or a train, train dataset list')

    print("Train dataset size: %d, Test dataset size: %d" % (len(train_file_paths), len(test_file_paths)))
    # loading,pre-processing and writing train data
    print("===Train data===")
    data_train, label_train, class_weights_train, weights_train, _ = du.load_dataset(train_file_paths,
                                                                                     orientation,
                                                                                     remap_config=remap_config,
                                                                                     return_weights=True,
                                                                                     reduce_slices=True,
                                                                                     remove_black=True)

    _write_h5(data_train, label_train, class_weights_train, weights_train, f, mode='train')

    # loading,pre-processing and writing test data
    print("===Test data===")
    data_test, label_test, class_weights_test, weights_test, _ = du.load_dataset(test_file_paths,
                                                                                 orientation,
                                                                                 remap_config=remap_config,
                                                                                 return_weights=True,
                                                                                 reduce_slices=True,
                                                                                 remove_black=True)

    _write_h5(data_test, label_test, class_weights_test, weights_test, f, mode='test')


if __name__ == "__main__":
    import sys
    print("* Start *")
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', '-dd', required=True,
                        help='Base directory of the data folder. This folder should contain one folder per volume.')
    parser.add_argument('--label_dir', '-ld', required=True,
                        help='Base directory of all the label files. This folder should have one file per volumn with same name as the corresponding volumn folder name inside data_dir')
    parser.add_argument('--data_split', '-ds', required=False,
                        help='Ratio to split data randomly into train and test. input e.g. 80,20')
    parser.add_argument('--train_volumes', '-trv', required=False,
                        help='Path to a text file containing the list of volumes to be used for training')
    parser.add_argument('--test_volumes', '-tev', required=False,
                        help='Path to a text file containing the list of volumes to be used for testing')
    parser.add_argument('--data_id', '-id', required=True, help='Valid options are "MALC", "ADNI", "CANDI" and "IBSR"')
    parser.add_argument('--remap_config', '-rc', required=False, help='Valid options are "FS" and "Neo"')
    parser.add_argument('--orientation', '-o', required=True, help='Valid options are COR, AXI, SAG')
    parser.add_argument('--destination_folder', '-df', required=True, help='Path where to generate the h5 files')

    sys.argv = [
        "convert_h5.py",
        "-dd", "/opt/localdata/data/usr-envs/ruiying/Code/NeurIPS/QuickNet/quickNAT_train/train_data/oasis_image",
        "-ld", "/opt/localdata/data/usr-envs/ruiying/Code/NeurIPS/QuickNet/quickNAT_train/train_data/oasis_label",
        "-trv", "/opt/localdata/data/usr-envs/ruiying/Code/NeurIPS/QuickNet/quickNAT_train/train_data/train_volumes.txt",
        "-tev", "/opt/localdata/data/usr-envs/ruiying/Code/NeurIPS/QuickNet/quickNAT_train/train_data/test_volumes.txt",
        "-id", "oasis",
        "-o", "AXI",
        "-df", "/opt/localdata/data/usr-envs/ruiying/Code/NeurIPS/QuickNet/quickNAT_train/train_h5/oasis/axial"
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

    convert_h5(args.data_dir, args.label_dir, args.data_split, args.train_volumes, args.test_volumes, f,
               args.data_id,
               args.remap_config,
               args.orientation)
    print("* Finish *")
