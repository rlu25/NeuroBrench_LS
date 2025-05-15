# Copyright 2019 Image Analysis Lab, German Center for Neurodegenerative Diseases (DZNE), Bonn
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import sys
sys.path.insert(0, "FastSurfer/FastSurfer_MAE_path")

import glob

# IMPORTS
import time
from collections import defaultdict
from os.path import join
from pathlib import Path

import h5py
import nibabel as nib
import numpy as np
from numpy import typing as npt
import os 
from FastSurferCNN.data_loader.data_utils import (
    create_weight_mask,
    filter_blank_slices_thick,
    get_labels_from_lut,
    get_thick_slices,
    map_aparc_aseg2label,
    read_classes_from_lut,
    transform_axial,
    transform_sagittal,
    unify_lateralized_labels,
)
from FastSurferCNN.utils import logging
from FastSurferCNN.utils.parser_defaults import FASTSURFER_ROOT

LOGGER = logging.getLogger(__name__)


class H5pyDataset:
    """
    Class representing H5py Dataset.

    Attributes
    ----------
    dataset_name : str
        Path and name of hdf5-data_loader
    data_path : str
        Directory with images to load
    slice_thickness : int
        Number of pre- and succeeding slices
    orig_name : str
        Default name of original images
    aparc_name : str
        Default name for ground truth segmentations.
    aparc_nocc : str
        Segmentation without corpus callosum (used to mask this segmentation in ground truth).
        If the used segmentation was already processed, do not set this argument
    available_sizes : int
        Sizes of images in the dataset.
    max_weight : int
        Overall max weight for any voxel in weight mask.
    edge_weight : int
        Weight for edges in weight mask.
    hires_weight : int
        Weight for hires elements (sulci, WM strands, cortex border) in weight mask.
    gradient : bool
        Turn on to only use median weight frequency (no gradient)
    gm_mask : bool
        Turn on to add cortex mask for hires-processing.
    lut : pd.Dataframe
        DataFrame with ids present, name of ids, color for plotting
    labels : np.ndarray
        full label list
    labels_sag : np.ndarray
        sagittal label list
    lateralization : Dict
        dictionary mapping between left and right hemispheres
    subject_dirs : List[str]
        list ob subject directory names
    search_pattern : str
        Pattern to match files in directory
    data_set_size : int
        Number of subjects
    processing : str
        Use aseg, aparc or no specific mapping processing (Default: "aparc")

    Methods
    -------
    __init__
        Constructor
    _load_volumes
        Load image and segmentation volume
    transform
        Transform image along axis
    _pad_image
        Pad image with zeroes
    create_hdf5_dataset
        Create a hdf5 file
    """

    def __init__(self, params: dict, processing: str = "aparc"):
        """
        Construct H5pyDataset object.

        Parameters
        ----------
        params : Dict
            A dictionary containing the following keys:
            - dataset_name (str): Path and name of hdf5-data_loader
            - data_path (str): Directory with images to load
            - thickness (int): Number of pre- and succeeding slices
            - image_name (str): Default name of original images
            - gt_name (str): Default name for ground truth segmentations.
            - gt_nocc (str): Segmentation without corpus callosum (used to mask this segmentation in ground truth).
                            If the used segmentation was already processed, do not set this argument.
            - sizes (int): Sizes of images in the dataset.
            - max_weight (int): Overall max weight for any voxel in the weight mask.
            - edge_weight (int): Weight for edges in the weight mask.
            - hires_weight (int): Weight for hires elements (sulci, WM strands, cortex border) in the weight mask.
            - gradient (bool): Turn on to only use median weight frequency (no gradient)
            - gm_mask (bool): Turn on to add cortex mask for hires-processing.
            - lut (str): FreeSurfer-style Color Lookup Table with labels to use in the final prediction.
                        Has to have columns: ID	LabelName	R	G	B	A
            - sag_mask (tuple[str, str]): Suffixes of labels names to mask for final sagittal labels.
            - combi (str): Suffixes of labels names to combine.
            - pattern (str): Pattern to match files in the directory.
        processing : str, optional
            Use aseg (Default value = "aparc").

        Returns
        -------
        None
            This is a constructor function, it returns nothing.
        """
        self.dataset_name = params["dataset_name"]
        self.data_path = params["data_path"]
        self.slice_thickness = params["thickness"]
        self.orig_name = params["image_name"]
        self.aparc_name = params["gt_name"]
        self.aparc_nocc = params["gt_nocc"]
        self.processing = processing
        self.plane = params["plane"]

        self.available_sizes = params["sizes"]
        self.max_weight = params["max_weight"]
        self.edge_weight = params["edge_weight"]
        self.hires_weight = params["hires_weight"]
        self.gradient = params["gradient"]
        self.gm_mask = params["gm_mask"]

        self.lut = read_classes_from_lut(params["lut"])
        self.labels, self.labels_sag = get_labels_from_lut(self.lut, params["sag_mask"])
        self.lateralization = unify_lateralized_labels(self.lut, params["combi"])

        self.image_label_pairs = []
        for entry in os.listdir(self.data_path):
            if entry.endswith("_img"):
                label_dir = entry.replace("_img", "_label")
                if not os.path.isdir(os.path.join(self.data_path, label_dir)):
                    continue
                img_dir = os.path.join(self.data_path, entry)
                lbl_dir = os.path.join(self.data_path, label_dir)
                for img_file in sorted(glob.glob(os.path.join(img_dir, "*.nii.gz"))):
                    base = os.path.basename(img_file).replace(".nii.gz", "")
                    seg_file = os.path.join(lbl_dir, base + "_seg.nii.gz")
                    if os.path.exists(seg_file):
                        self.image_label_pairs.append((img_file, seg_file))


        self.data_set_size = len(self.image_label_pairs)

    def _load_volumes(
        self, subject_path: str
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, tuple]:
        """
        Load the given image and segmentation and gets the zoom values.

        Checks if an aseg-nocc file is set and loads it instead

        Parameters
        ----------
        subject_path : str
            Path to subject file.

        Returns
        -------
        ndarray
            Original image.
        ndarray
            Segmentation ground truth.
        ndarray
            Segmentation ground truth without corpus callosum.
        tuple
            Zoom values.
        """
        # Load the orig and extract voxel spacing information (x, y, and z dim)
        LOGGER.info(
            f"Processing intensity image {self.orig_name} and ground truth segmentation {self.aparc_name}"
        )
        orig = nib.load(subject_path[0])
        # Load the segmentation ground truth
        aseg = nib.load(subject_path[1])

        zoom = orig.header.get_zooms()[:3]
        zoom = np.asarray(zoom, dtype=np.float32)  # Ensure consistent shape
        orig = np.asarray(orig.get_fdata(), dtype=np.uint16)
        aseg = np.asarray(aseg.get_fdata(), dtype=np.uint8)
        if orig.ndim == 4 and orig.shape[-1] == 1:
            orig = np.squeeze(orig, axis=-1)
        if aseg.ndim == 4 and aseg.shape[-1] == 1:
            aseg = np.squeeze(aseg, axis=-1)
        # orig = np.asarray(orig.get_fdata())
        # aseg = np.asarray(aseg.get_fdata())
        
        
        
        aseg_nocc = None

        return orig, aseg, aseg_nocc, zoom

    def transform(
        self, plane: str, imgs: npt.NDArray, zoom: npt.NDArray
    ) -> tuple[npt.NDArray, npt.NDArray]:
        """
        Transform the image and zoom along the given axis.

        Parameters
        ----------
        plane : str
            Plane (sagittal, axial, ).
        imgs : npt.NDArray
            Input image.
        zoom : npt.NDArray
            Zoom factors.

        Returns
        -------
        npt.NDArray
            Transformed image.
        npt.NDArray
            Transformed zoom factors.
        """
        for i in range(len(imgs)):
            if self.plane == "sagittal":
                imgs[i] = transform_sagittal(imgs[i])
                zooms = zoom[::-1][:2]
            elif self.plane == "axial":
                imgs[i] = transform_axial(imgs[i])
                zooms = zoom[1:]
            else:
                zooms = zoom[:2]
        return imgs, zooms

    # def _pad_image(self, img: npt.NDArray, max_out: int) -> np.ndarray:
    #     """
    #     Pad the margins of the input image with zeros.

    #     Parameters
    #     ----------
    #     img : npt.NDArray
    #         Image array.
    #     max_out : int
    #         Size of output image.

    #     Returns
    #     -------
    #     np.ndarray
    #         0-padded image to the given size.
    #     """
    #     # Get correct size = max along shape
    #     h, w, d = img.shape
    #     LOGGER.info(f"Padding image from {img.shape} to {max_out}x{max_out}x{max_out}")
    #     padded_img = np.zeros((max_out, max_out, max_out), dtype=img.dtype)
    #     padded_img[0:h, 0:w, 0:d] = img
    #     return padded_img
    def _pad_crop_image(self, img: npt.NDArray, max_out: int) -> np.ndarray:
       
        h, w, d = img.shape
        padded_img = np.zeros((max_out, max_out, d), dtype=img.dtype)

        # Compute crop/pad indices for axis 0
        if h <= max_out:
            start_out_h = (max_out - h) // 2
            end_out_h = start_out_h + h
            start_in_h = 0
            end_in_h = h
        else:
            start_out_h = 0
            end_out_h = max_out
            start_in_h = (h - max_out) // 2
            end_in_h = start_in_h + max_out

        # Compute crop/pad indices for axis 1
        if w <= max_out:
            start_out_w = (max_out - w) // 2
            end_out_w = start_out_w + w
            start_in_w = 0
            end_in_w = w
        else:
            start_out_w = 0
            end_out_w = max_out
            start_in_w = (w - max_out) // 2
            end_in_w = start_in_w + max_out

        padded_img[start_out_h:end_out_h, start_out_w:end_out_w, :] = img[start_in_h:end_in_h, start_in_w:end_in_w, :]

        return padded_img

    def norm_img(self, img, percentile=100):
        img = (img - np.min(img)) / (np.percentile(img, percentile) - np.min(img))
        return np.clip(img, 0, 1)
    
    def create_hdf5_dataset(self, blt: int):
        """
        Create a hdf5 dataset.

        Parameters
        ----------
        blt : int
            Blank slice threshold.
        """
        data_per_size = defaultdict(lambda: defaultdict(list))
        start_d = time.time()

        for idx, current_subject in enumerate(self.image_label_pairs):
            try:
                # start = time.time()

                LOGGER.info(
                    f"Volume Nr: {idx + 1} Processing MRI Data from {current_subject}/{self.orig_name}"
                )

                orig, aseg, aseg_nocc, zoom = self._load_volumes(current_subject)
                #size, _, _ = orig.shape
                size = 256

                # mapped_aseg, mapped_aseg_sag = map_aparc_aseg2label(
                #     aseg,
                #     self.labels,
                #     self.labels_sag,
                #     self.lateralization,
                #     aseg_nocc,
                #     processing=self.processing,
                # )

                # if self.plane == "sagittal":
                #     mapped_aseg = mapped_aseg_sag
                #     weights = create_weight_mask(
                #         mapped_aseg,
                #         max_weight=self.max_weight,
                #         ctx_thresh=19,
                #         max_edge_weight=self.edge_weight,
                #         max_hires_weight=self.hires_weight,
                #         cortex_mask=self.gm_mask,
                #         gradient=self.gradient,
                #     )

                # else:
                #     weights = create_weight_mask(
                #         mapped_aseg,
                #         max_weight=self.max_weight,
                #         ctx_thresh=33,
                #         max_edge_weight=self.edge_weight,
                #         max_hires_weight=self.hires_weight,
                #         cortex_mask=self.gm_mask,
                #         gradient=self.gradient,
                #     )

                # print(
                #     f"Created weights with max_w {self.max_weight}, gradient {self.gradient},"
                #     f" edge_w {self.edge_weight}, hires_w {self.hires_weight}, gm_mask {self.gm_mask}"
                # )

                # transform volumes to correct shape
                weights = None
                [orig, aseg], zoom = self.transform(
                    self.plane, [orig, aseg], zoom
                )
                orig = self.norm_img(orig,99.5)*255
                #pad and crop
                target_size = 256  # or 256
                orig = self._pad_crop_image(orig, target_size)
                aseg = self._pad_crop_image(aseg, target_size)

                # Create Thick Slices, filter out blanks
                orig_thick = get_thick_slices(orig, self.slice_thickness)

                

                orig, aseg = filter_blank_slices_thick(
                    orig_thick, aseg, threshold=blt/zoom[0]
                )

                num_batch = orig.shape[2]
                orig = np.transpose(orig, (2, 0, 1, 3))
                aseg = np.transpose(aseg, (2, 0, 1))
                # weights = np.transpose(weights, (2, 0, 1))
                weights = np.ones_like(aseg, dtype=np.float32)
               
                print(zoom)
                data_per_size[f"{size}"]["orig"].extend(orig)
                data_per_size[f"{size}"]["aseg"].extend(aseg)
                data_per_size[f"{size}"]["weight"].extend(weights)
                data_per_size[f"{size}"]["zoom"].extend((zoom,) * num_batch)
                sub_name = current_subject[1].split("/")[-1][:-11]
                
                data_per_size[f"{size}"]["subject"].append(
                    sub_name.encode("ascii", "ignore")
                )

            except Exception as e:
                LOGGER.info(f"Volume: {idx} Failed Reading Data. Error: {e}")
                continue

        for key, data_dict in data_per_size.items():
            data_per_size[key]["orig"] = np.asarray(data_dict["orig"], dtype=np.uint8)
            data_per_size[key]["aseg"] = np.asarray(data_dict["aseg"], dtype=np.uint8)
            data_per_size[key]["weight"] = np.asarray(data_dict["weight"], dtype=float)

        with h5py.File(self.dataset_name, "w") as hf:
            dt = h5py.special_dtype(vlen=str)
            for key, data_dict in data_per_size.items():
                group = hf.create_group(f"{key}")
                group.create_dataset("orig_dataset", data=data_dict["orig"])
                group.create_dataset("aseg_dataset", data=data_dict["aseg"])
                group.create_dataset("weight_dataset", data=data_dict["weight"])
                group.create_dataset("zoom_dataset", data=data_dict["zoom"])
                group.create_dataset("subject", data=data_dict["subject"], dtype=dt)

        end_d = time.time() - start_d
        LOGGER.info(
            f"Successfully written {self.dataset_name} in {end_d:.3f} seconds."
        )


def make_parser():
    import argparse

    # Training settings
    parser = argparse.ArgumentParser(description="HDF5-Creation")

    parser.add_argument(
        "--hdf5_name",
        type=str,
        default="../data/hdf5_set/Multires_coronal.hdf5",
        help="path and name of hdf5-data_loader (default: ../data/hdf5_set/Multires_coronal.hdf5)",
    )
    parser.add_argument(
        "--plane",
        type=str,
        default="axial",
        choices=["axial", "coronal", "sagittal"],
        help="Which plane to put into file (axial (default), coronal or sagittal)",
    )
    parser.add_argument(
        "--data_dir", type=str, default="/data", help="Directory with images to load"
    )
    parser.add_argument(
        "--thickness",
        type=int,
        default=3,
        help="Number of pre- and succeeding slices (default: 3)",
    )
    parser.add_argument(
        "--csv_file",
        type=str,
        default=None,
        help="Csv-file listing subjects to include in file",
    )
    parser.add_argument(
        "--pattern", type=str, help="Pattern to match files in directory."
    )
    parser.add_argument(
        "--image_name",
        type=str,
        default="mri/orig.mgz",
        help="Default name of original images. FreeSurfer orig.mgz is default (mri/orig.mgz)",
    )
    parser.add_argument(
        "--gt_name",
        type=str,
        default="mri/aparc.DKTatlas+aseg.mgz",
        help="Default name for ground truth segmentations. Default: mri/aparc.DKTatlas+aseg.mgz."
        " If Corpus Callosum segmentation is already removed, do not set gt_nocc."
        " (e.g. for our internal training set mri/aparc.DKTatlas+aseg.filled.mgz exists already"
        " and should be used here instead of mri/aparc.DKTatlas+aseg.mgz). ",
    )
    parser.add_argument(
        "--gt_nocc",
        type=str,
        default=None,
        help="Segmentation without corpus callosum (used to mask this segmentation in ground truth)."
        " If the used segmentation was already processed, do not set this argument."
        " For a normal FreeSurfer input, use mri/aseg.auto_noCCseg.mgz.",
    )
    parser.add_argument(
        "--lut",
        type=Path,
        default="/opt/localdata/data/usr-envs/ruiying/Code/NeurIPS/FastSurfer/FastSurfer-dev/FastSurferCNN/config/FastSurfer_ColorLUT.tsv",
        help="FreeSurfer-style Color Lookup Table with labels to use in final prediction. "
        "Has to have columns: ID	LabelName	R	G	B	A"
        "Default: FASTSURFERDIR/FastSurferCNN/config/FastSurfer_ColorLUT.tsv.",
    )
    parser.add_argument(
        "--combi",
        action="append",
        default=["Left-", "Right-"],
        help="Suffixes of labels names to combine. Default: Left- and Right-.",
    )
    parser.add_argument(
        "--sag_mask",
        default=("Left-", "ctx-rh"),
        help="Suffixes of labels names to mask for final sagittal labels. Default: Left- and ctx-rh.",
    )
    parser.add_argument(
        "--max_w",
        type=int,
        default=5,
        help="Overall max weight for any voxel in weight mask. Default=5",
    )
    parser.add_argument(
        "--edge_w",
        type=int,
        default=5,
        help="Weight for edges in weight mask. Default=5",
    )
    parser.add_argument(
        "--hires_w",
        type=int,
        default=None,
        help="Weight for hires elements (sulci, WM strands, cortex border) in weight mask. Default=None",
    )
    parser.add_argument(
        "--no_grad",
        action="store_true",
        default=False,
        help="Turn on to only use median weight frequency (no gradient)",
    )
    parser.add_argument(
        "--gm",
        action="store_true",
        default=False,
        help="Turn on to add cortex mask for hires-processing.",
    )
    parser.add_argument(
        "--processing",
        type=str,
        default="aparc",
        choices=["aparc", "aseg", "none"],
        help="Use aseg, aparc or no specific mapping processing",
    )
    parser.add_argument(
        "--blank_slice_thresh",
        type=float,
        default=0.99,
        help="Threshold value for function filter_blank_slices. Slices with number of"
        "labeled voxels below this threshold are discarded. Default: 50.",
    )
    parser.add_argument(
        "--sizes",
        nargs="+",
        type=int,
        default=256,
        help="Sizes of images in the dataset. Default: 256",
    )
    return parser


def main(args):
    dataset_params = {
        "dataset_name": args.hdf5_name,
        "data_path": args.data_dir,
        "thickness": args.thickness,
        "csv_file": args.csv_file,
        "pattern": args.pattern,
        "image_name": args.image_name,
        "gt_name": args.gt_name,
        "gt_nocc": args.gt_nocc,
        "sizes": args.sizes,
        "max_weight": args.max_w,
        "edge_weight": args.edge_w,
        "plane": args.plane,
        "lut": str(args.lut),
        "combi": args.combi,
        "sag_mask": args.sag_mask,
        "hires_weight": args.hires_w,
        "gm_mask": args.gm,
        "gradient": not args.no_grad,
    }

    dataset_generator = H5pyDataset(params=dataset_params, processing=args.processing)
    dataset_generator.create_hdf5_dataset(args.blank_slice_thresh)




if __name__ == "__main__":
    import sys
    parser = make_parser()
    
    args = parser.parse_args()
    main(args)
