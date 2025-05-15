

import os
import glob
import copy
from pathlib import Path
import numpy as np
import torch
import nibabel as nib


import sys
sys.path.insert(0, "FastSurfer_MAE_path")

from FastSurferCNN.utils import PLANES, Plane, logging, parser_defaults
from FastSurferCNN.utils.logging import setup_logging
from FastSurferCNN.utils.checkpoint import get_checkpoints, load_checkpoint_config_defaults
from FastSurferCNN.utils.common import SubjectDirectoryConfig, SubjectList
from FastSurferCNN.inference import Inference
from FastSurferCNN.run_prediction import RunModelOnData
import FastSurferCNN.reduce_to_aseg as rta
import yaml
from pathlib import Path

LOGGER = logging.getLogger(__name__)


with open("eval.yaml", "r") as f:
    cfg = yaml.safe_load(f)

input_root = cfg["input_root"]
output_root = cfg["output_root"]
lut_path = cfg["lut_path"]

ckpt_ax = Path(cfg["ckpt_ax"])
ckpt_cor = Path(cfg["ckpt_cor"])
ckpt_sag = Path(cfg["ckpt_sag"])

cfg_ax = Path(cfg["cfg_ax"])
cfg_cor = Path(cfg["cfg_cor"])
cfg_sag = Path(cfg["cfg_sag"])


# Set up logging
setup_logging("fastsurfer_batch.log")



# Load model once
model = RunModelOnData(
    lut=lut_path,
    ckpt_ax=ckpt_ax,
    ckpt_sag=ckpt_sag,
    ckpt_cor=ckpt_cor,
    cfg_ax=cfg_ax,
    cfg_sag=cfg_sag,
    cfg_cor=cfg_cor,
    device="cuda",
    batch_size=1,
    async_io=False,
)

# Loop through all datasets
for dataset in os.listdir(input_root):
    img_dir = os.path.join(input_root, dataset, "img")
    for img_path in glob.glob(os.path.join(img_dir, "*.nii.gz")):
        sid = Path(img_path).stem
        subj_out = os.path.join(output_root, dataset, sid)

        # Define paths
        orig_name = img_path
        pred_name = "mri/pred.nii.gz"
        conf_name = "mri/orig.nii.gz"
       

        # Subject I/O paths
        config = SubjectDirectoryConfig(
            orig_name=orig_name,
            pred_name=pred_name,
            conf_name=conf_name,
            sid=sid,
            in_dir=None,
            out_dir=subj_out,
        )
        config.copy_orig_name = "mri/orig/001.nii.gz"

        subjects = SubjectList(config, segfile="pred_name", copy_orig_name="copy_orig_name")
        subjects.make_subjects_dir()
        subject = subjects[0]


        # Predict
        try:
            # Conform
            orig_img, orig_data = model.conform_and_save_orig(subject)

            # Predict
            pred_data = model.get_prediction(orig_name, orig_data, orig_img.header.get_zooms())

            # Save prediction only
            model.save_img(subject.segfile, pred_data, orig_img, dtype=np.int16)

        except RuntimeError as e:
            LOGGER.warning(f"Skipping {sid} due to runtime error: {e}")
            continue
        
        
print("✅ All subjects processed.")
