# Overview

This directory contains all information needed to run inference with the trained FastSurferVINN_MAE or train it from scratch. Subcortical segmentation, NeuroBench_LS, 


The network was trained with NeuroBench_LS_subcortical_segmentation. link: https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/B5OU7H 

<!-- before inference -->
# 1. Inference
<!-- after inference heading -->

The *FastSurferCNN* directory contains all the source code and modules needed to run the scripts. A list of python libraries used within the code can be found in __requirements.txt__. The main script is called __eval_NeuroBenchLS.py__ within .yaml, datastructurw is 

## General
```
test_data/
├── domain1_img/     # Input images for domain 1
├── domain2_img/     # Input images for domain 2
├── ... 

domain1_img/
├── subject_001.nii.gz
├── subject_002.nii.gz
...
```
## Checkpoints and configs

* `--ckpt_sag`: path to sagittal network checkpoint
* `--ckpt_cor`: path to coronal network checkpoint
* `--ckpt_ax`: path to axial network checkpoint
* `--cfg_cor`: Path to the coronal config file
* `--cfg_sag`: Path to the axial config file
* `--cfg_ax`: Path to the sagittal config file

## Optional commands

* `--clean`: clean up segmentation after running it (optional)
* `--device <str>`:Device for processing (_auto_, _cpu_, _cuda_, _cuda:<device_num>_), where cuda means Nvidia GPU; you can select which one e.g. "cuda:1". Default: "auto", check GPU and then CPU
* `--viewagg_device <str>`: Define where the view aggregation should be run on. 
  Can be _auto_ or a device (see --device).
  By default (_auto_), the program checks if you have enough memory to run the view aggregation on the gpu. 
  The total memory is considered for this decision. 
  If this fails, or you actively overwrote the check with setting `--viewagg_device cpu`, view agg is run on the cpu. 
  Equivalently, if you define `--viewagg_device gpu`, view agg will be run on the gpu (no memory check will be done).
* `--batch_size`: Batch size for inference. Default=1

The output will be stored in:

- `../output/subjectX/mri/pred.nii.gz` (subcotical segmentation)
- `../output/subjectX/mri/ori.nii.gz` (original image)



## Example Command: Evaluation Single Subject
if you want to test single data


```bash
python3 run_prediction.py \
          --in_dir ../data \
          --sd ../output \
          --seg_log ../output/temp_Competitive.log
```

The output will be stored in:

- `../output/subjectX/mri/aparc.DKTatlas+aseg.deep.mgz` (large segmentation)
- `../output/subjectX/mri/mask.mgz` (brain mask)
- `../output/subjectX/mri/aseg_noCC.mgz` (reduced segmentation)
- and the log in `../output/temp_Competitive.log`

<!-- before generate_hdf5 -->

# 2. Hdf5-Trainingset Generation
<!-- after generate_hdf5 heading -->

The *FastSurferCNN* directory contains all the source code and modules needed to create a hdf5-file from given MRI volumes. 
A list of python libraries used within the code can be found in __requirements.txt__. The main script is called __generate_hdf5.py__ within which certain options can be selected and set via the command line:

### General

* `--hdf5_name`: Path and name of the to-be-created hdf5-file. Default: ../data/hdf5_set/Multires_coronal.hdf5
* `--data_dir`: Directory with images to load. Default: /data
Your data_dir would then look like this:
 ```
  data/
├── domain1_img/     # Input images for domain 1
├── domain1_label/   # Corresponding labels or segmentations for domain 1
├── domain2_img/     # Input images for domain 2
├── domain2_label/   # Corresponding labels or segmentations for domain 2
├── ...              # Additional domains (e.g., test_img, test_label, etc.)

domain1_img/
├── subject_001.nii.gz
├── subject_002.nii.gz
...

domain1_label/
├── subject_001_seg.nii.gz
├── subject_002_seg.nii.gz
...

 ```
* --lut: FreeSurfer-style Color Lookup Table with labels to use in final prediction. 


## Example Command: Axial (Single Resolution)

```bash
python3 generate_hdf5.py \
          --hdf5_name ../data/training_set_axial.hdf5 \
          --csv_file ../training_set_subjects_dirs.csv \
          --thickness 3 \
          --plane axial \
          --image_name mri/orig.mgz \
          --gt_name mri/aparc.DKTatlas+aseg.mgz \
          --gt_nocc mri/aseg.auto_noCCseg.mgz
          --max_w 5 \
          --edge_w 4 \
          --hires_w 4 \
          --sizes 256
```

## Example Command: Coronal (Single Resolution)

```bash
python3 generate_hdf5.py \
          --hdf5_name ../data/training_set_coronal.hdf5 \
          --csv_file ../training_set_subjects_dirs.csv \
          --plane coronal \
          --image_name mri/orig.mgz \
          --gt_name mri/aparc.DKTatlas+aseg.mgz \
          --gt_nocc mri/aseg.auto_noCCseg.mgz
          --max_w 5 \
          --edge_w 4 \
          --hires_w 4 \
          --sizes 256
```

## Example Command: Sagittal (Multiple Resolutions)

```bash
python3 generate_hdf5.py \
          --hdf5_name ../data/training_set_sagittal.hdf5 \
          --csv_file ../training_set_subjects_dirs.csv \
          --plane sagittal \
          --image_name mri/orig.mgz \
          --gt_name mri/aparc.DKTatlas+aseg.mgz \
          --gt_nocc mri/aseg.auto_noCCseg.mgz
          --max_w 5 \
          --edge_w 4 \
          --hires_w 4 \
          --sizes 256 311 320
```

## Example Command: Sagittal using --data_dir instead of --csv_file
`--data_dir` specifies the path in which the data is located, with `--pattern` we can select subjects from the specified path. By default the pattern is "*" meaning all subjects will be selected.
As an example, imagine you have 19 FreeSurfer processed subjects labeled subject1 to subject19 in the ../data directory:

```
/home/user/FastSurfer/data
├── subject1
├── subject2
├── subject3
…
│
├── subject19
    ├── mri
    │   ├── aparc.DKTatlas+aseg.mgz
    │   ├── aseg.auto_noCCseg.mgz
    │   ├── orig.mgz
    │   ├── …
    │   …
    ├── scripts
    ├── stats
    ├── surf
    ├── tmp
    ├── touch
    └── trash
```

Setting `--pattern` "*" will select all 19 subjects (subject1, ..., subject19).
Now, if only a subset should be used for the hdf5-file (e.g. subject 10 till subject19), this can be done by changing the `--pattern` flag to "subject1[0-9]": 

```bash
python3 generate_hdf5.py \
          --hdf5_name ../data/training_set_axial.hdf5 \
          --data_dir ../data \
          --plane sagittal \
          --blank_slice_thresh \ 0.8
```
<!-- before training -->
# 3. Training
<!-- after training heading -->

The *FastSurferCNN* directory contains all the source code and modules needed to run the scripts. A list of python libraries used within the code can be found in __requirements.txt__. The main training script is called __run_model.py__ whose options can be set through a configuration file and command line arguments:
* `--cfg`: Path to the configuration file. Default: config/FastSurferVINN_NeuroBench_LS.yaml
* `--aug`: List of augmentations to use. Default: None.
* `--opt`: List of class options to use.

The `--cfg` file configures the model to be trained. See config/FastSurferVINN.yaml for an example and config/defaults.py for all options and default values.

The configuration options include:


## Optimizer options

* `BASE_LR`: Base learning rate. Default: 0.01
* `OPTIMIZING_METHOD`: Optimization method [sgd, adam, adamW]. Default: adamW
* `MOMENTUM`: Momentum for optimizer. Default: 0.9
* `NESTEROV`: Enables Nesterov for optimizer. Default: True
* `LR_SCHEDULER`: Learning rate scheduler [step_lr, cosineWarmRestarts, reduceLROnPlateau]. Default: cosineWarmRestarts










## Example Command: Training FastSurferCNN
Trains FastSurferCNN using a provided configuration file and specifying no augmentations:

```bash
python3 run_model.py \
          --cfg custom_configs/FastSurferCNN_NeuroBench_LS.yaml \
          --aug None
```
