# QuickNat and Bayesian QuickNAT - Lifespan implementation

A fully convolutional network for quick and accurate segmentation combined with Lifespan MAE
-----------------------------------------------------------


## Getting Started

### Pre-requisites
Please install the required packages for smooth functioning of the tool by running
```
pip install -r requirements.txt
```

### Training your model

```
python run_train.py 
```

### Evaluating your model

```
python run_test.py 
```


* **device**: CPU or ID of GPU (0 or 1) you want to excecute your code.
* **coronal_model_path**: It is by default set to "saved_models/finetuned_alldata_coronal.pth.tar" which is our final model. You may also use "saved_models/IXI_fsNet_coronal.pth.tar" which is our pre-trained model.
* **axial_model_path**: Similar to above. It is only used for view_aggregation stage.
* * `--data_dir`: Directory with images to load. Default: /data
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

* **volumes_txt_file**: Path to the '.txt' file for the domain name.
* **batch_size**: Set this according the capacity of your GPU RAM.
* **save_predictions_dir**: Indicate the absolute path where you want to save the segmentation outputs along with the '.csv' files for volume and uncertainty estimates.
* **view_agg**: Valid options are "True" or "False". When "False", it uses coronal network by default.
* **estimate_uncertainty**: Valid options are "True" or "False". Indicates if you want to estimate the structure-wise uncertainty for segmentation Quality control. Refer to "Bayesian QuickNAT" paper for more details.
* **mc_samples**: Active only if **estimate_uncertainty** flag is "True". Indicates the number of Monte-Carlo samples used for uncertainty estimation. 
* **labels**: List of label names please find [here](https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/B5OU7H).

 
