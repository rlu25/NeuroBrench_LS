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
python run_model.py 
```

### Evaluating your model

```
python eval_NeuroBenchLS.py 
```


* **device**: CPU or ID of GPU (0 or 1) you want to excecute your code.
* **coronal_model_path**: It is by default set to "saved_models/finetuned_alldata_coronal.pth.tar" which is our final model. You may also use "saved_models/IXI_fsNet_coronal.pth.tar" which is our pre-trained model.
* **axial_model_path**: Similar to above. It is only used for view_aggregation stage.
* **data_dir**: Absolute path to the data directory where input volumes are present.
* **directory_struct**: Valid options are "FS" or "Linear". If you input data directory is similar to FreeSurfer, i.e. **data_dir**/<Data_id>/mri/orig.mgz then use "FS". If the entries are **data_dir**/<Data_id> use "Linear".
* **volumes_txt_file**: Path to the '.txt' file where the data_ID names are stored. If **directory_struct** is "FS" the entries should be only the folder names, whereas if it is "Linear" the entry name should be the file names with the file extensions.
* **batch_size**: Set this according the capacity of your GPU RAM.
* **save_predictions_dir**: Indicate the absolute path where you want to save the segmentation outputs along with the '.csv' files for volume and uncertainty estimates.
* **view_agg**: Valid options are "True" or "False". When "False", it uses coronal network by default.
* **estimate_uncertainty**: Valid options are "True" or "False". Indicates if you want to estimate the structure-wise uncertainty for segmentation Quality control. Refer to "Bayesian QuickNAT" paper for more details.
* **mc_samples**: Active only if **estimate_uncertainty** flag is "True". Indicates the number of Monte-Carlo samples used for uncertainty estimation. 
* **labels**: List of label names please find [here](https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/B5OU7H).

 
