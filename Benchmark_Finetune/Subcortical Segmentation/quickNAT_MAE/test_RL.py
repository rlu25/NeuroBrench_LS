import os
import glob
import nibabel as nib
import torch
quicknat_model = torch.load("/opt/localdata/data/usr-envs/ruiying/Code/NeurIPS/QuickNet/quickNAT_MAE/saved_models/mae_pretrian/model_final.pth")
print(quicknat_model)