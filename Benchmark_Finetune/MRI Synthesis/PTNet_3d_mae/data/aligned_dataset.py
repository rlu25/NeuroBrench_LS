### Copyright (C) 2017 NVIDIA Corporation. All rights reserved. 
### Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).

### This script was modified based on the pix2pixHD official implementation (see license above)
### https://github.com/NVIDIA/pix2pixHD

import os.path
from data.base_dataset import BaseDataset
from data.image_folder import make_dataset
import torch
import nibabel as nib
import numpy as np


class AlignedDataset(BaseDataset):
    def initialize(self, opt):
        self.opt = opt
        self.root = opt.dataroot

        ### input A (label maps)
        dir_A = '_A' if self.opt.label_nc == 0 else '_label'
        self.dir_A = os.path.join(opt.dataroot, opt.phase + dir_A)
        self.A_paths = sorted(make_dataset(self.dir_A))
        ### input B (real images)
        # if opt.isTrain:
        dir_B = '_B' if self.opt.label_nc == 0 else '_img'
        self.dir_B = os.path.join(opt.dataroot, opt.phase + dir_B)
        self.B_paths = sorted(make_dataset(self.dir_B))
        self.opt = opt
        self.dataset_size = len(self.A_paths)

    def __getitem__(self, index):
        ### input A (label maps)
        A_path = self.A_paths[index]
        A = np.load(A_path)
        B_path = self.B_paths[index]
        B = np.load(B_path)
        if not self.opt.no_flip:
            if np.random.uniform(0, 1) > 0.5:
                tmp_1 = np.random.uniform(0, 1)
                if 0 <= tmp_1 < 1 / 3:
                    A = np.flip(A, 0)
                    B = np.flip(B, 0)
                elif 1 / 3 <= tmp_1 < 2 / 3:
                    A = np.flip(A, 1)
                    B = np.flip(B, 1)
                elif tmp_1 >= 2 / 3:
                    A = np.flip(A, 2)
                    B = np.flip(B, 2)
        A_tensor = torch.from_numpy(A.copy()).to(dtype=torch.float)
        # if len(A_tensor.shape) == 2:
        A_tensor = torch.unsqueeze(A_tensor, 0)
        # B_tensor = inst_tensor = feat_tensor = 0
        ### input B (real images)
        # if self.opt.isTrain:

        B_tensor = torch.from_numpy(B.copy()).to(dtype=torch.float)
        # if len(B_tensor.shape) == 2:
        B_tensor = torch.unsqueeze(B_tensor, 0)

        # input_dict = {'label': A_tensor, 'inst': inst_tensor, 'image': B_tensor,
        #               'feat': feat_tensor, 'path': A_path}
        input_dict = {'label': A_tensor, 'image': B_tensor, 'path': A_path}

        return input_dict

    def __len__(self):
        return len(self.A_paths) // self.opt.batchSize * self.opt.batchSize

    def name(self):
        return 'AlignedDataset'
