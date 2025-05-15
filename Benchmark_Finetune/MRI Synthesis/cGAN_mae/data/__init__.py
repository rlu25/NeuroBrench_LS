import torch.utils.data
from data.base_data_loader import BaseDataLoader
import numpy as np
import h5py
import random
import os

def CreateDataLoader(opt):
    data_loader = CustomDatasetDataLoader()
    print(data_loader.name())
    data_loader.initialize(opt)
    return data_loader


def CreateDataset(opt):
    # ------------------------------
    # Load Data
    # ------------------------------
    target_file = os.path.join(opt.dataroot, opt.phase, 'data.mat')
    f = h5py.File(target_file, 'r')
    data_x = np.array(f['data_x'])
    # print(f"data_x shape: {data_x.shape}")
    data_y = np.array(f['data_y'])
    if opt.isTrain and data_x.shape[3] > 1 :
        # Reshape to (256, 256, N, 1)
        data_x= data_x.reshape(256, 256, -1, 1)
        data_y= data_y.reshape(256, 256, -1, 1)

    # load .mat files with shape (256, 256, N, 3)
    # test: load .mat files with shape (n, 1, 256, 256) (N, 1, H, W)
    # Determine slice indices
    # mid_slice = data_x.shape[3] // 2
    # if opt.isTrain:
    #     if opt.which_direction == 'AtoB':
    #         data_x = data_x[:, :, :, mid_slice - opt.input_nc // 2: mid_slice + opt.input_nc // 2 + 1]
    #         data_y = data_y[:, :, :, mid_slice - opt.output_nc // 2: mid_slice + opt.output_nc // 2 + 1]
    #     else:
    #         data_y = data_y[:, :, :, mid_slice - opt.input_nc // 2: mid_slice + opt.input_nc // 2 + 1]
    #         data_x = data_x[:, :, :, mid_slice - opt.output_nc // 2: mid_slice + opt.output_nc // 2 + 1]

    if not opt.isTrain and data_x.ndim == 4 and data_x.shape[1] == 1:
        # reshape from (N, 1, H, W) to (1, N, H, W) to align with training format
        data_x = np.transpose(data_x, ( 2, 3, 0, 1))
        data_y = np.transpose(data_y, ( 2, 3, 0, 1))

    # Shuffle if needed
    if opt.dataset_mode == 'unaligned_mat':
        print("Training phase" if opt.isTrain else "Testing phase")
        # indices = list(range(data_y.shape[2]))
        # if opt.isTrain:
        #     random.shuffle(indices)
        # data_y = data_y[:, :, indices, :]

    # ------------------------------
    # Preprocess
    # ------------------------------
    data_x = np.transpose(data_x, (3, 2, 0, 1))  # (N, C, H, W)
    data_y = np.transpose(data_y, (3, 2, 0, 1))

    data_x[data_x < 0] = 0
    data_y[data_y < 0] = 0

    # ------------------------------
    # Normalize and Package into Dataset
    # ------------------------------
    dataset = []
    num_samples = max(data_x.shape[1], data_y.shape[1])

    for i in range(num_samples):
        x = data_x[:, i, :, :] if i < data_x.shape[1] else None
        y = data_y[:, i, :, :] if i < data_y.shape[1] else None

        sample = {}
        if x is not None:
            x = (x - 0.5) / 0.5
            sample['A'] = torch.from_numpy(x)
            sample['A_paths'] = opt.dataroot
        if y is not None:
            y = (y - 0.5) / 0.5
            sample['B'] = torch.from_numpy(y)
            sample['B_paths'] = opt.dataroot

        dataset.append(sample)

    return dataset


class CustomDatasetDataLoader(BaseDataLoader):
    def name(self):
        return 'CustomDatasetDataLoader'

    def initialize(self, opt):
        BaseDataLoader.initialize(self, opt)
        self.dataset = CreateDataset(opt)
        self.dataloader = torch.utils.data.DataLoader(
            self.dataset,
            batch_size=opt.batchSize,
            shuffle=not opt.serial_batches,
            num_workers=int(opt.nThreads)
        )

    def load_data(self):
        return self

    def __len__(self):
        return min(len(self.dataset), self.opt.max_dataset_size)

    def __iter__(self):
        for i, data in enumerate(self.dataloader):
            if i >= self.opt.max_dataset_size:
                break
            yield data