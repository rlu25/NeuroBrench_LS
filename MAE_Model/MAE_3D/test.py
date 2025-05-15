import torch
import os
from cfg.default import get_cfg_defaults
from model.mpl_seg import EMA_MPL
from model.utils import util
import numpy as np
import pandas
import torchio as tio
import nibabel as nib
import medpy.metric.binary as mmb
from tqdm import tqdm


def infer_single_scan(model, cfg, tmp_scans):
    pad_flag = False
    model.eval()
    x, y, z = cfg.data.patch_size
    if cfg.data.normalize:
        tmp_scans = util.norm_img(tmp_scans, cfg.data.norm_perc)
    if min(tmp_scans.shape) < min(x, y, z):
        x_ori_size, y_ori_size, z_ori_size = tmp_scans.shape
        pad_flag = True
        x_diff = x-x_ori_size
        y_diff = y-y_ori_size
        z_diff = z-z_ori_size
        tmp_scans = np.pad(tmp_scans, ((max(0, int(x_diff/2)), max(0, x_diff-int(x_diff/2))), (max(0, int(
            y_diff/2)), max(0, y_diff-int(y_diff/2))), (max(0, int(z_diff/2)), max(0, z_diff-int(z_diff/2)))), constant_values=1e-4)  # cant pad with 0s, otherwise the local and global patches wont be the same location

    pred = np.zeros((cfg.train.cls_num,) + tmp_scans.shape)
    tmp_norm = np.zeros((cfg.train.cls_num,) + tmp_scans.shape)

    scan_patches, _, tmp_idx = util.patch_slicer(tmp_scans, tmp_scans, cfg.data.patch_size,
                                                 (x - 16, y -
                                                     16, z - 16),
                                                 remove_bg=cfg.data.remove_bg, test=True, ori_path=None)
    bound = util.get_bounds(torch.from_numpy(tmp_scans))
    global_scan = torch.unsqueeze(torch.from_numpy(
        tmp_scans).to(dtype=torch.float), dim=0)

    '''
    Sliding window implementation to go through the whole scans
    '''
    for idx, patch in enumerate(scan_patches):
        ipt = torch.from_numpy(patch).to(dtype=torch.float).cuda()
        ipt = ipt.reshape((1, 1,) + ipt.shape)

        patch_idx = tmp_idx[idx]
        location = torch.zeros_like(
            torch.from_numpy(tmp_scans)).float()
        location = torch.unsqueeze(location, 0)
        location[:, patch_idx[0]:patch_idx[1], patch_idx[2]:patch_idx[3], patch_idx[4]:patch_idx[5]] = 1

        sbj = tio.Subject(one_image=tio.ScalarImage(
            tensor=global_scan[:, bound[0]:bound[1], bound[2]:bound[3], bound[4]:bound[5]]),
            a_segmentation=tio.LabelMap(
                tensor=location[:, bound[0]:bound[1], bound[2]:bound[3], bound[4]:bound[5]]))
        transforms = tio.transforms.Resize(target_shape=(x, y, z))
        sbj = transforms(sbj)
        down_scan = sbj['one_image'].data
        loc = sbj['a_segmentation'].data
        tmp_coor = util.get_bounds(loc)
        coordinates_A = np.array([np.floor(tmp_coor[0] / 4),
                                  np.ceil(tmp_coor[1] / 4),
                                  np.floor(tmp_coor[2] / 4),
                                  np.ceil(tmp_coor[3] / 4),
                                  np.floor(tmp_coor[4] / 4),
                                  np.ceil(tmp_coor[5] / 4)
                                  ]).astype(int)
        coordinates_A = torch.unsqueeze(
            torch.from_numpy(coordinates_A), 0)
        tmp_pred, _ = model(ipt, down_scan.cuda().reshape([1, 1, x, y, z]),
                            coordinates_A)

        patch_idx = (slice(0, cfg.train.cls_num),) + (
            slice(patch_idx[0], patch_idx[1]), slice(
                patch_idx[2], patch_idx[3]),
            slice(patch_idx[4], patch_idx[5]))
        pred[patch_idx] += torch.squeeze(
            tmp_pred).detach().cpu().numpy()
        tmp_norm[patch_idx] += 1

    pred[tmp_norm > 0] = (pred[tmp_norm > 0]) / \
        tmp_norm[tmp_norm > 0]
    sf = torch.nn.Softmax(dim=0)
    pred_vol = sf(torch.from_numpy(pred)).numpy()
    pred_vol = np.argmax(pred_vol, axis=0)
    if pad_flag:
        pred_vol = pred_vol[max(0, int(x_diff/2)): max(0, int(x_diff/2))+x_ori_size,
                            max(0, int(y_diff/2)): max(0, int(y_diff/2))+y_ori_size,
                            max(0, int(z_diff/2)): max(0, int(z_diff/2))+z_ori_size]
        assert pred_vol.shape == (
            x_ori_size, y_ori_size, z_ori_size), 'pred_vol shape must be the same as the original scan shape'
    return pred_vol


'''
--- ckpt_dir
    --- proj_name
        --- exps_lst

About structure of test data directory, please refer to cfg/default.py
'''

ckpt_dir = ''
proj_name = ''
exps_lst = [
    
]
test_data_dir = ''
output_root = '' 
# if set as false, only save prediction (assuming no GT label existing in inference)
is_test = True

if __name__ == '__main__':
    print('Start testing:')
    print('Ckpt dir: ', ckpt_dir)
    print('Project name: ', proj_name)
    print('Experiments: ', exps_lst)
    print('Is test: ', is_test)
    for i in exps_lst:
        print(i)
        exp_dir = os.path.join(ckpt_dir, proj_name, i)
        cfg = get_cfg_defaults()
        cfg.merge_from_file(os.path.join(exp_dir, 'train_cfg.yaml'))
        model = EMA_MPL(cfg)
        model.cuda()
        model.load_state_dict(torch.load(
            os.path.join(exp_dir, 'best_model.pth')), strict=True)
        
        for dataset_name in sorted(os.listdir(test_data_dir)):
            dataset_dir = os.path.join(test_data_dir, dataset_name)
            img_dir = os.path.join(dataset_dir, 'img')

            if not os.path.exists(img_dir):
                continue  # Skip non-image folders

            print(f'  → Testing on {dataset_name}')
            output_dir = os.path.join(output_root, dataset_name)
            os.makedirs(output_dir, exist_ok=True)

            img_files = sorted([f for f in os.listdir(img_dir) if f.endswith(cfg.data.extension)])
            for fname in tqdm(img_files):
                img_path = os.path.join(img_dir, fname)
                img_nii = nib.load(img_path)
                img_data = np.squeeze(img_nii.get_fdata())
                test_data = img_data.copy()
                with torch.no_grad():
                    pred_vol = infer_single_scan(model, cfg, test_data)
                pred_vol = pred_vol.astype(np.uint8)
                pred = nib.Nifti1Image(
                    pred_vol, affine=img_nii.affine, header=img_nii.header)
                nib.save(pred, os.path.join(output_dir, fname))

        