import torch.utils.data as data
import os
from .data_utils import *
import torchio as tio
import torch
import numpy as np
import random


class mae_dataset(data.Dataset):
    def __init__(self, cfg):
        self.cfg = cfg
        self.patch_size = cfg.data.patch_size  # (x, y, z)
        self.aug_prob   = cfg.data.aug_prob
        self.normalize  = cfg.data.normalize
        self.norm_perc  = cfg.data.norm_perc
        self.remove_bg  = cfg.data.remove_bg
        # get all image paths (source and target)
        # folder should end with '_train'
        
        # 1) build domain→scan‑paths dict
        #self._build_path_index()
        
        # 2) prepare torchio transforms once
        self.affine = tio.RandomAffine(
            p=self.aug_prob,
            scales=(0.75, 1.5),
            degrees=40,
            # scales=(0.95, 1.05),
            # degrees=10,
            isotropic=True,
            default_pad_value=0,
            # default_pad_value='minimum',
            image_interpolation='linear'
        )
        self.resize = tio.transforms.Resize(target_shape=(self.patch_size[0],
                                                         self.patch_size[1],
                                                         1))        
        
        with open(cfg.data.mae_domain) as f:
            self.domains = f.read().splitlines()
        self.path_dic = {}
        self.subject_count = {}
        self.image_count = {}
        total_subjects = set()
        total_images = 0
        exclude_pairs = set()
        if os.path.exists(cfg.data.mae_test_list):
            df_exclude = pd.read_csv(cfg.data.mae_test_list)
            for _, row in df_exclude.iterrows():
                exclude_pairs.add((row["sub_id"], row["ses_id"]))
        for idx, domain in enumerate(self.domains):
            domain_path = os.path.join(cfg.data.mae_root, domain)
            nii_paths = []
            # Collect .nii.gz files from both skull-stripped and preprocessed folders
            search_dirs = []
            if domain in ["ADHD", "ABIDE"]:
                search_dirs += [
                    os.path.join(domain_path, "Preprocessed", "*", "sub-*", "ses-*", "anat", "*.nii.gz"),
                    os.path.join(domain_path, "Skull-stripped", "*", "sub-*", "ses-*", "anat",  "*.nii.gz")
                ]
            elif domain == "Infant_nonlabel":
                search_dirs += [os.path.join(domain_path, "*.nii.gz")]
            elif domain == "MOMMA_fetal":
                search_dirs += [os.path.join(domain_path, "sub-*", "ses-*", "anat", "*.nii.gz")]
            else:
                search_dirs += [
                    os.path.join(domain_path, "Preprocessed", "sub-*", "ses-*", "anat", "*.nii.gz"),
                    os.path.join(domain_path, "Skull-stripped", "sub-*", "ses-*", "anat", "*.nii.gz")
                ]
            # Gather files
            for pattern in search_dirs:
                matches = glob.glob(pattern, recursive=True)
                filtered = []
                for f in matches:
                    if f.endswith("_mask.nii.gz"):
                        continue
                    parts = f.split(os.sep)
                    sub_id = next((p for p in parts if p.startswith("sub-")), None)
                    ses_id = next((p for p in parts if p.startswith("ses-")), None)
                    if (sub_id, ses_id) in exclude_pairs:
                        continue
                    filtered.append(f)
                nii_paths.extend(filtered)
            # Store image paths
            self.path_dic[str(idx)] = sorted(nii_paths)
            self.image_count[domain] = len(nii_paths)
            total_images += len(nii_paths)
            # Track subjects
            domain_subjects = set()
            for path in nii_paths:
                parts = path.split(os.sep)
                sub_id = next((p for p in parts if p.startswith("sub-")), None)
                if sub_id:
                    domain_subjects.add(sub_id)
                    total_subjects.add(sub_id)
            self.subject_count[domain] = len(domain_subjects)
            # Domain summary
            print(f":file_folder: Domain: {domain}")
            print(f"  :brain: Subjects: {self.subject_count[domain]}")
            print(f"  :page_facing_up: Images : {self.image_count[domain]}")
        print("====================================")
        print(f":white_check_mark: Total domains  : {len(self.domains)}")
        print(f":bust_in_silhouette: Total subjects : {len(total_subjects)}")
        print(f":frame_with_picture:  Total images   : {total_images}")
        print("====================================")
        self.num_domain = len(self.domains)
        self.all_img = total_images
        self.all_subjects = len(total_subjects)
        # self.resize = tio.transforms.Resize(target_shape=self.patch_size[:2]  # only height and width
        # )
 
        
    def _load_and_norm(self, path):
        # print(path)
        if self.cfg.data.nips:
            arr = load_axial_aligned(path).squeeze()
        else:
            arr = nib.load(path).get_fdata().squeeze()
        arr = np.clip(arr, 0, None)
        arr = random_flip(arr)
        if self.cfg.data.normalize:
            if np.random.uniform() <= self.cfg.data.aug_prob:
                perc_dif = 100-self.cfg.data.norm_perc
                arr = norm_img(arr, np.random.uniform(
                    self.cfg.data.norm_perc-perc_dif, 100))
            else:
                arr = norm_img(arr, self.cfg.data.norm_perc)

        return self._pad_to_patch(arr)
    
    def __getitem__(self, _):
        while True:
            try:
                # 1) pick a random index *once*
                idx =  int(np.random.random_sample() // (1 / self.num_domain))
                # 2) pick one distinct domain
                tmp_path = self.path_dic[str(idx)]
                index = np.random.randint(0, len(tmp_path))


                # 3) fetch the *same* idx from each
                p1 = tmp_path[index]

                # 4) load, preprocess, augment, extract patches exactly as before
                scan1 = self._load_and_norm(p1)
                # --- Check for thin Z-dimension ---
                if scan1.shape[-1] < 4:
                    print(f"[DEBUG] Skipping due to thin Z ({scan1.shape}) at {p1}")
                    continue
                # --- Check for empty scan ---
                if scan1.size == 0 or scan1.shape[0] == 0 or scan1.shape[1] == 0:
                    print(f"[DEBUG] Skipping due to empty scan at {p1}")
                    continue
                # --- Check for NaN values ---
                if np.isnan(scan1).any():
                    print(f"[DEBUG] Skipping due to NaN values at {p1}")
                    continue
                t1 = torch.from_numpy(scan1).unsqueeze(0)  # Ensure shape is (1, H, W, Z)
                # print(f'→ {len(t1.shape)} {p1}')
                if torch.sum(t1) < 1e-5:
                    print(f"[DEBUG] Skipping due to blank tensor after affine at {p1}")
                    continue
                t1 = self.affine(t1)
                
                # local1, global1 = self._extract_patches(t1)
                local1 = self._extract_patches(t1)
                # print(local1.shape, global1.shape)
                return {
                    'local_patch': local1,
                    # 'global_img':  global1,
                }
            except Exception as e:
                print(f"[WARN] Skipping due to error: {e}")
                continue
                # return self.__getitem__(_)  # retry with a new sample
            
            
    def _pad_to_patch(self, scan: np.ndarray) -> np.ndarray:
        x, y, _ = self.patch_size
        h, w    = scan.shape[:2]
        if h < x:
            pad_h = x - h
            pad_top = pad_h // 2
            pad_bottom = pad_h - pad_top
            scan = np.pad(scan, ((pad_top, pad_bottom), (0, 0), (0, 0)), constant_values=0)
        elif h > x:
            crop_top = (h - x) // 2
            scan = scan[crop_top:crop_top + x, :, :]
        if w < y:
            pad_w = y - w
            pad_left = pad_w // 2
            pad_right = pad_w - pad_left
            scan = np.pad(scan, ((0, 0), (pad_left, pad_right), (0, 0)), constant_values=0)
        elif w > y:
            crop_left = (w - y) // 2
            scan = scan[:, crop_left:crop_left + y, :]
        return scan
    
    # def _has_foreground(self, volume: torch.Tensor):
    #     """Check if at least one slice has foreground."""
    #     return (volume > 1e-5).any().item()
    def _has_foreground(self, volume: torch.Tensor, threshold: float = 1e-5, min_ratio: float = 0.5):
        """Check if at least `min_ratio` of slices in the volume contain foreground."""
        num_slices = volume.shape[0]
        nonzero_slices = [(volume[i] > threshold).any().item() for i in range(num_slices)]
        ratio = sum(nonzero_slices) / num_slices
        return ratio >= min_ratio

    def _extract_patches(self, scan: torch.Tensor):
        C, H, W, Z = scan.shape
        x, y, z = self.patch_size  # target patch size: (x, y, z=7)
        for _ in range(10):
            center = random.randint(z // 2, Z - z // 2 - 1)
            z_start = center - z // 2
            z_end = center + z // 2 + 1
            slice_stack = scan[0, :, :, z_start:z_end]  # [H, W, z]
            slice_stack = slice_stack.permute(2, 0, 1).contiguous()  # [z, H, W]

            if self._has_foreground(slice_stack):
                break
        else:
            raise ValueError("No non-empty slice found after 10 tries")

        
        return slice_stack  # shape: [7, 256, 256]
    # , global_img

    def __len__(self):
        return 2000  # or: return max(2000, total_scans)


class mpl_dataset(data.Dataset):
    def __init__(self, cfg):
        self.cfg = cfg
        # get all image paths (source and target)
        # folder should end with '_train'

        # data from target domain, only img (folder name should end with '_train')

        tgt_dir, src_dir1 = list_finetune_domains(
            cfg.data.tgt_data, cfg.data.src_data)

        self.path_dic = {}
        for i in range(len(tgt_dir)):
            self.path_dic[str(i)] = sorted(
                list_scans(tgt_dir[i], self.cfg.data.extension))
        self.num_domain = len(tgt_dir)

        # data from source domain,  img + label (folder name should end with '_img' for img and '_label' for label)

        self.path_dic_B1 = {}
        self.path_dic_B2 = {}
        for i in range(len(src_dir1)):
            self.path_dic_B1[str(i)] = sorted(
                list_scans(src_dir1[i], self.cfg.data.extension))
            self.path_dic_B2[str(i)] = [i.replace(
                '_img', '_label') for i in self.path_dic_B1[str(i)]]
            self.path_dic_B2[str(i)] = [i.replace(
                '_te1_split', '_ref_mask_split') for i in self.path_dic_B2[str(i)]]

        self.num_domain_B = len(src_dir1)

        print('num of target domain: ' + str(self.num_domain))
        print('num of source domain: ' + str(self.num_domain_B))

    def __getitem__(self, index):
        idx = int(np.random.random_sample() // (1 / self.num_domain))
        tmp_path = self.path_dic[str(idx)]
        indexA = np.random.randint(0, len(tmp_path))

        idx = int(np.random.random_sample() // (1 / self.num_domain_B))
        tmp_path_B1 = self.path_dic_B1[str(idx)]
        tmp_path_B2 = self.path_dic_B2[str(idx)]

        indexB = np.random.randint(0, len(tmp_path_B1))
        x, y, z = self.cfg.data.patch_size
        '''
        getitem for training/validation
        '''

        '''
        load non-labeled data
        '''
        tmp_scansA = nib.load(tmp_path[indexA])
        # print("tmp_scansA shape: ", tmp_scansA.get_fdata().shape)
        tmp_scansA = np.squeeze(tmp_scansA.get_fdata())
        tmp_scansA[tmp_scansA < 0] = 0

        # normalization
        if self.cfg.data.normalize:
            if np.random.uniform() <= self.cfg.data.aug_prob:
                perc_dif = 100 - self.cfg.data.norm_perc
                tmp_scansA = norm_img(tmp_scansA, np.random.uniform(
                    self.cfg.data.norm_perc - perc_dif, 100))
            else:
                tmp_scansA = norm_img(tmp_scansA, self.cfg.data.norm_perc)
        # padding
        pad_h, pad_w = max(0, x - tmp_scansA.shape[0]), max(0, y - tmp_scansA.shape[1])
        # print("shape of tmp_scansA: ", tmp_scansA.shape)
        if pad_h > 0 or pad_w > 0:
            
            tmp_scansA = np.pad(tmp_scansA, ((pad_h // 2, pad_h - pad_h // 2), (pad_w // 2, pad_w - pad_w // 2),(0,0)), 
                         constant_values=1e-4)  # Avoid zero-padding
        tmp_scansA = np.expand_dims(tmp_scansA, axis=-1)
        tmp_scansA = torch.unsqueeze(torch.from_numpy(tmp_scansA), 0)
        if len(tmp_scansA.shape) != 4:
            return self.__getitem__((index + 1) % len(self))  # Try next valid sample        
        _, x1, y1, z1 = tmp_scansA.shape
        if self.cfg.data.aug:
            transforms = tio.Compose([tio.RandomAffine(p=self.cfg.data.aug_prob, scales=(0.7, 1.3), degrees=30,
                                                       isotropic=False,
                                                       default_pad_value=0, image_interpolation='linear',
                                                       label_interpolation='nearest')
                                      ])
            # tmp_scans = tio.ScalarImage(tensor=tmp_scansA)
            tmp_scans = transforms(tmp_scansA)
        else:
            # tmp_scans = tio.ScalarImage(tensor=tmp_scansA)
            tmp_scans = tmp_scansA
        # randomly select patch
        if self.cfg.data.remove_bg:
            bound = get_bounds(tmp_scans.data)
            if bound[1] - x > bound[0]:
                x_idx = np.random.randint(bound[0], bound[1] - x)
            else:
                if bound[1] - x >= 0:
                    x_idx = bound[1] - x
                else:
                    if bound[0] + x < x1:
                        x_idx = bound[0]
                    else:
                        x_idx = int((x1 - x) / 2)
            if bound[3] - y > bound[2]:
                y_idx = np.random.randint(bound[2], bound[3] - y)
            else:
                if bound[3] - y >= 0:
                    y_idx = bound[3] - y
                else:
                    if bound[2] + y < y1:
                        y_idx = bound[2]
                    else:
                        y_idx = int((y1 - y) / 2)
        else:
            bound = [0, x1, 0, y1]
            x_idx = 0 if x1 - x == 0 else np.random.randint(0, x1 - x)
        #     y_idx = 0 if y1 - y == 0 else np.random.randint(0, y1 - y)
        # print('tmp_scans shape (A): ', tmp_scans.data.shape)
        # print('bound (A): ', bound)
            
        location = torch.zeros_like(tmp_scans.data).float()
        location[:, x_idx:x_idx + x, y_idx:y_idx + y, ] = 1
        sbj = tio.Subject(one_image=tio.ScalarImage(tensor=tmp_scans.data[:, bound[0]:bound[1], bound[2]:bound[3],:]), a_segmentation=tio.LabelMap(
            tensor=location[:, bound[0]:bound[1],
                            bound[2]:bound[3], :]
        ))
        transforms = tio.transforms.Resize(target_shape=(x, y, 1))
        sbj = transforms(sbj)
        down_scan = sbj['one_image'].data[:,:,:,0]
        # print('sbj shape (A): ', sbj['one_image'].data.shape)
        locA = sbj['a_segmentation'].data

        tmp_coor = get_bounds(locA)
        coordinates_A = np.array([np.floor(tmp_coor[0] / 4),
                                  np.ceil(tmp_coor[1] / 4),
                                  np.floor(tmp_coor[2] / 4),
                                  np.ceil(tmp_coor[3] / 4),

                                  ]).astype(int)

        patchA = tmp_scans.data[:, x_idx:x_idx + x,
                                y_idx:y_idx + y,0].float()
        downA = down_scan.float()

        '''
        load annotated data
        '''

        tmp_scans = nib.load(tmp_path_B1[indexB])
        # print("tmp_scansB1 shape: ", tmp_scans.get_fdata().shape)
        tmp_scans = np.squeeze(tmp_scans.get_fdata())
        '''
        WARNING: HERE WE ONLY USE POSITIVE INTENSITY 
        FOR CT, USE PREPROCESSING TO turn negatives to positives 
        
        '''
        tmp_scans[tmp_scans < 0] = 0
        # print("tmp_scansB2 shape: ", nib.load(tmp_path_B2[indexB]).get_fdata().shape)
        tmp_label = np.squeeze(
            np.round(nib.load(tmp_path_B2[indexB]).get_fdata()))
        assert tmp_scans.shape == tmp_label.shape, 'scan and label must have the same shape'

        if self.cfg.data.normalize:
            if np.random.uniform() <= self.cfg.data.aug_prob:
                perc_dif = 100 - self.cfg.data.norm_perc
                tmp_scans = norm_img(tmp_scans, np.random.uniform(
                    self.cfg.data.norm_perc - perc_dif, 100))
            else:
                tmp_scans = norm_img(tmp_scans, self.cfg.data.norm_perc)
        
        pad_h, pad_w = max(0, x - tmp_scans.shape[0]), max(0, y - tmp_scans.shape[1])
        if pad_h > 0 or pad_w > 0:
            tmp_scans = np.pad(tmp_scans, ((pad_h // 2, pad_h - pad_h // 2), (pad_w // 2, pad_w - pad_w // 2),(0,0),), 
                         constant_values=1e-4)  # Avoid zero-padding
            tmp_label = np.pad(tmp_label, ((pad_h // 2, pad_h - pad_h // 2), (pad_w // 2, pad_w - pad_w // 2),(0,0)), 
                         constant_values=1e-4)  # Avoid zero-padding
        tmp_scans = np.expand_dims(tmp_scans, axis=-1)
        tmp_label = np.expand_dims(tmp_label, axis=-1)            
        tmp_scans = torch.unsqueeze(torch.from_numpy(tmp_scans), 0)
        tmp_label = torch.unsqueeze(torch.from_numpy(tmp_label), 0)        

        if len(tmp_scans.shape) != 4:
            return self.__getitem__((index + 1) % len(self))  # Try next valid sample

        _, x1, y1, z1 = tmp_scans.shape
        tmp_scans = tio.ScalarImage(tensor=tmp_scans)
        tmp_label = tio.LabelMap(tensor=tmp_label)
        sbj = tio.Subject(one_image=tmp_scans, a_segmentation=tmp_label)
        if self.cfg.data.aug:
            transforms = tio.Compose([tio.RandomAffine(p=self.cfg.data.aug_prob, scales=(0.7, 1.4), degrees=30,
                                                       isotropic=False,
                                                       default_pad_value=0, image_interpolation='linear',
                                                       label_interpolation='nearest'),
                                      tio.RandomBiasField(
                                      p=self.cfg.data.aug_prob),
                                      tio.RandomGamma(
                                      p=self.cfg.data.aug_prob, log_gamma=(-0.4, 0.4))
                                      ])
            sbj = transforms(sbj)
        tmp_scans = sbj['one_image'].data.float()
        tmp_label = sbj['a_segmentation'].data.float()

        if self.cfg.data.remove_bg:
            bound = get_bounds(tmp_scans.data)
            if bound[1] - x > bound[0]:
                x_idx = np.random.randint(bound[0], bound[1] - x)
            else:
                if bound[1] - x >= 0:
                    x_idx = bound[1] - x
                else:
                    if bound[0] + x < x1:
                        x_idx = bound[0]
                    else:
                        x_idx = int((x1 - x) / 2)
            if bound[3] - y > bound[2]:
                y_idx = np.random.randint(bound[2], bound[3] - y)
            else:
                if bound[3] - y >= 0:
                    y_idx = bound[3] - y
                else:
                    if bound[2] + y < y1:
                        y_idx = bound[2]
                    else:
                        y_idx = int((y1 - y) / 2)

        else:
            bound = [0, x1, 0, y1]
            x_idx = 0 if x1 - x == 0 else np.random.randint(0, x1 - x)
            y_idx = 0 if y1 - y == 0 else np.random.randint(0, y1 - y)
        # print('tmp_scans shape (B): ', tmp_scans.data.shape)
        # print('bound (B): ', bound)
        location_B = torch.zeros_like(tmp_scans.data).float()
        location_B[:, x_idx:x_idx + x,
                   y_idx:y_idx + y, ] = 1

        sbj = tio.Subject(one_image=tio.ScalarImage(tensor=tmp_scans[:, bound[0]:bound[1], bound[2]:bound[3],:]),
                          a_segmentation=tio.LabelMap(
            tensor=location_B[:, bound[0]:bound[1], bound[2]:bound[3], :])
        )
        transforms = tio.transforms.Resize(target_shape=(x, y, 1))
        sbj = transforms(sbj)
        # print('sbj shape (B): ', sbj['one_image'].data.shape)
        down_scan = sbj['one_image'].data[:,:,:,0].float()
        locB = sbj['a_segmentation'].data

        tmp_coor = get_bounds(locB)
        sbj = tio.Subject(one_image=tio.ScalarImage(tensor=tmp_scans[:, bound[0]:bound[1], bound[2]:bound[3],:]),
                          a_segmentation=tio.LabelMap(
            tensor=tmp_label[:, bound[0]:bound[1], bound[2]:bound[3], :])
        )
        sbj = transforms(sbj)
        aux_label = sbj['a_segmentation'].data[:,:,:,0] 

        coordinates_B = np.array([np.floor(tmp_coor[0] / 4),
                                  np.ceil(tmp_coor[1] / 4),
                                  np.floor(tmp_coor[2] / 4),
                                  np.ceil(tmp_coor[3] / 4),
                                ]).astype(int)
        
        input_dict = {'imgB': tmp_scans[:, x_idx:x_idx + x, y_idx:y_idx + y, 0],
                      'labelB': torch.squeeze(tmp_label[:, x_idx:x_idx + x, y_idx:y_idx + y, 0]),
                      'label_B_aux': torch.squeeze(aux_label),
                      'downB': down_scan,
                      'cord_B': coordinates_B,
                      'imgA': patchA,
                      'downA': downA,
                      'cord_A': coordinates_A}
        # print('imgB shape: ', input_dict['imgB'].shape)
        # print('labelB shape: ', input_dict['labelB'].shape)
        # print('downB shape: ', input_dict['downB'].shape)
        # print('cord_B shape: ', input_dict['cord_B'].shape)
        # print('imgA shape: ', input_dict['imgA'].shape)
        # print('downA shape: ', input_dict['downA'].shape)
        # print('cord_A shape: ', input_dict['cord_A'].shape)

        return input_dict

    def __len__(self):

        # we used fixed 100 steps for each epoch in finetuning
        # THIS PARAM WAS NEVER TUNED
        return 100

class mae_dataset_s(data.Dataset):
    def __init__(self, cfg):
        self.cfg = cfg
        self.patch_size = cfg.data.patch_size  # (x, y, z)
        self.aug_prob   = cfg.data.aug_prob
        self.normalize  = cfg.data.normalize
        self.norm_perc  = cfg.data.norm_perc
        self.remove_bg  = cfg.data.remove_bg
        # get all image paths (source and target)
        # folder should end with '_train'
        
        # 1) build domain→scan‑paths dict
        self._build_path_index()
        
        # 2) prepare torchio transforms once
        self.affine = tio.RandomAffine(
            p=self.aug_prob,
            scales=(0.75, 1.5),
            degrees=40,
            isotropic=False,
            default_pad_value=0,
            image_interpolation='linear'
        )
        self.resize = tio.transforms.Resize(target_shape=(self.patch_size[0],
                                                         self.patch_size[1],
                                                         1))        
    def _build_path_index(self):
        self.path_dict = {}
        total = 0
        for i, domain in enumerate(list_mae_domains(self.cfg.data.mae_root)):
            scans = sorted(list_scans(domain, self.cfg.data.extension))
            self.path_dict[i] = scans
            total += len(scans)

        self.num_domains = len(self.path_dict)
        self.length = min(len(v) for v in self.path_dict.values())
        print(f'→ {self.num_domains} domains, {total} scans in total')
        
    def _load_and_norm(self, path):
        arr = nib.load(path).get_fdata().squeeze()
        arr = np.clip(arr, 0, None)
        arr = random_flip(arr)
        if self.normalize and random.random() < self.aug_prob:
            low = self.norm_perc * 0.5
            p   = random.uniform(low, 100.0)
            arr = norm_img(arr, p)
        else:
            arr = norm_img(arr, self.norm_perc)
        return self._pad_to_patch(arr)
    
    def __getitem__(self, _):
        # 1) pick a random index *once*
        idx = random.randrange(self.length)

        # 2) pick two distinct domains
        d1 = random.randrange(self.num_domains)
        d2 = random.randrange(self.num_domains)
        while d2 == d1:
            d2 = random.randrange(self.num_domains)

        # 3) fetch the *same* idx from each
        p1 = self.path_dict[d1][idx]
        p2 = self.path_dict[d2][idx]
        # print(f'→ {d1} {p1}  {d2} {p2}')

        # 4) load, preprocess, augment, extract patches exactly as before
        scan1 = self._load_and_norm(p1)
        scan2 = self._load_and_norm(p2)
        
        t1, t2 = torch.from_numpy(scan1).unsqueeze(0), torch.from_numpy(scan2).unsqueeze(0)
        # Ensure input tensors are (1, H, W, Z) for TorchIO
        # 2) put into one subject:
        subject = tio.Subject(
            img1=tio.ScalarImage(tensor=t1),
            img2=tio.ScalarImage(tensor=t2),
        )
        subject = self.affine(subject)
        t1, t2 = subject['img1'].data, subject['img2'].data  # [1, H, W, Z]
        local1, global1 = self._extract_patches(t1)
        local2, global2 = self._extract_patches(t2)

        return {
            'local_patch1': local1,
            'global_img1':  global1,
            'local_patch2': local2,
            'global_img2':  global2,
        }

    def _pad_to_patch(self, scan: np.ndarray) -> np.ndarray:
        x, y, _ = self.patch_size
        h, w    = scan.shape[:2]
        pad_h   = max(0, x - h)
        pad_w   = max(0, y - w)
        if pad_h or pad_w:
            pad = (
                (pad_h//2, pad_h - pad_h//2),
                (pad_w//2, pad_w - pad_w//2),
                (0, 0)
            )
            scan = np.pad(scan, pad, constant_values=1e-4)
        return scan

    def _extract_patches(self, scan: torch.Tensor):
        C, H, W, Z = scan.shape
        x, y, z   = self.patch_size

        # pick a slice at random
        sl = random.randrange(Z)
        slice_2d = scan[0, :, :, sl]  # [H, W]

        # compute foreground bounds if requested
        bound = get_bounds(slice_2d)
        max_x = H - x
        max_y = W - y

        if self.remove_bg:
            x0 = np.clip(random.randint(bound[0], bound[1] - x), 0, max_x)
            y0 = np.clip(random.randint(bound[2], bound[3] - y), 0, max_y)
        else:
            x0 = random.randint(0, max_x) if max_x > 0 else 0
            y0 = random.randint(0, max_y) if max_y > 0 else 0

        # local patch
        local = slice_2d[x0:x0+x, y0:y0+y].unsqueeze(0)  # [1, x, y]

        # downsample the full slice
        subject = tio.Subject(
            image=tio.ScalarImage(tensor=slice_2d.unsqueeze(0).unsqueeze(-1))
        )
        subject = self.resize(subject)
        global_img = subject['image'].data[:, :, :, 0]     # [1, H', W']

        return local, global_img

    def __len__(self):
        return 2000  # or: return max(2000, total_scans)

