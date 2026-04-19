import random
from PIL import Image

import numpy as np
import torch
import os

from datasets.data import Dataset
from torchvision import transforms

import datasets
from datasets import register
from scipy.io import savemat, loadmat
from utils.geometry import make_coord_grid
import torch.nn.functional as F
from torchvision.utils import save_image
from utils.utils import to_pixel_samples, make_coord, make_cell, gridy2gridx_homography, celly2cellx_homography, to_pixel_samples_grid, make_cell_, gridy2gridx_erp2pers
from utils import Equirec2Perspec as E2P
from utils.projection_helper import erp2sphere

def resize_fn(img, size):
    return transforms.ToTensor()(
        transforms.Resize(size, Image.BICUBIC)(
            transforms.ToPILImage()(img)))

class BaseWrapperDiv2kWarp_:
    def __init__(self, dataset, arg_map_path = None, img_max=None, patch_max=None, ret_gt=True, resize_gt_lb=None, resize_gt_ub=None,
                 final_crop_gt=None, p_whole=0.0, p_max=0.0):
        self.dataset = datasets.make(dataset)
        self.img_max = img_max
        self.patch_max = patch_max
        self.ret_gt = ret_gt
        self.resize_gt_lb = resize_gt_lb
        self.resize_gt_ub = resize_gt_ub
        self.final_crop_gt = final_crop_gt
        self.p_whole = p_whole
        self.p_max = p_max
        self.face_vector_img = loadmat("./mesh/level9_vector_inverse.mat")["vector"]
        self.face_vector_img_tr = self.face_vector_img.reshape(-1, 3)
        self.arg_map_path = arg_map_path

        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(0.5, 0.5),
        ])

    def process(self, data, H, W, others = None):
        if 'transform' in data:
            hr = data['image_hr']
            img = data['image']
            transform_m = data['transform']
            text_data = data['text']
        elif 'text' in data:
            img = data['image']
            text_data = data['text']
            text_data_2 = data['text_2']
        else:
            img = data['image']

        pano_H, pano_W = img.shape[1:3]
        if others is not None:
            THETA = others[0]
            PHI = others[1]
        else:

            THETA_range = [-150, -120, -90, -60, -30, 0, 30, 60, 90, 120, 150, 180]
            PHI_range = [-90, -80, -70, -60, -50, -40, -30, -20, -10, 0 ,10, 20, 30, 40, 50, 60, 70, 80, 90]
            if pano_H == 1024:
                PHI_prob = [  0,   0,   0,   0,   1,   1,   2,   3,   4,  5,  4,  3,  2,  1,  1,  0,  0,  0,  0]
            elif pano_H == 1664:
                PHI_prob = [0.5, 0.5, 0.5,   1,   1,   1,   2,   3,   4,  5,  4,  3,  2,  1, 1,   1, 0.5, 0.5, 0.5]
            else:
                PHI_prob = [0.5, 0.5, 0.5,   1,   1,   1,   2,   3,   4,  5,  4,  3,  2,  1, 1,   1, 0.5, 0.5, 0.5]

            THETA = random.choice(THETA_range)
            PHI = random.choices(PHI_range, weights=PHI_prob)[0]

        ratio = H/W

        if pano_H == 1024:
            if PHI == 50 or PHI == -50:
                H_FOV = random.choice([70, 60])
            else:
                H_FOV = random.choice([100, 90, 80, 70])
        else:
            H_FOV = random.choice([100, 90, 80, 70, 60])

        V_FOV = int(H_FOV * ratio)

        file_name = "8_{}_{}_{}_{}_{}_{}_arg_new_v2.mat".format(H,W,V_FOV,H_FOV, THETA, PHI)
        arg_map_path = os.path.join(self.arg_map_path, file_name)
        arg_map_file = loadmat(arg_map_path)
        arg_map = torch.from_numpy(arg_map_file["arg_map"])
        dist_vector = torch.from_numpy(arg_map_file["dist_vector"])
        THETA = arg_map_file["THETA"]
        PHI = arg_map_file["PHI"]
        gt_h = H
        gt_w = W

        c,h,w = img.shape
        gridy = make_coord((gt_h,gt_w))
        grid_pers, mask = gridy2gridx_erp2pers(gridy.flip(-1), h, w, gt_h, gt_w, H_FOV, THETA, PHI)
        grid_pers = grid_pers.view(gt_h, gt_w,2)

        pad_size = 4
        PAN_H, PAN_W = img.shape[-2:]
        pad_img = F.pad(img.unsqueeze(0), [pad_size, pad_size, pad_size, pad_size], mode='replicate')
        scale_y = PAN_H / (PAN_H + 2*pad_size)
        scale_x =  PAN_W/ (PAN_W + 2*pad_size)
        grid_pers = grid_pers * torch.tensor([scale_y, scale_x])

        gt = F.grid_sample(pad_img, grid_pers.unsqueeze(0).flip(-1), mode='bicubic', align_corners=False)
        gt.clamp_(0,1)
        img = erp2sphere(img.unsqueeze(0), self.face_vector_img_tr).squeeze(0)
        gridy, hr_rgb = to_pixel_samples_grid(img.contiguous())
        cell = make_cell(gridy, img)
        ret = {}

        if 'transform' in data:
            ret.update({
            'inp': img,
            'gt': gt[0],
            'gt_coord': gridy, # p p 2
            'gt_cell': cell, # p p 10
            'text': text_data.clone().detach(),
            'metric': transform_m.clone().detach(),
            'mask': mask
            })
        elif 'text' in data:
            ret.update({
            'inp': img,
            'gt': gt[0],
            'arg_map': arg_map,
            'dist_vector': dist_vector,
            'gt_coord': gridy, # p p 2
            'gt_cell': cell, # p p 10
            'text': text_data.clone().detach(),
            'text_2': text_data_2.clone().detach(),
            'mask': mask
            })
        else:
            ret.update({
            'inp': img,
            'gt': gt[0], # 3 p p
            'arg_map': arg_map,
            'dist_vector': dist_vector,
            'gt_coord': gridy, # p p 2
            'gt_cell': cell, # p p 10
            'patch_max': torch.tensor(self.patch_max, dtype=torch.float32),
            'img_max': torch.tensor(self.img_max, dtype=torch.float32),
            })

        return ret

@register('360sp_pan_self_proj_fd')
class WrapperCAE(BaseWrapperDiv2kWarp_, Dataset):
    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx, H=None, W=None, others=None):
        data = self.dataset[idx]
        if H is None:
            H = data['H']
        if W is None:
            W = data['W']
        return self.process(data, H, W, others=others)

class BaseWrapperDiv2kWarp:
    def __init__(self, dataset, arg_map_path = None, img_max=None, patch_max=None, ret_gt=True, resize_gt_lb=None, resize_gt_ub=None,
                 final_crop_gt=None, p_whole=0.0, p_max=0.0):
        self.dataset = datasets.make(dataset)
        self.img_max = img_max
        self.patch_max = patch_max
        self.ret_gt = ret_gt
        self.resize_gt_lb = resize_gt_lb
        self.resize_gt_ub = resize_gt_ub
        self.final_crop_gt = final_crop_gt
        self.p_whole = p_whole
        self.p_max = p_max
        self.face_vector_img = loadmat("./mesh/level9_vector_inverse.mat")["vector"]
        self.face_vector_img_tr = self.face_vector_img.reshape(-1, 3)
        self.arg_map_path = arg_map_path
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(0.5, 0.5),
        ])

    def process(self, data, H, W, others = None):
        if 'transform' in data:
            hr = data['image_hr']
            img = data['image']
            transform_m = data['transform']
            text_data = data['text']
        elif 'text' in data:
            img = data['image']
            text_data = data['text']
            text_data_2 = data['text_2']
        else:
            recon = data['recon']
            img = data['image']

        pano_H, pano_W = img.shape[1:3]
        if others is not None:
            THETA = others[0]
            PHI = others[1]
        else:

            THETA_range = [-150, -120, -90, -60, -30, 0, 30, 60, 90, 120, 150, 180]
            PHI_range = [-90, -80, -70, -60, -50, -40, -30, -20, -10, 0 ,10, 20, 30, 40, 50, 60, 70, 80, 90]
            if pano_H == 1024:
                PHI_prob = [  0,   0,   0,   0,   1,   1,   2,   3,   4,  5,  4,  3,  2,  1,  1,  0,  0,  0,  0]
            elif pano_H == 1664:
                PHI_prob = [0.5, 0.5, 0.5,   1,   1,   1,   2,   3,   4,  5,  4,  3,  2,  1, 1,   1, 0.5, 0.5, 0.5]
            else:
                PHI_prob = [0.5, 0.5, 0.5,   1,   1,   1,   2,   3,   4,  5,  4,  3,  2,  1, 1,   1, 0.5, 0.5, 0.5]

            THETA = random.choice(THETA_range)
            PHI = random.choices(PHI_range, weights=PHI_prob)[0]

        ratio = H/W

        if pano_H == 1024:
            if PHI == 50 or PHI == -50:
                H_FOV = random.choice([70, 60])
            else:
                H_FOV = random.choice([100, 90, 80, 70])
        else:
            H_FOV = random.choice([100, 90, 80, 70, 60])

        V_FOV = int(H_FOV * ratio)

        file_name = "8_{}_{}_{}_{}_{}_{}_arg_new_v2.mat".format(H,W,V_FOV,H_FOV, THETA, PHI)
        arg_map_path = os.path.join(self.arg_map_path, file_name)
        arg_map_file = loadmat(arg_map_path)
        arg_map = torch.from_numpy(arg_map_file["arg_map"])
        dist_vector = torch.from_numpy(arg_map_file["dist_vector"])
        THETA = arg_map_file["THETA"]
        PHI = arg_map_file["PHI"]
        gt_h = H
        gt_w = W

        c,h,w = img.shape
        gridy = make_coord((gt_h,gt_w))
        grid_pers, mask = gridy2gridx_erp2pers(gridy.flip(-1), h, w, gt_h, gt_w, H_FOV, THETA, PHI)
        grid_pers = grid_pers.view(gt_h, gt_w,2)

        pad_size = 4
        PAN_H, PAN_W = img.shape[-2:]
        pad_img = F.pad(img.unsqueeze(0), [pad_size, pad_size, pad_size, pad_size], mode='replicate')
        scale_y = PAN_H / (PAN_H + 2*pad_size)
        scale_x =  PAN_W/ (PAN_W + 2*pad_size)
        grid_pers = grid_pers * torch.tensor([scale_y, scale_x])

        gt = F.grid_sample(pad_img, grid_pers.unsqueeze(0).flip(-1), mode='bicubic', align_corners=False)
        gt.clamp_(0,1)

        ret = {}

        if 'transform' in data:
            ret.update({
            'inp': img,
            'gt': gt[0],
            'gt_coord': gridy, # p p 2
            'text': text_data.clone().detach(),
            'metric': transform_m.clone().detach(),
            'mask': mask
            })
        elif 'text' in data:
            ret.update({
            'inp': img,
            'gt': gt[0],
            'arg_map': arg_map,
            'dist_vector': dist_vector,
            'gt_coord': gridy, # p p 2
            'text': text_data.clone().detach(),
            'text_2': text_data_2.clone().detach(),
            'mask': mask
            })
        else:
            ret.update({
            'gt': gt[0], # 3 p p
            'arg_map': arg_map,
            'recon': recon,
            'dist_vector': dist_vector,
            'gt_coord': gridy, # p p 2
            'patch_max': torch.tensor(self.patch_max, dtype=torch.float32),
            'img_max': torch.tensor(self.img_max, dtype=torch.float32),
            })
        return ret

@register('360sp_pan_self_proj_fd_lmdb')
class WrapperCAE(BaseWrapperDiv2kWarp, Dataset):
    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx, H=None, W=None, others=None):
        data = self.dataset[idx]
        if H is None:
            H = data['H']
        if W is None:
            W = data['W']
        return self.process(data, H, W, others=others)

class BaseWrapperDiv2kWarp_pano:
    def __init__(self, dataset, arg_map_path = None, img_max=None, patch_max=None, ret_gt=True, resize_gt_lb=None, resize_gt_ub=None,
                 final_crop_gt=None, p_whole=0.0, p_max=0.0):
        self.dataset = datasets.make(dataset)
        self.img_max = img_max
        self.patch_max = patch_max
        self.ret_gt = ret_gt
        self.resize_gt_lb = resize_gt_lb
        self.resize_gt_ub = resize_gt_ub
        self.final_crop_gt = final_crop_gt
        self.p_whole = p_whole
        self.p_max = p_max
        self.face_vector_img = loadmat("./mesh/level9_vector_inverse.mat")["vector"]
        self.face_vector_img_tr = self.face_vector_img.reshape(-1, 3)
        self.arg_map_path = arg_map_path
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(0.5, 0.5),
        ])

    def process(self, data, H, W, others = None):
        if 'transform' in data:
            hr = data['image_hr']
            img = data['image']
            transform_m = data['transform']
            text_data = data['text']
        elif 'text' in data:
            img = data['image']
            text_data = data['text']
            text_data_2 = data['text_2']
        else:
            recon = data['recon']
            img = data['image']

        pano_H, pano_W = img.shape[1:3]
        if others is not None:
            THETA = others[0]
            PHI = others[1]
        else:

            THETA_range = [-150, -120, -90, -60, -30, 0, 30, 60, 90, 120, 150, 180]
            PHI_range = [-90, -80, -70, -60, -50, -40, -30, -20, -10, 0 ,10, 20, 30, 40, 50, 60, 70, 80, 90]
            if pano_H == 1024:
                PHI_prob = [  0,   0,   0,   0,   1,   1,   2,   3,   4,  5,  4,  3,  2,  1,  1,  0,  0,  0,  0]
            elif pano_H == 1664:
                PHI_prob = [0.5, 0.5, 0.5,   1,   1,   1,   2,   3,   4,  5,  4,  3,  2,  1, 1,   1, 0.5, 0.5, 0.5]
            else:
                PHI_prob = [0.5, 0.5, 0.5,   1,   1,   1,   2,   3,   4,  5,  4,  3,  2,  1, 1,   1, 0.5, 0.5, 0.5]

            THETA = random.choice(THETA_range)
            PHI = random.choices(PHI_range, weights=PHI_prob)[0]

        ratio = H/W

        if pano_H == 1024:
            if PHI == 50 or PHI == -50:
                H_FOV = random.choice([70, 60])
            else:
                H_FOV = random.choice([100, 90, 80, 70])
        else:
            H_FOV = random.choice([100, 90, 80, 70, 60])

        V_FOV = int(H_FOV * ratio)

        arg_map_path = "/home/zwxionggroup/xurk/dataset/arg_map_panorama_level9/1024_2048_arg_new_v2.mat"
        arg_map_file = loadmat(arg_map_path)
        arg_map = torch.from_numpy(arg_map_file["arg_map"])
        dist_vector = torch.from_numpy(arg_map_file["dist_vector"])
        gt_h = H
        gt_w = W
        gridy = make_coord((gt_h,gt_w))

        gt = img.unsqueeze(0)
        gt.clamp_(0,1)

        ret = {}

        if 'transform' in data:
            ret.update({
            'inp': img,
            'gt': gt[0],
            'gt_coord': gridy, # p p 2
            'text': text_data.clone().detach(),
            'metric': transform_m.clone().detach(),
            'mask': mask
            })
        elif 'text' in data:
            ret.update({
            'inp': img,
            'gt': gt[0],
            'arg_map': arg_map,
            'dist_vector': dist_vector,
            'gt_coord': gridy, # p p 2
            'text': text_data.clone().detach(),
            'text_2': text_data_2.clone().detach(),
            'mask': mask
            })
        else:
            ret.update({
            'inp': img,
            'gt': gt[0], # 3 p p
            'arg_map': arg_map,
            'recon': recon,
            'dist_vector': dist_vector,
            'gt_coord': gridy, # p p 2
            'patch_max': torch.tensor(self.patch_max, dtype=torch.float32),
            'img_max': torch.tensor(self.img_max, dtype=torch.float32),
            })
        return ret

@register('360sp_pan_self_proj_fd_lmdb_pano')
class WrapperCAE(BaseWrapperDiv2kWarp_pano, Dataset):
    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx, H=None, W=None, others=None):
        data = self.dataset[idx]
        if H is None:
            H = data['H']
        if W is None:
            W = data['W']
        return self.process(data, H, W, others=others)
