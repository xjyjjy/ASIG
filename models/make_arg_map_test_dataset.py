import argparse
import math
import sys
from itertools import product
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from scipy.io import loadmat
from tqdm import tqdm

ROOT_DIR = Path(__file__).resolve().parents[1]
DATA_ROOT = Path('/home/zwxionggroup/xiajy/dataset')
sys.path.append(str(ROOT_DIR))

from utils.projection_helper import erp2sphere
from utils.utils import make_arg_map_dataset, make_coord, gridy2gridx_erp2pers

COMPARE_DIR = DATA_ROOT / 'arg_map_for_test' / 'compare_arg_pano_test_dataset_2048_512'
ARG_MAP_SAVE_DIR = DATA_ROOT / 'arg_map_for_test' / 'arg_map_2048_512'

img = cv2.imread(str(DATA_ROOT / '360/mv/rgb_test_big/0026d80fc3c84d91bd088b831d230738_rgb.png'))
face_vector_img_8 = loadmat(str(ROOT_DIR / 'mesh/level9_vector_inverse.mat'))['vector']
face_vector_img_tr_8 = face_vector_img_8.reshape(-1, 3)
img_level = 9
THETA_range = [0]
PHI_range = [0]
W = [1024]
H_FOV = [160, 120]
RATIO = [0.75]


def calculate_psnr_tensor(img1, img2):
    img1 = torch.from_numpy(img1 / 255.0).float()
    img2 = torch.from_numpy(img2 / 255.0).float()
    mse = torch.mean((img1 - img2) ** 2)
    if mse == 0:
        return float('inf')
    return 20 * math.log10(1.0 / math.sqrt(mse.item()))


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--W', type=int, nargs='+', default=[1024])
    return parser.parse_args()


def save_bicubic(image, width, h_fov, ratio, theta, phi, theta_save, phi_save):
    pan_h, pan_w = image.shape[:2]
    image = torch.from_numpy(image / 255).permute(2, 0, 1).float()
    height = int(width * ratio)
    gridy = make_coord((height, width))
    grid_pers, _ = gridy2gridx_erp2pers(gridy.flip(-1), 1664, 3328, int(width * ratio), width, h_fov, theta, phi)
    grid_pers = grid_pers.view(height, width, 2)
    pad_size = 4
    pad_img = F.pad(image.unsqueeze(0), [pad_size, pad_size, pad_size, pad_size], mode='replicate')
    scale_y = pan_h / (pan_h + 2 * pad_size)
    scale_x = pan_w / (pan_w + 2 * pad_size)
    grid_pers = grid_pers * torch.tensor([scale_y, scale_x])
    out = F.grid_sample(pad_img, grid_pers.unsqueeze(0).flip(-1), mode='bicubic', align_corners=False)
    out.clamp_(0, 1)
    out = (255 * out.squeeze(0).permute(1, 2, 0).cpu().numpy()).astype('uint8')
    cv2.imwrite(str(COMPARE_DIR / f'{int(theta_save)}_{int(phi_save)}_{width}_{h_fov}_{ratio}_bicubic.png'), out)
    return out


def save_arg_map(arg_map, image, width, h_fov, ratio, theta_save, phi_save):
    arg_map = arg_map.cuda()
    image = torch.from_numpy(image / 255).permute(2, 0, 1).float().unsqueeze(0).cuda()
    image = erp2sphere(image, face_vector_img_tr_8)
    bs, channels, _, _ = image.shape
    box_num, height, width_box, _ = face_vector_img_8.shape
    image = image.view(bs, channels, box_num, height, width_box).permute(0, 2, 1, 3, 4).contiguous().view(bs * box_num, channels, height, width_box)
    image = image.view(bs, 5, -1, height, width_box).permute(0, 1, 3, 4, 2).reshape(bs, 5 * height * width_box, -1)
    image = image[:, arg_map, :].permute(0, 3, 1, 2).contiguous()
    image.clamp_(0, 1)
    out = (255 * image.squeeze(0).permute(1, 2, 0).cpu().numpy()).astype('uint8')
    cv2.imwrite(str(COMPARE_DIR / f'{int(theta_save)}_{int(phi_save)}_{width}_{h_fov}_{ratio}_arg_map.png'), out)
    return out


args = parse_args()
W = args.W
COMPARE_DIR.mkdir(parents=True, exist_ok=True)
ARG_MAP_SAVE_DIR.mkdir(parents=True, exist_ok=True)
param_list = list(product(THETA_range, PHI_range, W, H_FOV, RATIO))
psnr_list = []
for theta, phi, width, h_fov, ratio in tqdm(param_list, desc='Processing'):
    theta_save = theta
    phi_save = phi
    arg_map, dist_vector = make_arg_map_dataset(
        face_vector_img_8, int(width * ratio), width, int(h_fov * ratio), h_fov,
        theta, phi, img_level, theta_save, phi_save, save_path=str(ARG_MAP_SAVE_DIR) + '/',
    )
    if arg_map is None:
        continue
    bicubic_img = save_bicubic(img, width, h_fov, ratio, theta, phi, theta_save, phi_save)
    arg_map_img = save_arg_map(arg_map, img, width, h_fov, ratio, theta_save, phi_save)
    psnr_list.append(calculate_psnr_tensor(bicubic_img, arg_map_img))
average_psnr = float(np.mean(psnr_list))
