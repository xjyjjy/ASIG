import sys
from pathlib import Path

import cv2
import torch
import torch.nn.functional as F
from scipy.io import loadmat
from tqdm import tqdm

ROOT_DIR = Path(__file__).resolve().parents[1]
DATA_ROOT = Path('/home/zwxionggroup/xiajy/dataset')
sys.path.append(str(ROOT_DIR))

from utils.projection_helper import erp2sphere
from utils.utils import make_coord, make_arg_map_dataset_panorama

PANORAMA_SAVE_DIR = DATA_ROOT / 'panorama_generation_com'

img = cv2.imread(str(DATA_ROOT / '360/360sp/panoramas/__EFZYuzVQTnCAqgyyrY3g.png'))
face_vector_img_8 = loadmat(str(ROOT_DIR / 'mesh/level9_vector_inverse.mat'))['vector']
face_vector_img_tr_8 = face_vector_img_8.reshape(-1, 3)


def save_bicubic(image, width):
    pan_h, pan_w = image.shape[:2]
    image = torch.from_numpy(image / 255).permute(2, 0, 1).float()
    height = width // 2
    grid_pers = make_coord((height, width)).view(height, width, 2).flip(-1)
    pad_size = 4
    pad_img = F.pad(image.unsqueeze(0), [pad_size, pad_size, pad_size, pad_size], mode='replicate')
    scale_y = pan_h / (pan_h + 2 * pad_size)
    scale_x = pan_w / (pan_w + 2 * pad_size)
    grid_pers = grid_pers * torch.tensor([scale_y, scale_x])
    out = F.grid_sample(pad_img, grid_pers.unsqueeze(0).flip(-1), mode='bicubic', align_corners=False)
    out.clamp_(0, 1)
    out = (255 * out.squeeze(0).permute(1, 2, 0).cpu().numpy()).astype('uint8')
    cv2.imwrite(str(PANORAMA_SAVE_DIR / f'{height}_{width}_bicubic.png'), out)


def save_arg_map(arg_map, image, width):
    arg_map = arg_map.cuda()
    image = torch.from_numpy(image / 255).permute(2, 0, 1).float().unsqueeze(0).cuda()
    image = erp2sphere(image, face_vector_img_tr_8)
    height = width // 2
    bs, channels, _, _ = image.shape
    box_num, feat_h, feat_w, _ = face_vector_img_8.shape
    image = image.view(bs, channels, box_num, feat_h, feat_w).permute(0, 2, 1, 3, 4).contiguous().view(bs * box_num, channels, feat_h, feat_w)
    image = image.view(bs, 5, -1, feat_h, feat_w).permute(0, 1, 3, 4, 2).reshape(bs, 5 * feat_h * feat_w, -1)
    image = image[:, arg_map, :].permute(0, 3, 1, 2).contiguous()
    image.clamp_(0, 1)
    out = (255 * image.squeeze(0).permute(1, 2, 0).cpu().numpy()).astype('uint8')
    cv2.imwrite(str(PANORAMA_SAVE_DIR / f'{height}_{width}_arg_map.png'), out)


PANORAMA_SAVE_DIR.mkdir(parents=True, exist_ok=True)
for width in tqdm([2048]):
    height = width // 2
    arg_map, dist_vector = make_arg_map_dataset_panorama(face_vector_img_8, height, width)
    save_bicubic(img, width)
    save_arg_map(arg_map, img, width)
