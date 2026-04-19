import os
import shutil
import time
import logging

import numpy as np
import gc
from torch.optim import SGD, Adam, AdamW
from bitsandbytes.optim import AdamW8bit
from tensorboardX import SummaryWriter
import os
import time
import shutil
import math

import torch
import numpy as np
from torch.optim import SGD, Adam
from torch.nn.parameter import UninitializedBuffer, UninitializedParameter
from tensorboardX import SummaryWriter
from torchvision.transforms import ToPILImage
from  .projection_helper import xyz2uv, remap, xyz2uv_panorama
import cv2
#from srwarp import transform
from scipy.io import savemat, loadmat


import torch.nn.functional as F


def ensure_path(path, replace=True, force_replace=False):
    is_temp = os.path.basename(path.rstrip('/')).startswith('_')
    if os.path.exists(path):
        # if replace and (is_temp or force_replace or input(f'{path} exists, replace? y/[n]') == 'y'):
        #     shutil.rmtree(path) 
        #     os.mkdir(path)
        print(f'resume from {path}.')
    else:
        os.makedirs(path)


def set_logger(file_path):
    logger = logging.getLogger()
    logger.setLevel('INFO')
    stream_handler = logging.StreamHandler()
    file_handler = logging.FileHandler(file_path, 'a')
    formatter = logging.Formatter('[%(asctime)s] %(message)s', '%m-%d %H:%M:%S')
    for handler in [stream_handler, file_handler]:
        handler.setFormatter(formatter)
        handler.setLevel('INFO')
        logger.addHandler(handler)
    return logger


def set_save_dir(save_dir, replace=True):
    ensure_path(save_dir, replace=replace)
    logger = set_logger(os.path.join(save_dir, 'log.txt'))
    writer = SummaryWriter(os.path.join(save_dir, 'tensorboard'))
    return logger, writer


def compute_num_params(model, text=True):
    tot = 0
    for p in model.parameters():
        if isinstance(p, (UninitializedParameter, UninitializedBuffer)):
            continue
        tot += int(np.prod(p.shape))
    if text:
        if tot >= 1e6:
            s = '{:.1f}M'.format(tot / 1e6)
        else:
            s = '{:.1f}K'.format(tot / 1e3)
        return f'{s} ({tot})'
    else:
        return tot


def make_optimizer(params, optimizer_spec, load_sd=False):
    optimizer = {
        'sgd': SGD,
        'adam': Adam,
        'adamw': AdamW,
        'adamw8bit': AdamW8bit,
    }[optimizer_spec['name']](params, **optimizer_spec['args'])
    if load_sd:
        optimizer.load_state_dict(optimizer_spec['sd'])
    return optimizer


class Averager():

    def __init__(self):
        self.n = 0.0
        self.v = 0.0

    def add(self, v, n=1.0):
        self.v = (self.v * self.n + v * n) / (self.n + n)
        self.n += n

    def item(self):
        return self.v


class EpochTimer():

    def __init__(self, max_epoch):
        self.max_epoch = max_epoch
        self.epoch = 0
        self.t_start = time.time()
        self.t_last = self.t_start

    def epoch_done(self):
        t_cur = time.time()
        self.epoch += 1
        epoch_time = t_cur - self.t_last
        tot_time = t_cur - self.t_start
        est_time = tot_time / self.epoch * self.max_epoch
        self.t_last = t_cur
        return time_text(epoch_time), time_text(tot_time), time_text(est_time)


def time_text(sec):
    if sec >= 3600:
        return f'{sec / 3600:.1f}h'
    elif sec >= 60:
        return f'{sec / 60:.1f}m'
    else:
        return f'{sec:.1f}s'



def str2bool(v):
    if isinstance(v, bool):
         return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')
        

class Averager():

    def __init__(self):
        self.n = 0.0
        self.v = 0.0

    def add(self, v, n=1.0):
        self.v = (self.v * self.n + v * n) / (self.n + n)
        self.n += n

    def item(self):
        return self.v


class Timer():

    def __init__(self):
        self.v = time.time()

    def s(self):
        self.v = time.time()

    def t(self):
        return time.time() - self.v


def time_text(t):
    if t >= 3600:
        return '{:.1f}h'.format(t / 3600)
    elif t >= 60:
        return '{:.1f}m'.format(t / 60)
    else:
        return '{:.1f}s'.format(t)


_log_path = None


def set_log_path(path):
    global _log_path
    _log_path = path


def log(obj, filename='log.txt'):
    print(obj)
    if _log_path is not None:
        with open(os.path.join(_log_path, filename), 'a') as f:
            print(obj, file=f)


def compute_num_params(model, text=False):
    tot = 0
    for p in model.parameters():
        if isinstance(p, (UninitializedParameter, UninitializedBuffer)):
            continue
        tot += int(np.prod(p.shape))
    if text:
        if tot >= 1e6:
            return '{:.1f}M'.format(tot / 1e6)
        else:
            return '{:.1f}K'.format(tot / 1e3)
    else:
        return tot


def make_optimizer(param_list, optimizer_spec, load_sd=False):
    Optimizer = {
        'sgd': SGD,
        'adam': Adam,
        'adamw': AdamW,
        'adamw8bit': AdamW8bit

    }[optimizer_spec['name']]
    optimizer = Optimizer(param_list, **optimizer_spec['args'])
    if load_sd:
        optimizer.load_state_dict(optimizer_spec['sd'])
    return optimizer


def make_coord(shape, ranges=None, flatten=True):
    """ Make coordinates at grid centers.
    """
    coord_seqs = []
    for i, n in enumerate(shape):
        if ranges is None:
            v0, v1 = -1, 1
        else:
            v0, v1 = ranges[i]
        r = (v1 - v0) / (2 * n)
        seq = v0 + r + (2 * r) * torch.arange(n).float()
        coord_seqs.append(seq)
    ret = torch.stack(torch.meshgrid(*coord_seqs, indexing='ij'), dim=-1)
    if flatten:
        ret = ret.view(-1, ret.shape[-1])
    ret = ret.flip(-1)
    return ret

def make_coord_sphere(shape, ranges=None, flatten=True):
    """ Make coordinates at grid centers.
    """
    coord_seqs = []
    for i, n in enumerate(shape):
        if ranges is None:
            v0, v1 = -1, 1
        else:
            v0, v1 = ranges[i]
        r = (v1 - v0) / (2 * n)
        seq = v0 + r + (2 * r) * torch.arange(n).float()
        coord_seqs.append(seq)
    ret = torch.stack(torch.meshgrid(*coord_seqs), dim=-1)
    if flatten:
        ret = ret.view(-1, ret.shape[-1])
    ret = ret.flip(-1)
    # print(ret.shape)
    # print(ret[0,0,::], ret[0,-1,::], ret[-1,0,::], ret[-1,-1,::])
    return ret

def make_sphere_coord(shape, ranges=None, flatten=True):
    """ Make coordinates at grid centers.
    """
    coord_seqs = []
    for i, n in enumerate(shape):
        if ranges is None:
            v0, v1 = -1, 1
        else:
            v0, v1 = ranges[i]
        r = (v1 - v0) / (2 * n)
        seq = v0 + r + (2 * r) * torch.arange(n).float()
        coord_seqs.append(seq)
    ret = torch.stack(torch.meshgrid(*coord_seqs), dim=-1)
    if flatten:
        ret = ret.view(-1, ret.shape[-1])
    return ret


def make_cell(coord, img):
    coord = coord.unsqueeze(0)
    coord_bot_left  = coord + torch.tensor([-1/img.shape[-2], -1/img.shape[-1]]).unsqueeze(0)
    coord_bot_right = coord + torch.tensor([-1/img.shape[-2],  1/img.shape[-1]]).unsqueeze(0)
    coord_top_left  = coord + torch.tensor([ 1/img.shape[-2], -1/img.shape[-1]]).unsqueeze(0)
    coord_top_right = coord + torch.tensor([ 1/img.shape[-2],  1/img.shape[-1]]).unsqueeze(0)

    coord_left  = coord + torch.tensor([-1/img.shape[-2], 0]).unsqueeze(0)
    coord_right = coord + torch.tensor([ 1/img.shape[-2], 0]).unsqueeze(0)
    coord_bot   = coord + torch.tensor([ 0, -1/img.shape[-1]]).unsqueeze(0)
    coord_top   = coord + torch.tensor([ 0,  1/img.shape[-1]]).unsqueeze(0)

    cell_side   = torch.cat((coord_left, coord_right, coord_bot, coord_top), dim=0)
    cell_corner = torch.cat((coord_bot_left, coord_bot_right, coord_top_left, coord_top_right), dim=0)

    cell = torch.cat((cell_corner, cell_side, coord), dim=0).permute(1,2,0,3)
    return cell

def make_cell_(coord, img):

    coord_bot_left  = coord + torch.tensor([-1/img.shape[-2], -1/img.shape[-1]]).unsqueeze(0)
    coord_bot_right = coord + torch.tensor([-1/img.shape[-2],  1/img.shape[-1]]).unsqueeze(0)
    coord_top_left  = coord + torch.tensor([ 1/img.shape[-2], -1/img.shape[-1]]).unsqueeze(0)
    coord_top_right = coord + torch.tensor([ 1/img.shape[-2],  1/img.shape[-1]]).unsqueeze(0)
    coord_left  = coord + torch.tensor([-1/img.shape[-2], 0]).unsqueeze(0)
    coord_right = coord + torch.tensor([ 1/img.shape[-2], 0]).unsqueeze(0)
    coord_bot   = coord + torch.tensor([ 0, -1/img.shape[-1]]).unsqueeze(0)
    coord_top   = coord + torch.tensor([ 0,  1/img.shape[-1]]).unsqueeze(0)

    cell_side   = torch.cat((coord_left, coord_right, coord_bot, coord_top), dim=0)
    cell_corner = torch.cat((coord_bot_left, coord_bot_right, coord_top_left, coord_top_right), dim=0)
    print(cell_side.shape, cell_corner.shape, coord.shape)
    cell = torch.cat((cell_corner, cell_side, coord), dim=0)
    print(cell.shape)
    return cell

def to_pixel_samples(img):
    """ Convert the image to coord-RGB pairs.
        img: Tensor, (3, H, W)
    """
    coord = make_coord(img.shape[-2:])
    rgb = img.view(3, -1).permute(1, 0)
    return coord, rgb
def to_pixel_samples_grid(img):
    """ Convert the image to coord-RGB pairs.
        img: Tensor, (3, H, W)
    """
    coord = make_coord(img.shape[-2:], flatten = False)
    rgb = img
    # rgb = img.view(3, -1).permute(1, 0)
    return coord, rgb
def ssim(img1, img2):
    C1 = (0.01 * 255)**2
    C2 = (0.03 * 255)**2

    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)
    kernel = cv2.getGaussianKernel(11, 1.5)
    window = np.outer(kernel, kernel.transpose())

    mu1 = cv2.filter2D(img1, -1, window)[5:-5, 5:-5]  # valid
    mu2 = cv2.filter2D(img2, -1, window)[5:-5, 5:-5]
    mu1_sq = mu1**2
    mu2_sq = mu2**2
    mu1_mu2 = mu1 * mu2
    sigma1_sq = cv2.filter2D(img1**2, -1, window)[5:-5, 5:-5] - mu1_sq
    sigma2_sq = cv2.filter2D(img2**2, -1, window)[5:-5, 5:-5] - mu2_sq
    sigma12 = cv2.filter2D(img1 * img2, -1, window)[5:-5, 5:-5] - mu1_mu2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) *
                                                            (sigma1_sq + sigma2_sq + C2))
    return ssim_map.mean()
def calculate_ssim(img1, img2):
    '''calculate SSIM
    the same outputs as MATLAB's
    img1, img2: [0, 255]
    '''
    if not img1.shape == img2.shape:
        raise ValueError('Input images must have the same dimensions.')
    if img1.ndim == 2:
        return ssim(img1, img2)
    elif img1.ndim == 3:
        if img1.shape[2] == 3:
            ssims = []
            for i in range(3):
                ssims.append(ssim(img1[:,:,i], img2[:,:,i]))
            return np.array(ssims).mean()
        elif img1.shape[2] == 1:
            return ssim(np.squeeze(img1), np.squeeze(img2))
    else:
        raise ValueError('Wrong input image dimensions.')

def calc_ssim(sr, hr, dataset=None, scale=1, rgb_range=1):
    sr = sr.cpu()
    hr = hr.cpu()
    if dataset is not None:
        if dataset == 'benchmark':
            shave = scale
            if diff.size(1) > 1:
                gray_coeffs = [65.738, 129.057, 25.064]
                #convert = diff.new_tensor(gray_coeffs).view(1, 3, 1, 1) / 256
                #diff = diff.mul(convert).sum(dim=1)
        elif dataset == 'div2k':
            shave = scale + 6
        else:
            raise NotImplementedError
        sr = sr[..., shave:-shave, shave:-shave]
        hr = hr[..., shave:-shave, shave:-shave]
    sr = 255*sr.squeeze().permute(1,2,0).numpy()
    hr = 255*hr.squeeze().permute(1,2,0).numpy()
    return calculate_ssim(sr, hr)
def calc_lpips(sr, hr, lpips_model, dataset=None, scale=1):
    sr = sr.cpu()
    hr = hr.cpu()
    if dataset is not None:
        if dataset == 'benchmark':
            shave = scale
            if diff.size(1) > 1:
                gray_coeffs = [65.738, 129.057, 25.064]
                #convert = diff.new_tensor(gray_coeffs).view(1, 3, 1, 1) / 256
                #diff = diff.mul(convert).sum(dim=1)
        elif dataset == 'div2k':
            shave = scale + 6
        else:
            raise NotImplementedError
        sr = sr[..., shave:-shave, shave:-shave]
        hr = hr[..., shave:-shave, shave:-shave]
    similarity = lpips_model.forward(sr, hr)
    return similarity

def calc_psnr(sr, hr, dataset=None, scale=1, rgb_range=1):
    diff = (sr - hr) / rgb_range
    if dataset is not None:
        if dataset == 'benchmark':
            shave = scale
            if diff.size(1) > 1:
                gray_coeffs = [65.738, 129.057, 25.064]
                convert = diff.new_tensor(gray_coeffs).view(1, 3, 1, 1) / 256
                diff = diff.mul(convert).sum(dim=1)
        elif dataset == 'div2k':
            shave = scale + 6
        else:
            raise NotImplementedError
        if isinstance(scale, list):
            valid = diff[..., shave[0]:-shave[0], shave[1]:-shave[1]]
        else:
            valid = diff[..., shave:-shave, shave:-shave]
    else:
        valid = diff
    mse = valid.pow(2).mean()
    return -10 * torch.log10(mse)


def calc_mpsnr(sr, hr, mask, dataset=None, rgb_range=1):
    diff = mask * (sr - hr) / rgb_range
    if dataset is not None:
        if dataset == 'benchmark':
            if diff.size(1) > 1:
                gray_coeffs = [65.738, 129.057, 25.064]
                convert = diff.new_tensor(gray_coeffs).view(1, 3, 1, 1) / 256
                diff = diff.mul(convert).sum(dim=1)
        elif dataset == 'div2k':
            pass
        else:
            raise NotImplementedError
        valid = diff
    else:
        valid = diff
    mask_factor = sr.shape[-2]*sr.shape[-1]/torch.sum(mask)
    mse = valid.pow(2).mean()*mask_factor
    return -10 * torch.log10(mse)

def gridy2gridx(gridy, H, W, h, w, cpu=True):
    # scaling
    gridy += 1
    gridy[:, 0] *= H / 2
    gridy[:, 1] *= W / 2
    gridy -= 0.5
    gridy = gridy.flip(-1)

    # rescaling
    gridx = gridy.flip(-1)
    gridx += 0.5
    gridx[:, 0] /= h / 2
    gridx[:, 1] /= w / 2
    gridx -= 1
    gridx = gridx.float()

    mask = torch.where(torch.abs(gridx) > 1, 0, 1)
    mask = mask[:, 0] * mask[:, 1]
    mask = mask.float()

    return gridx, mask
def gridy2gridx_homography(gridy, H, W, h, w, m, cpu=True):
    # scaling
    gridy += 1
    gridy[:, 0] *= H / 2
    gridy[:, 1] *= W / 2
    gridy -= 0.5
    gridy = gridy.flip(-1)
    
    # coord -> homogeneous coord
    if cpu:
        gridy = torch.cat((gridy, torch.ones(gridy.shape[0], 1)), dim=-1).double()
    else:
        gridy = torch.cat((gridy, torch.ones(gridy.shape[0], 1).cuda()), dim=-1).double()
    
    # transform
    if cpu:
        m = transform.inverse_3x3(m)
    else:
        m = transform.inverse_3x3(m).cuda()
    gridx = torch.mm(m, gridy.permute(1, 0)).permute(1, 0)

    # homogeneous coord -> coord
    gridx[:, 0] /= gridx[:, -1]
    gridx[:, 1] /= gridx[:, -1]
    gridx = gridx[:, 0:2]

    # rescaling
    gridx = gridx.flip(-1)
    gridx += 0.5
    gridx[:, 0] /= h / 2
    gridx[:, 1] /= w / 2
    gridx -= 1
    gridx = gridx.float()

    mask = torch.where(torch.abs(gridx) > 1, 0, 1)
    mask = mask[:, 0] * mask[:, 1]
    mask = mask.float()

    return gridx, mask



def gridy2gridx_erp2fish(gridy, H, W, h, w, FOV, THETA, PHI):    
    # scaling    
    wFOV = FOV
    hFOV = float(H) / W * wFOV
    h_len = h*np.sin(np.radians(hFOV / 2.0))
    w_len = w*np.sin(np.radians(wFOV / 2.0))
    
    gridy = gridy.float()
    gridy[:, 0] *= h_len / h
    gridy[:, 1] *= w_len / w
    gridy = gridy.double()
    
    # H -> negative z-axis, W -> y-axis, place Warepd_plane on x-axis
    gridy = gridy.flip(-1)
    hr_norm = torch.norm(gridy, p=2, dim=-1, keepdim=True)
    
    mask = torch.where(hr_norm > 1, 0.0, 1.0)
    hr_norm = torch.where(hr_norm > 1, 1.0, hr_norm)
    hr_xaxis = torch.sqrt(1 - hr_norm**2)
    gridy = torch.cat((hr_xaxis, gridy), dim=-1)
    
    # set center position (theta, phi)
    y_axis = np.array([0.0, 1.0, 0.0], np.float64)
    z_axis = np.array([0.0, 0.0, 1.0], np.float64)
    [R1, _] = cv2.Rodrigues(z_axis * np.radians(THETA))
    [R2, _] = cv2.Rodrigues(np.dot(R1, y_axis) * np.radians(PHI))

    gridy = torch.mm(torch.from_numpy(R1), gridy.permute(1, 0)).permute(1, 0)
    gridy = torch.mm(torch.from_numpy(R2), gridy.permute(1, 0)).permute(1, 0)

    # find corresponding sphere coordinate 
    lat = torch.arcsin(gridy[:, 2].clamp_(-1+1e-6, 1-1e-6)) / np.pi * 2 # clamping to prevent arcsin explosion
    lon = torch.atan2(gridy[:, 1], gridy[:, 0]) / np.pi
    
    gridx = torch.stack((lat, lon), dim=-1)
    gridx = gridx.float()
    
    # mask
    mask = mask.squeeze(-1).float()
    
    return gridx, mask


# def gridy2gridx_pers2erp(gridy, H, W, h, w):    
#     # scaling    
#     gridy = gridy.double()
#     lat = gridy[:, 0] * np.pi / 4
#     lon = gridy[:, 1] * np.pi
    
#     z0 = torch.sin(lat)
#     x0 = torch.cos(lon) * torch.sqrt(1 - z0**2)
#     y0 = torch.sin(lon) * torch.sqrt(1 - z0**2)
    
#     y0 = y0 / x0
#     z0 = z0 / x0
        
#     gridx = torch.stack((z0, y0), dim=-1)
#     gridx = gridx.float()
    
#     # mask
#     mask1 = torch.where(x0 < 0, 0, 1) # filtering in backplane
#     mask2 = torch.where(torch.abs(gridx) > 1, 0, 1)
#     mask = mask1 * mask2[:, 0] * mask2[:, 1]
#     mask = mask.float()
    
#     return gridx, mask


def gridy2gridx_fish2erp(gridy, H, W, h, w):    
    # scaling    
    gridy = gridy.double()
    lat = gridy[:, 0] * np.pi / 2
    lon = gridy[:, 1] * np.pi
    
    z0 = torch.sin(lat)
    x0 = torch.cos(lon) * torch.sqrt(1 - z0**2)
    y0 = torch.sin(lon) * torch.sqrt(1 - z0**2)

    z0 = z0/x0
    y0 = y0/x0
        
    gridx = torch.stack((z0, y0), dim=-1)
    gridx = gridx.float()
    
    # mask
    mask = torch.where(x0 < 0, 0, 1) # filtering in backplane
    mask = mask.float()
    
    return gridx, mask


def gridy2gridx_fish2pers(gridy, H, W, h, w, FOV, THETA, PHI):
    # scaling    
    wFOV = FOV
    hFOV = float(H) / W * wFOV
    h_len = h*np.tan(np.radians(hFOV / 2.0))
    w_len = w*np.tan(np.radians(wFOV / 2.0))
    
    gridy = gridy.float()
    gridy[:, 0] *= h_len / h
    gridy[:, 1] *= w_len / w
    gridy = gridy.double()
    
    # H -> negative z-axis, W -> y-axis, place Warepd_plane on x-axis
    gridy = gridy.flip(-1)
    gridy = torch.cat((torch.ones(gridy.shape[0], 1), gridy), dim=-1)
    
    # project warped planed onto sphere
    hr_norm = torch.norm(gridy, p=2, dim=-1, keepdim=True)
    gridy /= hr_norm
    
    # set center position (theta, phi)
    y_axis = np.array([0.0, 1.0, 0.0], np.float64)
    z_axis = np.array([0.0, 0.0, 1.0], np.float64)
    [R1, _] = cv2.Rodrigues(z_axis * np.radians(THETA))
    [R2, _] = cv2.Rodrigues(np.dot(R1, y_axis) * np.radians(PHI))
    
    gridy = torch.mm(torch.from_numpy(R1), gridy.permute(1, 0)).permute(1, 0)
    gridy = torch.mm(torch.from_numpy(R2), gridy.permute(1, 0)).permute(1, 0)

    # find corresponding sphere coordinate 
    lat = torch.arcsin(gridy[:, 2])
    lon = torch.atan2(gridy[:, 1] , gridy[:, 0])
    
    z0 = torch.sin(lat)
    x0 = torch.cos(lon) * torch.sqrt(1 - z0**2)
    y0 = torch.sin(lon) * torch.sqrt(1 - z0**2)
        
    gridx = torch.stack((z0, y0), dim=-1)
    gridx = gridx.float()
    
    # mask
    mask = torch.where(x0 < 0, 0, 1) # filtering in backplane
    mask = mask.float()
    
    return gridx, mask


def celly2cellx_homography(celly, H, W, h, w, m, cpu=True):
    cellx, _ = gridy2gridx_homography(celly, H, W, h, w, m, cpu) # backward mapping
    return shape_estimation(cellx)


def celly2cellx_erp2pers(celly, H, W, h, w, FOV, THETA, PHI):
    cellx, _ = gridy2gridx_erp2pers(celly, H, W, h, w, FOV, THETA, PHI) # backward mapping
    return shape_estimation(cellx)


def celly2cellx_erp2fish(celly, H, W, h, w, FOV, THETA, PHI):
    cellx, _ = gridy2gridx_erp2fish(celly, H, W, h, w, FOV, THETA, PHI) # backward mapping
    return shape_estimation(cellx)


def celly2cellx_pers2erp(celly, H, W, h, w):
    cellx, _ = gridy2gridx_pers2erp(celly, H, W, h, w) # backward mapping
    return shape_estimation(cellx)


def celly2cellx_fish2erp(celly, H, W, h, w):
    cellx, _ = gridy2gridx_fish2erp(celly, H, W, h, w) # backward mapping
    return shape_estimation(cellx)


def celly2cellx_fish2pers(celly, H, W, h, w, FOV, THETA, PHI):
    cellx, _ = gridy2gridx_fish2pers(celly, H, W, h, w, FOV, THETA, PHI) # backward mapping
    return shape_estimation(cellx)

def celly2cellx(celly, H, W, h, w):
    cellx, _ = gridy2gridx(celly, H, W, h, w) # backward mapping
    return shape_estimation(cellx)

def shape_estimation(cell):
    cell_1 = cell[7*cell.shape[0]//9:8*cell.shape[0]//9, :] \
                - cell[6*cell.shape[0]//9:7*cell.shape[0]//9, :]
    cell_2 = cell[5*cell.shape[0]//9:6*cell.shape[0]//9, :] \
                - cell[4*cell.shape[0]//9:5*cell.shape[0]//9, :] # Jacobian
    cell_3 = cell[7*cell.shape[0]//9:8*cell.shape[0]//9, :] \
              - 2*cell[8*cell.shape[0]//9:9*cell.shape[0]//9, :] \
                + cell[6*cell.shape[0]//9:7*cell.shape[0]//9, :]
    cell_4 = cell[5*cell.shape[0]//9:6*cell.shape[0]//9, :] \
              - 2*cell[8*cell.shape[0]//9:9*cell.shape[0]//9, :] \
                + cell[4*cell.shape[0]//9:5*cell.shape[0]//9, :] # Second-order derivatives in Hessian
    cell_5 = cell[3*cell.shape[0]//9:4*cell.shape[0]//9, :] \
                - cell[0*cell.shape[0]//9:1*cell.shape[0]//9, :] \
                - cell[1*cell.shape[0]//9:2*cell.shape[0]//9, :] \
                + cell[0*cell.shape[0]//9:1*cell.shape[0]//9, :] \
                - cell[2*cell.shape[0]//9:3*cell.shape[0]//9, :] \
                + cell[0*cell.shape[0]//9:1*cell.shape[0]//9, :] # Cross-term in Hessian
    shape = torch.cat((cell_1, cell_2, 4*cell_3, 4*cell_4, cell_5), dim=-1)
    return shape


def first_x_translation(a, b, e, f, g, x, y):
    num = a*f*y + a - e*(b*y + g)
    den = torch.square(f*y + e*x + 1)
    return num/den


def first_y_translation(a, b, e, f, g, x, y):
    num = -f*(a*x + g) + b + e*b*x
    den = torch.square(f*y + e*x + 1)
    return num/den


def second_x_translation(a, b, e, f, g, x, y):
    num = -2*e*(a*f*y + a - e*(b*y + g))
    den = torch.pow(f*y + e*x + 1, 3)
    return num/den


def second_y_translation(a, b, e, f, g, x, y):
    num = -2*f*(-f*(a*x + g) + e*b*x + b)
    den = torch.pow(f*y + e*x + 1, 3)
    return num/den


def second_xy_translation(a, b, e, f, g, x, y):
    num = -f*(a*f*y - e*a*x + a - 2*e*g) - e*b*(-f*y + e*x + 1)
    den = torch.pow(f*y + e*x + 1, 3)
    return num/den


def JacobTInv(transf, x, y, h, w):
    a = transf[0, 0]
    b = transf[0, 1]
    c = transf[1, 0]
    d = transf[1, 1]
    e = transf[2, 0]
    f = transf[2, 1]
    g = transf[0, 2]
    h = transf[1, 2]

    JacobT = torch.zeros(2,2)
    JacobT[0, 0] = first_x_translation(a, b, e, f, g, 2*y/w + 1/w - 1, 2*x/h + 1/h - 1)
    JacobT[1, 0] = first_x_translation(c, d, e, f, h, 2*y/w + 1/w - 1, 2*x/h + 1/h - 1)
    JacobT[0, 1] = first_y_translation(a, b, e, f, g, 2*y/w + 1/w - 1, 2*x/h + 1/h - 1)
    JacobT[1, 1] = first_y_translation(c, d, e, f, h, 2*y/w + 1/w - 1, 2*x/h + 1/h - 1)
    JacobTInv = torch.linalg.inv(JacobT)
    return JacobTInv


def quantize(x: torch.Tensor) -> torch.Tensor:
    x = 127.5 * (x + 1)
    x = x.clamp(min=0, max=255)
    x = x.round()
    x = x / 127.5 - 1
    return x

def get_mask_boundaries_np(mask):
    mask = mask[0,0]
    nonzero_indices = torch.nonzero(mask, as_tuple=False)
    if nonzero_indices.size(0) == 0:
        return {"top": None, "bottom": None, "left": None, "right": None}

    # 获取边界索引
    top = torch.min(nonzero_indices[:, 0]).item()      # 最小行索引
    bottom = torch.max(nonzero_indices[:, 0]).item()   # 最大行索引
    left = torch.min(nonzero_indices[:, 1]).item()     # 最小列索引
    right = torch.max(nonzero_indices[:, 1]).item()    # 最大列索引
    return top, bottom, left, right

def save_image(tensor_image, output_dir):
    to_pil = ToPILImage()
    pil_image = to_pil(tensor_image)
    output_path = os.path.join(output_dir)  # 保存路径
    pil_image.save(output_path)





########LTEW#############

def gridy2gridx_homography(gridy, H, W, h, w, m, cpu=True):
    # scaling
    gridy += 1
    gridy[:, 0] *= H / 2
    gridy[:, 1] *= W / 2
    gridy -= 0.5
    gridy = gridy.flip(-1)
    
    # coord -> homogeneous coord
    if cpu:
        gridy = torch.cat((gridy, torch.ones(gridy.shape[0], 1)), dim=-1).double()
    else:
        gridy = torch.cat((gridy, torch.ones(gridy.shape[0], 1).cuda()), dim=-1).double()
    
    # transform
    if cpu:
        m = transform.inverse_3x3(m)
    else:
        m = transform.inverse_3x3(m).cuda()
    gridx = torch.mm(m, gridy.permute(1, 0)).permute(1, 0)

    # homogeneous coord -> coord
    gridx[:, 0] /= gridx[:, -1]
    gridx[:, 1] /= gridx[:, -1]
    gridx = gridx[:, 0:2]

    # rescaling
    gridx = gridx.flip(-1)
    gridx += 0.5
    gridx[:, 0] /= h / 2
    gridx[:, 1] /= w / 2
    gridx -= 1
    gridx = gridx.float()

    mask = torch.where(torch.abs(gridx) > 1, 0, 1)
    mask = mask[:, 0] * mask[:, 1]
    mask = mask.float()
    
    return gridx, mask


def gridy2gridx_erp2pers(gridy, H, W, h, w, FOV, THETA, PHI):   
    # scaling    
    wFOV = FOV
    hFOV = float(h) / w * wFOV
    # print(hFOV, wFOV)
    # hFOV = FOV
    # wFOV = float(H) / W * hFOV
    h_len = h*np.tan(np.radians(hFOV / 2.0))
    w_len = w*np.tan(np.radians(wFOV / 2.0))
    
    gridy = gridy.float()
    gridy[:, 0] *= h_len / h
    gridy[:, 1] *= w_len / w
    gridy = gridy.double()
    # gridy = gridy.double()
    
    # H -> negative z-axis, W -> y-axis, place Warepd_plane on x-axis
    gridy = gridy.flip(-1)
    gridy = torch.cat((torch.ones(gridy.shape[0], 1), gridy), dim=-1)
    
    # project warped planed onto sphere
    hr_norm = torch.norm(gridy, p=2, dim=-1, keepdim=True)
    gridy /= hr_norm
    
    # set center position (theta, phi)
    y_axis = np.array([0.0, 1.0, 0.0], np.float64)
    z_axis = np.array([0.0, 0.0, 1.0], np.float64)
    [R1, _] = cv2.Rodrigues(z_axis * np.radians(THETA))
    [R2, _] = cv2.Rodrigues(np.dot(R1, y_axis) * np.radians(PHI))
    
    gridy = torch.mm(torch.from_numpy(R1), gridy.permute(1, 0)).permute(1, 0)
    gridy = torch.mm(torch.from_numpy(R2), gridy.permute(1, 0)).permute(1, 0)

    # find corresponding sphere coordinate 
    lat = torch.arcsin(gridy[:, 2]) / np.pi * 2
    lon = torch.atan2(gridy[:, 1] , gridy[:, 0]) / np.pi
        
    gridx = torch.stack((lat, lon), dim=-1)
    gridx = gridx.float()
    
    # mask
    mask = torch.where(torch.abs(gridx) > 1, 0, 1)
    mask = mask[:, 0] * mask[:, 1]
    mask = mask.float()
    
    return gridx, mask


def gridy2gridx_erp2erp(gridy, H, W, h, w, THETA, PHI):   
 
    wFOV = 360
    hFOV = float(h) / w * wFOV

    h_len = h*np.tan(np.radians(hFOV / 2.0))
    w_len = w*np.tan(np.radians(wFOV / 2.0))
    
    gridy = gridy.float()
    gridy[:, 0] *= h_len / h
    gridy[:, 1] *= w_len / w
    gridy = gridy.double()
    # gridy = gridy.double()
    
    # H -> negative z-axis, W -> y-axis, place Warepd_plane on x-axis
    gridy = gridy.flip(-1)
    gridy = torch.cat((torch.ones(gridy.shape[0], 1), gridy), dim=-1)
    
    # project warped planed onto sphere
    hr_norm = torch.norm(gridy, p=2, dim=-1, keepdim=True)
    gridy /= hr_norm
    
    # set center position (theta, phi)
    y_axis = np.array([0.0, 1.0, 0.0], np.float64)
    z_axis = np.array([0.0, 0.0, 1.0], np.float64)
    [R1, _] = cv2.Rodrigues(z_axis * np.radians(THETA))
    [R2, _] = cv2.Rodrigues(np.dot(R1, y_axis) * np.radians(PHI))
    
    gridy = torch.mm(torch.from_numpy(R1), gridy.permute(1, 0)).permute(1, 0)
    gridy = torch.mm(torch.from_numpy(R2), gridy.permute(1, 0)).permute(1, 0)

    # find corresponding sphere coordinate 
    # lat = torch.arcsin(gridy[:, 2]) / np.pi * 2
    # lon = torch.atan2(gridy[:, 1] , gridy[:, 0]) / np.pi
        
    # gridx = torch.stack((lat, lon), dim=-1)
    # gridx = gridx.float()
    
    # # mask
    # mask = torch.where(torch.abs(gridx) > 1, 0, 1)
    # mask = mask[:, 0] * mask[:, 1]
    # mask = mask.float()
    
    return gridy

def gridy2gridx_erp2pers_(gridy, H, W, h, w, FOV, THETA, PHI):    
    # scaling    
    wFOV = FOV
    hFOV = float(h) / w * wFOV
    # print(hFOV)
    h_len = (h)*np.tan(np.radians(hFOV / 2.0))
    w_len = (w)*np.tan(np.radians(wFOV / 2.0))
    
    gridy = gridy.float()
    gridy[:, 0] *= h_len / h
    gridy[:, 1] *= w_len / w
    gridy = gridy.double()
    
    # H -> negative z-axis, W -> y-axis, place Warepd_plane on x-axis
    gridy = gridy.flip(-1)
    gridy = torch.cat((torch.ones(gridy.shape[0], 1), gridy), dim=-1)
    
    # project warped planed onto sphere
    hr_norm = torch.norm(gridy, p=2, dim=-1, keepdim=True)
    gridy /= hr_norm
    
    # set center position (theta, phi)
    y_axis = np.array([0.0, 1.0, 0.0], np.float64)
    z_axis = np.array([0.0, 0.0, 1.0], np.float64)
    [R1, _] = cv2.Rodrigues(z_axis * np.radians(THETA))
    [R2, _] = cv2.Rodrigues(np.dot(R1, y_axis) * np.radians(PHI))
    
    gridy = torch.mm(torch.from_numpy(R1), gridy.permute(1, 0)).permute(1, 0)
    gridy = torch.mm(torch.from_numpy(R2), gridy.permute(1, 0)).permute(1, 0)

    # find corresponding sphere coordinate 
    lat = torch.arcsin(gridy[:, 2])  #/ np.pi * 2
    lon = torch.atan2(gridy[:, 1] , gridy[:, 0]) #/ np.pi
        
    gridx = torch.stack((lat, lon), dim=-1)
    gridx = gridx.float()
    
    # mask
    mask = torch.where(torch.abs(gridx) > 1, 0, 1)
    mask = mask[:, 0] * mask[:, 1]
    mask = mask.float()
    
    return gridx, mask

# def gridy2gridx_erp2erp_(gridy, H, W, h, w, FOV, THETA, PHI):    
#     # scaling    
#     wFOV = FOV
#     hFOV = float(h) / w * wFOV
#     # print(hFOV)
#     h_len = (h)*np.tan(np.radians(hFOV / 2.0))
#     w_len = (w)*np.tan(np.radians(wFOV / 2.0))
    
#     gridy = gridy.float()
#     gridy[:, 0] *= h_len / h
#     gridy[:, 1] *= w_len / w
#     gridy = gridy.double()
    
#     # H -> negative z-axis, W -> y-axis, place Warepd_plane on x-axis
#     gridy = gridy.flip(-1)
#     gridy = torch.cat((torch.ones(gridy.shape[0], 1), gridy), dim=-1)
    
#     # project warped planed onto sphere
#     hr_norm = torch.norm(gridy, p=2, dim=-1, keepdim=True)
#     gridy /= hr_norm
    
#     # set center position (theta, phi)
#     y_axis = np.array([0.0, 1.0, 0.0], np.float64)
#     z_axis = np.array([0.0, 0.0, 1.0], np.float64)
#     [R1, _] = cv2.Rodrigues(z_axis * np.radians(THETA))
#     [R2, _] = cv2.Rodrigues(np.dot(R1, y_axis) * np.radians(PHI))
    
#     gridy = torch.mm(torch.from_numpy(R1), gridy.permute(1, 0)).permute(1, 0)
#     gridy = torch.mm(torch.from_numpy(R2), gridy.permute(1, 0)).permute(1, 0)

#     # find corresponding sphere coordinate 
#     lat = torch.arcsin(gridy[:, 2])  #/ np.pi * 2
#     lon = torch.atan2(gridy[:, 1] , gridy[:, 0]) #/ np.pi
        
#     gridx = torch.stack((lat, lon), dim=-1)
#     gridx = gridx.float()
    
#     # mask
#     mask = torch.where(torch.abs(gridx) > 1, 0, 1)
#     mask = mask[:, 0] * mask[:, 1]
#     mask = mask.float()
    
#     return gridx, mask
    
def genuv(h, w, v_rot=0):
    assert -np.pi / 2 <= v_rot and v_rot <= np.pi / 2
    u, v = np.meshgrid(np.arange(w), np.arange(h))
    u = (u + 0.5) * 2 * np.pi / w - np.pi
    v = (v + 0.5) * np.pi / h - np.pi / 2
    uv = np.stack([u, v], axis=-1)

    if v_rot != 0:
        # rotation
        xyz = uv2xyz(uv.astype(np.float64))
        # Rx = np.array([
        #     [1, 0, 0],
        #     [0, np.cos(v_rot), np.sin(v_rot)],
        #     [0, -np.sin(v_rot), np.cos(v_rot)],
        # ])
        xyz_rot = xyz.copy()
        xyz_rot[..., 0] = xyz[..., 0]
        xyz_rot[..., 1] = np.cos(v_rot) * xyz[..., 1] + np.sin(v_rot) * xyz[..., 2]
        xyz_rot[..., 2] = -np.sin(v_rot) * xyz[..., 1] + np.cos(v_rot) * xyz[..., 2]
        uv = xyz2uv(xyz_rot)

    return uv

# def xyz2uv(xyz):
#     x, y, z = xyz[..., 0], xyz[..., 1], xyz[..., 2]
#     u = np.arctan2(x, z)
#     c = np.sqrt(x * x + z * z)
#     v = np.arctan2(y, c)
#     return np.stack([u, v], axis=-1)

def uv2xyz(uv):
    sin_u = np.sin(uv[..., 0])
    cos_u = np.cos(uv[..., 0])
    sin_v = np.sin(uv[..., 1])
    cos_v = np.cos(uv[..., 1])
    return np.stack([
        cos_v * sin_u,
        sin_v,
        cos_v * cos_u,
    ], axis=-1)

# def gridy2gridx_pers2erp(gridy, H, W, h, w, FOV, THETA, PHI):    
def gridy2gridx_pers2erp(gridy, H, W, h, w):
    # scaling    
    # h_fov = FOV
    # w_fov = float(H) / W * h_fov
    # H, W = 10,  10
    gridy = gridy.double()
    lat = gridy[:, 0] * np.pi / 2
    lon = gridy[:, 1] * np.pi

    # lat_min = PHI - h_fov / 2
    # lat_max = PHI + h_fov / 2
    # lon_min = THETA - w_fov / 2
    # lon_max = THETA + w_fov / 2

    z0 = torch.sin(lat)
    x0 = torch.cos(lon) * torch.cos(lat) #torch.cos(lat) ##torch.sqrt(1 - z0**2)
    y0 = torch.sin(lon) * torch.cos(lat) #torch.cos(lat) #torch.sqrt(1 - z0**2)
    # print("x0_1", x0)
    # print("y0_1", y0)
    # print("z0_1", z0)
    
    y0 = y0 / x0
    z0 = z0 / x0

    # print("x0_2", x0)
    # print("y0_2", y0)
    # print("z0_2", z0)
    # sphere_points = torch.stack((x, y, z), dim=-1)  # (H, W, 3)

    # # 计算逆旋转矩阵
    # def rotation_matrix(theta, phi):
    #     # 绕 y 轴旋转 (theta)
    #     R1 = torch.tensor([
    #         [np.cos(-theta), 0, np.sin(-theta)],
    #         [0, 1, 0],
    #         [-np.sin(-theta), 0, np.cos(-theta)]
    #     ])
    #     # 绕 x 轴旋转 (phi)
    #     R2 = torch.tensor([
    #         [1, 0, 0],
    #         [0, np.cos(-phi), -np.sin(-phi)],
    #         [0, np.sin(-phi), np.cos(-phi)]
    #     ])
    #     return R2 @ R1

    # theta = np.radians(THETA)
    # phi = np.radians(PHI)
    # R_inv = rotation_matrix(theta, phi)
        
    gridx = torch.stack((z0, y0), dim=-1)
    gridx = gridx.float()

    t = (gridx[:, 0] > -1) & (gridx[:, 0] < 1) & (gridx[:, 1] > -1) & (gridx[:, 1] < 1)

    
    # mask

    mask1 = torch.where(x0 < 0, 0, 1) # filtering in backplane
    mask2 = torch.where(torch.abs(gridx) > 1, 0, 1)
    mask = mask1 * mask2[:, 0] * mask2[:, 1]
    mask = mask.float()
    gridx = torch.where(torch.isnan(gridx) | torch.isinf(gridx), torch.tensor(-2.0), gridx)
    gridx = torch.clamp(gridx, min=-1, max=1)
    return gridx, mask
    
def gridy2gridx_fish2erp_(gridy, H, W, h, w):    
    # scaling    
    gridy = gridy.double()
    lat = gridy[:, 0] * np.pi / 2
    lon = gridy[:, 1] * np.pi
    
    z0 = torch.sin(lat)
    x0 = torch.cos(lon) * torch.sqrt(1 - z0**2)
    y0 = torch.sin(lon) * torch.sqrt(1 - z0**2)
        
    gridx = torch.stack((z0, y0), dim=-1)
    gridx = gridx.float()
    
    # mask
    mask = torch.where(x0 < 0, 0, 1) # filtering in backplane
    mask = mask.float()
    
    return gridx, mask


def gridy2gridx_fish2pers(gridy, H, W, h, w, FOV, THETA, PHI):
    # scaling    
    wFOV = FOV
    hFOV = float(H) / W * wFOV
    h_len = h*np.tan(np.radians(hFOV / 2.0))
    w_len = w*np.tan(np.radians(wFOV / 2.0))
    
    gridy = gridy.float()
    gridy[:, 0] *= h_len / h
    gridy[:, 1] *= w_len / w
    gridy = gridy.double()
    
    # H -> negative z-axis, W -> y-axis, place Warepd_plane on x-axis
    gridy = gridy.flip(-1)
    gridy = torch.cat((torch.ones(gridy.shape[0], 1), gridy), dim=-1)
    
    # project warped planed onto sphere
    hr_norm = torch.norm(gridy, p=2, dim=-1, keepdim=True)
    gridy /= hr_norm
    
    # set center position (theta, phi)
    y_axis = np.array([0.0, 1.0, 0.0], np.float64)
    z_axis = np.array([0.0, 0.0, 1.0], np.float64)
    [R1, _] = cv2.Rodrigues(z_axis * np.radians(THETA))
    [R2, _] = cv2.Rodrigues(np.dot(R1, y_axis) * np.radians(PHI))
    
    gridy = torch.mm(torch.from_numpy(R1), gridy.permute(1, 0)).permute(1, 0)
    gridy = torch.mm(torch.from_numpy(R2), gridy.permute(1, 0)).permute(1, 0)

    # find corresponding sphere coordinate 
    lat = torch.arcsin(gridy[:, 2])
    lon = torch.atan2(gridy[:, 1] , gridy[:, 0])
    
    z0 = torch.sin(lat)
    x0 = torch.cos(lon) * torch.sqrt(1 - z0**2)
    y0 = torch.sin(lon) * torch.sqrt(1 - z0**2)
        
    gridx = torch.stack((z0, y0), dim=-1)
    gridx = gridx.float()
    
    # mask
    mask = torch.where(x0 < 0, 0, 1) # filtering in backplane
    mask = mask.float()
    
    return gridx, mask


def celly2cellx_homography(celly, H, W, h, w, m, cpu=True):
    cellx, _ = gridy2gridx_homography(celly, H, W, h, w, m, cpu) # backward mapping
    return shape_estimation(cellx)


def celly2cellx_erp2pers(celly, H, W, h, w, FOV, THETA, PHI):
    cellx, _ = gridy2gridx_erp2pers(celly, H, W, h, w, FOV, THETA, PHI) # backward mapping
    return shape_estimation(cellx)


def celly2cellx_erp2fish(celly, H, W, h, w, FOV, THETA, PHI):
    cellx, _ = gridy2gridx_erp2fish(celly, H, W, h, w, FOV, THETA, PHI) # backward mapping
    return shape_estimation(cellx)


def celly2cellx_pers2erp(celly, H, W, h, w):
    cellx, _ = gridy2gridx_pers2erp(celly, H, W, h, w) # backward mapping
    return shape_estimation(cellx)


def celly2cellx_fish2erp(celly, H, W, h, w):
    cellx, _ = gridy2gridx_fish2erp(celly, H, W, h, w) # backward mapping
    return shape_estimation(cellx)


def celly2cellx_fish2pers(celly, H, W, h, w, FOV, THETA, PHI):
    cellx, _ = gridy2gridx_fish2pers(celly, H, W, h, w, FOV, THETA, PHI) # backward mapping
    return shape_estimation(cellx)


def shape_estimation(cell):
    cell_1 = cell[7*cell.shape[0]//9:8*cell.shape[0]//9, :] \
                - cell[6*cell.shape[0]//9:7*cell.shape[0]//9, :]
    cell_2 = cell[5*cell.shape[0]//9:6*cell.shape[0]//9, :] \
                - cell[4*cell.shape[0]//9:5*cell.shape[0]//9, :] # Jacobian
    cell_3 = cell[7*cell.shape[0]//9:8*cell.shape[0]//9, :] \
              - 2*cell[8*cell.shape[0]//9:9*cell.shape[0]//9, :] \
                + cell[6*cell.shape[0]//9:7*cell.shape[0]//9, :]
    cell_4 = cell[5*cell.shape[0]//9:6*cell.shape[0]//9, :] \
              - 2*cell[8*cell.shape[0]//9:9*cell.shape[0]//9, :] \
                + cell[4*cell.shape[0]//9:5*cell.shape[0]//9, :] # Second-order derivatives in Hessian
    cell_5 = cell[3*cell.shape[0]//9:4*cell.shape[0]//9, :] \
                - cell[0*cell.shape[0]//9:1*cell.shape[0]//9, :] \
                - cell[1*cell.shape[0]//9:2*cell.shape[0]//9, :] \
                + cell[0*cell.shape[0]//9:1*cell.shape[0]//9, :] \
                - cell[2*cell.shape[0]//9:3*cell.shape[0]//9, :] \
                + cell[0*cell.shape[0]//9:1*cell.shape[0]//9, :] # Cross-term in Hessian
    shape = torch.cat((cell_1, cell_2, 4*cell_3, 4*cell_4, cell_5), dim=-1)
    return shape


def first_x_translation(a, b, e, f, g, x, y):
    num = a*f*y + a - e*(b*y + g)
    den = torch.square(f*y + e*x + 1)
    return num/den


def first_y_translation(a, b, e, f, g, x, y):
    num = -f*(a*x + g) + b + e*b*x
    den = torch.square(f*y + e*x + 1)
    return num/den


def second_x_translation(a, b, e, f, g, x, y):
    num = -2*e*(a*f*y + a - e*(b*y + g))
    den = torch.pow(f*y + e*x + 1, 3)
    return num/den


def second_y_translation(a, b, e, f, g, x, y):
    num = -2*f*(-f*(a*x + g) + e*b*x + b)
    den = torch.pow(f*y + e*x + 1, 3)
    return num/den


def second_xy_translation(a, b, e, f, g, x, y):
    num = -f*(a*f*y - e*a*x + a - 2*e*g) - e*b*(-f*y + e*x + 1)
    den = torch.pow(f*y + e*x + 1, 3)
    return num/den


def JacobTInv(transf, x, y, h, w):
    a = transf[0, 0]
    b = transf[0, 1]
    c = transf[1, 0]
    d = transf[1, 1]
    e = transf[2, 0]
    f = transf[2, 1]
    g = transf[0, 2]
    h = transf[1, 2]

    JacobT = torch.zeros(2,2)
    JacobT[0, 0] = first_x_translation(a, b, e, f, g, 2*y/w + 1/w - 1, 2*x/h + 1/h - 1)
    JacobT[1, 0] = first_x_translation(c, d, e, f, h, 2*y/w + 1/w - 1, 2*x/h + 1/h - 1)
    JacobT[0, 1] = first_y_translation(a, b, e, f, g, 2*y/w + 1/w - 1, 2*x/h + 1/h - 1)
    JacobT[1, 1] = first_y_translation(c, d, e, f, h, 2*y/w + 1/w - 1, 2*x/h + 1/h - 1)
    JacobTInv = torch.linalg.inv(JacobT)
    return JacobTInv


def quantize(x: torch.Tensor) -> torch.Tensor:
    x = 127.5 * (x + 1)
    x = x.clamp(min=0, max=255)
    x = x.round()
    x = x / 127.5 - 1
    return x










#!!! # make arg_map level (n) on gpu ####

def distance(a1, a2, q=1):
    # print("a1, a2", a1.shape, a2.shape)
    # theta1 = torch.pi/2 - a1[...,1]
    # theta2 = torch.pi/2 - a2[...,1]
    theta1 = torch.pi/2 - a1[...,1]
    theta2 = torch.pi/2 - a2[...,1]
    phi1 = a1[...,0]
    phi2 = a2[...,0]
    dist = torch.sqrt(torch.Tensor([2]).cuda()) * torch.sqrt(1-(torch.cos(theta1-theta2)+torch.sin(theta1)*torch.sin(theta2)*(torch.cos(phi1-phi2)-1)))
    # dist = 1-(torch.cos(theta1-theta2)+torch.sin(theta1)*torch.sin(theta2)*(torch.cos(phi1-phi2)-1))
    # print(dist.shape)                             
    # d = (torch.cos(theta1-theta2)+torch.sin(theta1)*torch.sin(theta2)*(torch.cos(phi1-phi2)-1))
    # d.clamp_(max = 1)
    # dist = torch.sqrt(torch.Tensor([2]).cuda()) * torch.sqrt(1-d)
    # torch.set_printoptions(precision=15)  # 精度位数
    # print(dist)
    del theta1, theta2, phi1, phi2
    return dist

# def xyz2uv(xyz):
#     x, y, z = xyz[..., 0], xyz[..., 1], xyz[..., 2]
#     u = np.arctan2(x, z)
#     c = np.sqrt(x * x + z * z)
#     v = np.arctan2(y, c)
#     return np.stack([u, v], axis=-1)
def make_arg_map(face_vector, h, w, v_fov, h_fov):

    uv = xyz2uv(face_vector).reshape(-1,2)
    # print("uv_shape", uv.shape)
    # print("uv0", uv[:,0].max(), uv[:,0].min())
    # print("uv1", uv[:,1].max(), uv[:,1].min())

    p, q = uv.shape

    xy = np.zeros([h,w,2])
    pi = np.pi
    h_begin = 30*np.pi/180 #np.random.uniform(-np.pi/2, np.pi/2)  # 垂直方向 [-π/2, π/2] 
    v_begin = -60*np.pi/180  #np.random.uniform(-np.pi, np.pi)  # 水平方向 [-π, π]
    # print(h_begin, v_begin)
    shape = [h, w]
    coord_seqs = []
    for i, n in enumerate(shape):
        if i == 0:  
            v0, v1 = h_begin, h_begin + v_fov*torch.pi/180.
            r = (v1 - v0) / (2 * n)
            seq = v0 - r - (2 * r) * torch.arange(n).float()
        else:  
            v0, v1 = v_begin, v_begin + h_fov*torch.pi/180.
            r = (v1 - v0) / (2 * n)
            seq = v0 + r + (2 * r) * torch.arange(n).float()
        coord_seqs.append(seq)
    xy = torch.stack(torch.meshgrid(*coord_seqs), dim=-1)#[..., [1,0]]
    # print(xy[...,0].max(), xy[...,0].min()) 
    # print(xy[...,1].max(), xy[...,1].min())
    xy = torch.stack(torch.meshgrid(*coord_seqs), dim=-1)[..., [1,0]]
    # print("xy_", xy[...,0].max(), xy[...,0].min()) 
    # print("xy", xy[...,1].max(), xy[...,1].min())
    # for i in range(h):
    #     for j in range(w):
    #     #xy[i,j] = [ -pi + 2*pi/1024*j, pi/2 - pi/512*i]
    #         xy[i,j] = [-pi+pi*(2*j+1)/(1024+1),pi/2-pi/2*(2*i+1)/(512+1)]
    # arg_map = torch.zeros([512,1024]).cuda()
    arg_map = torch.zeros([h,w]).to(torch.int64).cuda()
    # arg_map_ = torch.zeros([h,w]).cuda()
    dist_vector = torch.zeros([h,w,2]).cuda()
    # dist_vector_ = torch.zeros([h,w,2]).cuda()
    xy = torch.Tensor(xy).cuda()
    uv = torch.Tensor(uv).cuda()

    uv[:,0] = (uv[:,0] - 0.0031)*0.9994
    uv[:,1] = (uv[:,1] + 0.0025)*0.9994
    # print("uv", uv.max(), uv.min(), uv.shape)
    # print("xy", xy.max(), xy.min(), xy.shape)

    # print("xy" ,xy.shape)
    # xy_h, xy_w, _= xy.shape
    # xy = xy.reshape(-1,1,2)
    # dist_list = distance(xy, uv)
    # print(dist_list.shape)
    # arg_map = dist_list.argmin()

    # for i in range(h):
    #     for j in range(w):
    #         dist_list = distance(xy[i,j].unsqueeze(0), uv)
    #         arg_map[i,j] = dist_list.argmin()
    #         dist_vector[i,j] = xy[i,j] - uv[int(arg_map[i,j])]
            
    xy_h, xy_w, _= xy.shape
    xy = xy.reshape(h,-1,1,2)
    # print("begin")
    for i in range(h):
        dist_list = distance(xy[i], uv)
        dist_i = dist_list.argmin(dim=1)
        dist_vector[i] = xy[i].squeeze() - uv[dist_i]
        arg_map[i] = dist_i
    
    tmp = {}
    tmp['arg_map'] = arg_map.cpu().numpy()
    tmp['dist_vector'] = dist_vector.cpu().numpy()
    savemat('/data2/xiajy/dataset/arg_map/'+str(h)+'_' + str(w)+'_' + str(v_fov)+'_' + str(h_fov) +'_arg_new.mat', tmp)

    savez_compressed('/data2/xiajy/dataset/arg_map/'+str(h)+'_' + str(w)+'_' + str(v_fov)+'_' + str(h_fov) +'_arg_new.npz', **tmp)
    # print(sss)
    return arg_map, dist_vector
        # if i%100==0:
        # print(i)
    # tmp = {}
    # tmp['arg_map'] = arg_map.cpu().numpy()
    # print(arg_map.shape, arg_map)
    # savemat('./level'+str(img_level)+'_arg.mat', tmp)

def make_arg_map_(face_vector, h, w, v_fov, h_fov, level=8):

    uv = xyz2uv(face_vector).reshape(-1,2)
    # print("uv_shape", uv.shape)
    # print("uv0", uv[:,0].max(), uv[:,0].min())
    # print("uv1", uv[:,1].max(), uv[:,1].min())

    p, q = uv.shape

    xy = np.zeros([h,w,2])
    pi = np.pi
    # h_begin = 30*np.pi/180 #np.random.uniform(-np.pi/2, np.pi/2)  # 垂直方向 [-π/2, π/2] 
    # v_begin = -60*np.pi/180  #np.random.uniform(-np.pi, np.pi)  # 水平方向 [-π, π]
    # print(h_begin, v_begin)
    shape = [h, w]
    coord_seqs = []
    gridy = make_coord((512,1024))
    xy, mask = gridy2gridx_erp2pers_(gridy.flip(-1),  1664, 3328, 512, 1024, 120, 0, 0)
    xy = xy.view(512,1024,2).flip(-1)
    # print("grid_pers", grid_pers.shape, grid_pers.max(), grid_pers.min())
    # print("xy_", xy[...,0].max(), xy[...,0].min()) 
    # print("xy", xy[...,1].max(), xy[...,1].min())
    # for i in range(h):
    #     for j in range(w):
    #     #xy[i,j] = [ -pi + 2*pi/1024*j, pi/2 - pi/512*i]
    #         xy[i,j] = [-pi+pi*(2*j+1)/(1024+1),pi/2-pi/2*(2*i+1)/(512+1)]
    # arg_map = torch.zeros([512,1024]).cuda()
    arg_map = torch.zeros([h,w]).to(torch.int64).cuda()
    # arg_map_ = torch.zeros([h,w]).cuda()
    dist_vector = torch.zeros([h,w,2]).cuda()
    # dist_vector_ = torch.zeros([h,w,2]).cuda()
    xy = torch.Tensor(xy).cuda()
    uv = torch.Tensor(uv).cuda()
    uv[:,0] = uv[:,0]*0.9994
    uv[:,1] = uv[:,1]*0.9997
    # uv[:,0] = (uv[:,0] - 0.0031)*0.9994
    # uv[:,1] = (uv[:,1] + 0.0025)*0.9994
    # print("uv", uv.max(), uv.min(), uv.shape)
    # print("xy", xy.max(), xy.min(), xy.shape)

    # print("xy" ,xy.shape)
    # xy_h, xy_w, _= xy.shape
    # xy = xy.reshape(-1,1,2)
    # dist_list = distance(xy, uv)
    # print(dist_list.shape)
    # arg_map = dist_list.argmin()

    # for i in range(h):
    #     for j in range(w):
    #         dist_list = distance(xy[i,j].unsqueeze(0), uv)
    #         arg_map[i,j] = dist_list.argmin()
    #         dist_vector[i,j] = xy[i,j] - uv[int(arg_map[i,j])]
            
    xy_h, xy_w, _= xy.shape
    xy = xy.reshape(h,-1,1,2)
    # print(xy.shape, h, w)
    # print("begin")
    for i in range(h):
        dist_list = distance(xy[i], uv)
        dist_i = dist_list.argmin(dim=1)
        dist_vector[i] = xy[i].squeeze() - uv[dist_i]
        arg_map[i] = dist_i
        print(i)
    
    tmp = {}
    tmp['arg_map'] = arg_map.cpu().numpy()
    tmp['dist_vector'] = dist_vector.cpu().numpy()
    savemat('/home/zwxionggroup/xurk/dataset/arg_map/img'+str(level)+'_'+str(h)+'_' + str(w)+'_' + str(v_fov)+'_' + str(h_fov) +'_arg_new_v2.mat', tmp)

    # savez_compressed('/data2/xiajy/dataset/arg_map/'+str(h)+'_' + str(w)+'_' + str(v_fov)+'_' + str(h_fov) +'_arg_new.npz', **tmp)
    # print(sss)
    return arg_map, dist_vector


def make_arg_map_dataset(face_vector, h, w, v_fov, h_fov, THETA, PHI, level=8, THETA_SAVE=0, PHI_SAVE=0, save_path=None):
    # name = '/home/zwxionggroup/xurk/dataset/arg_map_dataset_pano_level8/'+str(level)+'_'+str(h)+'_' + str(w)+'_' + str(v_fov)+'_' + str(h_fov)+ '_'+ str(int(THETA_SAVE))+'_'+ str(int(PHI_SAVE)) +'_arg_new_v2.mat'
    # if os.path.exists(name):
    #     # a = loadmat(name)
    #     # print(name)
    #     # print("yes")
    #     return None, None
    # print(name)


    uv = xyz2uv(face_vector).reshape(-1,2)
    # print("uv_shape", uv.shape)
    # print("uv0", uv[:,0].max(), uv[:,0].min())
    # print("uv1", uv[:,1].max(), uv[:,1].min())
 
    p, q = uv.shape

    xy = np.zeros([h,w,2])
    pi = np.pi
    # h_begin = 30*np.pi/180 #np.random.uniform(-np.pi/2, np.pi/2)  # 垂直方向 [-π/2, π/2] 
    # v_begin = -60*np.pi/180  #np.random.uniform(-np.pi, np.pi)  # 水平方向 [-π, π]
    # print(h_begin, v_begin)
    shape = [h, w]
    coord_seqs = []
    gridy = make_coord((h,w))
    xy, mask = gridy2gridx_erp2pers_(gridy.flip(-1),  1664, 3328, h, w, h_fov, THETA, PHI)
    xy = xy.view(h,w,2).flip(-1)
    arg_map = torch.zeros([h,w]).to(torch.int64).cuda()
    # arg_map_ = torch.zeros([h,w]).cuda()
    dist_vector = torch.zeros([h,w,2]).cuda()
    # dist_vector_ = torch.zeros([h,w,2]).cuda()
    xy = torch.Tensor(xy).cuda()
    uv = torch.Tensor(uv).cuda()
    # uv[:,0] = uv[:,0]*0.9994
    # uv[:,1] = uv[:,1]*0.9997
    uv[:,0] = uv[:,0]*0.9997
    uv[:,1] = uv[:,1]*0.9997    
    # uv[:,0] = (uv[:,0] - 0.0031)*0.9994
    # uv[:,1] = (uv[:,1] + 0.0025)*0.9994
    # print("uv", uv.max(), uv.min(), uv.shape)
    # print("xy", xy.max(), xy.min(), xy.shape)

    # print("xy" ,xy.shape)
    # xy_h, xy_w, _= xy.shape
    # xy = xy.reshape(-1,1,2)
    # dist_list = distance(xy, uv)
    # print(dist_list.shape)
    # arg_map = dist_list.argmin()

    # for i in range(h):
    #     for j in range(w):
    #         dist_list = distance(xy[i,j].unsqueeze(0), uv)
    #         arg_map[i,j] = dist_list.argmin()
    #         dist_vector[i,j] = xy[i,j] - uv[int(arg_map[i,j])]
            
    xy_h, xy_w, _= xy.shape
    xy = xy.reshape(h,-1,1,2)
    # print(xy.shape, h, w)
    # print("begin")
    # for i in range(h):
    #     dist_list = distance(xy[i], uv)
    #     dist_i = dist_list.argmin(dim=1)
    #     dist_vector[i] = xy[i].squeeze() - uv[dist_i]
    #     arg_map[i] = dist_i
    chunk_size = 256
    for i in range(h):
        dist_list = []
        xy_ = xy[i]
        for j in range(0, xy.shape[1], chunk_size):
            dist_list_ = distance(xy_[j:j+chunk_size], uv)
            dist_list.append(dist_list_)
        dist_list = torch.cat(dist_list, dim=0)
        # print("dist_list", dist_list.shape, dist_list.max(), dist_list.min())
        # print("dist_list", dist_list.shape, dist_list.max(), dist_list.min())
        dist_i = dist_list.argmin(dim=1)
        dist_vector[i] = xy[i].squeeze() - uv[dist_i]
        arg_map[i] = dist_i
    tmp = {}
    tmp['arg_map'] = arg_map.cpu().numpy()
    tmp['dist_vector'] = dist_vector.cpu().numpy()
    tmp['THETA'] = THETA
    tmp['PHI'] = PHI
    # print(save_path)
    savemat(save_path+str(level)+'_'+str(h)+'_' + str(w)+'_' + str(v_fov)+'_' + str(h_fov)+ '_'+ str(int(THETA_SAVE))+'_'+ str(int(PHI_SAVE)) +'_arg_new_v2.mat', tmp)

    return arg_map, dist_vector

def make_arg_map_dataset_panorama(face_vector, h, w):


    uv = xyz2uv_panorama(face_vector).reshape(-1,2)

    # print("uv_shape", uv.shape)
    print("uv0", uv[:,0].max(), uv[:,0].min())
    print("uv1", uv[:,1].max(), uv[:,1].min())
    p, q = uv.shape

    xy = np.zeros([h,w,2])
    pi = np.pi
    shape = [h, w]
    coord_seqs = []
    gridy = make_coord((h,w), ([-pi/2, pi/2], [-pi, pi]))
    xy = gridy.view(h,w,2)
    arg_map = torch.zeros([h,w]).to(torch.int64).cuda()
    dist_vector = torch.zeros([h,w,2]).cuda()
    xy = torch.Tensor(xy).cuda()
    uv = torch.Tensor(uv).cuda()
    uv[:,0] = uv[:,0]*0.9997
    uv[:,1] = uv[:,1]*0.9997
            
    xy = xy.reshape(h,-1,1,2)
    chunk_size = 32

    for i in range(h):
        dist_list = []
        xy_ = xy[i]
        for j in range(0, xy.shape[1], chunk_size):
            # dist_list_ = distance(xy_[j:j+chunk_size], uv)
            dist_list_ = distance(xy_[j:j+chunk_size], uv)
            dist_list.append(dist_list_)
        # gc.collect()
        # torch.cuda.empty_cache()
        # torch.cuda.ipc_collect()
        # print("dtype", dist_list[0].dtype)
        # print("dtype", dist_list[0].dtype)
        dist_list = torch.cat(dist_list, dim=0)
        dist_i = dist_list.argmin(dim=1)
        dist_vector[i] = xy[i].squeeze() - uv[dist_i]
        arg_map[i] = dist_i

    
    tmp = {}
    tmp['arg_map'] = arg_map.cpu().numpy()
    tmp['dist_vector'] = dist_vector.cpu().numpy()
    savemat('/home/zwxionggroup/xurk/dataset/arg_map_panorama_level9/'+str(h)+'_' + str(w) + '_arg_new_v2.mat', tmp)

    return arg_map, dist_vector

def pad_noise_nearest(x, pad_size):
        bs, box_num, c, h, w = x.shape
        x_0 = w
        x_1 = h//2
        pad_size = pad_size
        top_pad = x[:, :, :, :pad_size, :].view(bs * box_num, c, pad_size, w)
        top_pad = F.interpolate(top_pad, size=(pad_size, x_1), mode='nearest').view(bs*box_num, c, pad_size, x_1)
        top_pad = F.pad(top_pad, pad=(0, pad_size, 0, 0), mode='reflect').permute(0, 1, 3, 2).view(bs, box_num, c, x_1 + pad_size, pad_size).flip(-2)
        top_pad_ = top_pad.clone()


        bottom_pad = x[:, :, :, -pad_size:, :].view(bs * box_num, c, pad_size, w)  
        bottom_pad = F.interpolate(bottom_pad, size=(pad_size, x_1), mode='nearest').view(bs*box_num, c, pad_size, x_1)
        bottom_pad = F.pad(bottom_pad, pad=(pad_size, 0, 0, 0), mode='reflect').permute(0, 1, 3, 2).view(bs, box_num, c, x_1 + pad_size, pad_size).flip(-2)
        bottom_pad_ = bottom_pad.clone()


        left_pad = x[:, :, :, :, :pad_size].view(bs, box_num, c, h, pad_size)  
        # left_pad = F.interpolate(left_pad, size=(2*x_1, pad_size), mode='nearest').view(bs, box_num, c, 2*x_1, pad_size)
        # left_pad = F.pad(left_pad, pad=(0, 0, pad_size, pad_size))

        left_pad_top = left_pad[:, :, :, :x_1, :].view(bs * box_num, c, x_1, pad_size)
        left_pad_top = F.pad(left_pad_top, pad=(0, 0, 0, pad_size), mode='reflect').view(bs, box_num, c, x_1 + pad_size, pad_size)

        left_pad_bottom = left_pad[:, :, :, x_1:, :].view(bs * box_num, c, x_1, pad_size).permute(0, 1, 3, 2)
        left_pad_bottom = F.interpolate(left_pad_bottom, size=(pad_size, x_0), mode='nearest').view(bs, box_num, c, pad_size, x_0).flip(-1)

        left_pad_top_ = left_pad_top.clone()
        left_pad_bottom_ = left_pad_bottom.clone()

  
        right_pad = x[:, :, :, :, -pad_size:].view(bs * box_num, c, h, pad_size)  

        right_pad_top = right_pad[ :, :, :x_1, :].view(bs * box_num, c, x_1, pad_size).permute(0, 1, 3, 2)
        right_pad_top = F.interpolate(right_pad_top, size=(pad_size, x_0), mode='nearest').view(bs, box_num, c, pad_size, x_0).flip(-1)

        right_pad_bottom = right_pad[:, :, x_1:, :]
        right_pad_bottom = F.pad(right_pad_bottom, pad=(0, 0, pad_size, 0), mode='reflect').view(bs, box_num, c, x_1 + pad_size, pad_size)

        right_pad_top_ = right_pad_top.clone()
        right_pad_bottom_ = right_pad_bottom.clone()

        alpha = 0
        for i in range(5):
            top_pad_[::,i,::,::,::] = top_pad[::,(i+1)%5,::,::,::] + alpha*top_pad[::,(i+1)%5,::,::,::]

            bottom_pad_[::,i,::,::,::] = bottom_pad[::,(i-1)%5,::,::,::] + alpha*bottom_pad[::,(i-1)%5,::,::,::]

            left_pad_top_[::,i,::,::,::] = left_pad_top[::,(i+1)%5,::,::,::] + alpha*left_pad_top[::,(i+1)%5,::,::,::]
            left_pad_bottom_[::,i,::,::,::] = left_pad_bottom[::,(i+1)%5,::,::,::] + alpha*left_pad_bottom[::,(i+1)%5,::,::,::]

            right_pad_top_[::,i,::,::,::] = right_pad_top[::,(i-1)%5,::,::,::] + alpha*right_pad_top[::,(i-1)%5,::,::,::]
            right_pad_bottom_[::,i,::,::,::] = right_pad_bottom[::,(i-1)%5,::,::,::] + alpha*right_pad_bottom[::,(i-1)%5,::,::,::]


        left = torch.cat((right_pad_bottom_, bottom_pad_), dim=3)
        top = right_pad_top_
        right = torch.cat((top_pad_, left_pad_top_), dim=3)
        bottom = left_pad_bottom_


        x = torch.cat((top, x, bottom), dim=3)
        x = torch.cat((left, x, right), dim=4)
        bs, box_num, c, h, w = x.shape
        x = x.view(bs*box_num, c, h, w)
        return x
# def pad_noise_nearest(x, pad_size):
#         bs, box_num, c, h, w = x.shape
#         x_0 = w
#         x_1 = h//2
#         pad_size = pad_size
#         top_pad = x[:, :, :, :pad_size, :].view(bs * box_num, c, pad_size, w)
#         top_pad = F.interpolate(top_pad, size=(pad_size, x_1), mode='bicubic').view(bs*box_num, c, pad_size, x_1)
#         top_pad = F.pad(top_pad, pad=(0, pad_size, 0, 0), mode='constant', value=0).permute(0, 1, 3, 2).view(bs, box_num, c, x_1 + pad_size, pad_size).flip(-2)
#         top_pad_ = top_pad.clone()


#         bottom_pad = x[:, :, :, -pad_size:, :].view(bs * box_num, c, pad_size, w)  
#         bottom_pad = F.interpolate(bottom_pad, size=(pad_size, x_1), mode='bicubic').view(bs*box_num, c, pad_size, x_1)
#         bottom_pad = F.pad(bottom_pad, pad=(pad_size, 0, 0, 0), mode='constant', value=0).permute(0, 1, 3, 2).view(bs, box_num, c, x_1 + pad_size, pad_size).flip(-2)
#         bottom_pad_ = bottom_pad.clone()


#         left_pad = x[:, :, :, :, :pad_size].view(bs, box_num, c, h, pad_size)  
#         # left_pad = F.interpolate(left_pad, size=(2*x_1, pad_size), mode='nearest').view(bs, box_num, c, 2*x_1, pad_size)
#         # left_pad = F.pad(left_pad, pad=(0, 0, pad_size, pad_size))

#         left_pad_top = left_pad[:, :, :, :x_1, :].view(bs * box_num, c, x_1, pad_size)
#         left_pad_top = F.pad(left_pad_top, pad=(0, 0, 0, pad_size), mode='constant', value=0).view(bs, box_num, c, x_1 + pad_size, pad_size)

#         left_pad_bottom = left_pad[:, :, :, x_1:, :].view(bs * box_num, c, x_1, pad_size).permute(0, 1, 3, 2)
#         left_pad_bottom = F.interpolate(left_pad_bottom, size=(pad_size, x_0), mode='nearest').view(bs, box_num, c, pad_size, x_0).flip(-1)

#         left_pad_top_ = left_pad_top.clone()
#         left_pad_bottom_ = left_pad_bottom.clone()

  
#         right_pad = x[:, :, :, :, -pad_size:].view(bs * box_num, c, h, pad_size)  

#         right_pad_top = right_pad[ :, :, :x_1, :].view(bs * box_num, c, x_1, pad_size).permute(0, 1, 3, 2)
#         right_pad_top = F.interpolate(right_pad_top, size=(pad_size, x_0), mode='nearest').view(bs, box_num, c, pad_size, x_0).flip(-1)

#         right_pad_bottom = right_pad[:, :, x_1:, :]
#         right_pad_bottom = F.pad(right_pad_bottom, pad=(0, 0, pad_size, 0), mode='constant', value=0).view(bs, box_num, c, x_1 + pad_size, pad_size)

#         right_pad_top_ = right_pad_top.clone()
#         right_pad_bottom_ = right_pad_bottom.clone()

#         alpha = 0
#         for i in range(5):
#             top_pad_[::,i,::,::,::] = top_pad[::,(i+1)%5,::,::,::] + alpha*top_pad[::,(i+1)%5,::,::,::]

#             bottom_pad_[::,i,::,::,::] = bottom_pad[::,(i-1)%5,::,::,::] + alpha*bottom_pad[::,(i-1)%5,::,::,::]

#             left_pad_top_[::,i,::,::,::] = left_pad_top[::,(i+1)%5,::,::,::] + alpha*left_pad_top[::,(i+1)%5,::,::,::]
#             left_pad_bottom_[::,i,::,::,::] = left_pad_bottom[::,(i+1)%5,::,::,::] + alpha*left_pad_bottom[::,(i+1)%5,::,::,::]

#             right_pad_top_[::,i,::,::,::] = right_pad_top[::,(i-1)%5,::,::,::] + alpha*right_pad_top[::,(i-1)%5,::,::,::]
#             right_pad_bottom_[::,i,::,::,::] = right_pad_bottom[::,(i-1)%5,::,::,::] + alpha*right_pad_bottom[::,(i-1)%5,::,::,::]


#         left = torch.cat((right_pad_bottom_, bottom_pad_), dim=3)
#         top = right_pad_top_
#         right = torch.cat((top_pad_, left_pad_top_), dim=3)
#         bottom = left_pad_bottom_


#         x = torch.cat((top, x, bottom), dim=3)
#         x = torch.cat((left, x, right), dim=4)
#         bs, box_num, c, h, w = x.shape
#         x = x.view(bs*box_num, c, h, w)
#         return x


def pad_noise( x, pad_size):

        bs, box_num, c, h, w = x.shape
        x_0 = w
        x_1 = h//2
        pad_size = pad_size

        top_pad = x[:, :, :, :pad_size, :].view(bs * box_num, c, pad_size, w)
        top_pad = F.interpolate(top_pad, size=(pad_size, x_1), mode='bicubic', align_corners=False).view(bs*box_num, c, pad_size, x_1)
        top_pad = F.pad(top_pad, pad=(0, pad_size, 0, 0), mode = 'reflect').permute(0, 1, 3, 2).view(bs, box_num, c, x_1 + pad_size, pad_size).flip(-2)
        top_pad_ = top_pad.clone()


        bottom_pad = x[:, :, :, -pad_size:, :].view(bs * box_num, c, pad_size, w)  
        bottom_pad = F.interpolate(bottom_pad, size=(pad_size, x_1), mode='bicubic', align_corners=False).view(bs*box_num, c, pad_size, x_1)
        bottom_pad = F.pad(bottom_pad, pad=(pad_size, 0, 0, 0), mode = 'reflect').permute(0, 1, 3, 2).view(bs, box_num, c, x_1 + pad_size, pad_size).flip(-2)
        bottom_pad_ = bottom_pad.clone()


        left_pad = x[:, :, :, :, :pad_size].view(bs, box_num, c, h, pad_size)  
        # left_pad = F.interpolate(left_pad, size=(2*x_1, pad_size), mode='bicubic', align_corners=False).view(bs, box_num, c, 2*x_1, pad_size)
        # left_pad = F.pad(left_pad, pad=(0, 0, pad_size, pad_size))

        left_pad_top = left_pad[:, :, :, :x_1, :].view(bs * box_num, c, x_1, pad_size)
        left_pad_top = F.pad(left_pad_top, pad=(0, 0, 0, pad_size), mode = 'reflect').view(bs, box_num, c, x_1 + pad_size, pad_size)

        left_pad_bottom = left_pad[:, :, :, x_1:, :].view(bs * box_num, c, x_1, pad_size).permute(0, 1, 3, 2)
        left_pad_bottom = F.interpolate(left_pad_bottom, size=(pad_size, x_0), mode='bicubic', align_corners=False).view(bs, box_num, c, pad_size, x_0).flip(-1)

        left_pad_top_ = left_pad_top.clone()
        left_pad_bottom_ = left_pad_bottom.clone()

  
        right_pad = x[:, :, :, :, -pad_size:].view(bs * box_num, c, h, pad_size)  

        right_pad_top = right_pad[ :, :, :x_1, :].view(bs * box_num, c, x_1, pad_size).permute(0, 1, 3, 2)
        right_pad_top = F.interpolate(right_pad_top, size=(pad_size, x_0), mode='bicubic', align_corners=False).view(bs, box_num, c, pad_size, x_0).flip(-1)

        right_pad_bottom = right_pad[:, :, x_1:, :]
        right_pad_bottom = F.pad(right_pad_bottom, pad=(0, 0, pad_size, 0), mode = 'reflect').view(bs, box_num, c, x_1 + pad_size, pad_size)

        right_pad_top_ = right_pad_top.clone()
        right_pad_bottom_ = right_pad_bottom.clone()

        
        for i in range(5):
            top_pad_[::,i,::,::,::] = top_pad[::,(i+1)%5,::,::,::]

            bottom_pad_[::,i,::,::,::] = bottom_pad[::,(i-1)%5,::,::,::]

            left_pad_top_[::,i,::,::,::] = left_pad_top[::,(i+1)%5,::,::,::]
            left_pad_bottom_[::,i,::,::,::] = left_pad_bottom[::,(i+1)%5,::,::,::]

            right_pad_top_[::,i,::,::,::] = right_pad_top[::,(i-1)%5,::,::,::]
            right_pad_bottom_[::,i,::,::,::] = right_pad_bottom[::,(i-1)%5,::,::,::]


        left = torch.cat((right_pad_bottom_, bottom_pad_), dim=3)
        top = right_pad_top_
        right = torch.cat((top_pad_, left_pad_top_), dim=3)
        bottom = left_pad_bottom_


        x = torch.cat((top, x, bottom), dim=3)
        x = torch.cat((left, x, right), dim=4)
        bs, box_num, c, h, w = x.shape
        x = x.view(bs*box_num, c, h, w)
        return x



def return_noise(x, pad_size):
        bs, c, h, w = x.shape
        x = x.view(bs//5, 5, c, h, w)
        final_h = h - pad_size*2
        final_w = w - pad_size*2

        top_pad = x[:, :, :, :pad_size, pad_size:-pad_size].view(bs, c, pad_size, final_w)
        # top_pad = F.interpolate(top_pad, size=(pad_size, final_h//2), mode='bicubic', align_corners=False).permute(0, 1, 3, 2).contiguous().view(bs//5, 5, c, final_h//2, pad_size).flip(-2)
        top_pad = F.interpolate(top_pad, size=(pad_size, final_h//2), mode='nearest').permute(0, 1, 3, 2).contiguous().view(bs//5, 5, c, final_h//2, pad_size).flip(-2)

        bottom_pad = x[:, :, :, -pad_size:, pad_size:-pad_size].view(bs, c, pad_size, final_w)
        # bottom_pad = F.interpolate(bottom_pad, size=(pad_size, final_h//2), mode='bicubic', align_corners=False).permute(0, 1, 3, 2).contiguous().view(bs//5, 5, c, final_h//2, pad_size).flip(-2)
        bottom_pad = F.interpolate(bottom_pad, size=(pad_size, final_h//2), mode='nearest').permute(0, 1, 3, 2).contiguous().view(bs//5, 5, c, final_h//2, pad_size).flip(-2)

        left_pad_top = x[:, :, :, pad_size:final_h//2+pad_size, :pad_size]
        
        left_pad_bottom = x[:, :, :, -pad_size-final_h//2:-pad_size, :pad_size].view(bs, c, final_h//2, pad_size)
        # left_pad_bottom = F.interpolate(left_pad_bottom, size=(final_w, pad_size), mode='bicubic', align_corners=False).permute(0, 1, 3, 2).contiguous().view(bs//5, 5, c, pad_size, final_w).flip(-1)
        left_pad_bottom = F.interpolate(left_pad_bottom, size=(final_w, pad_size), mode='nearest').permute(0, 1, 3, 2).contiguous().view(bs//5, 5, c, pad_size, final_w).flip(-1)

        right_pad_top = x[:, :, :, pad_size:final_h//2+pad_size, -pad_size:].view(bs, c, final_h//2, pad_size)
        # right_pad_top = F.interpolate(right_pad_top, size=(final_w, pad_size), mode='bicubic', align_corners=False).permute(0, 1, 3, 2).contiguous().view(bs//5, 5, c, pad_size, final_w).flip(-1)
        right_pad_top = F.interpolate(right_pad_top, size=(final_w, pad_size), mode='nearest').permute(0, 1, 3, 2).contiguous().view(bs//5, 5, c, pad_size, final_w).flip(-1)

        right_pad_bottom = x[:, :, :, -pad_size-final_h//2:-pad_size, -pad_size:]

        top_pad_ = top_pad.clone()
        bottom_pad_ = bottom_pad.clone()
        left_pad_top_ = left_pad_top.clone()
        left_pad_bottom_ = left_pad_bottom.clone()
        right_pad_top_ = right_pad_top.clone()
        right_pad_bottom_ = right_pad_bottom.clone()

        
        for i in range(5):
            top_pad_[::,i,::,::] = top_pad[::,(i+1)%5,::,::]
            bottom_pad_[::,i,::,::] = bottom_pad[::,(i-1)%5,::,::]
            left_pad_top_[::,i,::,::] = left_pad_top[::,(i+1)%5,::,::]
            left_pad_bottom_[::,i,::,::] = left_pad_bottom[::,(i+1)%5,::,::]
            right_pad_top_[::,i,::,::] = right_pad_top[::,(i-1)%5,::,::]
            right_pad_bottom_[::,i,::,::] = right_pad_bottom[::,(i-1)%5,::,::]
        left = torch.cat((right_pad_bottom_, bottom_pad_), dim=3)
        top = right_pad_top_
        right = torch.cat((top_pad_, left_pad_top_), dim=3)
        bottom = left_pad_bottom_


        x = x[..., pad_size:-pad_size, pad_size:-pad_size]
        # A, B: (B, C, H, W)
        # x [..., :pad_size,::] = self.choose_random(x[..., :pad_size,::], top)
        # x [..., -pad_size:,::] = self.choose_random(x[..., -pad_size:,::], bottom)
        # x [..., ::, :pad_size] = self.choose_random(x[..., ::, :pad_size], left)
        # x [..., ::, -pad_size:] = self.choose_random(x[..., ::, -pad_size:], right)
        alpha = 0.1
        x [..., :pad_size,::] = alpha*x[..., :pad_size,::] + top*(1-alpha)
        x [..., -pad_size:,::] = alpha*(x[..., -pad_size:,::]) + bottom*(1-alpha)
        x [..., ::, :pad_size] = alpha*(x[..., ::, :pad_size]) + left*(1-alpha)
        x [..., ::, -pad_size:] = alpha*(x[..., ::, -pad_size:]) + right*(1-alpha)

        # x [..., :pad_size,::] = (x[..., :pad_size,::] + top)/2
        # x [..., -pad_size:,::] = (x[..., -pad_size:,::] + bottom) / 2
        # x [..., ::, :pad_size] = (x[..., ::, :pad_size] + left) / 2
        # x [..., ::, -pad_size:] = (x[..., ::, -pad_size:] + right) / 2
        x = x.view(bs, c, final_h, final_w)
        return x
def return_noise_2(x, pad_size):
        bs, c, h, w = x.shape
        x = x.view(bs//5, 5, c, h, w)
        final_h = h - pad_size*2
        final_w = w - pad_size*2

        top_pad = x[:, :, :, :pad_size, pad_size:-pad_size].view(bs, c, pad_size, final_w)
        # top_pad = F.interpolate(top_pad, size=(pad_size, final_h//2), mode='bicubic', align_corners=False).permute(0, 1, 3, 2).contiguous().view(bs//5, 5, c, final_h//2, pad_size).flip(-2)
        top_pad = F.interpolate(top_pad, size=(pad_size, final_h//2), mode='nearest').permute(0, 1, 3, 2).contiguous().view(bs//5, 5, c, final_h//2, pad_size).flip(-2)

        bottom_pad = x[:, :, :, -pad_size:, pad_size:-pad_size].view(bs, c, pad_size, final_w)
        # bottom_pad = F.interpolate(bottom_pad, size=(pad_size, final_h//2), mode='bicubic', align_corners=False).permute(0, 1, 3, 2).contiguous().view(bs//5, 5, c, final_h//2, pad_size).flip(-2)
        bottom_pad = F.interpolate(bottom_pad, size=(pad_size, final_h//2), mode='nearest').permute(0, 1, 3, 2).contiguous().view(bs//5, 5, c, final_h//2, pad_size).flip(-2)

        left_pad_top = x[:, :, :, pad_size:final_h//2+pad_size, :pad_size]
        
        left_pad_bottom = x[:, :, :, -pad_size-final_h//2:-pad_size, :pad_size].view(bs, c, final_h//2, pad_size)
        # left_pad_bottom = F.interpolate(left_pad_bottom, size=(final_w, pad_size), mode='bicubic', align_corners=False).permute(0, 1, 3, 2).contiguous().view(bs//5, 5, c, pad_size, final_w).flip(-1)
        left_pad_bottom = F.interpolate(left_pad_bottom, size=(final_w, pad_size), mode='nearest').permute(0, 1, 3, 2).contiguous().view(bs//5, 5, c, pad_size, final_w).flip(-1)

        right_pad_top = x[:, :, :, pad_size:final_h//2+pad_size, -pad_size:].view(bs, c, final_h//2, pad_size)
        # right_pad_top = F.interpolate(right_pad_top, size=(final_w, pad_size), mode='bicubic', align_corners=False).permute(0, 1, 3, 2).contiguous().view(bs//5, 5, c, pad_size, final_w).flip(-1)
        right_pad_top = F.interpolate(right_pad_top, size=(final_w, pad_size), mode='nearest').permute(0, 1, 3, 2).contiguous().view(bs//5, 5, c, pad_size, final_w).flip(-1)

        right_pad_bottom = x[:, :, :, -pad_size-final_h//2:-pad_size, -pad_size:]

        top_pad_ = top_pad.clone()
        bottom_pad_ = bottom_pad.clone()
        left_pad_top_ = left_pad_top.clone()
        left_pad_bottom_ = left_pad_bottom.clone()
        right_pad_top_ = right_pad_top.clone()
        right_pad_bottom_ = right_pad_bottom.clone()

        
        for i in range(5):
            top_pad_[::,i,::,::] = top_pad[::,(i+1)%5,::,::]
            bottom_pad_[::,i,::,::] = bottom_pad[::,(i-1)%5,::,::]
            left_pad_top_[::,i,::,::] = left_pad_top[::,(i+1)%5,::,::]
            left_pad_bottom_[::,i,::,::] = left_pad_bottom[::,(i+1)%5,::,::]
            right_pad_top_[::,i,::,::] = right_pad_top[::,(i-1)%5,::,::]
            right_pad_bottom_[::,i,::,::] = right_pad_bottom[::,(i-1)%5,::,::]
        left = torch.cat((right_pad_bottom_, bottom_pad_), dim=3)
        top = right_pad_top_
        right = torch.cat((top_pad_, left_pad_top_), dim=3)
        bottom = left_pad_bottom_

        alpha = 0.5
        x = x.clone()

        x [..., pad_size:pad_size*2,pad_size:-pad_size] = alpha*x[..., pad_size:pad_size*2,pad_size:-pad_size] + top*(1-alpha)
        x [..., -pad_size*2:-pad_size,pad_size:-pad_size] = alpha*(x[..., -pad_size*2:-pad_size,pad_size:-pad_size]) + bottom*(1-alpha)
        x [..., pad_size:-pad_size, pad_size:pad_size*2] = alpha*(x[..., pad_size:-pad_size, pad_size:pad_size*2]) + left*(1-alpha)
        x [..., pad_size:-pad_size, -pad_size*2:-pad_size] = alpha*(x[..., pad_size:-pad_size, -pad_size*2:-pad_size]) + right*(1-alpha)

        x = x.view(bs, c, h, w)
        return x
