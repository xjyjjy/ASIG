import cv2
import numpy as np

# def spherical_rotate(img, R):
#     """
#     对 equirectangular 图像 img 做三维旋转 R @ [x,y,z]
#     img: (H, W, 3) BGR / RGB
#     R:   (3,3) 旋转矩阵
#     返回：同样 shape 的旋转后图像
#     """
#     H, W = img.shape[:2]

#     j = np.arange(W, dtype=np.float32)
#     i = np.arange(H, dtype=np.float32)
#     jj, ii = np.meshgrid(j, i)
#     lon = 2 * np.pi * (jj / W - 0.5)      # [-π, π]
#     lat = np.pi * (0.5 - ii / H)          # [+π/2, -π/2]

#     # 2) 经纬度 → 单位球笛卡尔坐标
#     x = np.cos(lat) * np.cos(lon)
#     y = np.cos(lat) * np.sin(lon)
#     z = np.sin(lat)

#     # 3) 旋转
#     xyz = np.stack([x, y, z], axis=0).reshape(3, -1)  # (3, H*W)
#     xyz_rot = R @ xyz                                # (3, H*W)
#     x2, y2, z2 = xyz_rot

#     # 4) 旋转后笛卡尔 → 经纬度
#     lon2 = np.arctan2(y2, x2)                         # [-π, π]
#     # clip 确保数值误差不溢出
#     z2 = np.clip(z2, -1.0, 1.0)
#     lat2 = np.arcsin(z2)                             # [-π/2, +π/2]

#     # 5) 经纬度 → 像素坐标
#     jj2 = (lon2 / (2 * np.pi) + 0.5) * W             # [0, W)
#     ii2 = (0.5 - lat2 / np.pi) * H                   # [0, H)

#     map_x = jj2.reshape(H, W).astype(np.float32)
#     map_y = ii2.reshape(H, W).astype(np.float32)

#     # 6) 重采样
#     # - 横向环绕，纵向超出填常数 0（可换 BORDER_REFLECT）
#     out = cv2.remap(
#         img, map_x, map_y,
#         interpolation=cv2.INTER_LINEAR,
#         borderMode=cv2.BORDER_WRAP
#     )
#     return out

# def euler_to_R(pitch, yaw, roll):
#     p, y, r = np.deg2rad([pitch, yaw, roll])
#     Rx = np.array([[1,         0,          0],
#                    [0, np.cos(p), -np.sin(p)],
#                    [0, np.sin(p),  np.cos(p)]])
#     Ry = np.array([[ np.cos(y), 0, np.sin(y)],
#                    [         0, 1,         0],
#                    [-np.sin(y), 0, np.cos(y)]])
#     Rz = np.array([[np.cos(r), -np.sin(r), 0],
#                    [np.sin(r),  np.cos(r), 0],
#                    [        0,          0, 1]])
#     return Rz @ Ry @ Rx


import torch
import torch.nn.functional as F

def euler_to_R(pitch: float, yaw: float, roll: float,
                      device=None, dtype=torch.float32) -> torch.Tensor:
    """
    构造旋转矩阵 R = Rz(roll) @ Ry(yaw) @ Rx(pitch)
    参数单位：度
    返回 shape (3,3) Tensor
    """
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # device = device or torch.device("cpu")
    p = torch.deg2rad(torch.tensor(pitch, dtype=dtype, device=device))
    y = torch.deg2rad(torch.tensor(yaw,   dtype=dtype, device=device))
    r = torch.deg2rad(torch.tensor(roll,  dtype=dtype, device=device))

    Rx = torch.tensor([
        [1,         0,          0],
        [0, torch.cos(p), -torch.sin(p)],
        [0, torch.sin(p),  torch.cos(p)]
    ], dtype=dtype, device=device)

    Ry = torch.tensor([
        [ torch.cos(y), 0, torch.sin(y)],
        [           0,  1,           0],
        [-torch.sin(y), 0, torch.cos(y)]
    ], dtype=dtype, device=device)

    Rz = torch.tensor([
        [torch.cos(r), -torch.sin(r), 0],
        [torch.sin(r),  torch.cos(r), 0],
        [           0,             0, 1]
    ], dtype=dtype, device=device)

    return Rz @ Ry @ Rx


def spherical_rotate(img: torch.Tensor, R: torch.Tensor) -> torch.Tensor:
    """
    对 equirectangular 特征图做三维旋转
    img: shape (B, C, H, W) Tensor
    R:   shape (3,3) 旋转矩阵 Tensor
    返回: (B, C, H, W)
    """
    B, C, H, W = img.shape
    device, dtype = img.device, img.dtype

    # 构建经纬度网格
    i = torch.arange(H, dtype=dtype, device=device)
    j = torch.arange(W, dtype=dtype, device=device)
    ii, jj = torch.meshgrid(i, j, indexing='ij')
    lon = 2 * torch.pi * (jj / W - 0.5)   # [-π, +π]
    lat = torch.pi * (0.5 - ii / H)       # [+π/2, -π/2]

    # 经纬度 -> 笛卡尔
    x = torch.cos(lat) * torch.cos(lon)
    y = torch.cos(lat) * torch.sin(lon)
    z = torch.sin(lat)

    # 旋转
    xyz = torch.stack([x, y, z], dim=0).view(3, -1)  # (3, H*W)
    xyz_rot = R @ xyz                                 # (3, H*W)
    x2, y2, z2 = xyz_rot

    # 笛卡尔 -> 经纬度
    lon2 = torch.atan2(y2, x2)                         # [-π, +π]
    lat2 = torch.asin(torch.clamp(z2, -1.0, 1.0))      # [-π/2, +π/2]

    # 经纬度 -> 归一化 grid coords [-1,1]
    grid_x = lon2 / torch.pi        # lon2/π -> [-1,1]
    grid_y = -lat2 / (torch.pi/2)   # -lat2/(π/2) -> [-1,1]
    grid = torch.stack([grid_x, grid_y], dim=-1).view(H, W, 2)
    grid = grid.unsqueeze(0).repeat(B, 1, 1, 1)        # (B, H, W, 2)

    # 重映射
    out = F.grid_sample(
        img, grid,
        mode='nearest',
        padding_mode = 'zeros',
        align_corners=False
    )
    return out
