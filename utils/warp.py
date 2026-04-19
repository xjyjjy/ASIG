import torch
import numpy as np
import torch.nn as nn
from torch import Tensor
from torchvision import transforms
from torchvision.transforms import functional as F
from torchvision.transforms import InterpolationMode
class WarpingMapper:
    def __init__(self, w=32):
        self.w = w
        self.index_lookup = []
        self.warp(w)
    def warp(self, w=32):
        index_out = []
        index_in = []
        
        for i in range(2*w):
            if i < w:
                start_col = w - 1 - i
                end_col = w + i  # (w - 1 - i) + (2*i + 1) 
            else:
                start_col = i - w
                end_col = 3*w - 1 - i #(i - w) + (4*w -1 - 2*i)
            for j in range(start_col, end_col):
                index_out.append((i,j))
        
        for i in range(2*w):
            if i < w:
                raw_start = 0
                col_start = w - 1 - i
                limit = 2*i + 1
                j = 0
                while j < limit:
                    index_in.append((raw_start, col_start))
                    col_start += 1
                    raw_start += 1
                    j+=1
                    if raw_start % 2 == 1 and j < 2*i:
                        index_in.append((raw_start, col_start))
                        raw_start += 1
                        j+=1
            else:
                raw_start = 2 * (i - w + 1) - 1
                col_start = 0
                limit = 4*w - 1 - 2*i
                j = 0
                # for j in range(4*w - 1 - 2*i):
                while j < limit:
                    index_in.append((raw_start, col_start))
                    j+=1
                    if raw_start % 2 == 1 and j < 4*w - 1 - 2*i - 1:
                        index_in.append((raw_start + 1, col_start))
                        raw_start += 1
                        j+=1
                    raw_start += 1
                    col_start += 1

        index_in = np.array(index_in)
        index_out = np.array(index_out)
        self.index_lookup.append(torch.tensor(index_out, dtype=torch.long))
        self.index_lookup.append(torch.tensor(index_in, dtype=torch.long))

    def mapping(self, x):
        bs, c, h, w = x.shape
        x = x.view(bs, c, 2, h//2 , w).permute(0, 2, 1, 3, 4).contiguous().view(bs*2, c, h//2, w)
        t = torch.zeros((bs*2, c, 2*w, 2*w-1), dtype=torch.float32, device=x.device)
        t [:, :, self.index_lookup[0][:, 0], self.index_lookup[0][:, 1]]  = x [:, :, self.index_lookup[1][:, 0], self.index_lookup[1][:, 1]]
        return t
    
    def inverse_mapping(self, x):
        bs, c, h, w = x.shape
        t = torch.zeros((bs, c, 1024, 512), dtype=x.dtype, device=x.device)
        t [:, :, self.index_lookup[1][:, 0], self.index_lookup[1][:, 1]]  = x [:, :, self.index_lookup[0][:, 0], self.index_lookup[0][:, 1]]
        t = t.view(bs//2, 2, c, 1024, 512).permute(0, 2, 1, 3, 4).contiguous().view(bs//2, c, 2048, 512)
        return t