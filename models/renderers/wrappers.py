import copy

import torch
import torch.nn as nn
import torch.nn.functional as F

import models
from models import register
from utils.geometry import convert_liif_feat_coord_cell_sphere
from utils.warp import WarpingMapper


@register('renderer_concat_wrapper_pan')
class ConcatWrapper(nn.Module):

    def __init__(
        self,
        net,
        z_dec_channels,
        co_pe_dim=None,
        co_pe_w_max=None,
        ce_pe_dim=None,
        ce_pe_w_max=None,
        x_channels=None,
        num_feature_levels=0,
        decoder_feature_channels=None,
    ):
        super().__init__()
        del z_dec_channels, co_pe_dim, co_pe_w_max, ce_pe_dim, ce_pe_w_max, x_channels, decoder_feature_channels
        if num_feature_levels != 0:
            raise NotImplementedError('cleaned s1/s2 renderer only keeps num_feature_levels=0')

        self.use_decoder_features = False
        net_spec = copy.copy(net)
        net_spec['args'] = copy.copy(net_spec.get('args') or {})
        net_spec['args']['in_channels'] = 3
        self.net = models.make(net_spec)
        self.warp_mapper = WarpingMapper(w=512)

    def get_last_layer_weight(self):
        return self.net.get_last_layer_weight()

    def forward(self, z_dec, arg_map, dist_vector):
        spherical_rgb = z_dec['sample'] if isinstance(z_dec, dict) else z_dec
        spherical_out = self.net(spherical_rgb)
        spherical_out = F.interpolate(spherical_out, size=(1024, 1023), mode='bicubic', align_corners=False)
        spherical_out = self.warp_mapper.inverse_mapping(spherical_out)
        spherical_out = F.interpolate(spherical_out, size=(1024, 256), mode='bicubic', align_corners=False)
        pred = convert_liif_feat_coord_cell_sphere(spherical_out, arg_map)
        pred = pred.permute(0, 3, 1, 2).contiguous()

        dist_vector = dist_vector.to(device=pred.device, dtype=pred.dtype)
        dist_vector = dist_vector.permute(0, 3, 1, 2).contiguous()
        return self.net.sample(pred, dist_vector)
