import random

import torch
import torch.nn.functional as F

from models import register
from models.infd_base import INFDBase
from models.vqgan.discriminator import make_discriminator
from models.vqgan.lpips import LPIPS
from models.infd_utils import (
    calculate_adaptive_gan_weight,
    compute_batch_psnr,
    compute_disc_hinge_loss,
    detach_nested_tensors,
    downsample_loss_inputs,
)


@register('infd')
class INFD(INFDBase):

    def __init__(
        self,
        disc=True,
        disc_cond_scale=False,
        disc_use_custom=False,
        adaptive_gan_weight=False,
        perc_loss=True,
        **kwargs,
    ):
        super().__init__(**kwargs)
        if disc_cond_scale:
            raise NotImplementedError('cleaned s0/s1/s2 code only keeps disc_cond_scale=False')
        self.perc_loss = LPIPS().eval() if perc_loss else None
        self.disc = make_discriminator(use_custom=disc_use_custom, input_nc=3) if disc else None
        self.adaptive_gan_weight = adaptive_gan_weight

    def get_params(self, name):
        if name == 'disc':
            return self.disc.parameters()
        return super().get_params(name)

    def run_renderer(self, z_dec, arg_map=None, dist_vector=None, coord=None, cell=None):
        del coord, cell
        if self.renderer is None:
            return None
        return self.renderer(z_dec=z_dec, arg_map=arg_map, dist_vector=dist_vector)


    def _reconstruction_losses(self, pred, target, data, ret, downsample_scale):
        lcfg = self.loss_cfg
        if lcfg.get('l1_loss_downsample', False):
            pred_for_loss = downsample_loss_inputs(pred, data['inp'], data['gt_cell'])
            target_for_loss = downsample_loss_inputs(target, data['inp'], data['gt_cell'])
            l1_loss = sum(torch.abs(p - t).mean() for p, t in zip(pred_for_loss, target_for_loss))
            l1_loss = l1_loss / len(pred_for_loss)
        else:
            l1_loss = torch.abs(pred - target).mean()
        ret['l1_loss'] = l1_loss.item()
        ret['loss'] = ret['loss'] + l1_loss * lcfg.get('l1_loss', 1)

        if lcfg.get('perc_loss_downsample', False):
            pred_vis = pred * 0.5 + 0.5
            target_vis = target * 0.5 + 0.5
            pred_down = F.interpolate(
                pred_vis,
                scale_factor=downsample_scale,
                mode='bicubic',
                align_corners=False,
                antialias=True,
            )
            target_down = F.interpolate(
                target_vis,
                scale_factor=downsample_scale,
                mode='bicubic',
                align_corners=False,
                antialias=True,
            )
            pred_down = (pred_down - 0.5) / 0.5
            target_down = (target_down - 0.5) / 0.5
            perc_loss = self.perc_loss(pred_down, target_down).mean()
        elif lcfg.get('perc_loss', 1) == 0:
            perc_loss = pred.new_tensor(0.0)
        else:
            perc_loss = self.perc_loss(pred, target).mean()
        ret['perc_loss'] = perc_loss.item()
        ret['loss'] = ret['loss'] + perc_loss * lcfg.get('perc_loss', 1)
        return l1_loss, perc_loss

    def _generator_gan_loss(self, pred, l1_loss, perc_loss, ret, last_layer):
        lcfg = self.loss_cfg
        logits_fake = self.disc(pred)
        gan_g_loss = -torch.mean(logits_fake)
        ret['gan_g_loss'] = gan_g_loss.item()
        weight = lcfg.get('gan_g_loss', 1)
        if self.training and self.adaptive_gan_weight:
            nll_loss = l1_loss * lcfg.get('l1_loss', 1) + perc_loss * lcfg.get('perc_loss', 1)
            adaptive_g_w = calculate_adaptive_gan_weight(nll_loss, gan_g_loss, last_layer)
            ret['adaptive_g_w'] = adaptive_g_w.item()
            weight = weight * adaptive_g_w
        ret['loss'] = ret['loss'] + gan_g_loss * weight

    def _disc_loss(self, pred, target):
        return compute_disc_hinge_loss(self.disc, pred, target)

    def forward(self, data, mode, has_opt=None, ret=None, **kwargs):
        gd = self.get_gd_from_opt(has_opt)
        arg_map = data['arg_map']
        dist_vector = data['dist_vector']
        random_number = random.randint(0, 10)

        if mode == 'S0':
            pred, ret, target = super().forward(
                data,
                arg_map=arg_map,
                dist_vector=dist_vector,
                mode='z_dec_S0',
                has_opt=has_opt,
                random_number=random_number,
            )
            ret['pred_patch'] = pred.detach()
            ret['target_patch'] = target.detach()
            l1_loss, perc_loss = self._reconstruction_losses(pred, target.detach(), data, ret, 0.8)
            torch.cuda.empty_cache()
            if kwargs.get('use_gan', False):
                self._generator_gan_loss(pred, l1_loss, perc_loss, ret, self.decoder.decoder.conv_out.conv.weight)
            ret['psnr'] = compute_batch_psnr(pred, target.detach())
            return ret

        if mode == 'pred':
            if self.renderer is None:
                with torch.no_grad():
                    pred, _, target = super().forward(
                        data,
                        arg_map=arg_map,
                        dist_vector=dist_vector,
                        mode='z_dec_S0',
                        has_opt=has_opt,
                        random_number=random_number,
                    )
                return pred, target
            with torch.no_grad():
                z_dec, _ = super().forward(
                    data,
                    arg_map=arg_map,
                    dist_vector=dist_vector,
                    mode='z_dec',
                    has_opt=has_opt,
                    random_number=random_number,
                )
            return self.run_renderer(z_dec, arg_map, dist_vector)

        if mode == 'loss':
            if not gd['renderer']:
                _, ret = super().forward(data, arg_map=arg_map, dist_vector=dist_vector, mode='z', has_opt=has_opt)
                return ret

            z_dec, ret = super().forward(
                data,
                arg_map=arg_map,
                dist_vector=dist_vector,
                mode='z_dec',
                has_opt=has_opt,
                random_number=random_number,
            )
            pred = self.run_renderer(detach_nested_tensors(z_dec), arg_map, dist_vector)
            target = data['gt']
            ret['pred_patch'] = pred.detach()

            l1_loss, perc_loss = self._reconstruction_losses(pred, target, data, ret, 0.9)
            if kwargs.get('use_gan', False):
                self._generator_gan_loss(pred, l1_loss, perc_loss, ret, self.renderer.get_last_layer_weight())

            ret['psnr'] = compute_batch_psnr(pred, target)
            return ret

        if mode == 'disc_loss':
            return self._disc_loss(ret['pred_patch'].detach().float(), data['gt'])

        if mode == 'disc_loss_s0':
            return self._disc_loss(ret['pred_patch'].detach().float(), ret['target_patch'].detach().float())

        raise NotImplementedError(f'Unsupported INFD mode for cleaned s0/s1/s2 code: {mode}')

