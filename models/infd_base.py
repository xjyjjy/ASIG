import random

import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.io import loadmat

import models
from diffusion import make_diffusion_train_components_sd, get_dm_loss_sd_pad_v1
from model_utils.padding import padding_v4_0, padding_v4_1
from models.distributions import DiagonalGaussianDistribution
from utils.warp import WarpingMapper



class INFDBase(nn.Module):

    def __init__(
        self,
        encoder,
        quant_conv,
        post_quant_conv,
        z_shape,
        decoder,
        renderer=None,
        z_dm=None,
        z_dp=None,
        z_dm_ema=True,
        z_dm_ema_rate=0.9999,
        z_ddim_eta=1.0,
        z_gaussian=True,
        quantizer=False,
        n_embed=None,
        vq_beta=0.25,
        loss_cfg=None,
        pre_from_sd=False,
        encoder_trainable=False,
        img_level=4,
        train_method=None,
        scale_factor=0.13025,
        add_module=None,
        **kwargs,
    ):
        super().__init__()
        del z_shape, z_dp, z_ddim_eta, n_embed, vq_beta, img_level, kwargs

        if not z_gaussian:
            raise NotImplementedError('cleaned s0/s1/s2 code only keeps z_gaussian=True')
        if quantizer:
            raise NotImplementedError('cleaned s0/s1/s2 code only keeps quantizer=False')

        self.scale_factor = scale_factor
        self.encoder = models.make(encoder)
        self.quant_conv = models.make(quant_conv)
        self.post_quant_conv = models.make(post_quant_conv)
        self.decoder = models.make(decoder)
        self.renderer = models.make(renderer) if renderer is not None else None
        self.train_method = train_method
        self.pre_from_sd = pre_from_sd
        self.z_gaussian = True
        self.quantizer = None
        self.loss_cfg = loss_cfg if loss_cfg is not None else {}

        self.warp_mapper = WarpingMapper(w=512)
        self.z_shape = [10, 4, 128, 128]
        self.pad_v4_0 = padding_v4_0([5, 6, 7, 8, 9])
        self.pad_v4_1 = padding_v4_1([9])
        self.face_vector_img_8 = loadmat('./mesh/level9_vector_inverse.mat')['vector']

        self._set_vae_trainability(encoder_trainable)

        if z_dm is None:
            self.z_dm = None
            self.z_dm_ema = None
        else:
            self.z_dm = models.make(z_dm)
            self.z_dm_ema = models.make(z_dm) if z_dm_ema else None
            if self.z_dm_ema is not None:
                for p in self.z_dm_ema.parameters():
                    p.requires_grad = False
            self.z_dm_ema_rate = z_dm_ema_rate
            self.z_dp = make_diffusion_train_components_sd()

        self.to_unet_condition = None
        self.to_unet_uncondition = None
        self.out = None
        if pre_from_sd:
            self._setup_s2_trainable_modules(add_module)

    def _set_vae_trainability(self, encoder_trainable):
        self.encoder.requires_grad_(False)
        self.quant_conv.requires_grad_(False)
        if encoder_trainable:
            self.post_quant_conv.requires_grad_(True)
            self.decoder.requires_grad_(True)
        else:
            self.post_quant_conv.requires_grad_(False)
            self.decoder.requires_grad_(False)

    def _setup_s2_trainable_modules(self, add_module):
        if add_module != 'all':
            raise NotImplementedError('cleaned s2 code only keeps add_module=all')

        self.to_unet_condition = nn.Conv2d(4, 4, 1, 0, 1)
        self.to_unet_uncondition = nn.Conv2d(3, 4, 1, 0, 1)
        self.out = nn.Conv2d(4, 4, 1, 0, 1)

        self.encoder.requires_grad_(False)
        self.decoder.requires_grad_(False)
        self.post_quant_conv.requires_grad_(False)
        self.quant_conv.requires_grad_(False)
        if self.renderer is not None:
            self.renderer.requires_grad_(False)
        for p in self.z_dm.parameters():
            p.requires_grad = True

    def get_params(self, name):
        if name == 'encoder':
            return self.encoder.parameters()
        if name == 'decoder':
            return list(self.decoder.parameters()) + list(self.post_quant_conv.parameters())
        if name == 'renderer':
            return self.renderer.parameters()
        if name == 'z_dm':
            return [p for p in self.z_dm.parameters() if p.requires_grad]
        raise KeyError(f'Unsupported optimizer target: {name}')

    def run_renderer(self, z_dec, coord, cell):
        raise NotImplementedError

    def mapping(self, x):
        return self.warp_mapper.mapping(x)

    def inverse_mapping(self, x):
        return self.warp_mapper.inverse_mapping(x)

    def _prepare_panorama_faces(self, x):
        bs, c, _, _ = x.shape
        box_num, h, w, _ = self.face_vector_img_8.shape
        x = x.view(bs, c, box_num, h, w).permute(0, 2, 1, 3, 4).contiguous().view(bs * box_num, c, h, w)
        x = self.mapping(x)
        x_image_out = self.pad_v4_1.get_padding(x, 9)
        x_image_out = F.interpolate(x_image_out, size=(1024, 1024), mode='bicubic', align_corners=False).clamp(-1, 1)
        return x_image_out

    def encode_z(self, x, arg_map, ret_kl_loss=False, random_number=None, pad_size=16, image=False):
        del arg_map, random_number, pad_size, image
        x_image_out = self._prepare_panorama_faces(x)

        if self.pre_from_sd:
            with torch.no_grad():
                z_chunks = []
                for xi in torch.chunk(x_image_out, chunks=5, dim=0):
                    zi = self.encoder(xi)
                    zi = self.quant_conv(zi)
                    z_chunks.append(zi)
                z = torch.cat(z_chunks, dim=0)
        else:
            z = self.encoder(x_image_out)
            z = self.quant_conv(z)

        posterior = DiagonalGaussianDistribution(z)
        z = posterior.sample()
        kl_loss = posterior.kl().mean()
        if ret_kl_loss:
            return z, kl_loss, x_image_out
        return z

    def decode_z(self, z, arg_map, ret_quant_loss=False, gt_cell=None, random_number=None, pad_size=16, image=False, need_decoder_features=False):
        del arg_map, gt_cell, random_number, pad_size
        quant_loss = None
        z = self.post_quant_conv(z)
        decoder_out = self.decoder(z, return_features=need_decoder_features)
        if image:
            z_dec = decoder_out['sample'] if isinstance(decoder_out, dict) else decoder_out
        else:
            z_dec = decoder_out
        if ret_quant_loss:
            return z_dec, quant_loss
        return z_dec

    def need_decoder_features(self):
        return self.renderer is not None and getattr(self.renderer, 'use_decoder_features', False)

    def _add_kl_loss(self, loss, ret, kl_loss, data):
        if self.z_dm is None:
            if isinstance(kl_loss, float):
                kl_loss = torch.tensor(kl_loss, device=data['gt'].device)
            ret['kl_loss'] = kl_loss.item()
            loss = loss + kl_loss * self.loss_cfg.get('kl_loss', 1)
        return loss

    def _encode(self, inp, arg_map, requires_grad, **kwargs):
        if requires_grad:
            return self.encode_z(inp, arg_map, ret_kl_loss=True, **kwargs)
        with torch.no_grad():
            return self.encode_z(inp, arg_map, ret_kl_loss=True, **kwargs)

    def _z_dm_loss(self, z, data):
        if self.train_method != 'pad_v1':
            raise NotImplementedError('cleaned s2 code only keeps train_method=pad_v1')

        bs, c, h, w = z.shape
        z = z.view(bs // 10, 10, c, h, w).contiguous()
        return get_dm_loss_sd_pad_v1(
            self.z_dp,
            self.z_dm,
            self.to_unet_uncondition,
            self.out,
            z * self.scale_factor,
            text=data['text'],
            text_2=data['text_2'],
            model_kwargs={},
        )

    def forward(self, data, arg_map=None, dist_vector=None, mode=None, has_opt=None, random_number=None, pad_size=2, **kwargs):
        del dist_vector, random_number, kwargs
        gd = self.get_gd_from_opt(has_opt)
        loss = torch.tensor(0, dtype=torch.float32, device=data['gt'].device)
        ret = {}
        inp = data.get('inp')
        recon = data.get('recon')

        if mode == 'z_dec_S0':
            z, kl_loss, x_image_out = self._encode(
                inp,
                arg_map,
                requires_grad=gd['encoder'],
                random_number=random.randint(0, 1000),
                pad_size=pad_size * 8,
                image=True,
            )
            loss = self._add_kl_loss(loss, ret, kl_loss, data)
            z_aug = self.pad_v4_0.get_padding(z.detach().clone(), 5)
            x_image_out_aug = x_image_out.clone()
            torch.cuda.empty_cache()
            z_dec, quant_loss = self.decode_z(
                z_aug,
                arg_map=arg_map,
                ret_quant_loss=True,
                gt_cell=None,
                random_number=random.randint(0, 1000),
                pad_size=0,
                image=True,
            )
            del quant_loss
            ret['loss'] = loss
            return z_dec, ret, x_image_out_aug

        if mode == 'z_dec':
            if gd['decoder'] or recon is None:
                z, kl_loss, _ = self._encode(
                    inp,
                    arg_map,
                    requires_grad=gd['encoder'],
                    random_number=random.randint(0, 1000),
                    pad_size=pad_size * 8,
                    image=False,
                )
                loss = self._add_kl_loss(loss, ret, kl_loss, data)
            else:
                z = None

            if gd['decoder']:
                with torch.no_grad():
                    z_dec, quant_loss = self.decode_z(
                        z,
                        arg_map=arg_map,
                        ret_quant_loss=True,
                        gt_cell=data.get('gt_cell'),
                        random_number=random.randint(0, 1000),
                        pad_size=0,
                        need_decoder_features=self.need_decoder_features(),
                    )
            elif recon is None:
                with torch.no_grad():
                    z = self.pad_v4_0.get_padding(z.detach(), 6)
                    z_dec, quant_loss = self.decode_z(
                        z,
                        arg_map=arg_map,
                        ret_quant_loss=True,
                        gt_cell=data.get('gt_cell'),
                        random_number=random.randint(0, 1000),
                        pad_size=16 * 8,
                        need_decoder_features=self.need_decoder_features(),
                    )
            else:
                quant_loss = None
                bs, box_num, c, h, w = recon.shape
                z_dec = recon.view(bs * box_num, c, h, w)

            if self.renderer is None:
                z_dec = F.interpolate(z_dec, size=(1024, 1023), mode='bicubic', align_corners=False)
                z_dec = self.inverse_mapping(z_dec)
                z_dec = F.interpolate(z_dec, size=(2048, 512), mode='bicubic', align_corners=False)

            del quant_loss
            ret['loss'] = loss
            return z_dec, ret

        if mode == 'z':
            z, kl_loss, _ = self._encode(
                inp,
                arg_map,
                requires_grad=gd['encoder'],
                random_number=random.randint(0, 1000),
                pad_size=pad_size * 8,
                image=False,
            )
            loss = self._add_kl_loss(loss, ret, kl_loss, data)
            if gd['z_dm']:
                z_dm_loss = self._z_dm_loss(z, data)
                loss = loss + z_dm_loss * self.loss_cfg.get('z_dm_loss', 1)
                ret['z_dm_loss'] = z_dm_loss.item()
            ret['loss'] = loss
            return z, ret

        raise NotImplementedError(f'Unsupported INFDBase mode for cleaned s0/s1/s2 code: {mode}')

    def update_dm_ema(self):
        if self.z_dm is None or self.z_dm_ema is None:
            return
        with torch.no_grad():
            for ema_p, cur_p in zip(self.z_dm_ema.parameters(), self.z_dm.parameters()):
                ema_p.data = ema_p.data * self.z_dm_ema_rate + cur_p.data * (1 - self.z_dm_ema_rate)

    def get_gd_from_opt(self, opt):
        opt = opt or {}
        return {
            'encoder': opt.get('encoder', False),
            'decoder': opt.get('decoder', False),
            'renderer': opt.get('renderer', False),
            'z_dm': (self.z_dm is not None) and (opt.get('encoder', False) or opt.get('z_dm', False)),
        }

