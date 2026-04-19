import torch
import torch.nn as nn
from diffusers import UNet2DConditionModel

import importlib.util
import sys
from pathlib import Path

from model_utils.padding import padding_v4_0
from .gaussian_diffusion_ import GaussianDiffusion_


def register(name):
    def decorator(factory):
        registry_module = sys.modules.get('models.models')
        if registry_module is None:
            module_path = Path(__file__).resolve().parents[1] / 'models' / 'models.py'
            spec = importlib.util.spec_from_file_location('models.models', module_path)
            registry_module = importlib.util.module_from_spec(spec)
            sys.modules['models.models'] = registry_module
            spec.loader.exec_module(registry_module)
        registry_module.models[name] = factory
        return factory
    return decorator


class ConvPadField(nn.Module):
    def __init__(self, conv: nn.Conv2d):
        super().__init__()
        self.conv = nn.Conv2d(
            in_channels=conv.in_channels,
            out_channels=conv.out_channels,
            kernel_size=conv.kernel_size,
            stride=conv.stride,
            padding=conv.padding,
            dilation=conv.dilation,
            groups=conv.groups,
            bias=(conv.bias is not None),
        )
        self.conv.load_state_dict(conv.state_dict())
        self.padding = None

    def forward(self, x):
        height = x.shape[2]
        if height == 128:
            level = 6
        elif height == 64:
            level = 5
        elif height == 32:
            level = 4
        elif height == 16:
            level = 3
        elif height == 8:
            level = 2
        else:
            raise ValueError(f'unsupported feature height for spherical padding: {height}')

        if self.padding is None:
            self.padding = padding_v4_0([level])
        return self.conv(self.padding.get_padding(x, level))


def add_conv_field(model: nn.Module):
    for name, module in model.named_children():
        if isinstance(module, nn.Conv2d) and module.stride == (1, 1):
            setattr(model, name, ConvPadField(module))
        else:
            add_conv_field(module)


@register('stable_diffusion_xl_v2_1')
def make_sdxl_unet(**kwargs):
    del kwargs
    model_name = 'stabilityai/stable-diffusion-xl-base-1.0'
    unet = UNet2DConditionModel.from_pretrained(model_name, subfolder='unet')
    add_conv_field(unet)
    unet.enable_gradient_checkpointing()
    unet.enable_xformers_memory_efficient_attention()
    return unet


class StableDiffusionProcess(GaussianDiffusion_):
    def training_losses_sd_pad_v1(self, model, fuse=None, out=None, *args, **kwargs):
        del fuse, out
        return super().training_losses_sd_pad_v1(model, *args, **kwargs)

    def sample_loop_ddim_pad_v1(self, model, fuse=None, out=None, *args, **kwargs):
        del fuse, out
        return super().sample_loop_ddim_pad_v1(model, *args, **kwargs)


def make_diffusion_train_components_sd(dp_args=None):
    if dp_args is None:
        dp_args = {}
    return StableDiffusionProcess(**dp_args)


def get_dm_loss_sd_pad_v1(dp, dm, fuse, out, x, noise=None, text=None, text_2=None, t=None, x_cond=None, model_kwargs=None):
    dm_losses = dp.training_losses_sd_pad_v1(
        dm,
        fuse,
        out,
        x,
        text,
        text_2,
        t,
        noise=noise,
        x_cond=x_cond,
        model_kwargs=model_kwargs,
    )
    return dm_losses['loss']
