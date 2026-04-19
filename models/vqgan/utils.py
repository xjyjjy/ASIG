import torch
import torch.nn as nn

from models import register
from diffusers import AutoencoderKL

from model_utils.padding import *

# @register('stable_diffusion')
# def make_sd_unet(**kwargs):
#     model_name = "stabilityai/sd-x2-latent-upscaler"
#     # model_name = "echarlaix/tiny-random-stable-diffusion-xl-refiner"
#     unet = UNet2DConditionModel.from_pretrained(model_name, subfolder="unet")
#     # print(unet)
#     lora_config = LoraConfig(
#     r=8,  
#     lora_alpha=32,  
#     target_modules=[
#         "q_proj", "k_proj", "v_proj", 
#         "conv",  
#         "linear"  
#     ],  
#     lora_dropout=0.1,  
#     )

#     unet = get_peft_model(unet, lora_config)
    
#     return unet

# @register('encoder_sd')
# def make_sd_unet(**kwargs):
#     model_name = "CompVis/stable-diffusion-v1-4" 
#     subfolder = "vae"
#     vae = AutoencoderKL.from_pretrained(model_name, subfolder="vae")

#     return vae.encoder

class conv_pad_filed(nn.Module):
    def __init__(self, conv: nn.Conv2d, name: str):
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
            # self.padding_v4_0 = padding_v4_0([5,6,7,8,9])
            self.padding_v4_0 = None
    def forward(self, x):
        bs, c, h, w = x.shape
        if h == 64:
            level = 5
        elif h == 128:
            level = 6
        elif h == 256:
            level = 7
        elif h == 512:
            level = 8
        elif h == 1024:
            level = 9
        if self.padding_v4_0 is None:
            self.padding_v4_0 = padding_v4_0([level])
        x = self.padding_v4_0.get_padding(x, level) # conmon_conv 0, sphere_conv 1
        x = self.conv(x)
        return x
    
# def add_conv_filed(model: nn.Module, prefix: str = ""):
#     for i, (name, module) in enumerate(model.named_children()):
#         full_name = f"{prefix}.{name}" if prefix else name
#         if isinstance(module, nn.Conv2d) and module.stride == (1,1):
#             print(f"Replacing conv_filed {full_name}")
#             setattr(model, name, conv_pad_filed(module, full_name)) 
#         else:
#             add_conv_filed(module, full_name)  # 递归处理子模块
    
def add_conv_filed(model: nn.Module, prefix: str = ""):
    for i, (name, module) in enumerate(model.named_children()):
        full_name = f"{prefix}.{name}" if prefix else name
        # if isinstance(module, nn.Conv2d) and module.stride == (1,1) and "resnets.0" in full_name:
        if isinstance(module, nn.Conv2d) and module.stride == (1,1):
            # print(f"Replacing conv_filed {full_name}")
            setattr(model, name, conv_pad_filed(module, full_name)) 
        else:
            add_conv_filed(module, full_name)  # 递归处理子模块


class SphericalVAEDecoder(nn.Module):
    def __init__(self, decoder: nn.Module, num_feature_levels: int = 4):
        super().__init__()
        self.decoder = decoder
        self.num_feature_levels = num_feature_levels

    def enable_gradient_checkpointing(self):
        if hasattr(self.decoder, "gradient_checkpointing"):
            self.decoder.gradient_checkpointing = True

    def disable_gradient_checkpointing(self):
        if hasattr(self.decoder, "gradient_checkpointing"):
            self.decoder.gradient_checkpointing = False

    def forward(self, sample, latent_embeds=None, return_features: bool = False):
        if not return_features:
            return self.decoder(sample, latent_embeds)

        sample = self.decoder.conv_in(sample)

        upscale_dtype = next(iter(self.decoder.up_blocks.parameters())).dtype
        sample = self.decoder.mid_block(sample, latent_embeds)
        sample = sample.to(upscale_dtype)

        decoder_features = []
        for up_block in self.decoder.up_blocks:
            sample = up_block(sample, latent_embeds)
            decoder_features.append(sample)

        if latent_embeds is None:
            image = self.decoder.conv_norm_out(sample)
        else:
            image = self.decoder.conv_norm_out(sample, latent_embeds)
        image = self.decoder.conv_act(image)
        image = self.decoder.conv_out(image)

        if not return_features:
            return image

        if len(decoder_features) > self.num_feature_levels:
            decoder_features = decoder_features[-self.num_feature_levels:]

        return {
            "sample": image,
            "features": decoder_features,
        }
    
# @register('encoder_sd')
# def make_sd_unet(**kwargs):
#     model_name = "sd-legacy/stable-diffusion-v1-5" 
#     subfolder = "vae"
#     vae = AutoencoderKL.from_pretrained(model_name, subfolder="vae")

    
#     # lora_config = LoraConfig(
#     # r=2,  
#     # lora_alpha=2,  
#     # target_modules = ["to_q", "to_k", "to_v", "to_out.0", "conv_in", "conv_out"],
#     # init_lora_weights="gaussian"
#     # )



#     # encoder = get_peft_model(vae.encoder, lora_config)
#     # for name, module in encoder.named_modules():
#     #     print(name)
# # 
#     # return encoder
#     return vae.encoder
@register('encoder_sdxl')
def make_sd_encoder_sdxl(**kwargs):
    model_name = "stabilityai/stable-diffusion-xl-base-1.0"
    vae = AutoencoderKL.from_pretrained(model_name, subfolder="vae")

    return vae.encoder
    
@register('quant_conv_sdxl')
def make_sd_quant_conv_sdxl(**kwargs):
    model_name = "stabilityai/stable-diffusion-xl-base-1.0"
    vae = AutoencoderKL.from_pretrained(model_name, subfolder="vae")
    return vae.quant_conv
    
    
@register('post_quant_conv_sdxl')
def make_sd_post_quant_conv_sdxl(**kwargs):
    model_name = "stabilityai/stable-diffusion-xl-base-1.0"
    vae = AutoencoderKL.from_pretrained(model_name, subfolder="vae")
    return vae.post_quant_conv
    
    
@register('decoder_sdxl')
def make_sd_decoder_sdxl(gradient_checkpointing=True, **kwargs):
    model_name = "stabilityai/stable-diffusion-xl-base-1.0"
    vae = AutoencoderKL.from_pretrained(model_name, subfolder="vae")
    decoder = vae.decoder
    add_conv_filed(decoder)
    decoder = SphericalVAEDecoder(decoder)
    if gradient_checkpointing:
        decoder.enable_gradient_checkpointing()
    return decoder
