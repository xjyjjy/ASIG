import copy
import warnings

import torch


models = dict()


def register(name):
    def decorator(cls):
        models[name] = cls
        return cls
    return decorator


def load_sd_from_ckpt(ckpt, pop_keys=None):
    sd = torch.load(ckpt, map_location='cpu')["model"]["sd"]
    if pop_keys is not None:
        pop_keys_dot = tuple(_ + '.' for _ in pop_keys)
        pop_keys = set(pop_keys)
        for key in list(sd.keys()):
            if key in pop_keys or key.startswith(pop_keys_dot):
                sd.pop(key)
    return sd


def make(spec, args=None, load_sd=False, resumed=False):
    load_sd = True
    load_sd_vae = False
    load_sd_renderer = False
    load_sd_unet = False
    load_ema = False

    if spec.get('args') is None:
        spec['args'] = dict()
    model = models[spec['name']](**spec['args'])

    if spec.get('vae') is not None:
        sd_vae = load_sd_from_ckpt(spec['vae'], spec.get('load_ckpt_pop_keys'))
        load_sd_vae = True
    if spec.get('renderer') is not None:
        sd_renderer = load_sd_from_ckpt(spec['renderer'], spec.get('load_ckpt_pop_keys'))
        load_sd_renderer = True

    if spec.get('unet') is not None:
        sd_unet = load_sd_from_ckpt(spec['unet'], spec.get('load_ckpt_pop_keys'))
        load_sd_unet = True
    elif 'sd' in spec:
        sd_unet = spec['sd']
        load_sd_unet = True

    if spec.get('ema') is not None:
        sd_ema = load_sd_from_ckpt(spec['ema'], spec.get('load_ckpt_pop_keys'))
        load_ema = True
    elif 'sd' in spec:
        sd_ema = spec['sd']
        load_ema = True

    if load_sd:
        model_state_dict = model.state_dict()
        new_sd = {}

        if load_sd_vae:
            valid_keys = ["encoder", "quant_conv", "decoder", "post_quant_conv", "perc_loss", "disc"]
            for key, param in sd_vae.items():
                if not any(token in key for token in valid_keys):
                    continue
                if key not in model_state_dict:
                    continue
                model_param = model_state_dict[key]
                if param.shape == model_param.shape:
                    new_sd[key] = param
                else:
                    warnings.warn(f"Skipping key '{key}': Shape mismatch (ckpt: {param.shape}, model: {model_param.shape})")

        if load_sd_renderer:
            valid_keys = ["renderer", "perc_loss", "disc"]
            for key, param in sd_renderer.items():
                if not any(token in key for token in valid_keys):
                    continue
                if key not in model_state_dict:
                    continue
                model_param = model_state_dict[key]
                if param.shape == model_param.shape:
                    new_sd[key] = param
                else:
                    warnings.warn(f"Skipping key '{key}': Shape mismatch (ckpt: {param.shape}, model: {model_param.shape})")

        if load_sd_unet:
            for key, param in sd_unet.items():
                if 'z_dm.' not in key:
                    continue
                stripped_key = key.replace('z_dm.', '')
                for name, model_param in model_state_dict.items():
                    if stripped_key in name and 'z_dm.' in name:
                        if param.shape == model_param.shape:
                            new_sd[name] = param
                        else:
                            warnings.warn(f"Skipping unet key '{name}': Shape mismatch (ckpt: {param.shape}, model: {model_param.shape})")

        if load_ema:
            for key, param in sd_ema.items():
                if 'z_dm_ema' not in key:
                    continue
                stripped_key = key.replace('z_dm_ema.', '')
                for name, model_param in model_state_dict.items():
                    if stripped_key in name and 'z_dm_ema' in name:
                        if param.shape == model_param.shape:
                            new_sd[name] = param
                        else:
                            warnings.warn(f"Skipping EMA key '{name}': Shape mismatch (ckpt: {param.shape}, model: {model_param.shape})")

        model.load_state_dict(new_sd, strict=False)
    return model


def make_test_new_render(spec, args=None, load_sd=False):
    new_render = torch.load(
        '/home/zwxionggroup/xiajy/model_save/diffusion/sdxl_s1_bp_pad_v4_lmdb_panorama/ae_div2k_pan_arg_map_tri_loss_2_fd_S1_pad_v4_lmdb_panorama/last-model-40.pth',
        map_location='cpu'
    )['model']['sd']

    if load_sd:
        sd = spec['sd']
    elif spec.get('load_ckpt') is not None:
        sd = load_sd_from_ckpt(spec['load_ckpt'], spec.get('load_ckpt_pop_keys'))
        load_sd = True

    if spec.get('args') is None:
        spec['args'] = dict()
    if args is not None:
        model_args = copy.deepcopy(spec['args'])
        model_args.update(args)
        spec['args'] = model_args

    model = models[spec['name']](**spec['args'])

    if load_sd:
        model_state_dict = model.state_dict()
        new_sd = {}
        for key, param in sd.items():
            if key not in model_state_dict:
                warnings.warn(f"Key '{key}' not found in model state_dict.")
                continue
            model_param = model_state_dict[key]
            if 'renderer' in key:
                if key in new_render and new_render[key].shape == model_param.shape:
                    param = new_render[key]
                else:
                    warnings.warn(
                        f"[renderer] fallback: '{key}' missing or shape mismatch "
                        f"(new_render: {new_render.get(key).shape if key in new_render else 'N/A'} "
                        f"vs model: {model_param.shape})"
                    )
            if param.shape == model_param.shape:
                new_sd[key] = param
            else:
                warnings.warn(f"Skipping key '{key}': Shape mismatch (ckpt: {param.shape}, model: {model_param.shape})")
        model.load_state_dict(new_sd, strict=False)
    return model


@register('identity')
def make_identity():
    return torch.nn.Identity()
