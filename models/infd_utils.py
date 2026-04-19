import os

import torch
import torch.distributed as dist
import torch.nn.functional as F


def detach_nested_tensors(value):
    if isinstance(value, dict):
        return {key: detach_nested_tensors(child) for key, child in value.items()}
    if isinstance(value, list):
        return [detach_nested_tensors(child) for child in value]
    if isinstance(value, tuple):
        return tuple(detach_nested_tensors(child) for child in value)
    if torch.is_tensor(value):
        return value.detach()
    return value


def downsample_loss_inputs(x, d_inp, d_gt_cell):
    gt_patch_res = torch.tensor([d_gt_cell.shape[1], d_gt_cell.shape[2]], dtype=torch.float32, device=d_inp.device)
    gt_whole_res = 2 / d_gt_cell[:, 0, 0, :]
    inp_res = torch.tensor([d_inp.shape[-2], d_inp.shape[-1]], dtype=torch.float32, device=d_inp.device)

    res = (gt_patch_res.view(1, 2) / gt_whole_res * inp_res.view(1, 2)).round()
    x = x * 0.5 + 0.5
    ret = []
    for i in range(res.shape[0]):
        t = F.interpolate(
            x[i].unsqueeze(0),
            size=(round(res[i][0].item()), round(res[i][1].item())),
            recompute_scale_factor=False,
            mode='bicubic',
            align_corners=False,
            antialias=True,
        )[0]
        ret.append((t - 0.5) / 0.5)
    return ret


def compute_disc_hinge_loss(disc, pred, target):
    logits_real = disc(target)
    logits_fake = disc(pred)
    loss_real = torch.mean(F.relu(1.0 - logits_real))
    loss_fake = torch.mean(F.relu(1.0 + logits_fake))
    loss = (loss_real + loss_fake) / 2
    return {
        'loss': loss,
        'disc_logits_real': logits_real.mean().item(),
        'disc_logits_fake': logits_fake.mean().item(),
    }


def compute_batch_psnr(pred, target):
    mse = ((target - pred) / 2).pow(2).mean(dim=[1, 2, 3])
    return (-10 * torch.log10(mse)).mean().item()


def calculate_adaptive_gan_weight(nll_loss, g_loss, last_layer):
    nll_grads = torch.autograd.grad(nll_loss, last_layer, retain_graph=True)[0]
    g_grads = torch.autograd.grad(g_loss, last_layer, retain_graph=True)[0]
    world_size = int(os.environ.get('WORLD_SIZE', '1'))
    if world_size > 1:
        dist.all_reduce(nll_grads, op=dist.ReduceOp.SUM)
        nll_grads.div_(world_size)
        dist.all_reduce(g_grads, op=dist.ReduceOp.SUM)
        g_grads.div_(world_size)
    d_weight = torch.norm(nll_grads) / (torch.norm(g_grads) + 1e-8)
    return torch.clamp(d_weight, 0.0, 1e4).detach()
