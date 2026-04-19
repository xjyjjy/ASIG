import os
import random
from contextlib import nullcontext

import torch
import torch.nn.functional as F
import torch.optim.lr_scheduler as lr_scheduler
import torchvision
from torch.cuda.amp import autocast

import utils
from trainers.base_trainer import BaseTrainer
from utils.lr_scheduler import get_cosine_schedule_with_warmup
from trainers.infd_trainer_utils import (
    build_zoom_boxes,
    collect_visualization_samples,
    load_visualization_arg_maps,
)

from .trainers import register


@register('infd_trainer')
class INFDTrainer(BaseTrainer):
    VIS_ARG_MAP_PATHS = [
        '/home/zwxionggroup/xiajy/dataset/arg_map_dataset_pano_level8/8_768_1024_60_80_0_0_arg_new_v2.mat',
        '/home/zwxionggroup/xiajy/dataset/arg_map_dataset_pano_level8/8_768_1024_60_80_30_0_arg_new_v2.mat',
        '/home/zwxionggroup/xiajy/dataset/arg_map_dataset_pano_level8/8_768_1024_60_80_-30_0_arg_new_v2.mat',
        '/home/zwxionggroup/xiajy/dataset/arg_map_dataset_pano_level8/8_768_1024_75_100_0_0_arg_new_v2.mat',
    ]

    def prepare_visualize(self):
        self.vis_spec = {}
        self.vis_spec['ds_samples'] = self.cfg.visualize.get('ds_samples', 0)
        self.vis_ds_samples = {}
        if self.vis_spec['ds_samples'] > 0 and self.datasets.get('val') is not None:
            self.vis_ds_samples['train'] = collect_visualization_samples(self.datasets['val'], self.vis_spec['ds_samples'])
            self.vis_ds_samples['val'] = collect_visualization_samples(self.datasets['val'], self.vis_spec['ds_samples'])

        self.vis_ae_center_zoom_res = self.cfg.visualize.get('ae_center_zoom_res')
        self.vis_spec['z_dm_samples'] = self.cfg.visualize.get('z_dm_samples', 0)
        self.vis_spec['z_dm_samples_zoom'] = self.cfg.visualize.get('z_dm_samples_zoom')
        self.vis_spec['bs'] = self.cfg.visualize.get('bs', 2)

    def make_datasets(self):
        super().make_datasets()
        self.vis_resolution = self.cfg.visualize.resolution
        if isinstance(self.vis_resolution, int):
            self.vis_resolution = (self.vis_resolution, self.vis_resolution)

        random.seed(0)
        self.prepare_visualize()
        if self.cfg.random_seed is not None:
            random.seed(self.cfg.random_seed + self.rank)
        else:
            random.seed()

    def make_model(self, model_spec=None, resumed=False):
        super().make_model(model_spec, resumed=resumed)
        for name, module in self.model.named_children():
            self.log(f'  .{name} {utils.compute_num_params(module)}')

        self.has_opt = {}
        if self.cfg.get('optimizers') is not None:
            for name in self.cfg.optimizers.keys():
                self.has_opt[name] = True

    def _build_scheduler(self, optimizer):
        scheduler_type = self.cfg.get('lr_schedulers_types')
        if scheduler_type is None:
            return lr_scheduler.StepLR(optimizer, step_size=self.cfg.get('lr_schedulers'), gamma=0.5)
        if scheduler_type == 'cosine':
            return get_cosine_schedule_with_warmup(
                optimizer,
                max_lr=self.cfg.get('max_lr'),
                min_lr=self.cfg.get('min_lr'),
                total_steps=self.cfg.get('total_steps'),
                warmup_steps=self.cfg.get('warmup_steps'),
            )
        raise NotImplementedError(f'Unsupported lr scheduler: {scheduler_type}')

    def _log_optimizer_params(self):
        for opt_name, optimizer in self.optimizers.items():
            n_params = sum(p.numel() for group in optimizer.param_groups for p in group['params'])
            self.log(f'Optimizer {opt_name}: #params={n_params}')

    def make_optimizers(self):
        self.optimizers = {}
        self.schedulers = {}
        for name, spec in self.cfg.optimizers.items():
            optimizer = utils.make_optimizer(self.model.get_params(name), spec)
            self.optimizers[name] = optimizer
            self.schedulers[name] = self._build_scheduler(optimizer)
        self._log_optimizer_params()

    def _use_s0_gan(self):
        return self.cfg.get('gan_start_after_iters') == -1

    def _use_renderer_gan(self):
        gan_start_iter = self.cfg.get('gan_start_after_iters')
        return gan_start_iter is not None and gan_start_iter >= 0 and self.iter > gan_start_iter

    def _step_non_disc_modules(self, loss, bf16=False):
        use_scaler = loss.dtype == torch.float16 and not bf16
        self.model_ddp.zero_grad(set_to_none=(bf16 or use_scaler))

        if use_scaler:
            self.scaler.scale(loss).backward()
        else:
            loss.backward()

        for name, optimizer in self.optimizers.items():
            if name == 'disc':
                continue
            if bf16 or use_scaler:
                torch.cuda.empty_cache()
            if use_scaler:
                self.scaler.step(optimizer)
            else:
                optimizer.step()

        if use_scaler:
            self.scaler.update()

        for scheduler in self.schedulers.values():
            scheduler.step()

    def _step_discriminator(self, data, ret, mode, bf16=False):
        if 'disc' not in self.optimizers:
            return

        with autocast(enabled=bf16, dtype=torch.bfloat16):
            d_ret = self.model_ddp(data, mode=mode, has_opt=self.has_opt, use_gan=True, ret=ret)

        loss = d_ret.pop('loss')
        ret['disc_loss'] = loss.item()
        ret.update(d_ret)

        self.optimizers['disc'].zero_grad(set_to_none=bf16)
        loss.backward()
        self.optimizers['disc'].step()

    def train_step(self, data, mode='loss', bp=True):
        use_gan_s0 = self._use_s0_gan()
        use_gan = self._use_renderer_gan()
        if use_gan_s0:
            mode = 'S0'

        torch.cuda.empty_cache()
        bf16_s0 = use_gan_s0 and torch.cuda.is_available()
        with autocast(enabled=bf16_s0, dtype=torch.bfloat16):
            ret = self.model_ddp(data, mode=mode, has_opt=self.has_opt, use_gan=(use_gan or use_gan_s0))

        if not isinstance(ret, dict) or 'loss' not in ret:
            return ret

        loss = ret.pop('loss')
        ret['loss'] = loss.item()

        if bp:
            self._step_non_disc_modules(loss, bf16=bf16_s0)
            if use_gan:
                self._step_discriminator(data, ret, mode='disc_loss')
            if use_gan_s0:
                self._step_discriminator(data, ret, mode='disc_loss_s0', bf16=bf16_s0)

        if self.model.z_dm_ema is not None:
            self.model.update_dm_ema()
        return ret

    def train_iter_start(self):
        hrft_iter = self.cfg.get('hrft_start_after_iters')
        if hrft_iter is not None and self.iter == hrft_iter + 1:
            self.train_loader = self.loaders['train_hrft']
            self.train_loader_sampler = self.loader_samplers['train_hrft']
            self.train_loader_epoch = 0
            self.train_batch_id = len(self.train_loader) - 1

    def run_training(self):
        super().run_training()

    def visualize(self, data=None):
        del data
        self.model_ddp.eval()
        with torch.no_grad():
            if self.is_master and self.vis_spec['ds_samples'] > 0 and self.model.z_dm is None:
                self.visualize_ae()
            if self.model.z_dm is not None and self.vis_spec['z_dm_samples'] > 0:
                self.visualize_z_dm_samples()
        self._barrier()

    def _load_visualization_arg_maps(self):
        return load_visualization_arg_maps(self.VIS_ARG_MAP_PATHS)

    def _sample_visualization_latent(self, dm, text, text_2, batch_size):
        if self.model.train_method != 'pad_v1':
            raise NotImplementedError(
                f'Current cleaned trainer only keeps pad_v1 sampling, got: {self.model.train_method}'
            )

        with torch.no_grad():
            return self.model.z_dp.sample_loop_ddim_pad_v1(
                dm,
                self.model.to_unet_uncondition,
                self.model.out,
                text,
                text_2,
                (batch_size, *self.model.z_shape),
                strength=7.5,
                step=1,
                padd_noise_size=16,
            )

    def _ddp_no_sync_context(self):
        return self.model_ddp.no_sync() if self.distributed else nullcontext()

    def generate_samples(self, dm, n, bs=1, zoom_boxes=None, text=None, text_2=None, arg_map=None, dist_vector=None, mode='ret', save_path=None, save_prefix='', data=None):
        del zoom_boxes, dist_vector, data
        bs = 1
        model = self.model
        gens = []
        gens_zoom = []
        to_pil = torchvision.transforms.ToPILImage()
        arg_maps, dist_vectors = self._load_visualization_arg_maps()

        for i in range(0, n, bs):
            bs_ = min(i + bs, n) - i
            text_1 = text.repeat(bs_, 1, 1)
            text_2_ = text_2.repeat(bs_, 1, 1)
            z_gen = self._sample_visualization_latent(dm, text_1, text_2_, bs_)

            self._barrier()
            with self._ddp_no_sync_context():
                device = self.model_ddp.module.z_dm_ema.device if self.distributed else self.model.z_dm_ema.device
                if self.distributed:
                    self.model_ddp.module.z_dm_ema.to('cpu')
                    self.model_ddp.module.z_dm.to('cpu')
                else:
                    self.model.z_dm_ema.to('cpu')
                    self.model.z_dm.to('cpu')
            self._barrier()

            with torch.inference_mode():
                torch.cuda.empty_cache()
                z_gen = model.decode_z(z_gen / model.scale_factor, arg_map=arg_map, pad_size=16 * 8)
                z_gen = F.interpolate(z_gen, size=(1024, 1023), mode='bicubic', align_corners=False)
                z_gen = model.inverse_mapping(z_gen)
                z_gen = F.interpolate(z_gen, size=(1024, 256), mode='bicubic', align_corners=False)

            for t in range(4):
                render_arg_map = arg_maps[t].unsqueeze(0).repeat(bs_, 1, 1)
                render_dist_vector = dist_vectors[t].unsqueeze(0).repeat(bs_, 1, 1, 1)
                with torch.no_grad():
                    x_gen = model.run_renderer(z_gen, render_arg_map, render_dist_vector, render_arg_map, render_dist_vector)
                if mode == 'ret':
                    gens.append(x_gen)
                elif mode == 'save':
                    for j in range(bs_):
                        fid = i + j + t
                        to_pil(((x_gen[j] + 1) / 2).clamp(0, 1)).save(os.path.join(save_path, save_prefix + f'{fid}.png'))

            gens_zoom.append(z_gen)
            self._barrier()
            with self._ddp_no_sync_context():
                if self.distributed:
                    self.model_ddp.module.z_dm_ema.to(device)
                    self.model_ddp.module.z_dm.to(device)
                else:
                    self.model.z_dm_ema.to(device)
                    self.model.z_dm.to(device)
            self._barrier()

            if mode == 'ret':
                gens = torch.cat(gens, dim=0)
                gens_zoom = torch.cat(gens_zoom, dim=0)
                return gens, gens_zoom

    def visualize_ae_(self, name, data, bs=1):
        gt = data['gt']
        n = data['inp'].shape[0]
        pred = []
        changed_gt = []
        center_zoom = []
        for i in range(0, n, bs):
            batch = {k: v[i:min(i + bs, n)] for k, v in data.items()}
            out = self.model(batch, mode='pred')
            if isinstance(out, tuple) and len(out) == 2:
                pred.append(out[0])
                changed_gt.append(out[1])
            else:
                pred.append(out)

            if self.vis_ae_center_zoom_res is not None:
                center_zoom.append(out[0] if isinstance(out, tuple) and len(out) == 2 else out)

        pred = torch.cat(pred, dim=0)
        changed_gt = torch.cat(changed_gt, dim=0) if len(changed_gt) > 0 else None
        if self.is_master:
            vimg = []
            if gt.shape == pred.shape:
                for i in range(len(gt)):
                    vimg.extend([pred[i], gt[i]])
            else:
                for i in range(len(gt)):
                    vimg.extend([pred[i], changed_gt[i]])
            vimg = torch.stack(vimg)
            vimg = torchvision.utils.make_grid(vimg, nrow=4, normalize=True, value_range=(-1, 1))
            self.log_image(name, vimg)

        if self.vis_ae_center_zoom_res is not None:
            center_zoom = torch.cat(center_zoom, dim=0)
            if self.is_master:
                vimg = []
                for i in range(len(gt)):
                    vimg.extend([center_zoom[i], center_zoom[i]])
                vimg = torch.stack(vimg)
                vimg = torchvision.utils.make_grid(vimg, nrow=4, normalize=True, value_range=(-1, 1))
                self.log_image(name + '_center_zoom', vimg)

    def visualize_ae(self):
        for split in ['val']:
            if self.vis_ds_samples.get(split) is None:
                continue
            self.visualize_ae_(split, self.vis_ds_samples[split])

    def visualize_z_dm_samples_(self, name, dm, zoom_boxes, text, text_2, arg_map, dist_vector, bs=2, data=None):
        del bs, data
        ret = self.generate_samples(
            dm,
            self.vis_spec['z_dm_samples'],
            bs=self.vis_spec['bs'],
            zoom_boxes=zoom_boxes,
            text=text[0].unsqueeze(0),
            text_2=text_2[0].unsqueeze(0),
            arg_map=arg_map,
            dist_vector=dist_vector,
            mode='ret',
        )
        gens = ret[0] if isinstance(ret, tuple) else ret
        if self.is_master:
            vimg = torchvision.utils.make_grid(gens, nrow=2, normalize=True, value_range=(-1, 1))
            self.log_image(name, vimg)

    def visualize_z_dm_samples(self):
        data = self.vis_ds_samples['val']
        text = data['text']
        text_2 = data['text_2']
        arg_map = data['arg_map']
        dist_vector = data['dist_vector']
        zoom_boxes = build_zoom_boxes(self.vis_spec['z_dm_samples'], self.vis_spec.get('z_dm_samples_zoom'))

        for name, dm in [('z_dm_samples', self.model.z_dm), ('z_dm_ema_samples', self.model.z_dm_ema)]:
            if dm is not None:
                self.visualize_z_dm_samples_(name, dm, zoom_boxes, text, text_2, arg_map, dist_vector)
