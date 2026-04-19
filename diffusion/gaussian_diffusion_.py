from typing import List, Optional, Union

import torch
import torch.nn.functional as F
from diffusers import DDIMScheduler, DDPMScheduler, DPMSolverMultistepScheduler
from tqdm import tqdm
from transformers import CLIPTextModel, CLIPTextModelWithProjection

from model_utils.padding import padding_v4_0


def rescale_noise_cfg(noise_cfg, noise_pred_text, guidance_rescale=0.0):
    std_text = noise_pred_text.std(dim=list(range(1, noise_pred_text.ndim)), keepdim=True)
    std_cfg = noise_cfg.std(dim=list(range(1, noise_cfg.ndim)), keepdim=True)
    noise_pred_rescaled = noise_cfg * (std_text / std_cfg)
    return guidance_rescale * noise_pred_rescaled + (1 - guidance_rescale) * noise_cfg


def retrieve_timesteps(
    scheduler,
    num_inference_steps: Optional[int] = None,
    device: Optional[Union[str, torch.device]] = None,
    timesteps: Optional[List[int]] = None,
    **kwargs,
):
    if timesteps is not None:
        scheduler.set_timesteps(timesteps=timesteps, device=device, **kwargs)
        timesteps = scheduler.timesteps
        return timesteps, len(timesteps)

    scheduler.set_timesteps(num_inference_steps, device=device, **kwargs)
    return scheduler.timesteps, num_inference_steps


class GaussianDiffusion_:
    def __init__(self):
        model_name = 'stabilityai/stable-diffusion-xl-base-1.0'
        self.noise_scheduler = DDPMScheduler.from_pretrained(model_name, subfolder='scheduler')
        self.noise_scheduler_ddim = DDIMScheduler.from_pretrained(model_name, subfolder='scheduler')
        self.noise_scheduler_DPMSolver = DPMSolverMultistepScheduler.from_pretrained(
            model_name,
            subfolder='scheduler',
            algorithm_type='dpmsolver++',
            use_karras_sigmas=True,
        )
        self.text_encoder_one = CLIPTextModel.from_pretrained(model_name, subfolder='text_encoder').cuda()
        self.text_encoder_two = CLIPTextModelWithProjection.from_pretrained(model_name, subfolder='text_encoder_2').cuda()
        self.text_encoder_one.requires_grad_(False)
        self.text_encoder_two.requires_grad_(False)
        self.pad_v4_0 = padding_v4_0([6])

    def encode_prompt(self, text_1, text_2, text_encoders):
        prompt_embeds_list = []
        with torch.no_grad():
            for text, text_encoder in zip([text_1, text_2], text_encoders):
                prompt_embeds = text_encoder(
                    text,
                    output_hidden_states=True,
                    return_dict=False,
                )
                pooled_prompt_embeds = prompt_embeds[0]
                prompt_embeds = prompt_embeds[-1][-2]
                batch_size, seq_len, _ = prompt_embeds.shape
                prompt_embeds_list.append(prompt_embeds.view(batch_size, seq_len, -1))

        prompt_embeds = torch.cat(prompt_embeds_list, dim=-1)
        pooled_prompt_embeds = pooled_prompt_embeds.view(batch_size, -1)
        return prompt_embeds, pooled_prompt_embeds

    def _model_forward(self, model, x, timesteps, prompt_embeds, added_cond_kwargs):
        output = model(
            x,
            timesteps,
            prompt_embeds,
            added_cond_kwargs=added_cond_kwargs,
            return_dict=False,
        )
        if isinstance(output, (tuple, list)):
            output = output[0]
        return output

    def _encode_train_prompt(self, text, text_2, batch_size, device):
        self.text_encoder_one.to(device)
        self.text_encoder_two.to(device)
        prompt_embeds, pooled_prompt_embeds = self.encode_prompt(
            text[:, 0, :],
            text_2[:, 0, :],
            [self.text_encoder_one, self.text_encoder_two],
        )
        prompt_embeds = prompt_embeds.unsqueeze(1).expand(-1, 10, -1, -1).reshape(batch_size * 10, 77, 2048).contiguous()
        pooled_prompt_embeds = pooled_prompt_embeds.unsqueeze(1).expand(-1, 10, -1).reshape(batch_size * 10, 1280).contiguous()
        return prompt_embeds.to(device), pooled_prompt_embeds.to(device)

    def training_losses_sd_pad_v1(
        self,
        model,
        x_start,
        text,
        text_2,
        t,
        noise=None,
        x_cond=None,
        model_kwargs=None,
        pad_size=8,
        bf16=True,
    ):
        del x_cond, model_kwargs, pad_size
        batch_size = x_start.shape[0]
        device = x_start.device
        prompt_embeds, pooled_prompt_embeds = self._encode_train_prompt(text, text_2, batch_size, device)

        if noise is None:
            noise = torch.randn_like(x_start)
        timesteps = t
        if timesteps is None:
            timesteps = torch.randint(
                0,
                self.noise_scheduler.config.num_train_timesteps,
                (batch_size,),
                device=device,
            ).unsqueeze(1).repeat(1, 10).view(-1).long()

        batch_size, box_num, channels, height, width = x_start.shape
        x_start = x_start.view(batch_size * box_num, channels, height, width)
        noise = noise.view(batch_size * box_num, channels, height, width)
        x_t = self.noise_scheduler.add_noise(x_start, noise, timesteps)
        x_t = self.pad_v4_0.get_padding(x_t, 6)
        target = self.pad_v4_0.get_padding(noise, 6)

        add_time_ids = torch.tensor(
            [[1024, 1024, 256, 256, 512, 512]],
            dtype=pooled_prompt_embeds.dtype,
            device=device,
        ).repeat(batch_size * 10, 1)
        added_cond_kwargs = {'text_embeds': pooled_prompt_embeds, 'time_ids': add_time_ids}

        if bf16:
            with torch.cuda.amp.autocast(dtype=torch.bfloat16):
                model_pred = self._model_forward(model, x_t, timesteps, prompt_embeds, added_cond_kwargs)
        else:
            model_pred = self._model_forward(model, x_t, timesteps, prompt_embeds, added_cond_kwargs)

        return {'loss': F.mse_loss(model_pred.float(), target.float(), reduction='mean')}

    def _build_sample_condition(self, text, text_2, batch_size, guidance_scale, full_image):
        do_classifier_free_guidance = guidance_scale > 1
        self.text_encoder_one.to(text.device)
        self.text_encoder_two.to(text.device)
        text_encoders = [self.text_encoder_one, self.text_encoder_two]

        if do_classifier_free_guidance:
            prompt_embeds, pooled_prompt_embeds = self.encode_prompt(text[:, 0, :], text_2[:, 0, :], text_encoders)
            negative_prompt_embeds, negative_pooled_prompt_embeds = self.encode_prompt(text[:, 1, :], text_2[:, 1, :], text_encoders)
            prompt_embeds = prompt_embeds.unsqueeze(1).expand(-1, 10, -1, -1).reshape(batch_size * 10, 77, 2048)
            pooled_prompt_embeds = pooled_prompt_embeds.unsqueeze(1).expand(-1, 10, -1).reshape(batch_size * 10, 1280)
            negative_prompt_embeds = negative_prompt_embeds.unsqueeze(1).expand(-1, 10, -1, -1).reshape(batch_size * 10, 77, 2048)
            negative_pooled_prompt_embeds = negative_pooled_prompt_embeds.unsqueeze(1).expand(-1, 10, -1).reshape(batch_size * 10, 1280)
            prompt_embeds = torch.cat([negative_prompt_embeds, prompt_embeds], dim=0)
            add_text_embeds = torch.cat([negative_pooled_prompt_embeds, pooled_prompt_embeds], dim=0)
        else:
            prompt_embeds, pooled_prompt_embeds = self.encode_prompt(text[:, 0, :], text_2[:, 0, :], text_encoders)
            prompt_embeds = prompt_embeds.unsqueeze(1).expand(-1, 10, -1, -1).reshape(batch_size * 10, 77, 2048)
            add_text_embeds = pooled_prompt_embeds.unsqueeze(1).expand(-1, 10, -1).reshape(batch_size * 10, 1280)

        if full_image:
            time_ids = [1024, 1024, 0, 0, 1024, 1024]
        else:
            time_ids = [1024, 1024, 256, 256, 512, 512]
        add_time_ids = torch.tensor([time_ids], dtype=add_text_embeds.dtype, device=text.device).repeat(batch_size * 10, 1)
        if do_classifier_free_guidance:
            add_time_ids = torch.cat([add_time_ids, add_time_ids], dim=0)

        return do_classifier_free_guidance, prompt_embeds.to(text.device), {'text_embeds': add_text_embeds.to(text.device), 'time_ids': add_time_ids}

    def sample_loop_ddim_pad_v1(
        self,
        model,
        text,
        text_2,
        shape,
        latents=None,
        guidance_scale=0.1,
        step=99,
        prog_=False,
        guidance_rescale=0.0,
        full_image=True,
        schedule='ddim',
        bf16=False,
        strength=None,
        padd_noise_size=None,
    ):
        del padd_noise_size
        if strength is not None and guidance_scale == 0.1:
            guidance_scale = strength

        batch_size = shape[0]
        schedule = self.noise_scheduler_ddim if schedule == 'ddim' else self.noise_scheduler_DPMSolver
        timesteps, num_inference_steps = retrieve_timesteps(schedule, step, text.device)
        if latents is None:
            latents = torch.randn(shape, dtype=torch.float32, device=text.device)
        latents = latents.view(shape[0] * shape[1], *shape[2:])

        do_classifier_free_guidance, prompt_embeds, added_cond_kwargs = self._build_sample_condition(
            text,
            text_2,
            batch_size,
            guidance_scale,
            full_image,
        )

        prog = {'sample': [], 'pred_xstart': []}
        progress_bar = tqdm(range(num_inference_steps))
        for timestep in timesteps:
            latent_model_input = torch.cat([latents] * 2) if do_classifier_free_guidance else latents
            latent_model_input = schedule.scale_model_input(latent_model_input, timestep)
            latent_model_input = self.pad_v4_0.get_padding(latent_model_input, 6)
            timestep = timestep.long()

            if bf16:
                with torch.cuda.amp.autocast(dtype=torch.bfloat16):
                    noise_pred = self._model_forward(model, latent_model_input, timestep, prompt_embeds, added_cond_kwargs).float()
            else:
                noise_pred = self._model_forward(model, latent_model_input, timestep, prompt_embeds, added_cond_kwargs).float()

            if do_classifier_free_guidance:
                noise_pred_uncond, noise_pred_cond = noise_pred.chunk(2)
                noise_pred = noise_pred_cond + guidance_scale * (noise_pred_cond - noise_pred_uncond)
                if guidance_rescale > 0.0:
                    noise_pred = rescale_noise_cfg(noise_pred, noise_pred_cond, guidance_rescale=guidance_rescale)

            latents = schedule.step(noise_pred, timestep, latents, return_dict=False)[0]
            progress_bar.update(1)
            if prog_:
                prog['sample'].append(noise_pred)
                prog['pred_xstart'].append(latents)

        latents = self.pad_v4_0.get_padding(latents, 6)
        if prog_:
            prog['sample'] = torch.stack(prog['sample'], dim=1)
            prog['pred_xstart'] = torch.stack(prog['pred_xstart'], dim=1)
            return prog
        return latents
