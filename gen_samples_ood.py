import argparse
import multiprocessing
import os
import random

import numpy as np
import torch
import torch.multiprocessing as mp
import torch.nn.functional as F
from scipy.io import loadmat
from torchvision import transforms
from tqdm import tqdm
from transformers import AutoTokenizer

import models


def set_global_seed(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_prompt_sd(tokenizer, tokenizer_two, text_data):
    text_data = 'a photo of ' + text_data
    negative_prompt = 'blurry low quality bad anatomy disfigured watermark text cropped deformed noisy'

    tokenized = tokenizer(
        text_data,
        padding='max_length',
        max_length=tokenizer.model_max_length,
        truncation=True,
        return_tensors='pt',
    ).input_ids
    second_tokenized = tokenizer_two(
        text_data,
        padding='max_length',
        max_length=tokenizer_two.model_max_length,
        truncation=True,
        return_tensors='pt',
    ).input_ids
    negative_tokenized = tokenizer(
        negative_prompt,
        padding='max_length',
        max_length=tokenizer.model_max_length,
        truncation=True,
        return_tensors='pt',
    ).input_ids
    second_negative_tokenized = tokenizer_two(
        negative_prompt,
        padding='max_length',
        max_length=tokenizer_two.model_max_length,
        truncation=True,
        return_tensors='pt',
    ).input_ids

    text_token = torch.cat([tokenized, negative_tokenized], dim=0).unsqueeze(0)
    second_text_token = torch.cat([second_tokenized, second_negative_tokenized], dim=0).unsqueeze(0)
    return text_token.cuda(), second_text_token.cuda()


def load_arg_maps(mat_paths):
    arg_maps = []
    dist_vectors = []
    for mat_path in mat_paths:
        mat = loadmat(mat_path)
        arg_maps.append(torch.from_numpy(mat['arg_map']).cuda())
        dist_vectors.append(torch.from_numpy(mat['dist_vector']).cuda())
    return arg_maps, dist_vectors


def prepare_text_files(text_list, local_seed):
    txt_files = [filename for filename in os.listdir(text_list) if filename.endswith('.txt')]
    if local_seed is not None:
        rng = random.Random(local_seed)
        rng.shuffle(txt_files)
    return txt_files


def render_from_latent(model, z_dec, n, arg_maps, dist_vectors):
    z_dec = model.decode_z(z_dec / model.scale_factor, arg_map=arg_maps[0].unsqueeze(0).repeat(n, 1, 1), pad_size=16 * 8)
    zdec = F.interpolate(z_dec, size=(1024, 1023), mode='bicubic', align_corners=False)
    z_dec_big = model.inverse_mapping(zdec)
    z_dec_small = F.interpolate(z_dec_big, size=(1024, 256), mode='bicubic', align_corners=False)
    z_dec_big = F.interpolate(z_dec_big, size=(2048, 512), mode='bicubic', align_corners=False)

    outputs = []
    for index in range(len(arg_maps)):
        arg_map = arg_maps[index].unsqueeze(0).repeat(n, 1, 1)
        dist_vector = dist_vectors[index].unsqueeze(0).repeat(n, 1, 1, 1)
        source = z_dec_small if index <= 4 else z_dec_big
        outputs.append(model.run_renderer(source, arg_map, dist_vector, arg_map, dist_vector))
    return outputs


def build_mat_paths():
    mat_paths = ['/home/zwxionggroup/xiajy/dataset/arg_map_panorama/1024_2048_arg_new_v2.mat']
    for theta in np.linspace(180, -180, 4, endpoint=False):
        mat_paths.append(f'/home/zwxionggroup/xiajy/dataset/arg_map_dataset_pano_level8/8_768_1024_60_80_{int(theta)}_0_arg_new_v2.mat')
    mat_paths.append('/home/zwxionggroup/xiajy/dataset/arg_map_panorama_level9/1024_2048_arg_new_v2.mat')
    for theta in np.linspace(180, -180, 4, endpoint=False):
        mat_paths.append(f'/home/zwxionggroup/xiajy/dataset/arg_map_for_test/arg_map_2048_512/8_768_1024_60_80_{int(theta)}_0_arg_new_v2.mat')
    return mat_paths


def prepare_output_dirs(save_path):
    for dirname in ['pano_small', 'pano_big', 'pano_ema_small', 'pano_ema_big', 'persp_small', 'persp_big']:
        os.makedirs(os.path.join(save_path, dirname), exist_ok=True)


def save_outputs(outputs, to_pil_img, save_path, basename, ema):
    for index, output in enumerate(outputs):
        images = (output * 0.5 + 0.5).clamp(0, 1)
        if index == 0:
            dirname = 'pano_ema_small' if ema else 'pano_small'
            suffix = ''
        elif 1 <= index <= 4:
            dirname = 'persp_small'
            suffix = f'_view_{index - 1}'
        elif index == 5:
            dirname = 'pano_ema_big' if ema else 'pano_big'
            suffix = ''
        else:
            dirname = 'persp_big'
            suffix = f'_view_{index - 1}'
        for image_index in range(images.shape[0]):
            to_pil_img(images[image_index]).save(os.path.join(save_path, dirname, f'{basename}{suffix}.png'))


def dm_worker(rank, tokenizer, tokenizer_2, text_list, args, save_path, ema=True):
    torch.cuda.set_device(rank)
    model = models.make_test_new_render(torch.load(args.model, map_location='cpu')['model'], load_sd=True).cuda()
    model.eval()

    if ema:
        dm = model.z_dm_ema
        del model.z_dm
    else:
        dm = model.z_dm
        del model.z_dm_ema

    mat_paths = build_mat_paths()
    arg_maps, dist_vectors = load_arg_maps(mat_paths)
    txt_files = prepare_text_files(text_list, args.local_seed)
    to_pil_img = transforms.ToPILImage()

    for _, filename in tqdm(enumerate(txt_files)):
        basename = os.path.splitext(filename)[0]
        check_path = os.path.join(save_path, ('pano_ema_big' if ema else 'pano_big'), f'{basename}.png')
        if os.path.exists(check_path):
            continue

        with open(os.path.join(text_list, filename), 'r', encoding='utf-8') as file:
            content = file.read()
        text_1, text_2 = get_prompt_sd(tokenizer, tokenizer_2, content)
        n = 1

        with torch.no_grad():
            z_dec = model.z_dp.sample_loop_ddim_pad_v1(
                dm,
                model.to_unet_uncondition,
                model.out,
                text_1,
                text_2,
                (n, *model.z_shape),
                guidance_scale=args.cfg_scale,
                step=30,
                guidance_rescale=0,
                full_image=False,
                schedule='dpms',
                bf16=args.bf16,
            )
            outputs = render_from_latent(model, z_dec, n, arg_maps, dist_vectors)
            save_outputs(outputs, to_pil_img, save_path, basename, ema)
        torch.cuda.empty_cache()


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', '-m')
    parser.add_argument('--batch-size', '-b', type=int, default=1)
    parser.add_argument('--cfg_scale', type=int, default=5)
    parser.add_argument('--bf16', action='store_true')
    parser.add_argument('--outdir', '-o')
    parser.add_argument('--output-sizes')
    parser.add_argument('--latent-shape')
    parser.add_argument('--patch-size', type=int, default=1024)
    parser.add_argument('--patch-padding', type=int, default=8)
    parser.add_argument('--text-data')
    parser.add_argument('--ema', default='true')
    parser.add_argument('--mode')
    parser.add_argument('--FOV')
    parser.add_argument('--THETA')
    parser.add_argument('--PHI')
    parser.add_argument('--seed', type=int, default=None)
    parser.add_argument('--local-seed', type=int, default=None)
    return parser.parse_args()


def main():
    args = parse_args()
    if args.seed is not None:
        set_global_seed(int(args.seed))

    multiprocessing.set_start_method('spawn', force=True)
    save_path = os.path.join(args.outdir, args.output_sizes)
    os.makedirs(save_path, exist_ok=True)
    prepare_output_dirs(save_path)

    model_name = 'stabilityai/stable-diffusion-xl-base-1.0'
    tokenizer = AutoTokenizer.from_pretrained(model_name, subfolder='tokenizer')
    tokenizer_two = AutoTokenizer.from_pretrained(model_name, subfolder='tokenizer_2')

    n_gpus = 1
    use_ema = str(args.ema).lower() not in {'false', '0', 'no'}

    if n_gpus == 1:
        dm_worker(0, tokenizer, tokenizer_two, args.text_data, args, save_path, ema=use_ema)
    else:
        processes = []
        for rank in range(n_gpus):
            process = mp.Process(target=dm_worker, args=(rank, tokenizer, tokenizer_two, args.text_data, args, save_path, use_ema))
            process.start()
            processes.append(process)
        for process in processes:
            process.join()


if __name__ == '__main__':
    main()
