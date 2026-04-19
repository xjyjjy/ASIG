import io
import json
import os
import random

import lmdb
import numpy as np
import torch
from PIL import Image
from datasets.data import Dataset
from torchvision import transforms
from transformers import AutoTokenizer

from datasets import register


IMAGE_EXTS = ('.png', '.PNG', '.jpg', '.JPG', '.jpeg', '.JPEG', '.webp')
SDXL_MODEL_NAME = 'stabilityai/stable-diffusion-xl-base-1.0'
NEGATIVE_PROMPT = 'blurry, low quality, bad anatomy, disfigured, watermark, text, cropped, deformed, noisy'


def _load_split_filenames(root_path, split_file=None, split_key=None, first_k=None):
    if split_file is None:
        filenames = sorted(os.listdir(root_path))
    else:
        with open(split_file, 'r') as f:
            filenames = json.load(f)[split_key]
    if first_k is not None:
        filenames = filenames[:first_k]
    return filenames


def _filter_files(root_path, filenames, suffixes):
    return [os.path.join(root_path, filename) for filename in filenames if filename.endswith(suffixes)]


def _load_lmdb_entry(raw_data):
    if raw_data is None:
        raise KeyError('LMDB entry is missing.')

    loaded = np.load(io.BytesIO(raw_data), allow_pickle=True)
    if hasattr(loaded, 'files'):
        return loaded

    if isinstance(loaded, np.ndarray) and loaded.dtype == object:
        loaded = loaded.item()
        if isinstance(loaded, dict):
            return loaded

    raise TypeError(f'Unsupported LMDB payload type: {type(loaded)}')


@register('360sp_pan_ae_argmap_fd')
class PanoramaArgMapDataset(Dataset):

    def __init__(self, root_path, arg_map_path=None, split_file=None, split_key=None, first_k=None, repeat=1, cache='none'):
        del arg_map_path, cache
        self.repeat = repeat
        filenames = _load_split_filenames(root_path, split_file=split_file, split_key=split_key, first_k=first_k)
        self.files_image = _filter_files(root_path, filenames, IMAGE_EXTS)
        self.to_tensor = transforms.ToTensor()

    def __len__(self):
        return len(self.files_image) * self.repeat

    def __getitem__(self, idx, H=None, W=None):
        image = self.to_tensor(Image.open(self.files_image[idx % len(self.files_image)]).convert('RGB'))
        return {
            'H': H,
            'W': W,
            'image': image,
        }


@register('360sp_pan_ae_argmap_fd_lmdb')
class PanoramaArgMapLmdbDataset(Dataset):

    def __init__(self, root_path, lmdb_path, fixed_fields=None, split_file=None, split_key=None, first_k=None, repeat=1, cache='none'):
        del fixed_fields, cache
        self.repeat = repeat
        self.env = lmdb.open(lmdb_path, readonly=True, lock=False, readahead=False, max_readers=256)

        filenames = _load_split_filenames(root_path, split_file=split_file, split_key=split_key, first_k=first_k)
        image_files = _filter_files(root_path, filenames, IMAGE_EXTS)
        self.keys = [os.path.splitext(os.path.basename(path))[0].encode() for path in image_files]

    def __len__(self):
        return len(self.keys) * self.repeat

    def __getitem__(self, idx):
        for offset in range(len(self.keys)):
            key = self.keys[(idx + offset) % len(self.keys)]
            with self.env.begin() as txn:
                raw_data = txn.get(key)
            if raw_data is None:
                continue
            try:
                sample = _load_lmdb_entry(raw_data)
            except Exception:
                continue
            if 'image_ori' not in sample or 'recon' not in sample:
                continue
            return {
                'image': torch.from_numpy(sample['image_ori']).float(),
                'recon': torch.from_numpy(sample['recon']).float(),
            }
        raise KeyError('No valid LMDB entry could be loaded for the requested sample range.')


class _BasePanoramaTextDataset(Dataset):

    def __init__(self, root_path, text_path, split_file=None, split_key=None, first_k=None, repeat=1, cache='none', text_drop_prob=0.0):
        del cache
        self.repeat = repeat
        self.text_drop_prob = text_drop_prob
        filenames = _load_split_filenames(root_path, split_file=split_file, split_key=split_key, first_k=first_k)
        self.files_image = _filter_files(root_path, filenames, IMAGE_EXTS)
        self.files_text = [os.path.join(text_path, os.path.splitext(os.path.basename(path))[0] + '.txt') for path in self.files_image]
        self.to_tensor = transforms.ToTensor()
        self.tokenizer = None
        self.tokenizer_two = None

    def __len__(self):
        return len(self.files_image) * self.repeat

    def _init_tokenizers(self):
        if self.tokenizer is None:
            self.tokenizer = AutoTokenizer.from_pretrained(SDXL_MODEL_NAME, subfolder='tokenizer')
            self.tokenizer_two = AutoTokenizer.from_pretrained(SDXL_MODEL_NAME, subfolder='tokenizer_2')

    def _load_text(self, idx):
        with open(self.files_text[idx % len(self.files_text)], 'r', encoding='utf-8') as f:
            text_data = f.read().strip()
        if self.text_drop_prob > 0 and random.random() < self.text_drop_prob:
            return ''
        return 'a photo of ' + text_data

    def _tokenize(self, text_data):
        self._init_tokenizers()
        tokenized = self.tokenizer(
            text_data,
            padding='max_length',
            max_length=self.tokenizer.model_max_length,
            truncation=True,
            return_tensors='pt',
        ).input_ids
        tokenized_2 = self.tokenizer_two(
            text_data,
            padding='max_length',
            max_length=self.tokenizer_two.model_max_length,
            truncation=True,
            return_tensors='pt',
        ).input_ids
        negative_tokenized = self.tokenizer(
            NEGATIVE_PROMPT,
            padding='max_length',
            max_length=self.tokenizer.model_max_length,
            truncation=True,
            return_tensors='pt',
        ).input_ids
        negative_tokenized_2 = self.tokenizer_two(
            NEGATIVE_PROMPT,
            padding='max_length',
            max_length=self.tokenizer_two.model_max_length,
            truncation=True,
            return_tensors='pt',
        ).input_ids
        return torch.cat([tokenized, negative_tokenized], dim=0), torch.cat([tokenized_2, negative_tokenized_2], dim=0)

    def __getitem__(self, idx, H=None, W=None):
        image = self.to_tensor(Image.open(self.files_image[idx % len(self.files_image)]).convert('RGB'))
        text, text_2 = self._tokenize(self._load_text(idx))
        return {
            'H': H,
            'W': W,
            'image': image,
            'text': text,
            'text_2': text_2,
        }


@register('360sp_pan_text')
class PanoramaTextDataset(_BasePanoramaTextDataset):

    def __init__(self, root_path, text_path=None, split_file=None, split_key=None, first_k=None, repeat=1, cache='none'):
        if text_path is None:
            raise ValueError('text_path is required for 360sp_pan_text')
        super().__init__(
            root_path=root_path,
            text_path=text_path,
            split_file=split_file,
            split_key=split_key,
            first_k=first_k,
            repeat=repeat,
            cache=cache,
            text_drop_prob=0.0,
        )


@register('360sp_pan_text_drop')
class PanoramaTextDropDataset(_BasePanoramaTextDataset):

    def __init__(self, root_path, text_path=None, split_file=None, split_key=None, first_k=None, repeat=1, cache='none'):
        if text_path is None:
            raise ValueError('text_path is required for 360sp_pan_text_drop')
        super().__init__(
            root_path=root_path,
            text_path=text_path,
            split_file=split_file,
            split_key=split_key,
            first_k=first_k,
            repeat=repeat,
            cache=cache,
            text_drop_prob=0.1,
        )
