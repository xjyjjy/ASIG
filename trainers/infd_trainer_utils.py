import random

import torch
from scipy.io import loadmat


def collect_visualization_samples(dataset, sample_count):
    n = len(dataset)
    step = max(n // sample_count, 1)
    samples = [dataset.__getitem__(i, H=512, W=1024) for i in list(range(0, n, step))[:sample_count]]
    data = {key: torch.stack([sample[key] for sample in samples]).cuda() for key in samples[0].keys()}
    data['inp'] = (data['inp'] - 0.5) / 0.5
    data['gt'] = (data['gt'] - 0.5) / 0.5
    return data


def load_visualization_arg_maps(mat_paths):
    arg_maps = []
    dist_vectors = []
    for mat_path in mat_paths:
        mat = loadmat(mat_path)
        arg_maps.append(torch.from_numpy(mat['arg_map']).cuda().unsqueeze(0))
        dist_vectors.append(torch.from_numpy(mat['dist_vector']).cuda().unsqueeze(0))
    return torch.cat(arg_maps, dim=0), torch.cat(dist_vectors, dim=0)


def build_zoom_boxes(sample_count, zoom_range):
    if zoom_range is None:
        return None
    zoom_min, zoom_max = zoom_range
    zoom_boxes = []
    for _ in range(sample_count):
        length = 2 * (1 / random.uniform(zoom_min, zoom_max))
        x0 = random.uniform(-1, 1 - length)
        y0 = random.uniform(-1, 1 - length)
        zoom_boxes.append((x0, y0, length))
    return zoom_boxes
