# ASIG: Arbitrary-Shaped Image Generation via Spherical Neural Field Diffusion

![teaser](assets/teaser.png)

**ASIG** is the official codebase for **Arbitrary-Shaped Image Generation via Spherical Neural Field Diffusion**.
It is designed to support high-quality image synthesis across diverse image shapes, including **perspective**, **panoramic**, and **fisheye** views, while enabling explicit control over spatial attributes such as **viewpoint**, **field-of-view (FOV)**, and **resolution**.

The framework is built on two key ideas from the paper:

- **Mesh-based spherical latent diffusion**, which generates a complete scene representation on the sphere and uses seam-aware denoising to maintain semantic and spatial consistency across viewpoints.
- **Spherical neural field sampling**, which extracts arbitrary regions from the scene representation with coordinate conditions, enabling flexible and distortion-aware image generation at different shapes and resolutions.

## 🌟 Features

![result](assets/Visualization.png)

- **Panoramic, Perspective, and Fisheye Generation**: ASIG supports high-quality image generation across three image forms within a unified framework: panoramic images, perspective images, and fisheye images.
- **Controllable Spatial Attributes**: ASIG enables explicit control over image resolution, camera viewpoint, and field-of-view (FOV) during generation.
- **Arbitrary-Shaped Image Generation**: In ASIG, "arbitrary-shaped" means not only supporting different projection types, but also flexibly generating images with different spatial attributes under a single generative framework.

## 🔨 Installation

We recommend the following environment:

- Linux
- Python `3.9`
- CUDA `11.8` (or another version compatible with your local PyTorch build)
- PyTorch `2.0.1`
- torchvision `0.15.2`

Create a fresh conda environment:

```bash
conda create -n ASIG python=3.9 -y
conda activate ASIG
```

Install PyTorch first. For CUDA 11.8, a typical command is:

```bash
pip install torch==2.0.1 torchvision==0.15.2 --index-url https://download.pytorch.org/whl/cu118
```


Install the repository requirements:

```bash
pip install -r requirements.txt
```

## 🚀 Train

Run all commands from the repository root.

### Training Data

The training dataset can be downloaded from [resources](https://pan.baidu.com/s/10-q5hfiGoQWiCP7B2WV6Xg?pwd=1234) with extraction code `1234`.

After downloading, place the dataset files under `dataset/`, or set:

```bash
export ASIG_DATASET_ROOT=/path/to/dataset
```

### Arg-map Generation

If the downloaded dataset package does not include the required arg-map `.mat` files, you can generate them with:

```bash
bash make_arg_map.sh dataset
bash make_arg_map.sh panorama
```

This step generates the geometric index / mapping files used by ASIG. In practice, the generated arg-map determines the target image resolution, camera viewpoint, field-of-view (FOV), and image type used for sampling, such as panoramic, perspective, or fisheye-style outputs.

#### Arg-map Indexing with L-subdivision

![L_sub](assets/L_sub.png)

This figure shows how the sphere is discretized before arg-map generation. ASIG first builds a subdivided icosahedron on the sphere, then uses the vertices / faces on this structure as geometric indices for later sampling and rendering.

In practice, the subdivision makes the spherical representation denser and more uniform, so each image location can be matched to a stable spherical index. This is why the arg-map can serve as a lookup table between image-space coordinates and the underlying spherical scene representation.

### S1 Training

`S1` uses the config file `cfgs/s1.yaml`.

```bash
bash s1.sh
```

Equivalent command:

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 torchrun --standalone --nproc-per-node=8 run.py \
  --cfg cfgs/s1.yaml \
  --save-root save/s1
```

### S2 Training

`S2` uses the config file `cfgs/s2.yaml` and loads the stage-2 related checkpoints under `checkpoints/`.

```bash
bash s2.sh
```

Equivalent command:

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 torchrun --standalone --nproc-per-node=8 run.py \
  --cfg cfgs/s2.yaml \
  --save-root save/s2
```

## 📒 Inference

The inference resources can be downloaded from [resources](https://pan.baidu.com/s/10-q5hfiGoQWiCP7B2WV6Xg?pwd=1234) with extraction code `1234`. The corresponding checkpoints will be uploaded later.

### Test

Run with:

```bash
bash test_cfg.sh
```

If you want to override the default paths:

```bash
MODEL_PATH=checkpoints/inference/model.pth \
CFG_SCALE=6 \
bash test_cfg.sh
```
