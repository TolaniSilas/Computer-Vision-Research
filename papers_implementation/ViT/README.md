# Vision Transformer (ViT)

This directory contains the PyTorch implementation of the Vision Transformer (ViT) architecture from the paper: [*an image is worth 16x16 words*](https://arxiv.org/abs/2010.11929). The ViT model architecture was trained on the STL10 dataset for experimentation and model architecture replication.

---

## To Setup:
Follow thoroughly the installation and implementation guide to replicate this project.

### Installation Guide

#### 1. install uv

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

#### 2. clone the repository

```bash
git clone https://github.com/TolaniSilas/Computer-Vision-Research.git

cd papers_implementation/ViT
```

#### 3. initialize uv and create virtual environment

```bash
uv init

uv venv
```

#### 4. activate the virtual environment

```bash
source .venv/bin/activate
```

#### 5. install dependencies

```bash
uv sync
```

this installs all libraries defined in `pyproject.toml`.

---

## GPU Acceleration (recommended for faster training time)

this implementation was built and tested using cpu only. for faster training and longer experimentation, a gpu is strongly recommended. install the cuda-compatible pytorch version that matches the cpu version utilized in this project. Alternatively, you can check the notebook in this ViT directory for the libraries' cuda version.

```bash
# example for cuda 12.8.
uv add torch==2.10.0+cu128 --default-index https://download.pytorch.org/whl/cu128

uv add torchvision==0.25.0+cu128
```

check available cuda versions at [pytorch.org](https://pytorch.org/get-started/locally/).

---

## Training

run a sanity check first to confirm everything is working, as expected:

```bash
bash scripts/train.sh --sanity_check
```

once confirmed, run full training:

```bash
bash scripts/train.sh
```

training settings such as epochs, learning rate, and batch size can be configured via command line arguments:
```bash
bash scripts/train.sh --num_epochs 20 --lr 1e-4 --train_batch_size 32
```

run `python3 -m training.train --help` to see all available arguments.


## Testing

run all tests to verify the model and components are working correctly:
```bash
python3 -m pytest tests/ -v
```