# CIFAR-100 CNN Baseline

This project now includes a minimal PyTorch CNN training pipeline for CIFAR-100.

## What is included

- Data loading from CIFAR-100 pickle file (`src/data/loader.py`)
- Basic preprocessing utilities (`src/data/preprocess.py`)
- Baseline CNN model (`src/models/cnn.py`)
- Train + validation loop with model checkpointing (`src/training/train.py`)

## Setup

From `cifar100-project`:

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

## Train

Default dataset path expects:

- `../CV_dataset/train`

Run:

```bash
python -m src.training.train --epochs 10 --batch-size 128 --lr 0.001
```

If your dataset path is different:

```bash
python -m src.training.train --dataset "C:/path/to/train"
```

The best model is saved to:

- `models/simple_cnn_cifar100.pt`
