"""
cnn_data.py
-----------
Data loading for MNIST, FMNIST, and CIFAR-10.

All three datasets are loaded via torchvision for consistency.
torchvision downloads and caches each dataset automatically on first call.

Expected cache layout (created automatically):
    Datasets/MNIST/
    Datasets/FMNIST/
    Datasets/CIFAR10/

Key design decisions
--------------------
Normalisation:
    Pixel values are first scaled to [0,1] by torchvision's ToTensor()
    transform, then z-scored per channel using statistics computed on
    the TRAINING SPLIT ONLY to prevent data leakage into the test set.

Stratified subsampling:
    Training sizes {1000, 5000, 10000, 30000} are taken with stratified
    sampling so the class ratio is preserved across all sizes.
    The full test set (10 000 images for all three datasets) is always
    used for evaluation, regardless of the training size.

Tensor format:
    (N, C, H, W) float32 — ready for PyTorch Conv2d.
    Labels are int64 LongTensors required by CrossEntropyLoss.

DataLoader:
    make_dataloader() wraps any (X, y) tensor pair in a PyTorch DataLoader.

    What a DataLoader does:
      - Divides the dataset into mini-batches of size *batch_size*.
      - If shuffle=True, it randomly re-orders the samples at the start
        of every epoch (important for training; never used for evaluation).
      - It is an iterator: each call to next() or each loop iteration
        yields one (X_batch, y_batch) pair and moves to the next batch.
      - When all batches are exhausted (one full pass = one epoch) it stops.
      - pin_memory=True (when GPU is available) pre-loads batches into
        CPU pinned (page-locked) memory so GPU transfer is faster.
      - num_workers=0 keeps loading in the main process to avoid pickling
        issues with PyTorch tensors across platforms.

    Why batch_size=512 for evaluation?
      During prediction we never compute gradients, so memory usage is
      much lower.  512 is large enough to saturate GPU throughput without
      risk of OOM.
"""

import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split
import torchvision
import torchvision.transforms as T
from pathlib import Path


# Stratified subsampling

def stratified_subsample(X: np.ndarray, y: np.ndarray,
                          n: int, random_state: int = 42):
    """
    Return (X_sub, y_sub) with exactly *n* samples, class proportions preserved.
    If n >= len(X), the full array is returned unchanged.
    """
    if n >= len(X):
        return X, y
    X_sub, _, y_sub, _ = train_test_split(
        X, y, train_size=n, stratify=y, random_state=random_state
    )
    return X_sub, y_sub


# Per-channel normalisation  (no data leakage: stats from train only)

def normalize(X_train: np.ndarray, X_test: np.ndarray):
    """
    Z-score normalisation per channel.
    X_train / X_test : float32 arrays of shape (N, C, H, W), already in [0,1].

    For MNIST/FMNIST: C=1, so this is a single global mean/std.
    For CIFAR-10:     C=3, so each colour channel is standardised independently.

    mean and std are computed ONLY from X_train and applied to both splits.
    This is identical to what StandardScaler does for tabular data in main.py.
    """
    mean = X_train.mean(axis=(0, 2, 3), keepdims=True)   # shape (1, C, 1, 1)
    std  = X_train.std( axis=(0, 2, 3), keepdims=True) + 1e-7
    return (X_train - mean) / std, (X_test - mean) / std


# torchvision loaders  (MNIST / FMNIST / CIFAR-10)

def torchvision_to_numpy(ds) -> tuple:
    """
    Convert a torchvision dataset to (X, y) numpy arrays.
    ToTensor() already scales uint8 pixels to float32 in [0,1].
    """
    loader = DataLoader(ds, batch_size=512, shuffle=False, num_workers=0)
    xs, ys = [], []
    for xb, yb in loader:
        xs.append(xb.numpy())
        ys.append(yb.numpy())
    return np.concatenate(xs), np.concatenate(ys)


def load_raw(dataset: str, root: str):
    """
    Download / load one of {MNIST, FMNIST, CIFAR10} via torchvision.
    Returns (X_train, y_train, X_test, y_test) as float32/int64 numpy arrays.
    X shape: (N, C, H, W) with values in [0, 1].
    """
    # ToTensor converts HxWxC uint8 [0,255] → CxHxW float32 [0,1]
    transform = T.ToTensor()
    d = dataset.upper()

    if d == 'MNIST':
        cache = str(Path(root) / 'MNIST')
        tr = torchvision.datasets.MNIST(cache, train=True,  download=True, transform=transform)
        te = torchvision.datasets.MNIST(cache, train=False, download=True, transform=transform)

    elif d == 'FMNIST':
        cache = str(Path(root) / 'FMNIST')
        tr = torchvision.datasets.FashionMNIST(cache, train=True,  download=True, transform=transform)
        te = torchvision.datasets.FashionMNIST(cache, train=False, download=True, transform=transform)

    elif d == 'CIFAR10':
        cache = str(Path(root) / 'CIFAR10')
        tr = torchvision.datasets.CIFAR10(cache, train=True,  download=True, transform=transform)
        te = torchvision.datasets.CIFAR10(cache, train=False, download=True, transform=transform)

    else:
        raise ValueError(f"Unknown dataset '{dataset}'.")

    X_train, y_train = torchvision_to_numpy(tr)
    X_test,  y_test  = torchvision_to_numpy(te)
    return X_train, y_train, X_test, y_test


# Public API

def load_cnn_dataset(dataset: str,
                     train_size: int,
                     root: str = 'Datasets',
                     random_state: int = 42):
    """
    Load and prepare one of {MNIST, FMNIST, CIFAR10} for CNN training.

    Parameters
    ----------
    dataset      : 'MNIST', 'FMNIST', or 'CIFAR10'
    train_size   : number of training samples after stratified subsampling.
                   Supported: 1000, 5000, 10000, 30000 (any int is accepted).
                   Pass None or 0 to use the full training set.
    root         : directory where torchvision caches raw data
    random_state : seed for subsampling reproducibility

    Returns
    -------
    X_train_t : torch.FloatTensor  (train_size, C, H, W)
    y_train_t : torch.LongTensor   (train_size,)
    X_test_t  : torch.FloatTensor  (N_test, C, H, W)
    y_test_t  : torch.LongTensor   (N_test,)
    """
    # 1. Download / load raw data
    X_tr, y_tr, X_te, y_te = load_raw(dataset, root)

    # 2. Stratified subsample training set
    if train_size and train_size < len(X_tr):
        X_tr, y_tr = stratified_subsample(X_tr, y_tr, train_size, random_state)

    # 3. Normalise per channel (train stats only)
    X_tr, X_te = normalize(X_tr, X_te)

    # 4. Convert to PyTorch tensors
    X_train_t = torch.tensor(X_tr, dtype=torch.float32)
    y_train_t = torch.tensor(y_tr, dtype=torch.long)
    X_test_t  = torch.tensor(X_te, dtype=torch.float32)
    y_test_t  = torch.tensor(y_te, dtype=torch.long)

    print(f"[cnn_data] {dataset.upper()}  "
          f"train={tuple(X_train_t.shape)}  test={tuple(X_test_t.shape)}")
    return X_train_t, y_train_t, X_test_t, y_test_t


def make_dataloader(X: torch.Tensor, y: torch.Tensor,
                    batch_size: int, shuffle: bool = True) -> DataLoader:
    """
    Wrap a (features, labels) tensor pair in a PyTorch DataLoader.

    The DataLoader:
      - Splits data into mini-batches of size *batch_size* automatically.
      - shuffle=True  → used during training; randomises order every epoch
                        so the model does not memorise the sample order.
      - shuffle=False → used during evaluation; order preserved for
                        index-aligned metric computation.
      - Each iteration yields (X_batch, y_batch) ready to send to the GPU.
      - pin_memory=True pre-loads batches into CPU page-locked memory when
        a GPU is available, making host-to-device transfer faster.
    """
    ds = TensorDataset(X, y)
    return DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=0,
        pin_memory=torch.cuda.is_available()
    )
