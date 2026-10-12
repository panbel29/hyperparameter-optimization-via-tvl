"""
cnn_train.py
------------
Training engine for the CNN HPO framework.

Why not use sklearn's cross_val_score?
---------------------------------------
sklearn's cross_val_score / GridSearchCV / RandomizedSearchCV expects estimators
that implement the sklearn API (fit / predict / score methods).  PyTorch CNNs do
not implement this interface.  Our cross_val_cnn() does the same thing sklearn
does internally, just with PyTorch loops instead:

    sklearn GridSearchCV internally:
        for each (train_idx, val_idx) in StratifiedKFold.split(X, y):
            estimator.fit(X[train_idx], y[train_idx])
            score = scorer(estimator, X[val_idx], y[val_idx])
        mean_score = mean(scores)

    cross_val_cnn() here:
        for each (train_idx, val_idx) in StratifiedKFold.split(X_np, y_np):
            model = build_cnn(...)
            model is trained with _train_epoch() for `epochs` iterations
            y_pred = _predict(model, val_loader)
            score = accuracy_score(y_val, y_pred)  OR  _compute_tvl(...)
        mean_score = mean(scores)

The logic is identical. The only difference is that we manage the training
loop ourselves (required for PyTorch) and compute the score explicitly.

Why use tvl_scorer.py's logic directly?
-----------------------------------------
tvl_scorer.py's tvl_scorer() function expects a fitted sklearn estimator with a
.talent_ attribute (set by TalentWrapper.fit()).  PyTorch models have no such
attribute.  Rather than wrapping the CNN in a sklearn-compatible shim (which
would be complex and fragile), we replicate the identical TvL formula in
_compute_tvl() using the same numpy RNG and the same mathematical operations.
The result is numerically identical to what tvl_scorer() would produce.

Training curve tracking
-----------------------
train_cnn_with_history() extends train_cnn() to record per-epoch training
accuracy and validation accuracy.  These are returned as lists and saved
alongside the results, enabling learning-curve plots for the optimal
accuracy-tuned and TvL-tuned models.
"""

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score
from typing import Optional

from cnn_model import build_cnn
from cnn_data import make_dataloader


# ─────────────────────────────────────────────────────────────────────────────
# Device
# ─────────────────────────────────────────────────────────────────────────────

def get_device() -> torch.device:
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# ─────────────────────────────────────────────────────────────────────────────
# Low-level train / predict helpers
# ─────────────────────────────────────────────────────────────────────────────

def train_epoch(model, loader, optimizer, criterion, device):
    """One full pass over the training DataLoader (one epoch)."""
    model.train()
    for xb, yb in loader:
        xb, yb = xb.to(device), yb.to(device)
        optimizer.zero_grad() # clear gradients from previous batch
        loss = criterion(model(xb), yb)
        loss.backward() # compute gradients via backprop
        optimizer.step() # update weights: w ← w - lr * grad


def predict(model, loader, device) -> np.ndarray:
    """Return hard class predictions (argmax) as a numpy int64 array."""
    model.eval()
    preds = []
    with torch.no_grad(): # no gradient tracking needed at eval time
        for xb, _ in loader:
            logits = model(xb.to(device))
            preds.append(logits.argmax(dim=1).cpu().numpy())
    return np.concatenate(preds)


def predict_proba(model, loader, device) -> np.ndarray:
    """Return class probabilities (softmax) as a float32 numpy array."""
    model.eval()
    probs = []
    with torch.no_grad():
        for xb, _ in loader:
            p = torch.softmax(model(xb.to(device)), dim=1).cpu().numpy()
            probs.append(p)
    return np.concatenate(probs)


# ─────────────────────────────────────────────────────────────────────────────
# TvL calculation — same formula as tvl_scorer.py, adapted for numpy arrays
# ─────────────────────────────────────────────────────────────────────────────

def compute_tvl(talent: float,
                 y_true: np.ndarray,
                 y_pred: np.ndarray,
                 m: int,
                 L: int,
                 random_state: int = 42) -> float:
    """
    Compute the TvL measure.

    This replicates tvl_scorer() from tvl_scorer.py exactly, but operates
    on pre-computed numpy arrays rather than a fitted sklearn estimator.
    The formula, RNG, and numerical result are identical.

    Parameters
    ----------
    talent     : training accuracy of the model on its own training fold
                 (equivalent to TalentWrapper.talent_)
    y_true     : ground-truth labels for the validation fold
    y_pred     : predicted labels for the validation fold
    m          : cardinality of each random subset (same m as in main.py)
    L          : number of random subsets (same L as in main.py)
    """
    if len(y_true) < m:
        return 0.0
    rng = np.random.default_rng(random_state)
    c = 0
    for _ in range(L):
        idx = rng.choice(len(y_true), size=m, replace=False)
        if accuracy_score(y_true[idx], y_pred[idx]) >= talent:
            c += int(np.sum(y_true[idx] == y_pred[idx]))
    return c / (L * m)


# ─────────────────────────────────────────────────────────────────────────────
# Full training run (used by tuners for the final refit and by the default eval)
# ─────────────────────────────────────────────────────────────────────────────

def train_cnn(dataset: str,
              X_train: torch.Tensor,
              y_train: torch.Tensor,
              epochs: int,
              lr: float,
              batch_size: int,
              dropout_prob: Optional[float] = None,
              momentum: float = 0.9,
              random_state: int = 42) -> nn.Module:
    """
    Train a CNN on the full training set and return the fitted model.

    Optimiser: SGD with momentum=0.9 (fixed, per TvL paper §III-A).
    Loss: CrossEntropyLoss.
    """
    torch.manual_seed(random_state)
    device = get_device()
    model = build_cnn(dataset, dropout_prob=dropout_prob).to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=momentum)
    criterion = nn.CrossEntropyLoss()
    loader    = make_dataloader(X_train, y_train, batch_size=batch_size, shuffle=True)

    for _ in range(epochs):
        train_epoch(model, loader, optimizer, criterion, device)
    return model


def train_cnn_with_history(dataset: str,
                           X_train: torch.Tensor,
                           y_train: torch.Tensor,
                           X_val:   torch.Tensor,
                           y_val:   torch.Tensor,
                           epochs: int,
                           lr: float,
                           batch_size: int,
                           dropout_prob: Optional[float] = None,
                           momentum: float = 0.9,
                           random_state: int = 42):
    """
    Train a CNN and record training accuracy and validation accuracy after
    every epoch.  Used to generate learning-curve plots for the optimal models.

    Returns
    -------
    model         : trained nn.Module
    train_history : list of per-epoch training accuracy (length = epochs)
    val_history   : list of per-epoch validation accuracy (length = epochs)
    """
    torch.manual_seed(random_state)
    device = get_device()

    model     = build_cnn(dataset, dropout_prob=dropout_prob).to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=momentum)
    criterion = nn.CrossEntropyLoss()

    tr_loader  = make_dataloader(X_train, y_train, batch_size=batch_size, shuffle=True) # for training
    tr_loader_eval = make_dataloader(X_train, y_train, batch_size=512, shuffle=False) # for evaluation
    val_loader = make_dataloader(X_val,   y_val,   batch_size=512,        shuffle=False)

    train_history, val_history = [], []
    y_tr_np  = y_train.numpy()
    y_val_np = y_val.numpy()

    for epoch in range(epochs):
        train_epoch(model, tr_loader, optimizer, criterion, device)

        # Evaluate on training set (model.eval() disables dropout)
        tr_preds  = predict(model, tr_loader_eval,  device)
        val_preds = predict(model, val_loader, device)

        train_acc = accuracy_score(y_tr_np,  tr_preds)  * 100
        val_acc   = accuracy_score(y_val_np, val_preds) * 100

        train_history.append(train_acc)
        val_history.append(val_acc)

        print(f"    epoch {epoch+1:3d}/{epochs}  "
              f"train_acc={train_acc:.2f}%  val_acc={val_acc:.2f}%")

    return model, train_history, val_history

def get_val_split(dataset: str,
                  X_train: torch.Tensor,
                  y_train: torch.Tensor,
                  random_state: int = 42):
    """
    Split the training tensor into train and validation subsets.

    Split sizes (matching official dataset structure):
        MNIST  / FMNIST : 50,000 train | 10,000 val  (from 60,000)
        CIFAR-10        : 40,000 train | 10,000 val  (from 50,000)

    Stratified so class ratios are preserved in both splits.
    The split is fixed — same random_state always produces the same split.
    """
    d = dataset.upper()
    test_size = 0.2   # same for all three datasets

    # train_test_split accepts only numpy arrays, so convert to numpy and back
    X_tr_np = X_train.numpy()
    y_tr_np = y_train.numpy()

    X_t, X_v, y_t, y_v = train_test_split(
        X_tr_np, y_tr_np,
        test_size=test_size,
        stratify=y_tr_np,
        random_state=random_state
    )

    X_t = torch.tensor(X_t, dtype=torch.float32)
    y_t = torch.tensor(y_t, dtype=torch.long)
    X_v = torch.tensor(X_v, dtype=torch.float32)
    y_v = torch.tensor(y_v, dtype=torch.long)

    print(f"[val_split] {d}  train={tuple(X_t.shape)}  val={tuple(X_v.shape)}")
    return X_t, y_t, X_v, y_v


def val_score(dataset: str,
                    X_tr: torch.Tensor, y_tr: torch.Tensor,
                    X_va: torch.Tensor, y_va: torch.Tensor,
                    epochs: int,
                    lr: float,
                    batch_size: int,
                    dropout_prob=None,
                    momentum: float = 0.9,
                    scoring: str = 'accuracy',
                    m: int = 100,
                    L: int = 100,
                    random_state: int = 42) -> float:
    """
    Train on X_tr, evaluate on X_va, return a single score.
    Replaces cross_val_cnn() — one model trained, one evaluation.
    """
    torch.manual_seed(random_state)
    device    = get_device()
    criterion = nn.CrossEntropyLoss()

    model     = build_cnn(dataset, dropout_prob=dropout_prob).to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=momentum)

    tr_loader = make_dataloader(X_tr, y_tr, batch_size=batch_size, shuffle=True) # for training
    tr_loader_eval = make_dataloader(X_tr, y_tr, batch_size=512, shuffle=False) # for evaluation
    va_loader = make_dataloader(X_va, y_va, batch_size=512, shuffle=False)

    for _ in range(epochs):
        train_epoch(model, tr_loader, optimizer, criterion, device)

    y_tr_pred = predict(model, tr_loader_eval, device)
    y_va_pred = predict(model, va_loader, device)
    y_tr_np   = y_tr.numpy()
    y_va_np   = y_va.numpy()

    if scoring == 'accuracy':
        return accuracy_score(y_va_np, y_va_pred)

    elif scoring == 'tvl':
        talent = accuracy_score(y_tr_np, y_tr_pred)
        tvl = compute_tvl(talent, y_va_np, y_va_pred,
                            m=m, L=L, random_state=random_state)
        return tvl
    else:
        raise ValueError(f"Unknown scoring '{scoring}'. Use 'accuracy' or 'tvl'.")

# ─────────────────────────────────────────────────────────────────────────────
# Evaluation on held-out test set  (mirrors get_measures from tvl_scorer.py)
# ─────────────────────────────────────────────────────────────────────────────

def evaluate_cnn(model: nn.Module, # The final model (after training)
                 X_train: torch.Tensor, y_train: torch.Tensor,
                 X_test:  torch.Tensor, y_test:  torch.Tensor) -> dict:
    """
    Compute the same metrics as get_measures() in tvl_scorer.py:
        training accuracy, test accuracy, generalisation error, AUROC.
    """
    device = get_device()
    model  = model.to(device)

    tr_loader = make_dataloader(X_train, y_train, 512, shuffle=False)
    te_loader = make_dataloader(X_test,  y_test,  512, shuffle=False)

    y_tr_np   = y_train.numpy()
    y_te_np   = y_test.numpy()
    y_tr_pred = predict(model, tr_loader, device)
    y_te_pred = predict(model, te_loader, device)
    y_te_prob = predict_proba(model, te_loader, device)

    train_acc = accuracy_score(y_tr_np, y_tr_pred) * 100
    test_acc  = accuracy_score(y_te_np, y_te_pred) * 100
    gen_err   = train_acc - test_acc
    f1 = f1_score(y_te_np, y_te_pred, average='macro') * 100

    n_classes = len(np.unique(y_te_np))
    if n_classes == 2:
        auroc = roc_auc_score(y_te_np, y_te_prob[:, 1])
    else:
        auroc = roc_auc_score(y_te_np, y_te_prob, multi_class='ovr')

    measures = {
        'training_accuracy':    round(float(train_acc), 4),
        'test_accuracy':        round(float(test_acc),  4),
        'generalization_error': round(float(gen_err),   4),
        'AUROC':                round(float(auroc),     4),
        'F1_Score':             round(float(f1),        4)
    }
    for k, v in measures.items():
        print(f"  {k}: {v}")
    return measures
