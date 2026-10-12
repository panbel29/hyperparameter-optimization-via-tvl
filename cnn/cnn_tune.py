"""
cnn_tune.py
-----------
Hyperparameter optimisation for CNNs.

Three methods
  1. Random Search   — 50 iterations, log-uniform LR, consistent with
                       RandomizedSearchCV used for tabular classifiers.
  2. Bayesian (TPE)  — Optuna / Tree-structured Parzen Estimator,
                       same sampler as bayesian_mlp() / bayesian_svm().
  3. Nelder-Mead     — scipy derivative-free simplex search,
                       validated on MNIST and CIFAR-10 (Albelwi 2017).

Hyperparameter search space (identical across all three methods)
  epochs       : int   in [5, 50]
  lr           : float in [1e-4, 1e-1]  (log scale)
  batch_size   : categorical  {16, 32, 64, 128, 256}
  dropout_prob : categorical  {None, 0.4, 0.5, 0.6}

m values
--------
m is passed in from cnn_main.py via the CLI (--m flag).
Typical values used in the TvL paper and this project:
  m = 5, 10, 50, 100  (explored across experiments)
Default is m=100, L=100 (consistent with main.py defaults).
m is NOT tuned; it is a fixed experimental condition, the same way
test_size is fixed per experiment in the tabular study.

Training curve capture
----------------------
After tuning is complete, each optimal model (accuracy-tuned and TvL-tuned)
is re-trained with train_cnn_with_history(), which records per-epoch training
accuracy and validation accuracy.  These lists are saved in the results dict
and can be passed directly to plot.py for learning-curve graphs.
"""

import warnings
import numpy as np
import optuna
from optuna.samplers import TPESampler
from scipy.optimize import minimize
import torch
from typing import Optional

from cnn_train import (val_score, train_cnn,
                       train_cnn_with_history, evaluate_cnn)

optuna.logging.set_verbosity(optuna.logging.WARNING)
warnings.filterwarnings('ignore')


# ─────────────────────────────────────────────────────────────────────────────
# Search-space constants
# ─────────────────────────────────────────────────────────────────────────────

BATCH_SIZES   = [16, 32, 64, 128, 256]   # extended from original {32,64,128}
DROPOUT_PROBS = [None, 0.4, 0.5, 0.6]   # None ≡ A1; float ≡ A2
EPOCH_MIN, EPOCH_MAX = 15, 50  # EPOCH_MAX = 30 for MNIST/FMNIST, 50 for CIFAR-10
LR_MIN,    LR_MAX    = 1e-4, 1e-1


# ─────────────────────────────────────────────────────────────────────────────
# Shared helpers
# ─────────────────────────────────────────────────────────────────────────────

def refit(dataset, X_train, y_train, X_val, y_val, X_test, y_test,
                        best_params, random_state):
    """
    Refit on full training set, then evaluate.
    X_test is used as the 'validation' split for the training-curve plot
    (i.e. the held-out test set — same approach as TvL paper Fig. 1).
    Returns (model, measures, train_history, val_history).
    """
    # Re-train best hp configuration with train_cnn_with_history() to capture learning curves
    _, train_history, val_history = train_cnn_with_history(
        dataset=dataset,
        X_train=X_train, y_train=y_train,
        X_val=X_val, y_val=y_val,
        epochs=best_params['epochs'],
        lr=best_params['lr'],
        batch_size=best_params['batch_size'],
        dropout_prob=best_params['dropout_prob'],
        momentum=0.9,
        random_state=random_state # make sure to pass random_state for reproducibility of learning curves
    )

    X_train_full = torch.cat([X_train, X_val], dim=0)
    y_train_full = torch.cat([y_train, y_val], dim=0)
    # Refit on full training set (train + val) with best hyperparameters, then evaluate on test set
    model_final = train_cnn(
        dataset=dataset,
        X_train=X_train_full, y_train=y_train_full,
        epochs=best_params['epochs'],
        lr=best_params['lr'],
        batch_size=best_params['batch_size'],
        dropout_prob=best_params['dropout_prob'],
        momentum=0.9,
        random_state=random_state
    )
    measures = evaluate_cnn(model_final, X_train_full, y_train_full, X_test, y_test)
    return model_final, measures, train_history, val_history


# ─────────────────────────────────────────────────────────────────────────────
# 1. Random Search
# ─────────────────────────────────────────────────────────────────────────────

def random_search_cnn(dataset: str,
                      X_train: torch.Tensor, y_train: torch.Tensor,
                      X_val: torch.Tensor, y_val: torch.Tensor,
                      X_test:  torch.Tensor, y_test:  torch.Tensor,
                      n_iter: int = 50,
                      scoring: str = 'accuracy',
                      m: int = 2, L: int = 100,
                      random_state: int = 42):
    """
    Uniform random search — 50 iterations by default.

    Sampling strategy mirrors RandomizedSearchCV:
      epochs       : uniform integer in [EPOCH_MIN, EPOCH_MAX]
      lr           : log-uniform in [LR_MIN, LR_MAX]
                     (= uniform in log space, same as loguniform() in mlp_tune.py)
      batch_size   : uniform draw from BATCH_SIZES
      dropout_prob : uniform draw from DROPOUT_PROBS

    Returns
    -------
    model, best_params, best_score, tuned_measures, train_history, val_history, history
    """
    rng = np.random.default_rng(random_state)
    best_score  = -np.inf
    best_params = None
    history     = []

    print(f"\n[Random Search | {scoring}]  n_iter={n_iter}")

    for i in range(n_iter):
        epochs       = int(rng.integers(EPOCH_MIN, EPOCH_MAX + 1))
        lr           = float(np.exp(rng.uniform(np.log(LR_MIN), np.log(LR_MAX))))
        batch_size   = int(rng.choice(BATCH_SIZES))
        dropout_prob = rng.choice([str(d) for d in DROPOUT_PROBS])
        dropout_prob = None if dropout_prob == 'None' else float(dropout_prob)

        score = val_score(dataset, X_train, y_train, X_val, y_val, epochs, lr, batch_size,
                          dropout_prob, 0.9, scoring, m, L, random_state)
        history.append({'epochs': epochs, 'lr': lr, 'batch_size': batch_size,
                        'dropout_prob': dropout_prob, 'score': score})

        if score > best_score:
            best_score  = score
            best_params = {'epochs': epochs, 'lr': lr,
                           'batch_size': batch_size, 'dropout_prob': dropout_prob}

        print(f"  iter {i+1:3d}/{n_iter} | ep={epochs:2d} lr={lr:.2e} "
              f"bs={batch_size:3d} drop={str(dropout_prob):<5} | "
              f"score={score:.4f}{'  ← best' if score == best_score else ''}")

    print(f"\n  Best: {best_score:.4f}  params: {best_params}")
    best_model, tuned_measures, tr_hist, va_hist = refit(
        dataset, X_train, y_train, X_val, y_val, X_test, y_test, best_params, random_state)
    return best_model, best_params, best_score, tuned_measures, tr_hist, va_hist, history


# ─────────────────────────────────────────────────────────────────────────────
# 2. Bayesian Optimisation (TPE via Optuna)
# ─────────────────────────────────────────────────────────────────────────────

def bayesian_cnn(dataset: str,
                 X_train: torch.Tensor, y_train: torch.Tensor,
                 X_val: torch.Tensor, y_val: torch.Tensor,
                 X_test:  torch.Tensor, y_test:  torch.Tensor,
                 n_trials: int = 50,
                 scoring: str = 'accuracy',
                 m: int = 2, L: int = 100,
                 random_state: int = 42):
    """
    Bayesian optimisation (TPE).  Same sampler as bayesian_mlp().
    """
    def objective(trial):
        epochs       = trial.suggest_int('epochs', EPOCH_MIN, EPOCH_MAX)
        lr           = trial.suggest_float('lr', LR_MIN, LR_MAX, log=True)
        batch_size   = trial.suggest_categorical('batch_size', BATCH_SIZES)
        dropout_str  = trial.suggest_categorical(
            'dropout_prob', ['None', '0.4', '0.5', '0.6'])
        dropout_prob = None if dropout_str == 'None' else float(dropout_str)
        score = val_score(dataset, X_train, y_train, X_val, y_val, epochs, lr, batch_size,
                         dropout_prob, 0.9, scoring, m, L, random_state)
        return score

    sampler = TPESampler(seed=random_state)
    study   = optuna.create_study(direction='maximize', sampler=sampler)
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)

    raw         = study.best_trial.params
    best_score  = study.best_trial.value
    best_params = {
        'epochs':       raw['epochs'],
        'lr':           raw['lr'],
        'batch_size':   raw['batch_size'],
        'dropout_prob': None if raw['dropout_prob'] == 'None'
                             else float(raw['dropout_prob']),
    }
    print(f"\n[Bayesian | {scoring}]  best: {best_score:.4f}  params: {best_params}")
    best_model, tuned_measures, tr_hist, va_hist = refit(
        dataset, X_train, y_train, X_val, y_val, X_test, y_test, best_params, random_state)
    return best_model, best_params, best_score, tuned_measures, tr_hist, va_hist


# ─────────────────────────────────────────────────────────────────────────────
# 3. Nelder-Mead
# ─────────────────────────────────────────────────────────────────────────────
#
# Implementation explanation
# --------------------------
# Nelder-Mead operates on a continuous vector v of length n (= 4 hyperparams).
# Each element of v is a continuous proxy for one hyperparameter:
#
#   v[0]  epochs      — continuous proxy; decoded by rounding to [EPOCH_MIN, EPOCH_MAX]
#   v[1]  log(lr)     — continuous; decoded by exp(v[1]), clipped to [LR_MIN, LR_MAX]
#   v[2]  batch raw   — continuous; decoded by nearest value in BATCH_SIZES
#   v[3]  dropout idx — continuous; decoded by rounding to nearest index
#
# This encoding is necessary because scipy's NM works in real-valued R^n space.
# Discrete hyperparameters are handled by rounding at decode time, which is the
# standard practice (Albelwi et al. 2017 on CIFAR-10 and MNIST).
#
# The objective function
# ----------------------
# NM *minimises*, so we return -score.  Internally scipy tracks the negative
# score, so the vertex with the lowest negative score = highest actual score.
#
# The simplex
# -----------
# A simplex in R^4 has n+1 = 5 vertices.  We initialise:
#   - Row 0: the centre of each parameter's range (the starting guess)
#   - Rows 1-4: the centre with one dimension perturbed (standard NM init)
# This gives the algorithm a well-spread initial shape to work from.
#
# The four operations (applied to the worst vertex each iteration)
# ────────────────────────────────────────────────────────────────
#   Reflection  : mirror worst vertex through centroid of the rest
#                 → moves away from bad region
#   Expansion   : if reflection improved greatly, go further in that direction
#                 → accelerates towards promising region
#   Contraction : if reflection didn't help, pull towards centroid
#                 → zooms in near best region
#   Shrinkage   : if all above fail, shrink all vertices towards the best
#                 → last resort; collapses simplex around current best
#
# Stopping criteria (scipy defaults overridden here)
# ───────────────────────────────────────────────────
#   maxiter : max number of objective function calls (= CNN training runs)
#   xatol   : stop when simplex diameter (max vertex distance) < 1e-2
#   fatol   : stop when objective value improvement across iteration < 1e-3
#
# Best-tracking
# -------------
# scipy's NM does not explicitly track the best-ever solution across all calls
# (it only tracks the current simplex).  We track best_score / best_params
# manually inside the closure so we never lose a good solution found during
# exploration (reflection/expansion) that was later abandoned.
# ─────────────────────────────────────────────────────────────────────────────

STEP_SCALE = np.array([2.0, 0.5, 1.5, 1.5])  # [epochs, log(lr), batch_idx, dropout_idx]


def decode_nm_vector(v: np.ndarray):
    """
    Decode a continuous NM vector to valid discrete hyperparameter values.

    v[0] → epochs      : rounded int, clipped to [EPOCH_MIN, EPOCH_MAX]
    v[1] → lr          : exp(v[1]), clipped to [LR_MIN, LR_MAX]
    v[2] → batch_idx    : rounded int, clipped to [0, len(BATCH_SIZES)-1]   ← CHANGED
    v[3] → dropout_idx : rounded int, clipped to [0, len(DROPOUT_PROBS)-1]

    NOTE (fix, see thesis discussion §X): v[2] now indexes BATCH_SIZES directly,
    matching the dropout_idx encoding. The previous "nearest raw value" decode
    placed boundaries 16-64 units apart in raw space, far beyond the simplex's
    perturbation scale, which froze batch_size at its initial decode (128) for
    every NM run regardless of dataset (confirmed empirically in cluster logs).
    """
    epochs       = int(np.clip(round(v[0]), EPOCH_MIN, EPOCH_MAX))
    lr           = float(np.clip(np.exp(v[1]), LR_MIN, LR_MAX))
    bs_idx       = int(np.clip(round(v[2]), 0, len(BATCH_SIZES) - 1))
    batch_size   = BATCH_SIZES[bs_idx]
    dp_idx       = int(np.clip(round(v[3]), 0, len(DROPOUT_PROBS) - 1))
    dropout_prob = DROPOUT_PROBS[dp_idx]
    return epochs, lr, batch_size, dropout_prob


def nelder_mead_cnn(dataset: str,
                    X_train: torch.Tensor, y_train: torch.Tensor,
                    X_val: torch.Tensor, y_val: torch.Tensor,
                    X_test:  torch.Tensor, y_test:  torch.Tensor,
                    max_iter: int = 55,
                    scoring: str = 'accuracy',
                    m: int = 2, L: int = 100,
                    random_state: int = 42):
    """
    Nelder-Mead simplex optimisation for CNN hyperparameters.

    Each call to the objective function = one training run on the fixed
    80/20 train/validation split (val_score()).  max_iter=55 means at most
    55 such CNN training runs, comparable to the 50 RS/Bayesian iterations.
    """
    rng = np.random.default_rng(random_state)

    # Centre of each dimension's VALID RANGE (not raw hyperparameter values).
    # batch_idx and dropout_idx both live in index space [0, len-1], so both
    # centre at (len-1)/2 — consistent encoding across the two categorical dims.
    center = np.array([
        (EPOCH_MIN + EPOCH_MAX) / 2.0,            # epochs ≈ 17.5
        (np.log(LR_MIN) + np.log(LR_MAX)) / 2.0,  # log-LR centre
        (len(BATCH_SIZES) - 1) / 2.0,             # batch_idx ≈ 2.0  (was raw value ≈ 99.2)
        (len(DROPOUT_PROBS) - 1) / 2.0            # dropout_idx ≈ 1.5
    ])

    # Build initial simplex: n+1 = 5 rows
    # Row 0 = centre; rows 1-4 = centre with one dim perturbed by ±STEP_SCALE[i]
    n   = len(center)
    sim = [center.copy()]
    for i in range(n):
        v    = center.copy()
        step = rng.uniform(0.7, 1.3) * STEP_SCALE[i] * (1 if rng.random() > 0.5 else -1)
        v[i] += step
        sim.append(v)
    initial_simplex = np.array(sim)   # shape (5, 4)

    # Sanity check: confirm the 5 initial vertices decode to distinct
    # hyperparameter combinations before spending compute on them.
    print(f"\n[NM init check | {dataset}] decoded initial simplex:")
    for k, v in enumerate(initial_simplex):
        ep, lr_, bs, dp = decode_nm_vector(v)
        print(f"  vertex {k}: epochs={ep:2d} lr={lr_:.2e} bs={bs:3d} drop={str(dp):<5}")

    # Best-ever tracking (NM may abandon good vertices during exploration)
    call_count  = [0]
    best_score  = [-np.inf]
    best_params = [None]

    def objective(v):
        """Return NEGATIVE score (scipy minimises)."""
        epochs, lr, batch_size, dropout_prob = decode_nm_vector(v)
        score = val_score(dataset, X_train, y_train, X_val, y_val, epochs, lr, batch_size,
                          dropout_prob, 0.9, scoring, m, L, random_state)
        call_count[0] += 1

        if score > best_score[0]:
            best_score[0]  = score
            best_params[0] = {'epochs': epochs, 'lr': lr,
                              'batch_size': batch_size, 'dropout_prob': dropout_prob}
        print(f"  NM call {call_count[0]:3d} | ep={epochs:2d} lr={lr:.2e} "
              f"bs={batch_size:3d} drop={str(dropout_prob):<5} | "
              f"score={score:.4f}{'  ← best' if score == best_score[0] else ''}")
        return -score

    print(f"\n[Nelder-Mead | {scoring}]  max_iter={max_iter}")
    result = minimize(
        objective,
        x0=center,
        method='Nelder-Mead',
        options={
            'maxiter':         max_iter,
            'xatol':           1e-2,
            'fatol':           1e-3,
            'initial_simplex': initial_simplex,
            'disp':            False,
        }
    )
    print(f"\n  Best: {best_score[0]:.4f}  params: {best_params[0]}")
    print(f"  scipy: {result.message}")

    best_model, tuned_measures, tr_hist, va_hist = refit(
        dataset, X_train, y_train, X_val, y_val, X_test, y_test, best_params[0], random_state)
    return best_model, best_params[0], best_score[0], tuned_measures, tr_hist, va_hist

    