"""
cnn_main.py
-----------
End-to-end CNN HPO pipeline — entry point for all CNN experiments.

Usage
-----
# Single run: MNIST, random search, accuracy, train size 5000
python cnn_main.py --dataset MNIST --method random_search --score accuracy --mt 5000

# TvL scoring with m=100 (m is a fixed experimental condition, not tuned)
python cnn_main.py --dataset CIFAR10 --method bayesian --score tvl --mt 10000 --m 100 --L 100

# All train sizes, all methods (full batch mode)
python cnn_main.py --dataset FMNIST --method all --score tvl --mt all --m 100 --L 100

m values
--------
m is passed via --m.  Typical values from TvL paper experiments: 5, 10, 50, 100.
To sweep over m values, run cnn_main.py multiple times with different --m values
(the same way the tabular experiments sweep over test sizes and m separately).
m is saved in the results JSON so all runs are fully reproducible.

Training curves
---------------
Per-epoch training accuracy and validation accuracy are saved in the results JSON
under keys 'default_train_history', 'default_val_history',
'tuned_train_history', 'tuned_val_history'.
These lists can be plotted directly (x-axis = epoch number, y-axis = accuracy %).
"""

import argparse
import json
from datetime import datetime
from pathlib import Path

import numpy as np
import torch

from cnn_data import load_cnn_dataset
from cnn_train import get_val_split
from cnn_tune import (
    random_search_cnn,
    bayesian_cnn,
    nelder_mead_cnn
)
from tvl_scorer import convert_to_python_types


TRAIN_SIZES = [1000, 5000, 10000, 30000, 50000] # m_t values
M_VALUES = [2, 4, 8, 16, 32, 64, 128] # m values for TvL scoring (fixed experimental condition, not tuned)


# ─────────────────────────────────────────────────────────────────────────────
# Result saving (mirrors main.py folder structure)
# ─────────────────────────────────────────────────────────────────────────────

def save_results(out_root, dataset, method, score, mt, m, results):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename  = f"{dataset}_CNN_{method}_{score}_mt{mt}_{timestamp}"

    if score == 'tvl':
        folder = (Path(out_root) / dataset / 'CNN' / method / score
                  / f'm_{m}' / f'mt_{mt}')
    else:
        folder = Path(out_root) / dataset / 'CNN' / method / score / f'mt_{mt}'

    folder.mkdir(parents=True, exist_ok=True)
    json_path = folder / f'{filename}.json'
    txt_path  = folder / f'{filename}.txt'

    with open(json_path, 'w') as f:
        json.dump(convert_to_python_types(results), f, indent=4)

    with open(txt_path, 'w') as f:
        f.write('='*60 + '\n')
        f.write(f"CNN HPO  |  {dataset}  |  {method}  |  {score}\n")
        f.write(f"Train size: {mt}   Timestamp: {timestamp}\n")
        if score == 'tvl':
            f.write(f"TvL params: m={m}  L={results.get('L')}\n")
        f.write('='*60 + '\n\n')

        f.write("FIXED STRUCTURAL PARAMETERS (TvL paper):\n")
        f.write("  Optimiser: SGD  momentum=0.9\n")
        f.write("  Loss: CrossEntropyLoss\n")
        f.write("  CV: 5-fold stratified\n\n")

        for section, key in [
            ("BEST HYPERPARAMETERS",    "best_params"),
            ("TUNED MODEL METRICS",     "tuned_metrics")
        ]:
            f.write(f"{section}:\n")
            for k, v in results[key].items():
                f.write(f"  {k}: {v}\n")
            if key == "best_params":
                f.write(f"  best_val_score: {results['best_val_score']:.4f}\n")
            f.write('\n')

    print(f"  ✓ {json_path}")
    print(f"  ✓ {txt_path}")
    return str(json_path)


# ─────────────────────────────────────────────────────────────────────────────
# Single experiment
# ─────────────────────────────────────────────────────────────────────────────

def run_experiment(dataset, method, score, mt, m, L,
                   n_iter, n_trials, nm_iter,
                   root, out_root, seed):

    print(f"\n{'='*65}")
    print(f"  CNN HPO | {dataset}  {method}  {score}  mt={mt}  m={m}")
    print(f"{'='*65}")

    # Load data
    X_train, y_train, X_test, y_test = load_cnn_dataset(
        dataset=dataset, train_size=mt, root=root, random_state=seed)

    X_tr, y_tr, X_val, y_val = get_val_split(dataset, X_train, y_train, random_state=seed)

    # Tuning
    print(f"\n--- Tuning: {method} ---")

    if method == 'random_search':
        (_, best_params, best_score,
         tuned_metrics, tr_hist, va_hist, _) = random_search_cnn(
            dataset, X_tr, y_tr, X_val, y_val, X_test, y_test,
            n_iter=n_iter, scoring=score, m=m, L=L, random_state=seed)

    elif method == 'bayesian':
        (_, best_params, best_score,
         tuned_metrics, tr_hist, va_hist) = bayesian_cnn(
            dataset, X_tr, y_tr, X_val, y_val, X_test, y_test,
            n_trials=n_trials, scoring=score, m=m, L=L, random_state=seed)

    elif method == 'nelder_mead':
        (_, best_params, best_score,
         tuned_metrics, tr_hist, va_hist) = nelder_mead_cnn(
            dataset, X_tr, y_tr, X_val, y_val, X_test, y_test,
            max_iter=nm_iter, scoring=score, m=m, L=L, random_state=seed)

    else:
        raise ValueError(f"Unknown method '{method}'.")

    # Package results (including training curves for plotting)
    results = {
        'dataset':               dataset,
        'classifier':            'CNN',
        'method':                method,
        'scoring':               score,
        'train_size':            mt,
        'm':                     m if score == 'tvl' else None,
        'L':                     L if score == 'tvl' else None,
        'n_iter':                n_iter   if method == 'random_search' else None,
        'n_trials':              n_trials if method == 'bayesian'      else None,
        'nm_max_iter':           nm_iter  if method == 'nelder_mead'   else None,
        'momentum_fixed':        0.9,
        'seed':                  seed,
        'best_params':           best_params,
        'best_val_score':        best_score,
        'tuned_metrics':         tuned_metrics,
        'tuned_train_history':   tr_hist,
        'tuned_val_history':     va_hist,
    }

    save_results(out_root, dataset, method, score, mt, m, results)
    return results


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def parse_mt(s):
    if s.lower() == 'all':
        return TRAIN_SIZES
    v = int(s)
    if v not in TRAIN_SIZES:
        print(f"[warn] mt={v} not in standard sizes {TRAIN_SIZES}; using anyway.")
    return [v]

def parse_m(s):
    """Accept '--m all' to sweep all M_VALUES, or '--m <int>' for a single value."""
    if s.lower() == 'all':
        return M_VALUES
    return [int(s)]

def main():
    parser = argparse.ArgumentParser(description='CNN HPO pipeline')
    parser.add_argument('--dataset',  required=True,
                        choices=['MNIST', 'FMNIST', 'CIFAR10'])
    parser.add_argument('--method',   required=True,
                        choices=['random_search', 'bayesian', 'nelder_mead', 'all'])
    parser.add_argument('--score',    required=True,
                        choices=['accuracy', 'tvl'])
    parser.add_argument('--mt',       default='all',
                        help='Training size: 1000|5000|10000|30000|60000')
    parser.add_argument('--m',        default='all',
                        help='TvL cardinality m (e.g. 2,4,8,16,32,64,128)')
    parser.add_argument('--L',        type=int, default=100,
                        help='TvL sampling replicates L')
    parser.add_argument('--n_iter',   type=int, default=50)
    parser.add_argument('--n_trials', type=int, default=50)
    parser.add_argument('--nm_iter',  type=int, default=55)
    parser.add_argument('--seed',     type=int, default=42)
    parser.add_argument('--root',     default='Datasets')
    parser.add_argument('--out',      default='Results/CNN')
    args = parser.parse_args()

    train_sizes = parse_mt(args.mt)
    methods = (['random_search', 'bayesian', 'nelder_mead']
               if args.method == 'all' else [args.method])
    m_values = parse_m(args.m) 
    if args.score == 'accuracy':
        m_values = [None]   # m is irrelevant for accuracy; run the loop only once

    total = len(train_sizes) * len(methods) * len(m_values)
    done  = 0
    for mt in train_sizes:
        for method in methods:
            for m in m_values:
                run_experiment(
                    dataset=args.dataset, method=method, score=args.score,
                    mt=mt, m=m, L=args.L,
                    n_iter=args.n_iter, n_trials=args.n_trials,
                    nm_iter=args.nm_iter,
                    root=args.root, out_root=args.out, seed=args.seed,
                )
                done += 1
                print(f"\n  Progress: {done}/{total} runs complete.\n")

    print(f"\n{'='*65}")
    print(f"  All {total} experiment(s) finished.")
    print(f"{'='*65}\n")


if __name__ == '__main__':
    main()
