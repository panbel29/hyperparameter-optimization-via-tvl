import argparse
import json
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.calibration import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from knn_tune import bayesian_knn, grid_search_knn, random_search_knn
from mlp_tune import bayesian_mlp, grid_search_mlp, random_search_mlp
from rf_tune import bayesian_rf, grid_search_rf, random_search_rf
from svm_tune import bayesian_optimization_svm, grid_search_svm, random_search_svm
from tvl_scorer import convert_to_python_types, get_measures


# ─────────────────────────────────────────────────────────────────────────────
# Data loading
# ─────────────────────────────────────────────────────────────────────────────

def _read_tabular(base_path: str) -> pd.DataFrame:
    """
    Load a tabular dataset, trying .csv, .tsv, and .data extensions in order.
    The last column is assumed to be the target for headerless .data files.
    """
    candidates = [
        (base_path + '.csv',  dict(sep=',',  header='infer')),
        (base_path + '.tsv',  dict(sep='\t', header='infer')),
        (base_path + '.data', dict(sep=',',  header=None)),
    ]
    for path, kwargs in candidates:
        if Path(path).exists():
            print(f"Loading dataset from: {path}")
            return pd.read_csv(path, **kwargs)
    raise FileNotFoundError(
        f"No dataset file found for '{base_path}'. "
        f"Tried: {[c[0] for c in candidates]}"
    )


def _load_data(args):
    """
    Return (X_train, X_test, y_train, y_test).

    Fast path: if --cache points to a precomputed .npz file, just load it.
    Slow path: read the raw file, preprocess, and split from scratch.
    """
    # Fast path — precomputed split provided by parallel_runner1.py
    if args.cache:
        npz = np.load(args.cache)
        return npz['X_train'], npz['X_test'], npz['y_train'], npz['y_test']

    # Slow path — read, encode categoricals, split, scale
    data = _read_tabular(f'Datasets/{args.data}')
    data.columns = data.columns.str.strip()

    # Identify the target column (try common names, fall back to the last column)
    target_candidates = ['target', 'Class']
    tc = next((c for c in target_candidates if c in data.columns), data.columns[-1])

    drop_cols = [tc] + [c for c in ['loan_id', 'id'] if c in data.columns]
    X_raw = data.drop(columns=drop_cols, errors='ignore')

    numeric_cols  = X_raw.select_dtypes(include='number').columns.tolist()
    category_cols = [c for c in X_raw.columns if c not in numeric_cols]

    # One-hot encode any categorical columns
    if category_cols:
        X_df = pd.concat(
            [X_raw[numeric_cols], pd.get_dummies(X_raw[category_cols])], axis=1
        )
    else:
        X_df = X_raw[numeric_cols].copy()

    y = data[tc]
    if y.dtype.kind in ('O', 'S', 'U'):
        y = pd.Series(LabelEncoder().fit_transform(y), index=y.index)

    X_tr, X_te, y_tr, y_te = train_test_split(
        X_df.values, np.ravel(y),
        test_size=args.size, stratify=y, random_state=42
    )

    # Scale features (fit on training only to avoid data leakage)
    scaler = StandardScaler()
    X_tr = scaler.fit_transform(X_tr)
    X_te = scaler.transform(X_te)

    return X_tr, X_te, y_tr, y_te


# ─────────────────────────────────────────────────────────────────────────────
# Main experiment pipeline
# ─────────────────────────────────────────────────────────────────────────────

def main(args):
    DATA_NAME     = args.data
    CLASSIFIER    = args.clf
    TUNING_METHOD = args.method
    SCORING       = args.score
    m             = args.m
    test_size     = args.size
    L             = 100   # number of TvL sampling replicates (fixed)
    n_jobs        = args.n_jobs

    n_iter   = 50   # iterations for Random Search
    n_trials = 50   # trials for Bayesian Optimization

    # ── Load data ─────────────────────────────────────────────────────────────
    X_train, X_test, y_train, y_test = _load_data(args)

    print(f"\n{'='*60}")
    print(f"DATA VERIFICATION")
    print(f"Dataset: {args.data}")
    print(f"X_train shape: {X_train.shape} | X_test shape: {X_test.shape}")
    print(f"Classes: {np.unique(y_train)}")
    print(f"Train class distribution: {np.bincount(y_train)}")
    print(f"Test class distribution:  {np.bincount(y_test)}")
    print(f"Test size: {args.size}")
    print(f"{'='*60}\n")

    print(f"\n{'='*60}")
    print(
        f"Classifier: {CLASSIFIER} | Method: {TUNING_METHOD} | "
        f"Scoring: {SCORING} | Test Size: {test_size}"
    )
    print(f"{'='*60}\n")

    # ── Classifier & tuning ───────────────────────────────────────────────────
    if CLASSIFIER == 'SVM':
        print("Selected Classifier: Support Vector Machine (SVM)")

        default_model = SVC(probability=True)
        default_model.fit(X_train, y_train)
        selected_params = {k: default_model.get_params()[k] for k in ['C', 'gamma', 'kernel']}
        print(f"Default SVM parameters: {selected_params}")
        default_measures = get_measures(default_model, X_train, X_test, y_train, y_test)

        if TUNING_METHOD == 'grid_search':
            tuned_estimator, best_params, best_score = grid_search_svm(
                X_train, y_train, iterations=3,
                initial_c_range=(-2, 2), initial_gamma_range=(-4, 0),
                degrees=(2, 4), steps=5, kernel='rbf',
                scoring=SCORING, m=m, L=L, n_jobs=n_jobs)

        elif TUNING_METHOD == 'random_search':
            tuned_estimator, best_params, best_score = random_search_svm(
                X_train, y_train, n_iter=n_iter, kernel='rbf',
                c_range=(1e-2, 1e2), gamma_range=(1e-4, 1e0), degrees=(2, 4),
                scoring=SCORING, m=m, L=L, n_jobs=n_jobs)

        elif TUNING_METHOD == 'bayesian':
            tuned_estimator, best_params, best_score = bayesian_optimization_svm(
                X_train, y_train, n_trials=n_trials, kernel='rbf',
                scoring=SCORING, m=m, L=L)

        else:
            raise ValueError(f"Unknown tuning method: {TUNING_METHOD}")

    elif CLASSIFIER == 'KNN':
        print("Selected Classifier: K-Nearest Neighbors (KNN)")

        default_model = KNeighborsClassifier()
        default_model.fit(X_train, y_train)
        selected_params = {k: default_model.get_params()[k] for k in ['n_neighbors', 'weights']}
        print(f"Default KNN parameters: {selected_params}")
        default_measures = get_measures(default_model, X_train, X_test, y_train, y_test)

        if TUNING_METHOD == 'grid_search':
            tuned_estimator, best_params, best_score = grid_search_knn(
                X_train, y_train, kmin=1, kmax=51,
                scoring=SCORING, m=m, L=L, n_jobs=n_jobs)

        elif TUNING_METHOD == 'random_search':
            tuned_estimator, best_params, best_score = random_search_knn(
                X_train, y_train, n_iter=n_iter,
                n_neighbors_range=(1, 51), weights=['uniform', 'distance'],
                scoring=SCORING, m=m, L=L, n_jobs=n_jobs)

        elif TUNING_METHOD == 'bayesian':
            tuned_estimator, best_params, best_score = bayesian_knn(
                X_train, y_train, n_trials=n_trials,
                n_neighbors_range=(1, 51), weights=['uniform', 'distance'],
                scoring=SCORING, m=m, L=L)

        else:
            raise ValueError(f"Unknown tuning method: {TUNING_METHOD}")

    elif CLASSIFIER == 'RF':
        print("Selected Classifier: Random Forest (RF)")

        default_model = RandomForestClassifier(random_state=42)
        default_model.fit(X_train, y_train)
        selected_params = {k: default_model.get_params()[k]
                           for k in ['n_estimators', 'max_depth', 'min_samples_split']}
        print(f"Default RF parameters: {selected_params}")
        default_measures = get_measures(default_model, X_train, X_test, y_train, y_test)

        if TUNING_METHOD == 'grid_search':
            tuned_estimator, best_params, best_score = grid_search_rf(
                X_train, y_train, scoring=SCORING, m=m, L=L, n_jobs=n_jobs)

        elif TUNING_METHOD == 'random_search':
            tuned_estimator, best_params, best_score = random_search_rf(
                X_train, y_train, n_iter=n_iter,
                scoring=SCORING, m=m, L=L, n_jobs=n_jobs)

        elif TUNING_METHOD == 'bayesian':
            tuned_estimator, best_params, best_score = bayesian_rf(
                X_train, y_train, n_trials=n_trials,
                scoring=SCORING, m=m, L=L, random_seed=42)

        else:
            raise ValueError(f"Unknown tuning method: {TUNING_METHOD}")

    elif CLASSIFIER == 'MLP':
        print("Selected Classifier: Multi-Layer Perceptron (MLP)")

        default_model = MLPClassifier(random_state=42)
        default_model.fit(X_train, y_train)
        selected_params = {k: default_model.get_params()[k]
                           for k in ['learning_rate_init', 'batch_size', 'max_iter']}
        print(f"Default MLP parameters: {selected_params}")
        default_measures = get_measures(default_model, X_train, X_test, y_train, y_test)

        if TUNING_METHOD == 'grid_search':
            tuned_estimator, best_params, best_score = grid_search_mlp(
                X_train, y_train, scoring=SCORING, m=m, L=L, n_jobs=n_jobs)

        elif TUNING_METHOD == 'random_search':
            tuned_estimator, best_params, best_score = random_search_mlp(
                X_train, y_train, n_iter=n_iter,
                scoring=SCORING, m=m, L=L, n_jobs=n_jobs)

        elif TUNING_METHOD == 'bayesian':
            tuned_estimator, best_params, best_score = bayesian_mlp(
                X_train, y_train, n_trials=n_trials,
                scoring=SCORING, m=m, L=L, random_seed=42)

        else:
            raise ValueError(f"Unknown tuning method: {TUNING_METHOD}")

    else:
        raise ValueError(f"Unknown classifier: {CLASSIFIER}")

    # ── Evaluate tuned model ──────────────────────────────────────────────────
    print("\n" + "="*60)
    print("RESULTS")
    print("="*60)
    print("\nDefault Model Performance:")
    print(default_measures)
    print(f"\nTuned Model Performance ({TUNING_METHOD} with {SCORING}):")
    tuned_measures = get_measures(tuned_estimator, X_train, X_test, y_train, y_test)

    # ── Save results ──────────────────────────────────────────────────────────
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename  = f"{DATA_NAME}_{CLASSIFIER}_{TUNING_METHOD}_{SCORING}_{timestamp}"

    results = {
        'dataset name':            DATA_NAME,
        'classifier':              CLASSIFIER,
        'tuning_method':           TUNING_METHOD,
        'n_iter':                  n_iter   if TUNING_METHOD == 'random_search' else None,
        'n_trials':                n_trials if TUNING_METHOD == 'bayesian'      else None,
        'test_size':               test_size,
        'scoring':                 SCORING,
        'm':                       m        if SCORING == 'tvl' else None,
        'L':                       L        if SCORING == 'tvl' else None,
        'timestamp':               timestamp,
        'default hyperparameters': selected_params,
        'default model metrics':   default_measures,
        'best hyperparameters':    best_params,
        'best score':              best_score,
        'tuned model metrics':     tuned_measures,
    }

    # Build output folder: Results/<dataset>/<clf>/<method>/<scoring>/...
    if SCORING == 'tvl':
        dataset_folder = Path(
            f'Results/{DATA_NAME}/{CLASSIFIER}/{TUNING_METHOD}/{SCORING}'
            f'/m_{m}/test_size_{test_size}'
        )
    else:
        dataset_folder = Path(
            f'Results/{DATA_NAME}/{CLASSIFIER}/{TUNING_METHOD}/{SCORING}/test_size_{test_size}'
        )

    dataset_folder.mkdir(parents=True, exist_ok=True)
    json_path = dataset_folder / f'{filename}.json'
    txt_path  = dataset_folder / f'{filename}.txt'

    # JSON — machine-readable
    with open(json_path, 'w') as f:
        json.dump(convert_to_python_types(results), f, indent=4)
    print(f"\n✓ Results saved to {json_path}")

    # TXT — human-readable summary
    with open(txt_path, 'w') as f:
        f.write("="*60 + "\n")
        f.write(f"RESULTS — {CLASSIFIER} with {TUNING_METHOD} ({SCORING}) — test_size={test_size}\n")
        if TUNING_METHOD == 'random_search':
            f.write(f"Number of iterations: {n_iter}\n")
        if SCORING == 'tvl':
            f.write(f"TvL Parameters: m={m}, L={L}\n")
        f.write(f"Timestamp: {timestamp}\n")
        f.write("="*60 + "\n\n")

        f.write("DEFAULT HYPERPARAMETERS:\n" + "-"*40 + "\n")
        for key, value in selected_params.items():
            f.write(f"{key}: {value}\n")

        f.write("\n" + "="*60 + "\n")
        f.write("DEFAULT MODEL PERFORMANCE:\n" + "-"*40 + "\n")
        for key, value in default_measures.items():
            f.write(f"{key}: {value}\n")

        f.write("\n" + "="*60 + "\n")
        f.write("BEST HYPERPARAMETERS (after tuning):\n" + "-"*40 + "\n")
        for key, value in best_params.items():
            f.write(f"{key}: {value}\n")
        f.write(f"\nBest CV Score: {best_score}\n")

        f.write("\n" + "="*60 + "\n")
        f.write("TUNED MODEL PERFORMANCE:\n" + "-"*40 + "\n")
        for key, value in tuned_measures.items():
            f.write(f"{key}: {value}\n")
        f.write("\n" + "="*60 + "\n")

    print(f"✓ Results saved to {txt_path}")
    print("All results saved successfully!")


# ─────────────────────────────────────────────────────────────────────────────
# CLI entry point
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data',     type=str,   required=True)
    parser.add_argument('--clf',      type=str,   required=True)
    parser.add_argument('--method',   type=str,   required=True)
    parser.add_argument('--score',    type=str,   required=True)
    parser.add_argument('--m',        type=int,   default=2)
    parser.add_argument('--size',     type=float, default=0.2)
    parser.add_argument('--cache',    type=str,   default=None,
                        help='Path to a precomputed .npz split (skips data loading).')
    parser.add_argument('--n_jobs',   type=int,   default=-1,
                        help='Parallel CV jobs. Set by the runner to avoid core thrashing.')
    arguments = parser.parse_args()
    main(arguments)
