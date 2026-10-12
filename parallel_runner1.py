"""
parallel_runner.py  –  redesigned for speed

Key improvements over the original:
1. Precompute & cache preprocessed splits once per (dataset, mt).
   Every tuning job just does np.load() instead of re-reading the raw
   file, running PCA, scaling, etc.
2. Inner n_jobs is capped so Pool workers and GridSearchCV don't fight
   over the same cores (avoids 16 × 16 = 256 competing threads).
3. Dataset loop is in the runner, so you can sweep all datasets in one
   sbatch submission.
4. test_size is an explicit dimension – main.py receives --test_size.
5. Accuracy-only jobs share the same cached split as TVL jobs.
"""

import itertools
import os
import subprocess
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler

# ── cluster / machine settings ────────────────────────────────────────────────
TOTAL_CORES   = 32           # matches --cpus-per-task in the SBATCH header
POOL_SIZE     = 16          # how many jobs run simultaneously
# Each job gets TOTAL_CORES // POOL_SIZE cores for its inner parallelism.
# E.g. 32 cores / 16 jobs = 2 cores per job  →  no thrashing.
CORES_PER_JOB = max(1, TOTAL_CORES // POOL_SIZE)

# ── experiment grid ────────────────────────────────────────────────────────────
DATASETS    = ['Raisin_Dataset']#, 'german','messidor_features','wdbc']  # datasets to run in parallel
CLASSIFIERS = ['MLP']
METHODS     = ['grid_search', 'random_search', 'bayesian']
SCORINGS    = ['accuracy', 'tvl']
TEST_SIZES  = [0.2, 0.3, 0.4, 0.5]   # fractions of the full dataset
M_VALUES    = [2, 4, 8, 16, 32, 64]  # TVL cardinalities

# Where preprocessed numpy arrays are cached
CACHE_DIR = Path('Cache')

# ── MNIST helpers ──────────────────────────────────────────────────────────────
import struct

def _load_mnist_images(path):
    with open(path, 'rb') as f:
        _, n, r, c = struct.unpack(">IIII", f.read(16))
        return np.frombuffer(f.read(n * r * c), dtype=np.uint8).reshape(n, r * c)

def _load_mnist_labels(path):
    with open(path, 'rb') as f:
        _, n = struct.unpack(">II", f.read(8))
        return np.frombuffer(f.read(n), dtype=np.uint8)


# ── file loading ───────────────────────────────────────────────────────────────
def _read_tabular(base_path: str) -> pd.DataFrame:
    """
    Locate and load a tabular dataset, trying three file extensions in order:
      .csv   → comma-separated, first row is header
      .tsv   → tab-separated, first row is header
      .data  → comma-separated, no header (UCI-style); columns named 0,1,2,...
               The last column is assumed to be the target.

    Args:
        base_path: path without extension, e.g. 'Datasets/air_quality'

    Raises:
        FileNotFoundError if none of the three extensions are found
    """
    candidates = [
       (base_path + '.csv', dict(sep='[,;]', engine='python', header='infer')),
        (base_path + '.tsv',  dict(sep='\t', header='infer')),
        (base_path + '.data', dict(sep=None, engine='python', header=None)),
    ]

    for path, kwargs in candidates:
        if Path(path).exists():
            print(f"Loading dataset from: {path}")
            return pd.read_csv(path, **kwargs)

    raise FileNotFoundError(
        f"No dataset file found for '{base_path}'. "
        f"Tried: {[c[0] for c in candidates]}"
    )


# ── preprocessing ──────────────────────────────────────────────────────────────
def preprocess_dataset(data_name: str):
    """
    Load and preprocess a dataset, returning (X, y) ready for splitting.
    Keeps all logic that was previously duplicated in every main.py call.
    """
    if data_name == 'MNIST':
        base = Path('Datasets/MNIST')
        X = _load_mnist_images(base / 'train-images.idx3-ubyte').astype(np.float32) / 255.0
        y = _load_mnist_labels(base / 'train-labels.idx1-ubyte')
        X_test_full = _load_mnist_images(base / 't10k-images.idx3-ubyte').astype(np.float32) / 255.0
        y_test_full = _load_mnist_labels(base / 't10k-labels.idx1-ubyte')

        # StandardScaler then PCA (matches original main.py flow)
        scaler = StandardScaler()
        X = scaler.fit_transform(X)
        X_test_full = scaler.transform(X_test_full)

        pca = PCA(n_components=100, random_state=42)
        X = pca.fit_transform(X)
        X_test_full = pca.transform(X_test_full)

        # For MNIST we keep the canonical train/test split; test_size dimension
        # is handled via mt (training sub-sample size) rather than a ratio.
        return X, y, X_test_full, y_test_full, 'fixed_split'

    else:
        data = _read_tabular(f'Datasets/{data_name}')
        data.columns = data.columns.str.strip()
        data = data.dropna().reset_index(drop=True)
        
        target_candidates = [
            'target','Air Quality','Hydration Level','Target','Class', 'eyeDetection','loan_status','RiskLevel',
              'NObeyesdad', 'Revenue', 'price_range', 'Potability'
        ]
        target_col = next(
            (c for c in target_candidates if c in data.columns),
            data.columns[-1]
        )

        cols_to_drop = [target_col] + [c for c in ['loan_id', 'id'] if c in data.columns]
        X_raw = data.drop(columns=cols_to_drop, errors='ignore')

        numeric_cols  = X_raw.select_dtypes(include='number').columns.tolist()
        category_cols = [c for c in X_raw.columns if c not in numeric_cols]

        if category_cols:
            X_df = pd.concat(
                [X_raw[numeric_cols], pd.get_dummies(X_raw[category_cols])], axis=1
            )
        else:
            X_df = X_raw[numeric_cols].copy()

        y = data[target_col]
        if y.dtype.kind in ('O', 'S', 'U'):
            y = pd.Series(
                LabelEncoder().fit_transform(y), index=y.index, name=target_col
            )

        return X_df.values, np.ravel(y), None, None, 'ratio_split'


# ── cache helpers ──────────────────────────────────────────────────────────────
def cache_key(data_name: str, test_size, mt: int) -> Path:
    """Return the .npz path for a preprocessed split."""
    ts_tag = f'ts{str(test_size).replace(".", "p")}' if test_size != 'fixed' else 'fixed'
    return CACHE_DIR / data_name / f'mt{mt}_{ts_tag}.npz'


def ensure_cache(data_name: str, test_sizes, mt_values):
    """
    Precompute and save all required (test_size, mt) splits for a dataset.
    Called once before the Pool is launched – completely serial, no race conditions.
    """
    X, y, X_test_fix, y_test_fix, split_mode = preprocess_dataset(data_name)
    mt_list = mt_values if split_mode == 'fixed_split' else [None]  # mt is only relevant for MNIST
    for test_size in test_sizes:
        for mt in mt_list:
            path = cache_key(data_name, test_size if split_mode == 'ratio_split' else 'fixed', mt)
            if path.exists():
                continue  # already cached

            path.parent.mkdir(parents=True, exist_ok=True)

            if split_mode == 'fixed_split':
                # MNIST: subsample training set; test set is always the canonical one
                n_classes = len(np.unique(y))
                actual_mt = min(mt, len(X))
                if actual_mt >= n_classes:
                    X_tr, _, y_tr, _ = train_test_split(
                        X, y, train_size=actual_mt, stratify=y, random_state=42
                    )
                else:
                    X_tr, y_tr = X, y
                X_te, y_te = X_test_fix, y_test_fix

            else:
                # CSV datasets: split by test_size ratio, then subsample training
                X_tr, X_te, y_tr, y_te = train_test_split(
                    X, y, test_size=test_size, stratify=y, random_state=42
                )

                # Scale (fit on training, apply to test)
                numeric_count = X_tr.shape[1]  # after get_dummies everything is numeric
                scaler = StandardScaler()
                X_tr = scaler.fit_transform(X_tr)
                X_te = scaler.transform(X_te)

            np.savez_compressed(path, X_train=X_tr, X_test=X_te, y_train=y_tr, y_test=y_te)
            print(f"  Cached: {path}")


# ── job description & completion check ────────────────────────────────────────
def result_folder(data_name, clf, method, score, mt, m, test_size) -> Path:
    base   = Path('Results') / data_name / clf / method / score
    ts_tag = f'test_size_{test_size}' if test_size != 'fixed' else 'fixed'
    mt_tag = f'm_t_{mt}' if mt is not None else 'full'

    if score == 'tvl':
        if mt is not None:
            return base / f'm_{m}' / mt_tag
        else:
            return base / f'm_{m}' / ts_tag

    # accuracy
    if mt is not None:
        return base / mt_tag
    else:
        return base / ts_tag


def is_completed(job) -> bool:
    folder = result_folder(*job)
    return folder.exists() and any(folder.glob('*.json'))


# ── run one job ────────────────────────────────────────────────────────────────
def run_job(job):
    data_name, clf, method, score, mt, m, test_size = job

    if is_completed(job):
        print(f"  Skip (done): {job}")
        return

    # Resolve which cache file to load
    split_mode = 'fixed_split' if data_name == 'MNIST' else 'ratio_split'
    ts_key = 'fixed' if split_mode == 'fixed_split' else test_size
    npz_path = cache_key(data_name, ts_key, mt)

    if not npz_path.exists():
        print(f"  ERROR: cache missing for {npz_path}. Skipping.")
        return

    cmd = [
        'python', 'Python files/main.py',
        '--data',      data_name,
        '--clf',       clf,
        '--method',    method,
        '--score',     score,
        '--mt',        str(mt),
        '--m',         str(m),
        '--size', str(test_size),
        '--cache',     str(npz_path),
        '--n_jobs',    str(CORES_PER_JOB),
    ]
    subprocess.run(cmd, check=False)


# ── main ───────────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    all_jobs = []

    for data_name in DATASETS:
        print(f"\n=== Preparing cache for: {data_name} ===")
        
        # Determine which test_sizes are meaningful for this dataset
        ts_list = ['fixed'] if data_name == 'MNIST' else TEST_SIZES
        mt_values = [5000, 10000, 30000, 60000] if data_name == 'MNIST' else [None]  # match main.py defaults and ensure_cache calls
        ensure_cache(data_name, ts_list, mt_values=mt_values)

        # Build job list for this dataset
        for clf, method, score, mt, m, ts in itertools.product(
            CLASSIFIERS, METHODS, SCORINGS,
            mt_values,
            M_VALUES, ts_list
        ):
            # Accuracy jobs don't use m — keep only one canonical combo
            if score == 'accuracy' and m != M_VALUES[0]:
                continue

            all_jobs.append((data_name, clf, method, score, mt, m, ts))

    pending = [j for j in all_jobs if not is_completed(j)]
    print(f"\nTotal jobs : {len(all_jobs)}")
    print(f"Done       : {len(all_jobs) - len(pending)}")
    print(f"Pending    : {len(pending)}")
    print(f"Pool size  : {POOL_SIZE}  |  cores/job : {CORES_PER_JOB}\n")

    with Pool(processes=POOL_SIZE) as pool:
        pool.map(run_job, pending)
