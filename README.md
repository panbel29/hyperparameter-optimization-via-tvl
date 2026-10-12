# Hyperparameter Optimization using the TvL Metric

Code for the diploma thesis *"Hyperparameter Optimization using the TvL Metric"* (School of Electrical and Computer Engineering, Aristotle University of Thessaloniki) and the derived journal manuscript submitted to *Pattern Recognition Letters*.

The project investigates whether tuning hyperparameters with the **TvL (Talent vs. Luck)** metric, introduced by Petrantonakis & Kompatsiaris (2019), yields models that generalize better than the standard practice of tuning for validation accuracy.

---

## Overview

For every configuration, two optimal models are obtained, one tuned for **accuracy** and one tuned for **TvL**. Both are compared against each other and against a model with **default hyperparameters**.

The study is run on two pipelines.

### Tabular pipeline

| Component | Values |
|---|---|
| Classifiers | SVM, KNN, Random Forest, MLP |
| Search methods | Grid Search, Random Search, Bayesian Optimization (Optuna, TPE) |
| Datasets | 36, grouped by number of samples, number of features and number of classes |
| Test sizes | multiple train/test splits (e.g. 0.3) |
| TvL cardinality *m* | multiple values |
| Validation | 5-fold cross-validation |

### CNN pipeline (case study)

| Component | Values |
|---|---|
| Datasets | MNIST, Fashion-MNIST, CIFAR-10 |
| Search methods | Random Search, Bayesian Optimization, Nelder-Mead |
| TvL parameters | multiple values of *m* and *mt* |

### Evaluation metrics

Test Accuracy, Generalization Error, macro F1-score and AUROC.

### Statistical validation

For the tabular pipeline, Friedman tests with Nemenyi post-hoc analysis and Critical Difference (CD) diagrams compare the tuning criteria within each (classifier, search method, test size) context, with datasets as blocks.

---

## Workflow

1. Load and preprocess each dataset (one-hot encoding of categorical features, feature scaling).
2. Split into training and test sets for each test size.
3. Train SVM, KNN, RF and MLP with default hyperparameters and evaluate with 5-fold CV.
4. Tune hyperparameters with Grid Search, Random Search and Bayesian Optimization, once with accuracy and once with TvL as the objective.
5. Select the optimal model for each tuning metric.
6. Fit on the training set and predict on the test set.
7. Compare the accuracy-tuned, TvL-tuned and default models using the metrics above.
8. Repeat for every search method, test size, value of *m* and dataset.
9. Aggregate results and run the statistical analysis.

---

## Repository structure

<!-- TODO: adjust names to match the uploaded files -->

```
.
├── tabular/              # Tabular pipeline: preprocessing, tuning, evaluation
├── cnn/
│   ├── cnn_model.py      # CNN architectures
│   ├── cnn_data.py       # Data loading for MNIST / Fashion-MNIST / CIFAR-10
│   ├── cnn_train.py      # Training loop
│   ├── cnn_tune.py       # RS / BO / Nelder-Mead tuning with accuracy and TvL
│   └── cnn_main.py       # Entry point
├── analysis.py           # Tabular analysis: plots, Friedman/Nemenyi, CD diagrams
├── cnn_analysis.py       # CNN analysis: heatmaps, line plots, learning curves
├── parallel_runner1.py   # Parallel execution of experiment configurations
├── check_results.py      # Completeness check of experiment outputs
├── slurm/                # SLURM job scripts
├── results/              # Final result tables (CSV)
├── requirements.txt
└── README.md
```

---

## Installation

```bash
git clone https://github.com/<user>/<repo>.git
cd <repo>
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

Main dependencies: Python 3, scikit-learn, PyTorch, Optuna, SciPy, scikit-posthocs, pandas, NumPy, Matplotlib.

---

## Usage

<!-- TODO: replace the tabular entry-point name and confirm arguments -->

**Quick test run** (small budget, CPU):

```bash
python cnn_main.py --n_trials 2 --mt 1000
```

**Analysis**

```bash
python analysis.py                  # tabular results
python cnn_analysis.py              # CNN results
python analysis.py --out <folder>   # write outputs to a separate folder
```

Individual plot groups can be skipped with the `--no_*` flags (see `--help`).

**HPC**

The full experiments were run on the AUTh "Aristotle" HPC cluster using SLURM. Example job scripts are in `slurm/`; adjust paths, partitions and time limits to your own cluster.

---

## Results

Final result tables are in `results/`. Each table reports, per configuration, the tuning measure (accuracy or TvL), the value of *m*, CV score, training and test accuracy, generalization error, AUROC, F1-score and the selected hyperparameters.

Main observations (tabular pipeline):

- TvL-based tuning matches or outperforms accuracy-based tuning in a large share of configurations, on both Test Accuracy and Generalization Error.
- The effect is most consistent with Bayesian Optimization.
- The cardinality *m* acts as a regulator of the trade-off between accuracy and generalization.
- MLP and Random Forest benefit the most.

In the CNN case study, TvL tends to select fewer training epochs and lower learning rates than accuracy-based tuning.

---

## Known limitations

- The contracting Grid Search implementation does not track the best configuration across iterations. This applies equally to both tuning criteria.
- Nelder-Mead uses roughly three times the compute budget of Random Search and Bayesian Optimization in the CNN pipeline.

---

## Citation

<!-- TODO: add paper citation once published -->

If you use this code, please cite the thesis or the associated paper, as well as the original TvL work:

> P. C. Petrantonakis and I. Kompatsiaris, "Detecting Talent vs. Luck in Machine Learning," 2019. <!-- TODO: verify exact title, venue and DOI -->

---

## Author

Panos — School of Electrical and Computer Engineering, Aristotle University of Thessaloniki.
<!-- TODO: full name, supervisor, contact -->
