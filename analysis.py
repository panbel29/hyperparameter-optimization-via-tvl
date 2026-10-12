"""
Statistical analysis module for comparing TvL vs Accuracy hyperparameter-tuning across datasets,
classifiers, search methods, test sizes and TvL cardinalities m.

Outputs (written to --output_dir)
----------------------------------
full_results.csv
aggregated_by_method.csv
ranks/ranks_{metric}.csv                          (4 files)
relative_scores/relative_{metric}.csv             (4 files)
boxplot/{clf}__{method}__{ts}.png                 (3 metrics per file)
cd_diagrams/{clf}__{method}__{ts}.png             (one CD diagram per context)
stats/friedman_nemenyi_{clf}__{method}__{ts}.csv  (one table per context)
"""

from __future__ import annotations

import glob
import json
import math
import re
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd
from scipy.stats import friedmanchisquare

try:
    import scikit_posthocs as sp
    _POSTHOCS_AVAILABLE = True
except ImportError:
    _POSTHOCS_AVAILABLE = False
    warnings.warn(
        "scikit-posthocs not installed – Nemenyi test will be skipped. "
        "Install with:  pip install scikit-posthocs",
        ImportWarning,
    )

# Metric configuration
METRICS      = ["Test_Accuracy", "Gen_Error", "AUROC", "F1_Score"]
PLOT_METRICS = ["Test_Accuracy", "Gen_Error", "AUROC", "F1_Score"]   # metrics used in box plots

# Only these m values are kept for TvL — any other value (e.g. 128, 256) is
# dropped at load time regardless of the --m_values CLI argument.
VALID_M_VALUES: set[int] = {2, 4, 8, 16, 32, 64}


# Dataset groups 
DATASET_GROUPS: dict[str, list[str]] = {
    "group_A1": ['heart','profb','tic-tac-toe','Raisin_Dataset','loan_approval_dataset','Daily_Water_Intake','phoneme','EEG_Eye_State','magic',
    'tokyo1','messidor_features','ring','hypothyroid','agaricus-lepiota','churn','dis','twonorm','kr-vs-kp'], # binary datasets
    "group_A2": ['maternal_health_risk','nursery','splice','waveform-21','ann-thyroid','vehicle','air_quality','vowel','soybean',
    'classast - pha','satimage','Dry_Bean_Dataset','obesity','segmentation','yeast','pendigits','mfeat_zernike','optdigits'], # multiclass datasets
    "group_B1": ['profb', 'soybean','vehicle','Raisin_Dataset','tic-tac-toe','tokyo1','vowel','maternal_health_risk','heart',
                'messidor_features','yeast','classast - pha'],   # <2000 samples
    "group_B2": ['mfeat_zernike','obesity','segmentation','hypothyroid','splice','kr-vs-kp','dis','loan_approval_dataset','churn',
               'waveform-21','air_quality','phoneme'],  # (2000,5500) samples
    "group_B3": ['optdigits','satimage','ann-thyroid','ring','twonorm','agaricus-lepiota','pendigits','nursery','Dry_Bean_Dataset',
                'EEG_Eye_State','magic','Daily_Water_Intake'],  # (5500,30000) samples
    "group_C1": ['Raisin_Dataset','segmentation', 'mfeat_zernike','vowel','twonorm','ring','waveform-21','heart','optdigits',
                 'agaricus-lepiota','kr-vs-kp','vehicle','pendigits','messidor_features','EEG_Eye_State','obesity','maternal_health_risk'], # balanced datasets
    "group_C2": ['loan_approval_dataset','tokyo1','magic','tic-tac-toe','profb','splice','phoneme','satimage','Daily_Water_Intake',
                'air_quality','churn','soybean','Dry_Bean_Dataset','nursery','hypothyroid','yeast','ann-thyroid','dis','classast - pha'], # imbalanced datasets
    
    "group_D": ['Raisin_Dataset','segmentation', 'mfeat_zernike','vowel','twonorm','ring','waveform-21','heart','optdigits',
                 'agaricus-lepiota','kr-vs-kp','vehicle','pendigits','messidor_features','EEG_Eye_State','obesity','maternal_health_risk',
                 'loan_approval_dataset','tokyo1','magic','tic-tac-toe','profb','splice','phoneme','satimage','Daily_Water_Intake',
               'air_quality','churn','soybean','Dry_Bean_Dataset','nursery','hypothyroid','yeast','ann-thyroid','dis','classast - pha'] # all datasets
} 

LOWER_IS_BETTER: dict[str, bool] = {
    "Test_Accuracy": False,
    "Gen_Error":     True,
    "AUROC":         False,
    "F1_Score":      False,
}

METRIC_LABELS: dict[str, str] = {
    "Test_Accuracy": "Test Accuracy (%)",
    "Gen_Error":     "Generalization Error (%)",
    "AUROC":         "AUROC",
    "F1_Score":      "F1",
}

# Human-readable subplot column titles
PLOT_METRIC_TITLES: dict[str, str] = {
    "Test_Accuracy": "Test Accuracy",
    "Gen_Error":     "Generalization Error",
    "AUROC":         "AUROC",
    "F1_Score":      "F1",
}

# Studentized range critical values q_α for α=0.05 (Demšar 2006, Table 5)
# Index = number of methods k (2 … 25)
_Q_ALPHA_05 = {
    2: 2.772, 3: 3.314, 4: 3.633, 5: 3.858, 6: 4.030,
    7: 2.949, 8: 4.286, 9: 4.387
}


# 0.  Data loading  (JSON only, Sampling=False skipped)

def _extract_f1(report_str: str | None) -> float | None:
    if not report_str or not isinstance(report_str, str):
        return None
    match = re.search(r"weighted avg\s+[\d.]+\s+[\d.]+\s+([\d.]+)", report_str)
    return float(match.group(1)) if match else None


def load_from_json(
    results_root: str | Path,
    datasets: list[str] | None = None,
) -> pd.DataFrame:
    """
    Recursively load every *.json experiment file under *results_root*.

    Rules
    -----
    * TVL files with sampling=False are skipped entirely.
    * Accuracy files are always kept.
    * De-duplicates by keeping the latest timestamp per unique configuration.
    """
    results_root = Path(results_root)
    json_files   = glob.glob(str(results_root / "**" / "*.json"), recursive=True)

    rows: list[dict] = []
    skipped_sampling_false = 0

    for fpath in json_files:
        try:
            with open(fpath) as fh:
                data = json.load(fh)
        except Exception as exc:
            warnings.warn(f"Could not parse {fpath}: {exc}")
            continue

        ds = data.get("dataset name", "")
        if datasets is not None and ds not in datasets:
            continue

        scoring  = data.get("scoring", "")
        sampling = data.get("sampling")
        m_val    = data.get("m")

        # Skip Sampling=False TVL files entirely (used at the start of experiments - now they are skipped)
        if scoring == "tvl" and sampling is False:
            skipped_sampling_false += 1
            continue

        # Skip TVL files whose m is not in the allowed set {2,4,8,16,32,64}
        if scoring == "tvl":
            try:
                if int(m_val) not in VALID_M_VALUES:
                    continue
            except (TypeError, ValueError):
                pass  # m is None or non-numeric → let through, will become NaN

        tuned   = data.get("tuned model metrics",   {})
        default = data.get("default model metrics", {})

        row = {
            "dataset":       ds,
            "classifier":    data.get("classifier"),
            "tuning_method": data.get("tuning_method"),
            "scoring":       scoring,
            "m":             data.get("m")   if scoring == "tvl" else None,
            "sampling":      sampling        if scoring == "tvl" else None,
            "test_size":     data.get("test_size") or data.get("test size"),
            "Test_Accuracy": tuned.get("test accuracy"),
            "Gen_Error":     tuned.get("generalization error"),
            "AUROC":         tuned.get("AUROC"),
            "F1_Score":      _extract_f1(tuned.get("classification report", "")),
            "Default_Test_Accuracy": default.get("test accuracy"),
            "Default_Gen_Error":     default.get("generalization error"),
            "Default_AUROC":         default.get("AUROC"),
            "Default_F1_Score":      _extract_f1(default.get("classification report", "")),
            "_timestamp": data.get("timestamp", ""),
            "_file":      fpath,
        }
        rows.append(row)

    if skipped_sampling_false:
        print(f"  Skipped {skipped_sampling_false} Sampling=False TVL file(s).")

    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows)
    dedup_cols = ["dataset", "classifier", "tuning_method", "scoring", "m", "sampling", "test_size"]
    df = (
        df.sort_values(["_timestamp", "_file"])
          .drop_duplicates(subset=dedup_cols, keep="last")
          .drop(columns=["_timestamp", "_file"])
          .reset_index(drop=True)
    )
    return df


# 1.  Relative scores

def compute_relative_score(tuned: pd.Series, baseline: pd.Series) -> pd.Series:
    """
    (tuned - baseline) / |baseline| * 100.
    Returns NaN when baseline == 0 (handles Gen_Error edge case).
    """
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        denom = baseline.abs().replace(0, np.nan)
        return (tuned - baseline) / denom * 100


def add_relative_scores(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for metric in METRICS:
        bc = f"Default_{metric}"
        if metric in df.columns and bc in df.columns:
            t = pd.to_numeric(df[metric], errors="coerce")
            b = pd.to_numeric(df[bc],     errors="coerce")
            df[f"Rel_{metric}"] = compute_relative_score(t, b)
    return df


# 2.  Ranking  (absolute values only)

def rank_methods(df: pd.DataFrame, group_cols: list[str], metric: str) -> pd.DataFrame:
    """Rank 1 = best.  Direction taken from LOWER_IS_BETTER."""
    if metric not in METRICS:
        raise ValueError(f"rank_methods() only accepts {METRICS}. Got '{metric}'.")
    ascending = LOWER_IS_BETTER[metric]
    df = df.copy()
    df[f"Rank_{metric}"] = (
        df.groupby(group_cols, group_keys=False)[metric]
          .transform(lambda s: s.rank(ascending=ascending, method="average"))
    )
    return df


def rank_all_metrics(df: pd.DataFrame, group_cols: list[str]) -> pd.DataFrame:
    for metric in METRICS:
        if metric in df.columns:
            df = rank_methods(df, group_cols, metric)
    return df

# 3.  Aggregate across datasets

def aggregate_by_method(
    df: pd.DataFrame,
    method_cols: list[str],
    value_cols: list[str] | None = None,
) -> pd.DataFrame:
    if value_cols is None:
        exclude = set(method_cols) | {"dataset", "sampling"}
        value_cols = [
            c for c in df.columns
            if c not in exclude and pd.api.types.is_numeric_dtype(df[c])
        ]
    records: list[dict] = []
    for key, grp in df.groupby(method_cols, dropna=False):
        row: dict = dict(zip(method_cols, key if isinstance(key, tuple) else (key,)))
        for col in value_cols:
            vals = pd.to_numeric(grp[col], errors="coerce").dropna()
            row[f"{col}_mean"]  = vals.mean()
            row[f"{col}_std"]   = vals.std(ddof=1)
            row[f"{col}_count"] = len(vals)
        records.append(row)
    return pd.DataFrame(records).reset_index(drop=True)


# 4.  Friedman test

def friedman_test(
    df: pd.DataFrame,
    method_col: str,
    dataset_col: str,
    metric: str,
    rank_col: str | None = None,
) -> dict:
    """
    Non-parametric Friedman test.

    Parameters
    ----------
    df:           Long-format DataFrame.
    method_col:   Column identifying the method.
    dataset_col:  Column identifying the dataset (blocking factor).
    metric:       Raw metric column — used to build the pivot for the statistic.
    rank_col:     If given, average ranks in the output are taken from this
                  pre-computed rank column (e.g. ``"Rank_Test_Accuracy"``)
                  so that CD diagram and heatmap display identical values.

    Returns
    -------
    dict with keys: statistic, p_value, n_methods, n_datasets, significant,
                    avg_ranks, pivot, rank_pivot.
    """
    # Raw metric pivot — used for the Friedman chi-squared statistic
    pivot = df.pivot_table(
        index=dataset_col, columns=method_col, values=metric, aggfunc="mean"
    ).dropna()

    if pivot.shape[0] < 2:
        raise ValueError(f"Friedman needs ≥2 complete blocks; got {pivot.shape[0]}.")
    if pivot.shape[1] < 2:
        raise ValueError(f"Friedman needs ≥2 methods; got {pivot.shape[1]}.")

    stat, p_val = friedmanchisquare(*[pivot[c].values for c in pivot.columns])

    # Average ranks — use pre-computed rank column if provided
    if rank_col is not None and rank_col in df.columns:
        rank_pivot = df.pivot_table(
            index=dataset_col, columns=method_col, values=rank_col, aggfunc="mean"
        ).reindex(index=pivot.index, columns=pivot.columns)
        avg_ranks = rank_pivot.mean()
    else:
        ascending  = LOWER_IS_BETTER.get(metric, False)
        rank_pivot = pivot.rank(axis=1, ascending=ascending, method="average")
        avg_ranks  = rank_pivot.mean()

    return {
        "statistic":   stat,
        "p_value":     p_val,
        "n_methods":   pivot.shape[1],
        "n_datasets":  pivot.shape[0],
        "significant": bool(p_val < 0.05),
        "avg_ranks":   avg_ranks,
        "pivot":       pivot,
        "rank_pivot":  rank_pivot,
    }

def nemenyi_test(
    df: pd.DataFrame,
    method_col: str,
    dataset_col: str,
    metric: str,
    rank_col: str | None = None,
) -> pd.DataFrame:
    """
    Pairwise Nemenyi test (Zar 1999, via scikit-posthocs).

    If *rank_col* is provided the test operates on the pre-computed rank
    matrix — consistent with the heatmap.  Otherwise falls back to
    re-ranking the raw metric pivot.
    Returns a symmetric DataFrame of p-values.
    """
    if not _POSTHOCS_AVAILABLE:
        raise ImportError("pip install scikit-posthocs")

    if rank_col is not None and rank_col in df.columns:
        pivot = df.pivot_table(
            index=dataset_col, columns=method_col, values=rank_col, aggfunc="mean"
        ).dropna()
    else:
        pivot = df.pivot_table(
            index=dataset_col, columns=method_col, values=metric, aggfunc="mean"
        ).dropna()

    if pivot.shape[0] < 2 or pivot.shape[1] < 2:
        raise ValueError("Nemenyi needs ≥2 blocks and ≥2 methods.")

    p_mat: pd.DataFrame = sp.posthoc_nemenyi_friedman(pivot.values)
    p_mat.index   = pivot.columns
    p_mat.columns = pivot.columns
    return p_mat


def run_stats_per_context(
    df: pd.DataFrame,
    friedman_metric: str = "Test_Accuracy",
) -> dict[str, dict]:
    """
    Run Friedman + Nemenyi for every (classifier, tuning_method, test_size)
    context independently, using pre-computed Rank_{metric} columns so that
    average ranks match the heatmap exactly.

    Returns
    -------
    dict keyed by "{clf}__{method}__{ts}" → {
        "friedman":  result dict (or None),
        "nemenyi":   p-value DataFrame (or None),
        "skipped":   reason string if skipped,
    }
    """
    rank_col = f"Rank_{friedman_metric}"
    results: dict[str, dict] = {}

    for (clf, method, ts), ctx in df.groupby(
        ["classifier", "tuning_method", "test_size"]
    ):
        key = f"{clf}__{method}__{ts}"
        entry: dict = {"friedman": None, "nemenyi": None, "skipped": None}

        try:
            fr = friedman_test(ctx, "method_label", "dataset", friedman_metric, rank_col=rank_col)
            entry["friedman"] = fr

            if fr["significant"] and _POSTHOCS_AVAILABLE:
                try:
                    entry["nemenyi"] = nemenyi_test(
                        ctx, "method_label", "dataset", friedman_metric,
                        rank_col=rank_col,
                    )
                except ValueError as exc:
                    entry["skipped"] = f"Nemenyi skipped: {exc}"

            elif not fr["significant"]:
                entry["skipped"] = "Friedman not significant – Nemenyi not run"

            elif not _POSTHOCS_AVAILABLE:
                entry["skipped"] = "scikit-posthocs not installed"

        except ValueError as exc:
            entry["skipped"] = str(exc)

        results[key] = entry

    return results


# 7.  Method label helper

def make_method_label(df: pd.DataFrame) -> pd.DataFrame:
    """
    accuracy row     → "accuracy"
    tvl, m=8, S=True → "tvl_m8"   (sampling always True after Change 1)
    """
    df = df.copy()

    def _label(row: pd.Series) -> str:
        if row["scoring"] == "accuracy":
            return "accuracy"
        try:
            return f"tvl_m{int(row['m'])}"
        except (ValueError, TypeError):
            return "tvl_mNA"

    df["method_label"] = df.apply(_label, axis=1)
    return df


def _sorted_method_labels(labels: list[str]) -> list[str]:
    """
    Return labels ordered as: accuracy first, then tvl_m* ascending by m.
    """
    acc   = [l for l in labels if l == "accuracy"]
    tvl   = sorted(
        [l for l in labels if l.startswith("tvl_m")],
        key=lambda s: int(re.search(r"tvl_m(\d+)", s).group(1))
        if re.search(r"tvl_m(\d+)", s) else 0,
    )
    return acc + tvl


def _xaxis_label(method_label: str) -> str:
    """Convert internal label to human-readable x-axis tick."""
    if method_label == "accuracy":
        return "accuracy"
    m = re.search(r"tvl_m(\d+)", method_label)
    return f"tvl_m={m.group(1)}" if m else method_label

# 8.  Shared box-plot helpers

REL_YLABELS: dict[str, str] = {
    "Test_Accuracy": "NTAcc (%)",
    "Gen_Error":     "NGE (%)",
    "AUROC":         "NAUROC (%)",
    "F1_Score":      "NF1 (%)",
}


def _draw_boxes(ax, ctx, metric, ordered, cmap):
    """Draw boxes for one metric subplot on *ax*. Returns True if drawn."""
    rel_col = f"Rel_{metric}"
    if rel_col not in ctx.columns:
        ax.set_visible(False)
        return False

    data_per_label, colors = [], []
    tvl_idx = 0
    for label in ordered:
        vals = pd.to_numeric(
            ctx.loc[ctx["method_label"] == label, rel_col], errors="coerce"
        ).dropna().tolist()
        data_per_label.append(vals)
        if label == "accuracy":
            colors.append("#4C72B0")
        else:
            colors.append(cmap[tvl_idx])
            tvl_idx += 1

    bp = ax.boxplot(
        data_per_label,
        patch_artist=True,
        medianprops=dict(color="black", linewidth=1.8),
        whiskerprops=dict(linewidth=1.2),
        capprops=dict(linewidth=1.2),
        flierprops=dict(marker="o", markersize=4, alpha=0.5),
    )
    for patch, color in zip(bp["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.8)

    ax.axhline(0, color="red", linewidth=1.0, linestyle="--", alpha=0.7)
    x_ticks = [_xaxis_label(l) for l in ordered]
    ax.set_xticks(range(1, len(ordered) + 1))
    ax.set_xticklabels(x_ticks, rotation=35, ha="right", fontsize=8)
    ax.set_ylabel(REL_YLABELS.get(metric, f"Relative {metric} (%)"), fontsize=9)
    ax.grid(axis="y", alpha=0.35, linestyle="--")

    y_bottom = ax.get_ylim()[0]
    for i, vals in enumerate(data_per_label, start=1):
        ax.text(i, y_bottom, f"n={len(vals)}",
                ha="center", va="bottom", fontsize=6, color="grey")
    return True


def _shared_legend(fig, cmap):
    handles = [
        mpatches.Patch(color="#4C72B0", label="Accuracy tuning"),
        mpatches.Patch(color=cmap[len(cmap) // 2], label="TvL tuning (m varies)"),
        plt.Line2D([0], [0], color="red", linestyle="--", linewidth=1.0,
                   label="Default model baseline (0%)"),
    ]
    fig.legend(handles=handles, loc="lower center",
               ncol=3, fontsize=8, bbox_to_anchor=(0.5, 0.0))


# 8a.  Option 1 — combined figure per (classifier, tuning_method)
#       Rows = test_sizes, Columns = metrics

def plot_boxplots_combined(
    df: pd.DataFrame,
    output_dir: Path,
) -> None:
    """
    One PNG per (classifier, tuning_method).
    Rows = test sizes (sorted).  Columns = PLOT_METRICS.
    Each cell: box plot of relative scores across datasets.

    File: {output_dir}/boxplot/{clf}__{method}.png
    """
    bp_dir = output_dir / "boxplot"
    bp_dir.mkdir(parents=True, exist_ok=True)

    for (clf, method), grp in df.groupby(["classifier", "tuning_method"]):
        test_sizes = sorted(grp["test_size"].dropna().unique().tolist())
        n_rows = len(test_sizes)
        n_cols = len(PLOT_METRICS)

        fig, axes = plt.subplots(
            n_rows, n_cols,
            figsize=(5.5 * n_cols, 4.5 * n_rows),
            squeeze=False,
        )
        fig.suptitle(
            f"Classifier: {clf}  |  Tuning method: {method}",
            fontsize=13, fontweight="bold",
        )

        all_labs = grp["method_label"].dropna().unique().tolist()
        ordered  = _sorted_method_labels(all_labs)
        n_tvl    = sum(1 for l in ordered if l.startswith("tvl"))
        cmap     = plt.cm.Oranges(np.linspace(0.35, 0.85, max(n_tvl, 1)))

        for row_i, ts in enumerate(test_sizes):
            ctx = grp[grp["test_size"] == ts]
            for col_i, metric in enumerate(PLOT_METRICS):
                ax = axes[row_i][col_i]
                _draw_boxes(ax, ctx, metric, ordered, cmap)
                if row_i == 0:
                    ax.set_title(PLOT_METRIC_TITLES.get(metric, metric.replace("_", " ")), fontsize=11, pad=6)
                # Show metric abbreviation on every column; prepend test_size only on leftmost
                ylabel = REL_YLABELS.get(metric, f"N{metric} (%)")
                if col_i == 0:
                    ax.set_ylabel(f"test_size={ts}\n{ylabel}", fontsize=9)
                else:
                    ax.set_ylabel(ylabel, fontsize=9)

        _shared_legend(fig, cmap)
        fig.tight_layout(rect=[0, 0.04, 1, 0.97])
        out_path = bp_dir / f"{clf}__{method}.png"
        fig.savefig(out_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"  Box plot (combined) saved: {out_path.name}")



def plot_rank_heatmap(
    df: pd.DataFrame,
    output_dir: Path,
) -> None:
    """
    One PNG per (classifier, tuning_method).
    Layout: one subplot per metric in PLOT_METRICS side by side.
    Each subplot is a heatmap where:
        rows    = methods ordered as accuracy, tvl_m=2, tvl_m=4, …
        columns = test sizes (sorted)
        colour  = mean rank across datasets (green=1=best, red=k=worst)
    Cell text shows the mean rank value rounded to 1 decimal place.

    File: {output_dir}/rank_heatmap/{clf}__{method}.png
    """
    import matplotlib.colors as mcolors

    hm_dir = output_dir / "rank_heatmap"
    hm_dir.mkdir(parents=True, exist_ok=True)

    for (clf, method), grp in df.groupby(["classifier", "tuning_method"]):
        all_labs   = grp["method_label"].dropna().unique().tolist()
        ordered    = _sorted_method_labels(all_labs)
        test_sizes = sorted(grp["test_size"].dropna().unique().tolist())
        x_labels   = [str(ts) for ts in test_sizes]
        y_labels   = [_xaxis_label(l) for l in ordered]
        k          = len(ordered)      # number of methods = max rank

        fig, axes = plt.subplots(
            1, len(PLOT_METRICS),
            figsize=(3.2 * len(test_sizes) * len(PLOT_METRICS) / len(PLOT_METRICS) + 1,
                     0.6 * k + 2.5),
            squeeze=False,
        )
        axes = axes[0]   # single row

        fig.suptitle(
            f"Mean rank heatmap\nClassifier: {clf}  |  Method: {method}",
            fontsize=12, fontweight="bold",
        )

        # Colour map: green (rank 1, best) → red (rank k, worst)
        cmap = mcolors.LinearSegmentedColormap.from_list(
            "rank_cmap",
            ["#2ecc71", "#f9f9a0", "#e74c3c"],   # green → yellow → red
        )

        for ax, metric in zip(axes, PLOT_METRICS):
            rank_col = f"Rank_{metric}"
            if rank_col not in grp.columns:
                ax.set_visible(False)
                continue

            # Build matrix: rows = methods, cols = test_sizes
            matrix = np.full((k, len(test_sizes)), np.nan)
            for row_i, label in enumerate(ordered):
                for col_i, ts in enumerate(test_sizes):
                    vals = pd.to_numeric(
                        grp.loc[
                            (grp["test_size"] == ts) &
                            (grp["method_label"] == label),
                            rank_col
                        ], errors="coerce"
                    ).dropna()
                    if len(vals):
                        matrix[row_i, col_i] = vals.mean()

            # Draw heatmap — rank range always 1..k
            im = ax.imshow(
                matrix,
                cmap=cmap,
                vmin=1, vmax=k,
                aspect="auto",
            )

            # Cell annotations
            for row_i in range(k):
                for col_i in range(len(test_sizes)):
                    val = matrix[row_i, col_i]
                    if not np.isnan(val):
                        # Choose text colour for readability
                        bg_norm = (val - 1) / max(k - 1, 1)
                        txt_color = "black" if 0.25 < bg_norm < 0.75 else "black"
                        ax.text(col_i, row_i, f"{val:.2f}",
                                ha="center", va="center",
                                fontsize=9, fontweight="bold",
                                color=txt_color)

            # Axes formatting
            ax.set_xticks(range(len(test_sizes)))
            ax.set_xticklabels(x_labels, fontsize=9)
            ax.set_xlabel("Test size", fontsize=9)
            ax.set_yticks(range(k))
            ax.set_yticklabels(y_labels, fontsize=9)
            ax.set_title(PLOT_METRIC_TITLES.get(metric, metric.replace("_", " ")), fontsize=11, pad=6)

            # Colour-bar
            cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            cbar.set_label("Mean rank", fontsize=8)
            cbar.set_ticks([1, (k + 1) / 2, k])
            cbar.set_ticklabels(["1 (best)", f"{(k+1)/2:.1f}", f"{k} (worst)"])
            cbar.ax.tick_params(labelsize=7)

            # Grid lines between cells
            ax.set_xticks(np.arange(-0.5, len(test_sizes), 1), minor=True)
            ax.set_yticks(np.arange(-0.5, k, 1), minor=True)
            ax.grid(which="minor", color="white", linewidth=1.5)
            ax.tick_params(which="minor", bottom=False, left=False)

        fig.tight_layout(rect=[0, 0, 1, 0.93])
        out_path = hm_dir / f"{clf}__{method}.png"
        fig.savefig(out_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"  Rank heatmap saved: {out_path.name}")


# 9.  Critical-difference diagram per context  

def _critical_difference(k: int, n: int, alpha: float = 0.05) -> float:
    """
    CD = q_α * sqrt(k(k+1) / 6N)

    q_α from the Studentized range distribution table (Demšar 2006).
    Only α=0.05 supported via _Q_ALPHA_05 lookup.
    """
    if alpha != 0.05:
        raise ValueError("Only α=0.05 is supported for the CD diagram.")
    if k < 2 or k > 25:
        raise ValueError(f"k must be between 2 and 25; got {k}.")
    q = _Q_ALPHA_05[k]
    return q * math.sqrt(k * (k + 1) / (6 * n))


def _draw_cd_diagram(
    avg_ranks: pd.Series,
    cd: float,
    n_datasets: int,
    title: str,
    ax: plt.Axes,
) -> None:
    """
    Draw a Demšar-style CD diagram on *ax*.

    Horizontal axis  : average rank (left = rank 1 = best).
    Each method       : a dot + label.
    Horizontal bars   : connect methods that are NOT significantly different
                        (|rank_i - rank_j| < CD).
    CD bar            : shown in the top-left corner.
    """
    # Sort by average rank ascending (rank 1 = best on the left)
    ranks = avg_ranks.sort_values()
    labels = [_xaxis_label(l) for l in ranks.index]
    rank_vals = ranks.values
    k = len(ranks)

    #  axis limits 
    r_min = max(1.0, rank_vals.min() - 0.5)
    r_max = rank_vals.max() + 0.5
    ax.set_xlim(r_min - 0.3, r_max + 0.3)
    ax.set_ylim(-1, k + 2)
    ax.invert_xaxis()           # rank 1 on the left
    ax.axis("off")

    #  vertical positions for method labels (evenly spaced) 
    y_positions = np.linspace(k - 1, 0, k)

    # dots + labels 
    for i, (label, rv, yp) in enumerate(zip(labels, rank_vals, y_positions)):
        color = "#4C72B0" if label == "accuracy" else "#E07B39"
        # Horizontal line from label to dot
        ax.plot([r_max + 0.2, rv], [yp, yp], color="grey",
                linewidth=0.8, linestyle="--", zorder=1)
        ax.scatter([rv], [yp], color=color, s=60, zorder=3)
        # Shift label slightly above the dashed line so it does not overlap
        ax.text(r_max + 0.22, yp + 0.18, label, va="bottom", ha="left",
                fontsize=9, color=color)
        ax.text(rv, yp + 0.35, f"{rv:.2f}", ha="center", fontsize=7, color="dimgrey")

    #  CD bracket in top-left corner 
    cd_y    = k + 1.0
    cd_left = r_max + 0.1
    ax.annotate("", xy=(cd_left - cd, cd_y),
                xytext=(cd_left, cd_y),
                arrowprops=dict(arrowstyle="<->", color="black", lw=1.4))
    ax.text(cd_left - cd / 2, cd_y + 0.35,
            f"CD = {cd:.3f}", ha="center", fontsize=8)

    #  horizontal bars connecting non-significantly-different pairs 
    # Group into cliques: methods whose ranks all lie within CD of each other.
    # Simple algorithm: for each method, extend a bar to the furthest method
    # reachable within CD.
    bar_y_step = 0.22
    bar_y_base = -0.5
    plotted_bars: list[tuple[float, float]] = []

    for i in range(k):
        j = i
        while j + 1 < k and abs(rank_vals[j + 1] - rank_vals[i]) < cd:
            j += 1
        if j > i:
            bar_left  = rank_vals[i]
            bar_right = rank_vals[j]
            # stack bars vertically to avoid overlap
            bar_y = bar_y_base - len(plotted_bars) * bar_y_step
            ax.plot([bar_left, bar_right], [bar_y, bar_y],
                    color="black", linewidth=2.5, solid_capstyle="round", zorder=2)
            plotted_bars.append((bar_left, bar_right))

    ax.set_title(title, fontsize=10, pad=4)


def plot_cd_diagrams_per_context(
    df: pd.DataFrame,
    stats_per_context: dict[str, dict],
    friedman_metric: str,
    output_dir: Path,
) -> None:
    """
    For each (classifier, tuning_method, test_size) context, draw a CD diagram.
    *output_dir* is the exact directory where PNGs are written (caller sets the
    metric sub-folder, e.g. ``out / "cd_diagrams" / "Test_Accuracy"``).
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    for (clf, method, ts), ctx in df.groupby(
        ["classifier", "tuning_method", "test_size"]
    ):
        key   = f"{clf}__{method}__{ts}"
        entry = stats_per_context.get(key, {})
        fr    = entry.get("friedman")

        fig, ax = plt.subplots(figsize=(9, 5))
        title   = (
            f"CD Diagram — {PLOT_METRIC_TITLES.get(friedman_metric, friedman_metric.replace('_',' '))} | "
            f"{clf} · {method} · test_size={ts}"
        )

        if fr is None:
            ax.text(0.5, 0.5, f"Tests could not run:\n{entry.get('skipped','')}",
                    ha="center", va="center", transform=ax.transAxes, fontsize=11)
            ax.axis("off")
            ax.set_title(title, fontsize=10)
        else:
            avg_ranks  = fr["avg_ranks"]
            n_datasets = fr["n_datasets"]
            k          = fr["n_methods"]

            try:
                cd = _critical_difference(k, n_datasets)
            except ValueError as exc:
                ax.text(0.5, 0.5, f"CD cannot be computed:\n{exc}",
                        ha="center", va="center", transform=ax.transAxes)
                ax.axis("off")
                ax.set_title(title, fontsize=10)
                fig.tight_layout()
                fig.savefig(output_dir / f"{key}.png", dpi=150, bbox_inches="tight")
                plt.close(fig)
                continue

            sig_str = (
                f"Friedman χ²={fr['statistic']:.3f}, p={fr['p_value']:.3e} "
                f"({'significant' if fr['significant'] else 'NOT significant'})"
            )
            _draw_cd_diagram(avg_ranks, cd, n_datasets,
                             title=f"{title}\n{sig_str}", ax=ax)

        fig.tight_layout()
        out_path = output_dir / f"{key}.png"
        fig.savefig(out_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"  CD diagram saved: {out_path.name}")


# 10.  Build boxplot data (CSV helper, kept for reference)

def build_all_boxplot_data(df: pd.DataFrame) -> dict[str, dict[str, pd.DataFrame]]:
    """
    Return nested dict {metric: {context_key: DataFrame}} for downstream use.
    Each DataFrame contains all scoring rows (accuracy + tvl_m*) for the context.
    """
    rel_col_map = {m: f"Rel_{m}" for m in METRICS if f"Rel_{m}" in df.columns}
    result: dict[str, dict[str, pd.DataFrame]] = {m: {} for m in rel_col_map}

    for (clf, method, ts), ctx in df.groupby(["classifier", "tuning_method", "test_size"]):
        key = f"{clf}__{method}__{ts}"
        for metric, rel_col in rel_col_map.items():
            keep = ["dataset", "scoring", "m", "sampling", "method_label",
                    metric, rel_col]
            keep = [c for c in keep if c in ctx.columns]
            slc  = ctx[keep].dropna(subset=[rel_col]).copy().reset_index(drop=True)
            if not slc.empty:
                result[metric][key] = slc

    return result


# Main pipeline

def main(
    results_root:    str = "Results",
    datasets:        list[str] | None = None,
    classifiers:     list[str] | None = None,
    tuning_methods:  list[str] | None = None,
    test_sizes:      list[float] | None = None,
    m_values:        list[int] | None = None,
    friedman_metrics: list[str] | None = None,
    output_dir:      str | None = None,
    do_boxplots:     bool = True,
    do_rank_heatmap:  bool = True,
    do_cd:           bool = True,
) -> dict:
    """
    End-to-end analysis pipeline.

    Parameters
    ----------
    results_root:     Root folder with JSON experiment results.
    datasets:         Dataset names to include (None = all).
    classifiers:      Filter by classifier(s).
    tuning_methods:   Filter by search method(s).
    test_sizes:       Filter by test-size value(s).
    m_values:         Additional m filter on top of VALID_M_VALUES.
                      None = keep all values in VALID_M_VALUES.
    friedman_metrics: List of metrics to run Friedman+Nemenyi+CD for.
                      Defaults to all four METRICS.
    output_dir:       Directory to write all outputs (None = no files written).
    """
    if friedman_metrics is None:
        friedman_metrics = METRICS

    # Load  (Sampling=False automatically skipped inside load_from_json)
    print(f"Loading JSON results from '{results_root}' …")
    df = load_from_json(results_root, datasets=datasets)

    if df.empty:
        print("No results found.  Check results_root and dataset names.")
        return {}

    print(f"  {len(df)} rows from {df['dataset'].nunique()} dataset(s).")

  
    # Filters
    if classifiers:
        df = df[df["classifier"].isin(classifiers)]
    if tuning_methods:
        df = df[df["tuning_method"].isin(tuning_methods)]
    if test_sizes:
        df = df[df["test_size"].isin(test_sizes)]
    if m_values is not None:
        df = df[df["scoring"].eq("accuracy") | df["m"].isin(m_values)]

    
    # Step 1 – relative scores
    print("Computing relative scores …")
    df = add_relative_scores(df)

    # Step 2 – method labels
    df = make_method_label(df)


    # Step 3 – ranks on absolute values
    print("Ranking methods …")
    rank_group_cols = ["dataset", "classifier", "tuning_method", "test_size"]
    df = rank_all_metrics(df, rank_group_cols)


    # Step 4 – aggregate across datasets
    print("Aggregating across datasets …")
    method_cols = ["classifier", "tuning_method", "scoring", "m", "sampling", "test_size"]
    agg_value_cols = (
        [m for m in METRICS         if m            in df.columns] +
        [f"Rel_{m}"  for m in METRICS if f"Rel_{m}"  in df.columns] +
        [f"Rank_{m}" for m in METRICS if f"Rank_{m}" in df.columns]
    )
    aggregated = aggregate_by_method(df, method_cols, value_cols=agg_value_cols)


    # Step 5 – Friedman + Nemenyi for ALL requested metrics
    all_stats: dict[str, dict[str, dict]] = {}
    for fmetric in friedman_metrics:
        print(f"\nRunning Friedman + Nemenyi for '{fmetric}' …")
        all_stats[fmetric] = run_stats_per_context(df, fmetric)
        n_sig = sum(
            1 for v in all_stats[fmetric].values()
            if v["friedman"] and v["friedman"]["significant"]
        )
        print(f"  {len(all_stats[fmetric])} contexts; {n_sig} significant.")


    # Step 6 – boxplot data dict (CSV reference)
    boxplot_data = build_all_boxplot_data(df)


    # Save outputs
    if output_dir:
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)

        # Full + aggregated
        df.to_csv(out / "full_results.csv", index=False)
        aggregated.to_csv(out / "aggregated_by_method.csv", index=False)

        # Rank CSVs
        ranks_dir = out / "ranks"
        ranks_dir.mkdir(exist_ok=True)
        id_cols = ["dataset", "classifier", "tuning_method", "test_size",
                   "scoring", "m", "sampling", "method_label"]
        for metric in METRICS:
            rc = f"Rank_{metric}"
            if rc in df.columns:
                cols = [c for c in id_cols if c in df.columns] + [metric, rc]
                df[cols].to_csv(ranks_dir / f"ranks_{metric}.csv", index=False)

        # Relative-score CSVs
        rel_dir = out / "relative_scores"
        rel_dir.mkdir(exist_ok=True)
        for metric in METRICS:
            rc = f"Rel_{metric}"
            if rc in df.columns:
                cols = [c for c in id_cols if c in df.columns] + [metric, rc]
                df[cols].to_csv(rel_dir / f"relative_{metric}.csv", index=False)

        # Boxplot CSVs
        bp_csv_dir = out / "boxplot_csv"
        bp_csv_dir.mkdir(exist_ok=True)
        for metric, ctx_dict in boxplot_data.items():
            md = bp_csv_dir / metric
            md.mkdir(exist_ok=True)
            for ctx_key, bdf in ctx_dict.items():
                bdf.to_csv(md / f"boxplot_{metric}__{ctx_key}.csv", index=False)

        # Stats + CD diagrams — one sub-folder per metric
        for fmetric, stats_per_context in all_stats.items():
            stats_dir = out / "stats" / fmetric
            stats_dir.mkdir(parents=True, exist_ok=True)

            for ctx_key, entry in stats_per_context.items():
                fr  = entry["friedman"]
                nem = entry["nemenyi"]
                rows_out: list[dict] = []
                if fr is not None:
                    for method, rank in fr["avg_ranks"].items():
                        rows_out.append({
                            "method":          method,
                            "avg_rank":        round(rank, 4),
                            "friedman_stat":   round(fr["statistic"], 4),
                            "friedman_pvalue": round(fr["p_value"], 6),
                            "significant":     fr["significant"],
                            "n_datasets":      fr["n_datasets"],
                            "skipped":         entry.get("skipped", ""),
                        })
                if rows_out:
                    pd.DataFrame(rows_out).to_csv(
                        stats_dir / f"friedman_{ctx_key}.csv", index=False
                    )
                if nem is not None:
                    nem.round(4).to_csv(
                        stats_dir / f"nemenyi_{ctx_key}.csv"
                    )

            # CD diagrams for this metric
            if do_cd:
                plot_cd_diagrams_per_context(
                    df, stats_per_context, fmetric,
                    out / "cd_diagrams" / fmetric,
                )
    
        # Box-plot PNGs — Option 1: combined 4-row figure per (clf, method)
        if do_boxplots:
            plot_boxplots_combined(df, out)
    
        # Summary

        # Rank heatmap — Option B: mean rank grid (methods x test_sizes)
        if do_rank_heatmap:
            print("Generating rank heatmap plots …")
            plot_rank_heatmap(df, out)
        n_bp  = len(list((out / "boxplot").glob("*.png")))       if (out / "boxplot").exists()       else 0
        n_hm  = len(list((out / "rank_heatmap").glob("*.png"))) if (out / "rank_heatmap").exists() else 0
        n_cd  = sum(
            len(list(p.glob("*.png")))
            for p in (out / "cd_diagrams").iterdir()
            if p.is_dir()
        ) if (out / "cd_diagrams").exists() else 0
        print(f"\nAll outputs written to '{out}':")
        print(f"  full_results.csv")
        print(f"  aggregated_by_method.csv")
        print(f"  ranks/                     ({len(METRICS)} CSVs)")
        print(f"  relative_scores/           ({len(METRICS)} CSVs)")
        print(f"  stats/{{metric}}/            (friedman_*.csv + nemenyi_*.csv per context, {len(friedman_metrics)} metrics)")
        print(f"  boxplot/                   ({n_bp} PNGs — rows=test_size, cols=metric)")
        print(f"  rank_heatmap/              ({n_hm} PNGs — mean rank heatmap methods × test sizes)")
        print(f"  cd_diagrams/{{metric}}/      ({n_cd} PNGs total, {len(friedman_metrics)} metrics)")


    # Terminal summary
    print("\n" + "=" * 70)
    print("AGGREGATED RESULTS  (mean ± std across datasets)")
    print("=" * 70)
    summary_cols = (
        [c for c in method_cols if c in aggregated.columns]
        + [f"{m}_mean"      for m in METRICS if f"{m}_mean"      in aggregated.columns]
        + [f"Rel_{m}_mean"  for m in METRICS if f"Rel_{m}_mean"  in aggregated.columns]
        + [f"Rank_{m}_mean" for m in METRICS if f"Rank_{m}_mean" in aggregated.columns]
    )
    print(aggregated[summary_cols].to_string(index=False))

    return {
        "df":          df,
        "aggregated":  aggregated,
        "all_stats":   all_stats,
        "boxplot_data": boxplot_data,
    }


# Group-aware runner

def run_all_groups(
    results_root:    str = "Results",
    output_root:     str = "Analysis_Output",
    classifiers:     list[str] | None = None,
    tuning_methods:  list[str] | None = None,
    test_sizes:      list[float] | None = None,
    m_values:        list[int] | None = None,
    friedman_metrics: list[str] | None = None,
    groups:          list[str] | None = None,
    do_boxplots:     bool = True,
    do_rank_heatmap:  bool = True,
    do_cd:           bool = True,
) -> dict[str, dict]:
    """
    Run the full analysis pipeline once per dataset group defined in
    DATASET_GROUPS, saving each group's outputs to its own sub-folder:

        {output_root}/{group_name}/

    Parameters
    ----------
    results_root:     Root folder with JSON experiment results.
    output_root:      Root output folder; one sub-folder is created per group.
    classifiers:      Optional filter applied to every group.
    tuning_methods:   Optional filter applied to every group.
    test_sizes:       Optional filter applied to every group.
    m_values:         Optional extra m filter (on top of VALID_M_VALUES).
    friedman_metrics: Metrics for Friedman/Nemenyi/CD (default: all 4).
    groups:           Subset of group names to run (default: all non-empty groups).

    Returns
    -------
    dict keyed by group name → return value of main() for that group.
    """
    all_results: dict[str, dict] = {}

    # Determine which groups to run
    target_groups = {
        name: datasets
        for name, datasets in DATASET_GROUPS.items()
        if datasets  # skip empty groups
    }
    if groups:
        target_groups = {k: v for k, v in target_groups.items() if k in groups}

    if not target_groups:
        print("No groups to run — fill in DATASET_GROUPS first.")
        return {}

    print(f"Running analysis for {len(target_groups)} group(s): "
          f"{list(target_groups.keys())}")

    for group_name, datasets in target_groups.items():
        print(f"\n{'='*70}")
        print(f"GROUP: {group_name}  ({len(datasets)} datasets: {datasets})")
        print(f"{'='*70}")

        group_output = str(Path(output_root) / group_name)

        result = main(
            results_root=results_root,
            datasets=datasets,
            classifiers=classifiers,
            tuning_methods=tuning_methods,
            test_sizes=test_sizes,
            m_values=m_values,
            friedman_metrics=friedman_metrics,
            output_dir=group_output,
            do_boxplots=do_boxplots,
            do_cd=do_cd,
            do_rank_heatmap=do_rank_heatmap,
        )

    print(f"\n{'='*70}")
    print(f"All groups completed. Outputs written under '{output_root}/'")
    for g in target_groups:
        print(f"  {output_root}/{g}/")
    return all_results


# CLI
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="TvL vs Accuracy statistical analysis — group-aware"
    )
    parser.add_argument("--results_root",  default="Results",
                        help="Root folder containing JSON experiment results")
    parser.add_argument("--output_root",   default="Analysis_Output",
                        help="Root output folder; sub-folders are created per group")
    parser.add_argument("--groups",        nargs="*", default=None,
                        help="Group names to run (default: all non-empty groups in DATASET_GROUPS)")
    parser.add_argument("--classifiers",   nargs="*")
    parser.add_argument("--tuning_methods",nargs="*")
    parser.add_argument("--test_sizes",    nargs="*", type=float)
    parser.add_argument("--m_values",      nargs="*", type=int)
    parser.add_argument("--friedman_metrics", nargs="*", default=None,
                        choices=METRICS,
                        help="Metrics for Friedman/Nemenyi/CD (default: all 4)")

    # Legacy single-group mode: pass --datasets + --output_dir directly
    parser.add_argument("--datasets",      nargs="*", default=None,
                        help="Run a single ad-hoc group (bypasses DATASET_GROUPS)")
    parser.add_argument("--output_dir",    default=None,
                        help="Output dir for single ad-hoc run (used with --datasets)")

    # Plot selection flags — set to 0/false to skip individual plot types
    parser.add_argument("--no_boxplots",  action="store_true",
                        help="Skip combined box-plot PNGs")
    parser.add_argument("--no_rank_heatmap",  action="store_true",
                        help="Skip rank heatmap plots")
    parser.add_argument("--no_cd",        action="store_true",
                        help="Skip CD diagram PNGs")

    args = parser.parse_args()

    plot_kw = dict(
        do_boxplots     = not args.no_boxplots,
        do_rank_heatmap  = not args.no_rank_heatmap,
        do_cd           = not args.no_cd,
    )

    if args.datasets:
        #  Single ad-hoc run (legacy mode, no groups) 
        main(
            results_root=args.results_root,
            datasets=args.datasets,
            classifiers=args.classifiers,
            tuning_methods=args.tuning_methods,
            test_sizes=args.test_sizes,
            m_values=args.m_values,
            friedman_metrics=args.friedman_metrics,
            output_dir=args.output_dir or args.output_root,
            **plot_kw,
        )
    else:
        #  Group mode 
        run_all_groups(
            results_root=args.results_root,
            output_root=args.output_root,
            classifiers=args.classifiers,
            tuning_methods=args.tuning_methods,
            test_sizes=args.test_sizes,
            m_values=args.m_values,
            friedman_metrics=args.friedman_metrics,
            groups=args.groups,
            **plot_kw,
        )