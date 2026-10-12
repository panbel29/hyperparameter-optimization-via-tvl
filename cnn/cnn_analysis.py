"""
Visualisation of CNN HPO results stored as JSON files.

Directory layout expected
-------------------------
Results/CNN/<DATASET>/CNN/<method>/<scoring>/mt_<mt>/<...>.json        (accuracy)
Results/CNN/<DATASET>/CNN/<method>/tvl/m_<m>/mt_<mt>/<...>.json        (tvl)

Three plot types are produced per dataset
-----------------------------------------
1. HEATMAP  — one figure per tuning method
   Rows    : mt values (train sizes)
   Columns : scoring keys  (accuracy, tvl_m2, tvl_m4, …)
   Cells   : metric value (test_accuracy / gen_error / f1 / auroc)
   One heatmap figure = 4 subplots (one per metric), saved per method.

2. LINE PLOT — one figure per tuning method
   X-axis  : mt values
   Lines   : one per scoring key (tvl_m=2, tvl_m=4, …)
   Horizontal dashed line : value of the accuracy-tuned model
   One figure = 4 subplots (one per metric), saved per method.

3. LEARNING CURVES — one figure per (scoring_key, method)
   X-axis  : epoch
   Two lines per figure: train_history, val_history
   Saved per (method, scoring_key, mt).

Usage
-----
    python cnn_analysis.py                       # all datasets
    python cnn_analysis.py --dataset MNIST       # single dataset
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import seaborn as sns

# Paths 
RESULTS_ROOT = Path("Results") / "CNN"
PLOTS_ROOT   = Path("Plots")   / "CNN"

# Constants 
DATASETS  = ["FMNIST"] # Change for the other two datasets
METHODS   = ["random_search", "bayesian", "nelder_mead"]
M_VALUES  = [2, 4, 8, 16, 32, 64, 128]
METRICS   = ["test_accuracy", "generalization_error", "f1_score", "auroc"]
METRIC_LABELS = {
    "test_accuracy":        "Test Accuracy (%)",
    "generalization_error": "Generalization Error (%)",
    "f1_score":             "F1 Score (%)",
    "auroc":                "AUROC",
}
METHOD_LABELS = {
    "random_search": "Random Search",
    "bayesian":      "Bayesian Optimization",
    "nelder_mead":   "Nelder-Mead",
}

# Colour palette 
ACCURACY_COLOR = "#2c7bb6"          # blue  — accuracy-tuned baseline
TVL_CMAP       = "tab10"


# 1.  JSON loading helpers

def _normalise_metrics(raw: dict[str, Any]) -> dict[str, float]:
    """Return a flat dict with lower-case metric names."""
    tm = raw.get("tuned_metrics", {})
    return {
        "test_accuracy":        float(tm.get("test_accuracy",        float("nan"))),
        "generalization_error": float(tm.get("generalization_error", float("nan"))),
        "f1_score":             float(tm.get("F1_Score",             float("nan"))),
        "auroc":                float(tm.get("AUROC",                float("nan"))),
    }


def load_accuracy_result(dataset: str, method: str, mt: int) -> dict[str, Any] | None:
    """Load the JSON for an accuracy-scored run."""
    base = RESULTS_ROOT / dataset / "CNN" / method / "accuracy" / f"mt_{mt}"
    jsons = sorted(base.glob("*.json"))
    if not jsons:
        return None
    with open(jsons[-1]) as f:
        raw = json.load(f)
    result = _normalise_metrics(raw)
    result["train_history"] = raw.get("tuned_train_history", [])
    result["val_history"]   = raw.get("tuned_val_history",   [])
    return result


def load_tvl_result(dataset: str, method: str, mt: int, m: int) -> dict[str, Any] | None:
    """Load the JSON for a tvl-scored run."""
    base = RESULTS_ROOT / dataset / "CNN" / method / "tvl" / f"m_{m}" / f"mt_{mt}"
    jsons = sorted(base.glob("*.json"))
    if not jsons:
        return None
    with open(jsons[-1]) as f:
        raw = json.load(f)
    result = _normalise_metrics(raw)
    result["train_history"] = raw.get("tuned_train_history", [])
    result["val_history"]   = raw.get("tuned_val_history",   [])
    return result


def discover_mt_values(dataset: str, method: str) -> list[int]:
    """Return sorted list of mt values that have at least one result file."""
    base = RESULTS_ROOT / dataset / "CNN" / method / "accuracy"
    mts: list[int] = []
    for d in sorted(base.iterdir()) if base.exists() else []:
        m = re.match(r"mt_(\d+)$", d.name)
        if m and any(d.glob("*.json")):
            mts.append(int(m.group(1)))
    return sorted(mts)



# 2.  Build data tables

def build_scoring_keys(m_values: list[int] = M_VALUES) -> list[str]:
    """Return ordered list of scoring labels: ['accuracy', 'tvl_m2', ...]."""
    return ["accuracy"] + [f"tvl_m{m}" for m in m_values]


def build_data_table(
    dataset: str,
    method:  str,
    mt_values: list[int],
    m_values:  list[int] = M_VALUES,
) -> dict[str, dict[str, dict[str, float]]]:
    """
    Returns nested dict:
        data[scoring_key][mt_str][metric] = float value
    """
    keys = build_scoring_keys(m_values)
    data: dict[str, dict[str, dict[str, float]]] = {k: {} for k in keys}

    for mt in mt_values:
        mt_str = str(mt)

        # accuracy
        res = load_accuracy_result(dataset, method, mt)
        if res:
            data["accuracy"][mt_str] = {m: res[m] for m in METRICS}

        # tvl
        for m in m_values:
            key = f"tvl_m{m}"
            res = load_tvl_result(dataset, method, mt, m)
            if res:
                data[key][mt_str] = {m: res[m] for m in METRICS}

    return data



# 3.  Plot 1 — Heatmaps

def plot_heatmap(
    dataset:   str,
    method:    str,
    mt_values: list[int],
    m_values:  list[int] = M_VALUES,
    plots_root: Path = PLOTS_ROOT,
) -> None:
    """
    One figure per method with 4 subplots (one per metric).
    Rows = mt values, Columns = scoring keys.
    """
    data    = build_data_table(dataset, method, mt_values, m_values)
    keys    = build_scoring_keys(m_values)
    mt_strs = [str(mt) for mt in mt_values]

    fig, axes = plt.subplots(2, 2, figsize=(max(14, len(keys) * 0.9), 10))
    axes = axes.flatten()

    col_labels = ["Accuracy"] + [f"TvL m={m}" for m in m_values]

    for ax_idx, metric in enumerate(METRICS):
        ax = axes[ax_idx]
        matrix = np.full((len(mt_strs), len(keys)), np.nan)

        for col_i, key in enumerate(keys):
            for row_i, mt_str in enumerate(mt_strs):
                val = data.get(key, {}).get(mt_str, {}).get(metric, np.nan)
                matrix[row_i, col_i] = val

        # choose diverging vs sequential colormap per metric
        cmap = "RdYlGn" if metric != "generalization_error" else "RdYlGn_r"

        sns.heatmap(
            matrix,
            ax=ax,
            xticklabels=col_labels,
            yticklabels=[f"mt={mt}" for mt in mt_values],
            annot=True,
            fmt=".2f",
            cmap=cmap,
            linewidths=0.5,
            linecolor="white",
            cbar_kws={"shrink": 0.7},
            annot_kws={"size": 8},
        )
        ax.set_title(METRIC_LABELS[metric], fontsize=11, fontweight="bold")
        ax.set_xlabel("Scoring", fontsize=9)
        ax.set_ylabel("Train size (mt)", fontsize=9)
        ax.tick_params(axis="x", rotation=45, labelsize=7)
        ax.tick_params(axis="y", rotation=0,  labelsize=8)

    fig.suptitle(
        f"{dataset}  |  {METHOD_LABELS[method]}  —  Metric Heatmap",
        fontsize=13, fontweight="bold", y=1.01,
    )
    fig.tight_layout()

    out_dir = plots_root / dataset / "heatmaps"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"heatmap_{method}.png"
    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)
    print(f"  [heatmap]  {out_path}")


# 4.  Plot 2 — Line plots (metric vs m, horizontal line = accuracy baseline)

def plot_lines(
    dataset:   str,
    method:    str,
    mt_values: list[int],
    m_values:  list[int] = M_VALUES,
    plots_root: Path = PLOTS_ROOT,
) -> None:
    """
    One figure per (method, mt) with 4 subplots (one per metric).
    X-axis  : m values (2, 4, 8, ..., 512)
    Line    : TvL metric values across m values
    H-line  : accuracy-tuned model value (constant horizontal dashed line)
 
    Saved under: plots_root/dataset/line_plots/<method>/lines_mt<mt>.png
    """
    for mt in mt_values:
        acc_res = load_accuracy_result(dataset, method, mt)
 
        fig, axes = plt.subplots(2, 2, figsize=(13, 9))
        axes = axes.flatten()
 
        for ax_idx, metric in enumerate(METRICS):
            ax = axes[ax_idx]
 
            # TvL line: metric vs m 
            tvl_x, tvl_y = [], []
            for m in m_values:
                res = load_tvl_result(dataset, method, mt, m)
                if res is not None:
                    tvl_x.append(m)
                    tvl_y.append(res[metric])
 
            if tvl_x:
                ax.plot(
                    tvl_x, tvl_y,
                    color="#e74c3c", linewidth=2, marker="o", markersize=6,
                    label="TvL tuned",
                )
 
            #  accuracy baseline: horizontal line 
            if acc_res is not None:
                acc_val = acc_res[metric]
                if not np.isnan(acc_val):
                    ax.axhline(
                        y=acc_val,
                        color=ACCURACY_COLOR, linewidth=2,
                        linestyle="--",
                        label="Accuracy tuned",
                    )
 
            ax.set_title(METRIC_LABELS[metric], fontsize=11, fontweight="bold")
            ax.set_xlabel("m  (log₂ scale)", fontsize=9)
            ax.set_ylabel(METRIC_LABELS[metric], fontsize=9)
            ax.set_xscale("log", base=2)
            ax.set_xticks(m_values)
            ax.set_xticklabels([str(m) for m in m_values], rotation=45, fontsize=8)
            ax.xaxis.set_minor_locator(plt.NullLocator())
            ax.grid(True, alpha=0.3)
            ax.legend(fontsize=9)
 
        fig.suptitle(
            f"{dataset}  |  {METHOD_LABELS[method]}  |  mt={mt}  —  Metrics vs m",
            fontsize=13, fontweight="bold",
        )
        fig.tight_layout()
 
        out_dir = plots_root / dataset / "line_plots" / method
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / f"lines_mt{mt}.png"
        fig.savefig(out_path, dpi=220, bbox_inches="tight")
        plt.close(fig)
        print(f"  [lines]    {out_path}")


# 5.  Plot 3 — Learning curves

def plot_learning_curves(
    dataset:   str,
    method:    str,
    mt_values: list[int],
    m_values:  list[int] = M_VALUES,
    plots_root: Path = PLOTS_ROOT,
) -> None:
    """
    For each (method, mt, m) combination produce one figure with 4 lines:
      - Train accuracy — accuracy tuned  (red solid)
      - Val   accuracy — accuracy tuned  (red dashed)
      - Train accuracy — TvL m={m} tuned (blue solid)
      - Val   accuracy — TvL m={m} tuned (blue dashed)

    Saved under:
        plots_root/dataset/learning_curves/<method>/mt_<mt>/lc_m<m>.png
    """
    for mt in mt_values:
        # load accuracy result once per mt
        acc_res = load_accuracy_result(dataset, method, mt)

        for m in m_values:
            tvl_res = load_tvl_result(dataset, method, mt, m)

            # skip if both are missing
            if acc_res is None and tvl_res is None:
                continue

            fig, ax = plt.subplots(figsize=(9, 5))

            # accuracy-tuned curves 
            if acc_res is not None:
                tr_h = acc_res.get("train_history", [])
                va_h = acc_res.get("val_history",   [])
                ep   = list(range(1, max(len(tr_h), len(va_h)) + 1))
                if tr_h:
                    ax.plot(ep[:len(tr_h)], tr_h,
                            color="#c0392b", linewidth=2, linestyle="-",
                            marker="o", markersize=3,
                            label="Train — Accuracy tuned")
                if va_h:
                    ax.plot(ep[:len(va_h)], va_h,
                            color="#c0392b", linewidth=2, linestyle="--",
                            marker="s", markersize=3,
                            label="Val — Accuracy tuned")

            # tvl-tuned curves 
            if tvl_res is not None:
                tr_h = tvl_res.get("train_history", [])
                va_h = tvl_res.get("val_history",   [])
                ep   = list(range(1, max(len(tr_h), len(va_h)) + 1))
                if tr_h:
                    ax.plot(ep[:len(tr_h)], tr_h,
                            color="#2980b9", linewidth=2, linestyle="-",
                            marker="o", markersize=3,
                            label=f"Train — TvL m={m}")
                if va_h:
                    ax.plot(ep[:len(va_h)], va_h,
                            color="#2980b9", linewidth=2, linestyle="--",
                            marker="s", markersize=3,
                            label=f"Val — TvL m={m}")

            ax.set_title(
                f"{dataset}  |  {METHOD_LABELS[method]}  |  mt={mt}  |  TvL m={m}",
                fontsize=10, fontweight="bold",
            )
            ax.set_xlabel("Epoch", fontsize=9)
            ax.set_ylabel("Accuracy (%)", fontsize=9)
            ax.legend(fontsize=8, loc="lower right")
            ax.grid(True, alpha=0.3)
            fig.tight_layout()

            out_dir = plots_root / dataset / "learning_curves" / method / f"mt_{mt}"
            out_dir.mkdir(parents=True, exist_ok=True)
            out_path = out_dir / f"lc_m{m}.png"
            fig.savefig(out_path, dpi=200, bbox_inches="tight")
            plt.close(fig)
            print(f"  [lc]       {out_path}")


# 6.  Main

def run_dataset(dataset: str, plots: list[str]) -> None:
    print(f"\n{'='*60}")
    print(f"  Dataset: {dataset}")
    print(f"{'='*60}")

    for method in METHODS:
        mt_values = discover_mt_values(dataset, method)
        if not mt_values:
            print(f"  [skip] No results found for {method}")
            continue

        print(f"\n  Method: {METHOD_LABELS[method]}  |  mt values: {mt_values}")

        if "heatmap" in plots:
            plot_heatmap(dataset, method, mt_values)
        if "lines" in plots:
            plot_lines(dataset, method, mt_values)
        if "lc" in plots:
            plot_learning_curves(dataset, method, mt_values)


def main() -> None:
    global RESULTS_ROOT, PLOTS_ROOT
    parser = argparse.ArgumentParser(description="CNN HPO result visualisation")
    parser.add_argument(
        "--dataset", choices=DATASETS + ["all"], default="all",
        help="Which dataset to plot (default: all)",
    )
    parser.add_argument(
        "--plots", nargs="+",
        choices=["heatmap", "lines", "lc"], default=["heatmap", "lines", "lc"],
        help=(
            "Which plot types to generate. Choose one or more of: "
            "heatmap  lines  lc  (default: all three). "
            "Example: --plots lines lc"
        ),
    )
    parser.add_argument(
        "--results", default=str(RESULTS_ROOT),
        help="Root directory of CNN results",
    )
    parser.add_argument(
        "--out", default=str(PLOTS_ROOT),
        help="Root directory for output plots",
    )

    args = parser.parse_args()

    RESULTS_ROOT = Path(args.results)
    PLOTS_ROOT   = Path(args.out)

    datasets = DATASETS if args.dataset == "all" else [args.dataset]
    for ds in datasets:
        run_dataset(ds, plots=args.plots)

    print("\nDone. All plots saved.")


if __name__ == "__main__":
    main()