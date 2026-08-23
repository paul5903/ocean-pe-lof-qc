from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_curve, precision_recall_curve, confusion_matrix
from .. import config

plt.rcParams.update({
    "font.size": 11,
    "axes.labelsize": 12,
    "axes.titlesize": 13,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 10,
    "figure.titlesize": 14,
    "font.family": "sans-serif"
})

def plot_roc_and_pr_curves(
    y_true: np.ndarray,
    y_score: np.ndarray,
    output_path: Path,
    title_suffix: str = ""
):
    fpr, tpr, _ = roc_curve(y_true, y_score)
    prec, rec, _ = precision_recall_curve(y_true, y_score)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5), dpi=config.DPI)

    axes[0].plot(fpr, tpr, color="#1f77b4", lw=2, label="PE-LOF Detector")
    axes[0].plot([0, 1], [0, 1], color="grey", linestyle="--", lw=1, label="Random Guess")
    axes[0].set_title(f"ROC Curve {title_suffix}".strip())
    axes[0].set_xlabel("False Positive Rate")
    axes[0].set_ylabel("True Positive Rate")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend(loc="lower right")

    base_rate = np.mean(y_true)
    axes[1].plot(rec, prec, color="#d62728", lw=2, label="PE-LOF Detector")
    axes[1].axhline(y=base_rate, color="grey", linestyle="--", lw=1, label=f"Base Rate ({base_rate:.1%})")
    axes[1].set_title(f"Precision-Recall Curve {title_suffix}".strip())
    axes[1].set_xlabel("Recall")
    axes[1].set_ylabel("Precision")
    axes[1].grid(True, alpha=0.3)
    axes[1].legend(loc="upper right")

    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=config.DPI)
    plt.close(fig)

def plot_confusion_matrix_heatmap(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    output_path: Path,
    title: str = "Confusion Matrix"
):
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    labels = ["Normal", "Anomaly"]

    fig, ax = plt.subplots(figsize=(6, 5), dpi=config.DPI)
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=labels,
        yticklabels=labels,
        cbar=False,
        ax=ax
    )
    ax.set_title(title)
    ax.set_xlabel("Predicted Label")
    ax.set_ylabel("Ground Truth")

    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=config.DPI)
    plt.close(fig)
