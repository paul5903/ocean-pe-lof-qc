# -*- coding: utf-8 -*-
"""
Scientific Visualization Suite for Ocean PE-LOF QC:
Generates publication-quality 300 DPI figures including ROC/PR curves,
Confusion Matrices, Anomaly Score distributions, and Ocean Vertical Profiles.
"""
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_curve, precision_recall_curve, confusion_matrix
from . import config

# Set global publication styling
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
    """Plots dual-panel ROC and Precision-Recall Curves."""
    fpr, tpr, _ = roc_curve(y_true, y_score)
    prec, rec, _ = precision_recall_curve(y_true, y_score)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5), dpi=config.DPI)

    # 1. ROC Curve
    axes[0].plot(fpr, tpr, color="#1f77b4", lw=2, label="PE-LOF Detector")
    axes[0].plot([0, 1], [0, 1], color="grey", linestyle="--", lw=1, label="Random Guess")
    axes[0].set_title(f"ROC Curve {title_suffix}".strip())
    axes[0].set_xlabel("False Positive Rate (1 - Specificity)")
    axes[0].set_ylabel("True Positive Rate (Recall / Sensitivity)")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend(loc="lower right")

    # 2. PR Curve
    base_rate = np.mean(y_true)
    axes[1].plot(rec, prec, color="#d62728", lw=2, label="PE-LOF Detector")
    axes[1].axhline(y=base_rate, color="grey", linestyle="--", lw=1, label=f"Base Anomaly Rate ({base_rate:.1%})")
    axes[1].set_title(f"Precision-Recall Curve {title_suffix}".strip())
    axes[1].set_xlabel("Recall (Sensitivity)")
    axes[1].set_ylabel("Precision (Positive Predictive Value)")
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
    title: str = "PE-LOF Confusion Matrix"
):
    """Plots annotated Confusion Matrix."""
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    labels = ["Normal (Good QC)", "Anomaly (Flagged)"]

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
    ax.set_ylabel("Ground Truth Quality Flag")

    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=config.DPI)
    plt.close(fig)


def plot_anomaly_score_distribution(
    y_true: np.ndarray,
    scores: np.ndarray,
    threshold: float,
    output_path: Path,
    title: str = "Anomaly Score Distribution & Threshold"
):
    """Plots histogram & density of anomaly scores for clean vs anomalous samples."""
    fig, ax = plt.subplots(figsize=(8, 5), dpi=config.DPI)

    scores_clean = scores[y_true == 0]
    scores_anom = scores[y_true == 1]

    ax.hist(scores_clean, bins=40, alpha=0.6, color="#2ca02c", density=True, label=f"Normal Observations (N={len(scores_clean)})")
    if len(scores_anom) > 0:
        ax.hist(scores_anom, bins=30, alpha=0.6, color="#d62728", density=True, label=f"Sensor Anomalies (N={len(scores_anom)})")

    ax.axvline(threshold, color="black", linestyle="--", lw=2, label=f"Calibrated Threshold ({threshold:.4f})")
    ax.set_title(title)
    ax.set_xlabel("Physics-Embedded LOF Anomaly Score")
    ax.set_ylabel("Density")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper right")

    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=config.DPI)
    plt.close(fig)


def plot_vertical_profile_qc(
    df: pd.DataFrame,
    profile_id: str,
    output_path: Path
):
    """Plots ocean vertical profile with flagged anomalies highlighted."""
    p_df = df[df["profile_id"] == profile_id].sort_values("depth")
    if len(p_df) == 0:
        return

    fig, axes = plt.subplots(1, 2, figsize=(10, 6), sharey=True, dpi=config.DPI)

    # Temperature Profile
    axes[0].plot(p_df["temperature"], p_df["depth"], color="#1f77b4", marker="o", markersize=3, label="Temperature Profile")
    if "qc_ai_suspect" in p_df.columns:
        anom_t = p_df[p_df["qc_ai_suspect"] == 1]
        axes[0].scatter(anom_t["temperature"], anom_t["depth"], color="red", s=50, zorder=5, label="Flagged Anomaly")
    axes[0].set_xlabel("Temperature (°C)")
    axes[0].set_ylabel("Depth (m)")
    axes[0].set_title("Temperature vs Depth")
    axes[0].invert_yaxis()
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()

    # Salinity Profile
    axes[1].plot(p_df["salinity"], p_df["depth"], color="#ff7f0e", marker="o", markersize=3, label="Salinity Profile")
    if "qc_ai_suspect" in p_df.columns:
        anom_s = p_df[p_df["qc_ai_suspect"] == 1]
        axes[1].scatter(anom_s["salinity"], anom_s["depth"], color="red", s=50, zorder=5, label="Flagged Anomaly")
    axes[1].set_xlabel("Salinity (PSU)")
    axes[1].set_title("Salinity vs Depth")
    axes[1].grid(True, alpha=0.3)
    axes[1].legend()

    plt.suptitle(f"CTD Vertical Profile QC: {profile_id}")
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=config.DPI)
    plt.close(fig)
