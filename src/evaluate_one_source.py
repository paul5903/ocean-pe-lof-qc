# -*- coding: utf-8 -*-
"""
Module đánh giá khoa học chuyên sâu cho một nguồn dữ liệu độc lập (Single Source Evaluator).
Tính toán ROC-AUC, PR-AUC, Confusion Matrix, Bootstrap 95% CI và phân rã đa chiều.
"""
from typing import Dict, Any, List, Optional, Tuple
import numpy as np
import pandas as pd
from sklearn.metrics import (
    roc_auc_score, precision_recall_curve, roc_curve, auc,
    accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
)
from . import config

def compute_binary_metrics(
    y_true: np.ndarray,
    scores: np.ndarray,
    threshold: float
) -> Dict[str, Any]:
    """
    Tính toàn bộ chỉ số đánh giá nhị phân với một ngưỡng điểm cụ thể.
    """
    y_pred = (scores >= threshold).astype(int)
    
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    
    try:
        auc_roc = roc_auc_score(y_true, scores)
    except Exception:
        auc_roc = 0.5
        
    p_curve, r_curve, _ = precision_recall_curve(y_true, scores)
    auc_pr = auc(r_curve, p_curve)

    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()

    spec = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
    fnr = fn / (fn + tp) if (fn + tp) > 0 else 0.0

    return {
        "threshold": float(threshold),
        "accuracy": float(acc),
        "precision": float(prec),
        "recall": float(rec),
        "f1_score": float(f1),
        "roc_auc": float(auc_roc),
        "pr_auc": float(auc_pr),
        "specificity": float(spec),
        "false_positive_rate": float(fpr),
        "false_negative_rate": float(fnr),
        "confusion_matrix": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
        "total_samples": int(len(y_true)),
        "total_anomalies": int(y_true.sum())
    }

def compute_bootstrap_ci(
    df: pd.DataFrame,
    group_col: str,
    target_col: str,
    score_col: str,
    threshold: float,
    n_iterations: int = 500,
    ci: float = 0.95,
    seed: int = config.RANDOM_SEED
) -> Dict[str, Tuple[float, float]]:
    """
    Bootstrap lấy mẫu lại theo nhóm (Profile hoặc Cruise) để tính khoảng tin cậy 95%.
    """
    np.random.seed(seed)
    unique_groups = df[group_col].unique()
    n_groups = len(unique_groups)

    f1_list, auc_list, rec_list, prec_list = [], [], [], []

    for _ in range(n_iterations):
        sample_groups = np.random.choice(unique_groups, size=n_groups, replace=True)
        sample_df = df[df[group_col].isin(sample_groups)]
        
        y_true = sample_df[target_col].values
        scores = sample_df[score_col].values
        
        if len(np.unique(y_true)) < 2:
            continue

        metrics = compute_binary_metrics(y_true, scores, threshold)
        f1_list.append(metrics["f1_score"])
        auc_list.append(metrics["roc_auc"])
        rec_list.append(metrics["recall"])
        prec_list.append(metrics["precision"])

    alpha = (1.0 - ci) / 2.0
    def get_bounds(arr):
        if not arr:
            return (0.0, 0.0)
        return (float(np.percentile(arr, alpha * 100)), float(np.percentile(arr, (1.0 - alpha) * 100)))

    return {
        "f1_score_ci": get_bounds(f1_list),
        "roc_auc_ci": get_bounds(auc_list),
        "recall_ci": get_bounds(rec_list),
        "precision_ci": get_bounds(prec_list)
    }

def analyze_by_depth_zones(
    df: pd.DataFrame,
    target_col: str,
    pred_col: str
) -> pd.DataFrame:
    """
    Phân rã kết quả theo 4 tầng nước hải dương học:
    - Bề mặt (0 - 50m)
    - Nhảy nhiệt (50 - 200m)
    - Trung gian (200 - 1000m)
    - Tầng sâu (> 1000m)
    """
    zones = []
    for d in df['depth']:
        if d <= 50:
            zones.append("Surface (0-50m)")
        elif d <= 200:
            zones.append("Thermocline (50-200m)")
        elif d <= 1000:
            zones.append("Intermediate (200-1000m)")
        else:
            zones.append("Deep (>1000m)")
    
    df_temp = df.copy()
    df_temp['depth_zone'] = zones

    results = []
    for zone, sub_df in df_temp.groupby('depth_zone'):
        y_t = sub_df[target_col].values
        y_p = sub_df[pred_col].values
        total = len(sub_df)
        anom = int(y_t.sum())
        detected = int((y_t & y_p).sum())
        
        prec = precision_score(y_t, y_p, zero_division=0)
        rec = recall_score(y_t, y_p, zero_division=0)
        f1 = f1_score(y_t, y_p, zero_division=0)
        
        results.append({
            "Depth Zone": zone,
            "Total Samples": total,
            "Anomalies": anom,
            "Detected": detected,
            "Precision (%)": round(prec * 100, 2),
            "Recall (%)": round(rec * 100, 2),
            "F1-Score": round(f1, 4)
        })

    return pd.DataFrame(results)

def analyze_by_season_and_group(
    df: pd.DataFrame,
    group_col: str,
    target_col: str,
    pred_col: str
) -> pd.DataFrame:
    """
    Phân rã kết quả theo mùa/tháng hoặc nhóm trạm/phao.
    """
    results = []
    for grp_val, sub_df in df.groupby(group_col):
        y_t = sub_df[target_col].values
        y_p = sub_df[pred_col].values
        total = len(sub_df)
        anom = int(y_t.sum())
        detected = int((y_t & y_p).sum())
        
        f1 = f1_score(y_t, y_p, zero_division=0)
        rec = recall_score(y_t, y_p, zero_division=0)
        
        results.append({
            group_col: str(grp_val),
            "Total Samples": total,
            "Anomalies": anom,
            "Detected": detected,
            "Recall (%)": round(rec * 100, 2),
            "F1-Score": round(f1, 4)
        })
    return pd.DataFrame(results)
