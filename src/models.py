# -*- coding: utf-8 -*-
"""
Physics-Embedded Local Outlier Factor (PE-LOF) Model Architecture:
Provides novelty detection, robust scaling, dynamic threshold calibration,
and statistical bootstrap evaluation.
"""
import numpy as np
import pandas as pd
from typing import Dict, Any, Tuple, Optional
from sklearn.preprocessing import RobustScaler
from sklearn.neighbors import LocalOutlierFactor
from sklearn.metrics import (
    roc_auc_score,
    precision_recall_curve,
    auc,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    accuracy_score,
    matthews_corrcoef,
    balanced_accuracy_score
)
from . import config


class PhysicsEmbeddedLOF:
    """
    Physics-Embedded Local Outlier Factor model wrapper for oceanographic QC.
    """

    def __init__(
        self,
        n_neighbors: int = config.N_NEIGHBORS,
        contamination: float = config.CONTAMINATION,
        threshold: float = config.OPTIMAL_THRESHOLD,
        metric: str = "minkowski",
        p: int = 2
    ):
        self.n_neighbors = n_neighbors
        self.contamination = contamination
        self.threshold = threshold
        self.metric = metric
        self.p = p

        self.scaler = RobustScaler()
        self.lof = LocalOutlierFactor(
            n_neighbors=self.n_neighbors,
            contamination=self.contamination,
            novelty=True,
            metric=self.metric,
            p=self.p
        )
        self.is_fitted = False

    def fit(self, X_train: np.ndarray) -> "PhysicsEmbeddedLOF":
        """
        Fits RobustScaler and LOF novelty estimator on clean training observations.
        """
        X_scaled = self.scaler.fit_transform(X_train)
        self.lof.fit(X_scaled)
        self.is_fitted = True
        return self

    def compute_anomaly_scores(self, X: np.ndarray) -> np.ndarray:
        """
        Computes positive anomaly scores where higher values indicate higher abnormality:
        Score = -score_samples(X)
        """
        if not self.is_fitted:
            raise RuntimeError("Model must be fitted before computing anomaly scores.")
        X_scaled = self.scaler.transform(X)
        # scikit-learn's score_samples returns opposite of LOF (inliers large, outliers small negative)
        scores = -self.lof.score_samples(X_scaled)
        return scores

    def predict(self, X: np.ndarray, threshold: Optional[float] = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        Predicts binary anomaly flags based on decision threshold.
        Returns: (y_pred_binary [0 for Normal, 1 for Anomaly], anomaly_scores)
        """
        t = self.threshold if threshold is None else threshold
        scores = self.compute_anomaly_scores(X)
        y_pred = (scores >= t).astype(int)
        return y_pred, scores

    @staticmethod
    def calibrate_optimal_threshold(
        y_true: np.ndarray,
        scores: np.ndarray,
        beta: float = 1.0
    ) -> Tuple[float, Dict[str, float]]:
        """
        Finds optimal decision threshold maximizing F_beta score on Precision-Recall curve.
        """
        precisions, recalls, thresholds = precision_recall_curve(y_true, scores)

        # Compute F-beta for all points
        beta_sq = beta ** 2
        denom = (beta_sq * precisions) + recalls
        f_scores = np.zeros_like(precisions)
        valid = denom > 0
        f_scores[valid] = (1 + beta_sq) * (precisions[valid] * recalls[valid]) / denom[valid]

        best_idx = np.argmax(f_scores)
        # thresholds array has length len(precisions) - 1
        best_threshold = float(thresholds[min(best_idx, len(thresholds) - 1)])

        best_stats = {
            "optimal_threshold": best_threshold,
            "max_f_score": float(f_scores[best_idx]),
            "precision_at_best": float(precisions[best_idx]),
            "recall_at_best": float(recalls[best_idx])
        }
        return best_threshold, best_stats

    @staticmethod
    def calculate_evaluation_metrics(
        y_true: np.ndarray,
        y_pred: np.ndarray,
        y_score: np.ndarray
    ) -> Dict[str, Any]:
        """
        Calculates full suite of classification and diagnostic metrics.
        """
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()

        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
        sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        precision = precision_score(y_true, y_pred, zero_division=0)
        f1 = f1_score(y_true, y_pred, zero_division=0)
        acc = accuracy_score(y_true, y_pred)
        bal_acc = balanced_accuracy_score(y_true, y_pred)
        mcc = matthews_corrcoef(y_true, y_pred)

        # ROC-AUC
        try:
            roc_auc = float(roc_auc_score(y_true, y_score))
        except Exception:
            roc_auc = 0.5

        # PR-AUC
        try:
            p_arr, r_arr, _ = precision_recall_curve(y_true, y_score)
            pr_auc = float(auc(r_arr, p_arr))
        except Exception:
            pr_auc = 0.0

        return {
            "roc_auc": roc_auc,
            "pr_auc": pr_auc,
            "precision": float(precision),
            "recall": float(sensitivity),
            "specificity": float(specificity),
            "f1_score": float(f1),
            "accuracy": float(acc),
            "balanced_accuracy": float(bal_acc),
            "mcc": float(mcc),
            "tp": int(tp),
            "fp": int(fp),
            "tn": int(tn),
            "fn": int(fn),
            "total_samples": int(len(y_true)),
            "anomaly_rate": float(np.mean(y_true))
        }

    @staticmethod
    def bootstrap_ci(
        y_true: np.ndarray,
        y_score: np.ndarray,
        threshold: float,
        n_iterations: int = 1000,
        ci: float = 0.95,
        random_seed: int = 42
    ) -> Dict[str, Tuple[float, float]]:
        """
        Computes 95% Confidence Intervals via non-parametric bootstrapping.
        """
        rng = np.random.RandomState(random_seed)
        n = len(y_true)
        metrics_dict = {"roc_auc": [], "pr_auc": [], "f1": [], "precision": [], "recall": []}

        for _ in range(n_iterations):
            indices = rng.randint(0, n, n)
            b_true = y_true[indices]
            b_score = y_score[indices]
            b_pred = (b_score >= threshold).astype(int)

            if len(np.unique(b_true)) < 2:
                continue

            try:
                metrics_dict["roc_auc"].append(roc_auc_score(b_true, b_score))
                p_arr, r_arr, _ = precision_recall_curve(b_true, b_score)
                metrics_dict["pr_auc"].append(auc(r_arr, p_arr))
                metrics_dict["f1"].append(f1_score(b_true, b_pred, zero_division=0))
                metrics_dict["precision"].append(precision_score(b_true, b_pred, zero_division=0))
                metrics_dict["recall"].append(recall_score(b_true, b_pred, zero_division=0))
            except Exception:
                continue

        alpha = (1.0 - ci) / 2.0
        ci_results = {}
        for k, vals in metrics_dict.items():
            if len(vals) > 0:
                low = float(np.percentile(vals, alpha * 100))
                high = float(np.percentile(vals, (1.0 - alpha) * 100))
                ci_results[k] = (low, high)
            else:
                ci_results[k] = (0.0, 0.0)

        return ci_results
