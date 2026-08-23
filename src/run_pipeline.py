# -*- coding: utf-8 -*-
"""
End-to-End Orchestration Pipeline for Ocean PE-LOF QC:
Executes full 8-step standardized evaluation:
1. Ingestion & Schema Normalization
2. Physics-Embedded 14-Feature Extraction
3. Profile-Aware 60:20:20 Partition (Zero Data Leakage)
4. Semi-Supervised Model Training (Clean Train subset)
5. Validation Calibration (PR-curve optimal threshold tau*)
6. Independent Test Evaluation
7. Bootstrap 95% Confidence Intervals
8. Publication-ready Tables & 300 DPI Figures Export
"""
import argparse
import sys
from pathlib import Path
import numpy as np
import pandas as pd

from . import config, data_fetchers, data_generator, feature_builder, qc_rules, models, visualizer, utils


def run_benchmark_pipeline(
    source: str = "synthetic",
    data_path: Path = None,
    output_dir: Path = None,
    n_profiles: int = 50,
    seed: int = config.RANDOM_SEED,
    export_plots: bool = True,
    bootstrap_iterations: int = 500
):
    out_dir = output_dir or (config.OUTPUTS_DIR / f"{source}_run")
    out_dir.mkdir(parents=True, exist_ok=True)
    logger = utils.setup_logger(f"PE_LOF_{source.upper()}", log_file=out_dir / "pipeline.log")

    logger.info("=" * 70)
    logger.info(f"STARTING PE-LOF BENCHMARK PIPELINE [Source: {source.upper()}]")
    logger.info("=" * 70)

    # 1. Ingestion & Data Preparation
    logger.info("[Step 1/8] Ingesting dataset...")
    if source == "synthetic":
        df_raw = data_generator.generate_synthetic_benchmark_dataset(
            n_profiles=n_profiles,
            levels_per_profile=35,
            anomaly_profile_ratio=0.25,
            random_seed=seed
        )
    elif source == "custom" and data_path is not None and data_path.exists():
        df_raw = data_fetchers.load_tabular_dataset(data_path, source_name="Custom_Survey")
    elif data_path is not None and data_path.is_file() and data_path.suffix.lower() == ".nc":
        if "woce" in source.lower():
            df_raw = data_fetchers.load_woce_netcdf(data_path)
        else:
            df_raw = data_fetchers.load_argo_netcdf(data_path)
    else:
        logger.warning(f"No specific file provided for {source}. Generating synthetic ocean profile benchmark...")
        df_raw = data_generator.generate_synthetic_benchmark_dataset(n_profiles=n_profiles, random_seed=seed)

    logger.info(f"Ingested {len(df_raw)} records across {df_raw['profile_id'].nunique()} unique profiles.")

    # 2. Physics & Feature Engineering
    logger.info("[Step 2/8] Extracting 14 Physics-Embedded features...")
    df_with_feat, df_feat_only = feature_builder.extract_pe_lof_features(df_raw)
    df_with_rules = qc_rules.apply_domain_rules(df_with_feat)

    # 3. Profile-Aware 60:20:20 Partition (Prevents Data Leakage)
    logger.info("[Step 3/8] Partitioning into Train (60%), Val (20%), Test (20%) by profile_id...")
    unique_profiles = df_with_rules["profile_id"].unique()
    rng = np.random.RandomState(seed)
    rng.shuffle(unique_profiles)

    n_p = len(unique_profiles)
    n_train = max(1, int(n_p * 0.60))
    n_val = max(1, int(n_p * 0.20))

    train_p = set(unique_profiles[:n_train])
    val_p = set(unique_profiles[n_train:n_train + n_val])
    test_p = set(unique_profiles[n_train + n_val:])

    df_train = df_with_rules[df_with_rules["profile_id"].isin(train_p)].copy()
    df_val = df_with_rules[df_with_rules["profile_id"].isin(val_p)].copy()
    df_test = df_with_rules[df_with_rules["profile_id"].isin(test_p)].copy()

    logger.info(f"Split sizes: Train={len(df_train)} obs ({len(train_p)} profiles), "
                f"Val={len(df_val)} obs ({len(val_p)} profiles), "
                f"Test={len(df_test)} obs ({len(test_p)} profiles)")

    # 4. Model Fitting on Clean Training Observations
    logger.info("[Step 4/8] Fitting Physics-Embedded LOF on clean training set (QC <= 2)...")
    clean_train_mask = (df_train["is_ground_truth_anomaly"] == 0)
    if clean_train_mask.sum() == 0:
        clean_train_mask = np.ones(len(df_train), dtype=bool)

    X_train = df_train.loc[clean_train_mask, config.LOCKED_FEATURES].values
    model = models.PhysicsEmbeddedLOF()
    model.fit(X_train)

    # 5. Validation Calibration
    logger.info("[Step 5/8] Calibrating optimal decision threshold on Validation split...")
    X_val = df_val[config.LOCKED_FEATURES].values
    y_val = df_val["is_ground_truth_anomaly"].values
    scores_val = model.compute_anomaly_scores(X_val)

    if len(np.unique(y_val)) > 1:
        calib_thresh, calib_stats = model.calibrate_optimal_threshold(y_val, scores_val, beta=1.0)
        logger.info(f"Calibrated Threshold: tau* = {calib_thresh:.4f} (Val Max F1 = {calib_stats['max_f_score']:.4f})")
    else:
        calib_thresh = config.OPTIMAL_THRESHOLD
        logger.info(f"Validation single class present. Using locked threshold tau* = {calib_thresh:.4f}")

    # 6. Independent Test Evaluation
    logger.info("[Step 6/8] Evaluating on Independent Test Split...")
    X_test = df_test[config.LOCKED_FEATURES].values
    y_test = df_test["is_ground_truth_anomaly"].values
    y_test_pred, scores_test = model.predict(X_test, threshold=calib_thresh)

    df_test["pe_lof_score"] = scores_test
    df_test["qc_ai_suspect"] = y_test_pred
    df_test = qc_rules.assign_final_qc_status(df_test)

    test_metrics = model.calculate_evaluation_metrics(y_test, y_test_pred, scores_test)

    # 7. Bootstrap 95% Confidence Intervals
    logger.info(f"[Step 7/8] Running Bootstrap {bootstrap_iterations} iterations for 95% Confidence Intervals...")
    ci_results = model.bootstrap_ci(y_test, scores_test, threshold=calib_thresh, n_iterations=bootstrap_iterations, random_seed=seed)

    # Log Core Results
    logger.info("-" * 50)
    logger.info(f"INDEPENDENT TEST RESULTS (Source: {source.upper()}):")
    logger.info(f"  ROC-AUC  : {test_metrics['roc_auc']:.4f}  [95% CI: {ci_results['roc_auc'][0]:.4f} - {ci_results['roc_auc'][1]:.4f}]")
    logger.info(f"  PR-AUC   : {test_metrics['pr_auc']:.4f}  [95% CI: {ci_results['pr_auc'][0]:.4f} - {ci_results['pr_auc'][1]:.4f}]")
    logger.info(f"  F1-Score : {test_metrics['f1_score']:.4f}  [95% CI: {ci_results['f1'][0]:.4f} - {ci_results['f1'][1]:.4f}]")
    logger.info(f"  Recall   : {test_metrics['recall']:.4f}  [95% CI: {ci_results['recall'][0]:.4f} - {ci_results['recall'][1]:.4f}]")
    logger.info(f"  Precision: {test_metrics['precision']:.4f}  [95% CI: {ci_results['precision'][0]:.4f} - {ci_results['precision'][1]:.4f}]")
    logger.info(f"  Specific.: {test_metrics['specificity']:.4f}")
    logger.info(f"  Balanced Acc: {test_metrics['balanced_accuracy']:.4f} | MCC: {test_metrics['mcc']:.4f}")
    logger.info(f"  Confusion Matrix: TP={test_metrics['tp']}, FP={test_metrics['fp']}, TN={test_metrics['tn']}, FN={test_metrics['fn']}")
    logger.info("-" * 50)

    # 8. Export Outputs & Visualizations
    logger.info("[Step 8/8] Exporting evaluation tables and figures...")
    # Export CSV & Excel
    df_test.to_csv(out_dir / "test_predictions.csv", index=False)
    summary_df = pd.DataFrame([{
        "source": source,
        "test_samples": len(df_test),
        "anomalies_ground_truth": int(y_test.sum()),
        "calibrated_threshold": calib_thresh,
        "roc_auc": test_metrics["roc_auc"],
        "roc_auc_ci_low": ci_results["roc_auc"][0],
        "roc_auc_ci_high": ci_results["roc_auc"][1],
        "pr_auc": test_metrics["pr_auc"],
        "pr_auc_ci_low": ci_results["pr_auc"][0],
        "pr_auc_ci_high": ci_results["pr_auc"][1],
        "f1_score": test_metrics["f1_score"],
        "recall": test_metrics["recall"],
        "precision": test_metrics["precision"],
        "specificity": test_metrics["specificity"],
        "mcc": test_metrics["mcc"],
        "tp": test_metrics["tp"],
        "fp": test_metrics["fp"],
        "tn": test_metrics["tn"],
        "fn": test_metrics["fn"]
    }])
    summary_df.to_csv(out_dir / "benchmark_metrics_summary.csv", index=False)

    if export_plots and len(np.unique(y_test)) > 1:
        visualizer.plot_roc_and_pr_curves(y_test, scores_test, out_dir / "figure_roc_pr_curve.png", title_suffix=f"({source.upper()})")
        visualizer.plot_confusion_matrix_heatmap(y_test, y_test_pred, out_dir / "figure_confusion_matrix.png", title=f"Confusion Matrix ({source.upper()})")
        visualizer.plot_anomaly_score_distribution(y_test, scores_test, calib_thresh, out_dir / "figure_score_distribution.png", title=f"Score Distribution ({source.upper()})")

        sample_prof = list(test_p)[0]
        visualizer.plot_vertical_profile_qc(df_test, sample_prof, out_dir / "figure_sample_vertical_profile.png")

    logger.info(f"[SUCCESS] Benchmark complete! Results saved in: {out_dir}")
    return test_metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Ocean PE-LOF QC Independent Benchmark Runner")
    parser.add_argument("--source", type=str, default="synthetic", choices=["synthetic", "argo", "woce", "custom"], help="Data source type")
    parser.add_argument("--data-path", type=str, default=None, help="Path to input NetCDF/CSV/Excel file")
    parser.add_argument("--output-dir", type=str, default=None, help="Output directory for reports and figures")
    parser.add_argument("--profiles", type=int, default=40, help="Number of profiles if synthetic")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    parser.add_argument("--no-plots", action="store_true", help="Disable plotting")
    args = parser.parse_args()

    dp = Path(args.data_path) if args.data_path else None
    op = Path(args.output_dir) if args.output_dir else None

    run_benchmark_pipeline(
        source=args.source,
        data_path=dp,
        output_dir=op,
        n_profiles=args.profiles,
        seed=args.seed,
        export_plots=(not args.no_plots)
    )
