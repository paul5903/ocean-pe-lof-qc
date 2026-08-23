import argparse
from pathlib import Path
import numpy as np
import pandas as pd

from .. import config
from ..core import feature_builder, qc_rules, models, visualizer, utils, schema
from . import data_generator

def run_regional_pipeline(
    data_path: Path = None,
    output_dir: Path = None,
    n_profiles: int = 50,
    seed: int = config.RANDOM_SEED
):
    out_dir = output_dir or (config.OUTPUTS_DIR / "regional_run")
    out_dir.mkdir(parents=True, exist_ok=True)
    logger = utils.setup_logger("REGIONAL_PIPELINE", log_file=out_dir / "pipeline.log")

    logger.info("Executing Regional Pipeline...")
    if data_path is not None and data_path.exists():
        if data_path.suffix.lower() in [".xlsx", ".xls"]:
            raw_df = pd.read_excel(data_path)
        else:
            raw_df = pd.read_csv(data_path)
        df_raw = schema.standardize_dataframe(raw_df, source_name="Regional_Field_Survey")
    else:
        logger.info("No custom file provided. Generating synthetic regional dataset...")
        df_raw = data_generator.generate_synthetic_benchmark_dataset(n_profiles=n_profiles, random_seed=seed)

    df_with_feat, df_feat_only = feature_builder.extract_pe_lof_features(df_raw)
    df_with_rules = qc_rules.apply_domain_rules(df_with_feat)

    grp_summary = df_with_rules.groupby("profile_id").agg(
        has_anom=("is_ground_truth_anomaly", lambda x: int((x > 0).any()))
    ).reset_index()

    anom_p = grp_summary[grp_summary["has_anom"] == 1]["profile_id"].tolist()
    norm_p = grp_summary[grp_summary["has_anom"] == 0]["profile_id"].tolist()

    rng = np.random.RandomState(seed)
    rng.shuffle(anom_p)
    rng.shuffle(norm_p)

    def split_lst(lst, r_tr=0.6, r_val=0.2):
        n = len(lst)
        n_tr = max(1, int(n * r_tr)) if n > 1 else n
        n_val = max(1, int(n * r_val)) if n > 2 else 0
        return lst[:n_tr], lst[n_tr:n_tr + n_val], lst[n_tr + n_val:]

    tr_a, val_a, te_a = split_lst(anom_p, 0.6, 0.2)
    tr_n, val_n, te_n = split_lst(norm_p, 0.6, 0.2)

    if not te_a and anom_p:
        te_a = [anom_p[-1]]
        if anom_p[-1] in tr_a:
            tr_a.remove(anom_p[-1])

    train_p = set(tr_a + tr_n)
    val_p = set(val_a + val_n)
    test_p = set(te_a + te_n)

    df_train = df_with_rules[df_with_rules["profile_id"].isin(train_p)].copy()
    df_val = df_with_rules[df_with_rules["profile_id"].isin(val_p)].copy()
    df_test = df_with_rules[df_with_rules["profile_id"].isin(test_p)].copy()

    clean_train_mask = (df_train["is_ground_truth_anomaly"] == 0)
    if clean_train_mask.sum() == 0:
        clean_train_mask = np.ones(len(df_train), dtype=bool)

    X_train = df_train.loc[clean_train_mask, config.LOCKED_FEATURES].values
    model = models.PhysicsEmbeddedLOF()
    model.fit(X_train)

    X_val = df_val[config.LOCKED_FEATURES].values
    y_val = df_val["is_ground_truth_anomaly"].values
    scores_val = model.compute_anomaly_scores(X_val)

    if len(np.unique(y_val)) > 1:
        calib_thresh, _ = model.calibrate_optimal_threshold(y_val, scores_val, beta=1.0)
    else:
        calib_thresh = config.OPTIMAL_THRESHOLD

    X_test = df_test[config.LOCKED_FEATURES].values
    y_test = df_test["is_ground_truth_anomaly"].values
    y_test_pred, scores_test = model.predict(X_test, threshold=calib_thresh)

    df_test["pe_lof_score"] = scores_test
    df_test["qc_ai_suspect"] = y_test_pred
    df_test = qc_rules.assign_final_qc_status(df_test)

    metrics = model.calculate_evaluation_metrics(y_test, y_test_pred, scores_test)
    ci_results = model.bootstrap_ci(y_test, scores_test, threshold=calib_thresh, n_iterations=500, random_seed=seed)

    logger.info(f"Regional Test ROC-AUC: {metrics['roc_auc']:.4f}, F1: {metrics['f1_score']:.4f}, Recall: {metrics['recall']:.4f}")

    df_test.to_csv(out_dir / "test_predictions.csv", index=False)
    summary_df = pd.DataFrame([{
        "pipeline": "Regional_Survey",
        "test_samples": len(df_test),
        "roc_auc": metrics["roc_auc"],
        "roc_auc_ci_low": ci_results["roc_auc"][0],
        "roc_auc_ci_high": ci_results["roc_auc"][1],
        "pr_auc": metrics["pr_auc"],
        "f1_score": metrics["f1_score"],
        "recall": metrics["recall"],
        "precision": metrics["precision"],
        "specificity": metrics["specificity"],
        "mcc": metrics["mcc"]
    }])
    summary_df.to_csv(out_dir / "metrics_summary.csv", index=False)

    if len(np.unique(y_test)) > 1:
        visualizer.plot_roc_and_pr_curves(y_test, scores_test, out_dir / "roc_pr_curve.png", title_suffix="(Regional)")
        visualizer.plot_confusion_matrix_heatmap(y_test, y_test_pred, out_dir / "confusion_matrix.png", title="Regional Confusion Matrix")

    logger.info(f"Pipeline completed. Outputs saved to {out_dir}")
    return metrics

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Regional Survey Pipeline")
    parser.add_argument("--data-path", type=str, default=None, help="Input data file")
    parser.add_argument("--output-dir", type=str, default=None, help="Output directory")
    parser.add_argument("--profiles", type=int, default=50, help="Number of synthetic profiles")
    args = parser.parse_args()

    dp = Path(args.data_path) if args.data_path else None
    op = Path(args.output_dir) if args.output_dir else None
    run_regional_pipeline(data_path=dp, output_dir=op, n_profiles=args.profiles)
