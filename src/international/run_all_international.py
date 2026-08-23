import argparse
from pathlib import Path
import numpy as np
import pandas as pd

from .. import config
from ..core import feature_builder, qc_rules, models, visualizer, utils, schema
from ..regional import data_generator

def parse_argo_netcdf(nc_file: Path) -> pd.DataFrame:
    import netCDF4 as nc
    ds = nc.Dataset(str(nc_file), "r")
    try:
        n_prof = ds.dimensions["N_PROF"].size
        n_levels = ds.dimensions["N_LEVELS"].size
        lats = ds.variables["LATITUDE"][:]
        lons = ds.variables["LONGITUDE"][:]
        pres = ds.variables["PRES"][:] if "PRES" in ds.variables else ds.variables["PRES_ADJUSTED"][:]
        temp = ds.variables["TEMP"][:] if "TEMP" in ds.variables else ds.variables["TEMP_ADJUSTED"][:]
        psal = ds.variables["PSAL"][:] if "PSAL" in ds.variables else ds.variables["PSAL_ADJUSTED"][:]
        temp_qc = ds.variables.get("TEMP_QC", None)

        rows = []
        for p_idx in range(n_prof):
            lat_val = float(lats[p_idx])
            lon_val = float(lons[p_idx])
            for lvl in range(n_levels):
                p_val = float(pres[p_idx, lvl]) if pres.ndim == 2 else float(pres[lvl])
                t_val = float(temp[p_idx, lvl]) if temp.ndim == 2 else float(temp[lvl])
                s_val = float(psal[p_idx, lvl]) if psal.ndim == 2 else float(psal[lvl])
                if np.isnan(p_val) or np.isnan(t_val) or p_val > 9999 or t_val > 99:
                    continue
                d_val = p_val / 1.019716
                qc_flag = 1
                if temp_qc is not None:
                    try:
                        q_char = temp_qc[p_idx, lvl] if temp_qc.ndim == 2 else temp_qc[lvl]
                        if isinstance(q_char, (bytes, str)) and str(q_char).strip() in ['3', '4']:
                            qc_flag = 4
                    except Exception:
                        pass
                rows.append({
                    "profile_id": f"argo_{nc_file.stem}_{p_idx}",
                    "lat": lat_val,
                    "lon": lon_val,
                    "depth": d_val,
                    "pressure": p_val,
                    "temperature": t_val,
                    "salinity": s_val,
                    "qc_raw_flag": qc_flag
                })
        return schema.standardize_dataframe(pd.DataFrame(rows), source_name="Argo_GDAC")
    finally:
        ds.close()

def run_international_pipeline(
    source: str = "argo",
    data_dir: Path = None,
    output_dir: Path = None,
    seed: int = config.RANDOM_SEED
):
    out_dir = output_dir or (config.OUTPUTS_DIR / f"international_{source}")
    out_dir.mkdir(parents=True, exist_ok=True)
    logger = utils.setup_logger("INTL_PIPELINE", log_file=out_dir / "pipeline.log")

    logger.info(f"Running International Pipeline for {source}...")
    in_dir = data_dir or (config.DATA_DIR / "argo_profiles")
    nc_files = list(in_dir.glob("*.nc")) if in_dir.exists() else []

    if nc_files:
        dfs = [parse_argo_netcdf(f) for f in nc_files]
        df_raw = pd.concat(dfs, ignore_index=True)
    else:
        logger.info("No NetCDF files found. Generating international synthetic equivalent...")
        df_raw = data_generator.generate_synthetic_benchmark_dataset(n_profiles=60, levels_per_profile=40, random_seed=seed)

    df_with_feat, _ = feature_builder.extract_pe_lof_features(df_raw)
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

    metrics = model.calculate_evaluation_metrics(y_test, y_test_pred, scores_test)
    ci_results = model.bootstrap_ci(y_test, scores_test, threshold=calib_thresh, n_iterations=500, random_seed=seed)

    logger.info(f"International {source.upper()} ROC-AUC: {metrics['roc_auc']:.4f}, F1: {metrics['f1_score']:.4f}")

    df_test["pe_lof_score"] = scores_test
    df_test["qc_ai_suspect"] = y_test_pred
    df_test = qc_rules.assign_final_qc_status(df_test)
    df_test.to_csv(out_dir / "test_predictions.csv", index=False)

    summary_df = pd.DataFrame([{
        "source": source,
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
        visualizer.plot_roc_and_pr_curves(y_test, scores_test, out_dir / "roc_pr_curve.png", title_suffix=f"({source.upper()})")
        visualizer.plot_confusion_matrix_heatmap(y_test, y_test_pred, out_dir / "confusion_matrix.png", title=f"Confusion Matrix ({source.upper()})")

    logger.info("International pipeline execution complete.")
    return metrics

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="International Pipeline Runner")
    parser.add_argument("--source", type=str, default="argo", choices=["argo", "woce"], help="Data source")
    args = parser.parse_args()
    run_international_pipeline(source=args.source)
