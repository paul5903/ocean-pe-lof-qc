# -*- coding: utf-8 -*-
import pytest
from src import run_pipeline


def test_end_to_end_pipeline_synthetic(tmp_path):
    metrics = run_pipeline.run_benchmark_pipeline(
        source="synthetic",
        output_dir=tmp_path,
        n_profiles=20,
        export_plots=True,
        bootstrap_iterations=50
    )

    assert "roc_auc" in metrics
    assert "f1_score" in metrics
    assert metrics["roc_auc"] > 0.70
    assert (tmp_path / "test_predictions.csv").exists()
    assert (tmp_path / "benchmark_metrics_summary.csv").exists()
