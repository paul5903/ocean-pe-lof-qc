from src.regional import run_regional_pipeline
from src.international import run_all_international

def test_regional_pipeline_execution(tmp_path):
    metrics = run_regional_pipeline.run_regional_pipeline(
        output_dir=tmp_path / "regional",
        n_profiles=15
    )
    assert "roc_auc" in metrics
    assert metrics["roc_auc"] > 0.70

def test_international_pipeline_execution(tmp_path):
    metrics = run_all_international.run_international_pipeline(
        source="argo",
        output_dir=tmp_path / "international"
    )
    assert "roc_auc" in metrics
    assert metrics["roc_auc"] > 0.70
