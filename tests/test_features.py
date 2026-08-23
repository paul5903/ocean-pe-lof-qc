from src.core import feature_builder
from src.regional import data_generator
from src import config

def test_feature_extraction():
    df_raw = data_generator.generate_synthetic_benchmark_dataset(n_profiles=5, levels_per_profile=10)
    df_full, df_feat = feature_builder.extract_pe_lof_features(df_raw)
    assert len(df_feat) == 50
    for col in config.LOCKED_FEATURES:
        assert col in df_feat.columns
    assert df_feat.isna().sum().sum() == 0
