# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import pytest
from src import feature_builder, data_generator, config


def test_extract_pe_lof_features():
    df_raw = data_generator.generate_synthetic_benchmark_dataset(n_profiles=5, levels_per_profile=10)
    df_full, df_feat = feature_builder.extract_pe_lof_features(df_raw)

    # Verify output dimensions and column specifications
    assert len(df_feat) == 50
    for col in config.LOCKED_FEATURES:
        assert col in df_feat.columns

    # Verify zero NaNs in extracted feature matrix
    assert df_feat.isna().sum().sum() == 0
