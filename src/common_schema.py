# -*- coding: utf-8 -*-
"""
Common Oceanographic Schema Normalization:
Standardizes heterogeneous raw formats (Argo GDAC NetCDF, WOCE CCHDO NetCDF/CSV,
WOA23 Climatology, and generic CTD tables) into a unified tabular schema.
"""
import numpy as np
import pandas as pd
from typing import Dict

STANDARD_COLUMNS = [
    "source",
    "profile_id",
    "station_id",
    "timestamp",
    "lat",
    "lon",
    "depth",
    "temperature",
    "salinity",
    "pressure",
    "sound_vel_direct",
    "qc_raw_flag",
    "is_ground_truth_anomaly"
]


def standardize_dataframe(
    df: pd.DataFrame,
    source_name: str,
    column_mapping: Dict[str, str] = None
) -> pd.DataFrame:
    """
    Maps and standardizes raw DataFrames into canonical schema.
    """
    df_std = pd.DataFrame()
    col_map = column_mapping or {}

    # Apply mapping
    renamed = df.rename(columns=col_map)

    df_std["source"] = source_name
    df_std["profile_id"] = renamed["profile_id"] if "profile_id" in renamed.columns else (renamed["station_id"] if "station_id" in renamed.columns else np.arange(len(renamed)))
    df_std["station_id"] = renamed["station_id"] if "station_id" in renamed.columns else df_std["profile_id"]
    df_std["timestamp"] = renamed["timestamp"] if "timestamp" in renamed.columns else "2023-01-01"

    # Coordinates & Depths
    df_std["lat"] = pd.to_numeric(renamed["lat"], errors="coerce") if "lat" in renamed.columns else 0.0
    df_std["lon"] = pd.to_numeric(renamed["lon"], errors="coerce") if "lon" in renamed.columns else 0.0
    df_std["depth"] = pd.to_numeric(renamed["depth"], errors="coerce") if "depth" in renamed.columns else 0.0

    # Core Parameters
    df_std["temperature"] = pd.to_numeric(renamed["temperature"], errors="coerce") if "temperature" in renamed.columns else np.nan
    df_std["salinity"] = pd.to_numeric(renamed["salinity"], errors="coerce") if "salinity" in renamed.columns else np.nan
    df_std["pressure"] = pd.to_numeric(renamed["pressure"], errors="coerce") if "pressure" in renamed.columns else np.nan
    df_std["sound_vel_direct"] = pd.to_numeric(renamed["sound_vel_direct"], errors="coerce") if "sound_vel_direct" in renamed.columns else np.nan

    # QC Flags
    if "qc_raw_flag" in renamed.columns:
        df_std["qc_raw_flag"] = renamed["qc_raw_flag"]
    elif "qc_flag" in renamed.columns:
        df_std["qc_raw_flag"] = renamed["qc_flag"]
    else:
        df_std["qc_raw_flag"] = 1

    # Ground Truth Anomaly definition
    if "is_ground_truth_anomaly" in renamed.columns:
        df_std["is_ground_truth_anomaly"] = renamed["is_ground_truth_anomaly"].astype(int)
    else:
        # Default oceanographic QC standard: QC > 2 is suspect/bad
        numeric_qc = pd.to_numeric(df_std["qc_raw_flag"], errors="coerce").fillna(1)
        df_std["is_ground_truth_anomaly"] = (numeric_qc > 2).astype(int)

    return df_std
