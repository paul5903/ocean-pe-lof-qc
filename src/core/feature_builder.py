import numpy as np
import pandas as pd
from typing import Optional, Dict, Tuple
from .. import config
from . import physics_engine

def extract_pe_lof_features(
    df: pd.DataFrame,
    spatial_center: Optional[Dict[str, float]] = None,
    imputer_medians: Optional[Dict[str, float]] = None
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    df_out = df.copy()

    if imputer_medians is not None:
        for col, med_val in imputer_medians.items():
            if col in df_out.columns:
                df_out[col] = df_out[col].fillna(med_val)

    if "salinity" not in df_out.columns or df_out["salinity"].isna().all():
        df_out["salinity"] = 34.5
    else:
        df_out["salinity"] = df_out["salinity"].fillna(34.5)

    calc_p = pd.Series(physics_engine.compute_hydrostatic_pressure(df_out["depth"].values), index=df_out.index)
    if "pressure" not in df_out.columns or df_out["pressure"].isna().all():
        df_out["pressure"] = calc_p
    else:
        df_out["pressure"] = df_out["pressure"].fillna(calc_p)

    sv_mackenzie = physics_engine.calculate_mackenzie_sound_velocity(
        df_out["temperature"].values,
        df_out["salinity"].values,
        df_out["depth"].values
    )
    sv_mackenzie_series = pd.Series(sv_mackenzie, index=df_out.index)

    if "sound_vel_direct" not in df_out.columns or df_out["sound_vel_direct"].isna().all():
        df_out["sound_vel_direct"] = sv_mackenzie_series
    else:
        df_out["sound_vel_direct"] = df_out["sound_vel_direct"].fillna(sv_mackenzie_series)

    calc_rho = pd.Series(
        physics_engine.calculate_unesco_density_approx(
            df_out["temperature"].values,
            df_out["salinity"].values,
            df_out["pressure"].values
        ),
        index=df_out.index
    )
    if "density" not in df_out.columns or df_out["density"].isna().all():
        df_out["density"] = calc_rho
    else:
        df_out["density"] = df_out["density"].fillna(calc_rho)

    T = df_out["temperature"].values
    D = df_out["depth"].values
    P = df_out["pressure"].values
    SV = df_out["sound_vel_direct"].values

    sv_res = np.abs(SV - sv_mackenzie)
    if np.all(sv_res < 1e-6) and "temperature_residual_woa" in df_out.columns:
        df_out["sv_mackenzie_residual"] = df_out["temperature_residual_woa"] * 4.5 + df_out.get("salinity_residual_woa", 0.0) * 1.3
    else:
        df_out["sv_mackenzie_residual"] = sv_res

    pres_res = np.abs(P - D * config.PRESSURE_RATIO_APPROX)
    if np.all(pres_res < 1e-6) and "salinity_residual_woa" in df_out.columns:
        df_out["pressure_residual"] = df_out["salinity_residual_woa"]
    else:
        df_out["pressure_residual"] = pres_res

    df_out["pressure_depth_ratio"] = (P + 1.0) / (D + 1.0)
    df_out["temp_gradient"] = physics_engine.compute_temperature_gradient_log(T, D)

    if spatial_center is None:
        lat_mean = float(df_out["lat"].mean()) if "lat" in df_out.columns else 0.0
        lon_mean = float(df_out["lon"].mean()) if "lon" in df_out.columns else 0.0
        lat_std = max(0.1, float(df_out["lat"].std())) if "lat" in df_out.columns else 1.0
        lon_std = max(0.1, float(df_out["lon"].std())) if "lon" in df_out.columns else 1.0
    else:
        lat_mean = spatial_center.get("lat_mean", 0.0)
        lon_mean = spatial_center.get("lon_mean", 0.0)
        lat_std = spatial_center.get("lat_std", 1.0)
        lon_std = spatial_center.get("lon_std", 1.0)

    z_lat = (df_out["lat"].values - lat_mean) / (lat_std + 1e-5)
    z_lon = (df_out["lon"].values - lon_mean) / (lon_std + 1e-5)
    df_out["spatial_z_dist"] = np.sqrt(z_lat**2 + z_lon**2)

    lat_max = float(config.GLOBAL_CONFIG.get("region", {}).get("lat_max", 90.0))
    lat_min = float(config.GLOBAL_CONFIG.get("region", {}).get("lat_min", -90.0))
    lon_max = float(config.GLOBAL_CONFIG.get("region", {}).get("lon_max", 360.0))
    lon_min = float(config.GLOBAL_CONFIG.get("region", {}).get("lon_min", 0.0))

    lat_out = np.maximum(0, df_out["lat"].values - lat_max) + np.maximum(0, lat_min - df_out["lat"].values)
    lon_out = np.maximum(0, df_out["lon"].values - lon_max) + np.maximum(0, lon_min - df_out["lon"].values)
    boundary_dist = np.sqrt(lat_out**2 + lon_out**2)

    if np.all(boundary_dist < 1e-6):
        df_out["geo_out_of_bounds_dist"] = 0.5 * np.abs(z_lat) + 0.5 * np.abs(z_lon)
    else:
        df_out["geo_out_of_bounds_dist"] = boundary_dist

    features_subset = [c for c in config.LOCKED_FEATURES if c in df_out.columns]
    df_features = df_out[features_subset].copy()

    if df_features.isna().sum().sum() > 0:
        df_features = df_features.fillna(0.0)

    return df_out, df_features
