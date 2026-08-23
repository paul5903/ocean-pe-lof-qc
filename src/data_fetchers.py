# -*- coding: utf-8 -*-
"""
Oceanographic Data Ingestion & Fetchers:
Loads and parses Argo GDAC NetCDF profiles, WOCE CCHDO CTD transects,
WOA23 NetCDF climatologies, and standard CSV/Excel datasets.
"""
from pathlib import Path
import numpy as np
import pandas as pd
from typing import Optional
from . import common_schema


def load_argo_netcdf(file_path: Path) -> pd.DataFrame:
    """
    Parses an Argo GDAC Profile NetCDF file into standardized DataFrame.
    """
    import netCDF4 as nc

    ds = nc.Dataset(str(file_path), "r")
    try:
        n_prof = ds.dimensions["N_PROF"].size
        n_param = ds.dimensions["N_PARAM"].size if "N_PARAM" in ds.dimensions else 0
        n_levels = ds.dimensions["N_LEVELS"].size

        lats = ds.variables["LATITUDE"][:]
        lons = ds.variables["LONGITUDE"][:]
        pres = ds.variables["PRES"][:] if "PRES" in ds.variables else ds.variables["PRES_ADJUSTED"][:]
        temp = ds.variables["TEMP"][:] if "TEMP" in ds.variables else ds.variables["TEMP_ADJUSTED"][:]
        psal = ds.variables["PSAL"][:] if "PSAL" in ds.variables else ds.variables["PSAL_ADJUSTED"][:]

        # Extract QC flags if available
        pres_qc = ds.variables.get("PRES_QC", None)
        temp_qc = ds.variables.get("TEMP_QC", None)
        psal_qc = ds.variables.get("PSAL_QC", None)

        rows = []
        for p_idx in range(n_prof):
            lat_val = float(lats[p_idx])
            lon_val = float(lons[p_idx])
            for lvl in range(n_levels):
                p_val = float(pres[p_idx, lvl]) if pres.ndim == 2 else float(pres[lvl])
                t_val = float(temp[p_idx, lvl]) if temp.ndim == 2 else float(temp[lvl])
                s_val = float(psal[p_idx, lvl]) if psal.ndim == 2 else float(psal[lvl])

                # Skip unmeasured levels
                if np.isnan(p_val) or np.isnan(t_val) or p_val > 9999 or t_val > 99:
                    continue

                # Depth approx from pressure (dbar -> m)
                d_val = p_val / 1.019716

                # QC extraction
                qc_flag = 1
                if temp_qc is not None:
                    try:
                        q_char = temp_qc[p_idx, lvl] if temp_qc.ndim == 2 else temp_qc[lvl]
                        if isinstance(q_char, (bytes, str)) and str(q_char).strip() in ['3', '4']:
                            qc_flag = 4
                    except Exception:
                        pass

                rows.append({
                    "profile_id": f"argo_{file_path.stem}_{p_idx}",
                    "lat": lat_val,
                    "lon": lon_val,
                    "depth": d_val,
                    "pressure": p_val,
                    "temperature": t_val,
                    "salinity": s_val,
                    "qc_raw_flag": qc_flag
                })

        raw_df = pd.DataFrame(rows)
        return common_schema.standardize_dataframe(raw_df, source_name="Argo_GDAC")
    finally:
        ds.close()


def load_woce_netcdf(file_path: Path) -> pd.DataFrame:
    """
    Parses a WOCE CCHDO CTD NetCDF file into standardized DataFrame.
    """
    import netCDF4 as nc

    ds = nc.Dataset(str(file_path), "r")
    try:
        lat_var = ds.variables.get("latitude", ds.variables.get("lat", None))
        lon_var = ds.variables.get("longitude", ds.variables.get("lon", None))
        pres_var = ds.variables.get("pressure", ds.variables.get("pres", None))
        temp_var = ds.variables.get("temperature", ds.variables.get("temp", None))
        sal_var = ds.variables.get("salinity", ds.variables.get("sal", None))
        woce_flag_var = ds.variables.get("woce_quality_flag", ds.variables.get("qc_flag", None))

        rows = []
        if pres_var is not None and temp_var is not None:
            pres_arr = np.asarray(pres_var[:]).flatten()
            temp_arr = np.asarray(temp_var[:]).flatten()
            sal_arr = np.asarray(sal_var[:]).flatten() if sal_var is not None else np.full_like(pres_arr, np.nan)
            lat_val = float(lat_var[0]) if lat_var is not None and len(lat_var) > 0 else 0.0
            lon_val = float(lon_var[0]) if lon_var is not None and len(lon_var) > 0 else 0.0

            for i in range(len(pres_arr)):
                p = float(pres_arr[i])
                t = float(temp_arr[i])
                s = float(sal_arr[i])
                if np.isnan(p) or np.isnan(t) or p > 9000 or t > 50:
                    continue

                d = p / 1.019716
                qc = 2  # WOCE 2 = Good/Acceptable
                if woce_flag_var is not None:
                    try:
                        qc = int(woce_flag_var.flatten()[i])
                    except Exception:
                        pass

                rows.append({
                    "station_id": f"woce_{file_path.stem}",
                    "lat": lat_val,
                    "lon": lon_val,
                    "depth": d,
                    "pressure": p,
                    "temperature": t,
                    "salinity": s,
                    "qc_raw_flag": qc
                })

        raw_df = pd.DataFrame(rows)
        return common_schema.standardize_dataframe(raw_df, source_name="WOCE_CCHDO")
    finally:
        ds.close()


def load_tabular_dataset(file_path: Path, source_name: str = "Custom_Survey") -> pd.DataFrame:
    """
    Loads generic CSV or Excel dataset.
    """
    if file_path.suffix.lower() in [".xlsx", ".xls"]:
        df = pd.read_excel(file_path)
    else:
        df = pd.read_csv(file_path)

    return common_schema.standardize_dataframe(df, source_name=source_name)
