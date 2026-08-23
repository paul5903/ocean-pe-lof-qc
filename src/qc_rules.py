# -*- coding: utf-8 -*-
"""
Oceanographic Quality Control Domain Rules:
Implements standard physical and range tests for ocean observation quality flags.
"""
import numpy as np
import pandas as pd
from typing import Tuple
from . import config


def evaluate_domain_rules_row(row: pd.Series) -> Tuple[bool, str]:
    """
    Evaluates physical oceanographic domain checks on a single measurement point.
    Returns: (is_violated, violated_rules_string)
    """
    violated = []

    # R1: Geographic Bounds
    lat, lon = row.get("lat"), row.get("lon")
    if pd.isna(lat) or pd.isna(lon):
        violated.append("R1_Missing_Coordinates")

    # R2: Seawater Temperature Range
    temp = row.get("temperature")
    if pd.notna(temp) and not (config.TEMP_MIN <= temp <= config.TEMP_MAX):
        violated.append("R2_Temperature_Out_Of_Range")

    # R3: Practical Salinity Range
    sal = row.get("salinity")
    if pd.notna(sal) and not (config.SAL_MIN <= sal <= config.SAL_MAX):
        violated.append("R3_Salinity_Out_Of_Range")

    # R4: Depth Range
    depth = row.get("depth")
    if pd.notna(depth) and not (config.DEPTH_MIN <= depth <= config.DEPTH_MAX):
        violated.append("R4_Depth_Out_Of_Range")

    # R5: Hydrostatic Pressure vs Depth Consistency
    pres = row.get("pressure")
    if pd.notna(pres) and pd.notna(depth):
        expected_p = depth * config.PRESSURE_RATIO_APPROX
        allowed_diff = config.PRESSURE_TOL_ABS + (depth * config.PRESSURE_TOL_REL)
        if abs(pres - expected_p) > allowed_diff:
            violated.append("R5_Pressure_Depth_Inconsistency")

    # R6: Sound Velocity Physical Range
    sv = row.get("sound_vel_direct")
    if pd.notna(sv) and not (config.SOUND_VEL_MIN <= sv <= config.SOUND_VEL_MAX):
        violated.append("R6_Sound_Velocity_Out_Of_Range")

    # R7: Acoustic Cross-Check Discrepancy
    sv_calc = row.get("sound_vel_calc")
    if pd.notna(sv) and pd.notna(sv_calc):
        if abs(sv - sv_calc) > config.SOUND_VEL_DIFF_MAX:
            violated.append("R7_Acoustic_Equation_Discrepancy")

    is_failed = len(violated) > 0
    return is_failed, "; ".join(violated) if is_failed else "None"


def apply_domain_rules(df: pd.DataFrame) -> pd.DataFrame:
    """Applies all oceanographic domain checks across an entire DataFrame."""
    df_out = df.copy()
    rule_fails = []
    rule_details = []

    for _, row in df_out.iterrows():
        is_fail, details = evaluate_domain_rules_row(row)
        rule_fails.append(is_fail)
        rule_details.append(details)

    df_out["qc_rule_fail"] = rule_fails
    df_out["qc_rule_details"] = rule_details
    return df_out


def assign_final_qc_status(
    df: pd.DataFrame,
    rule_col: str = "qc_rule_fail",
    ai_col: str = "qc_ai_suspect"
) -> pd.DataFrame:
    """
    Synthesizes rule-based checks and AI anomaly detection into a 4-tier QC status:
    1. QC_PASS: Passed both physical rules and AI anomaly model
    2. QC_RULE_FAIL: Failed domain rule, AI passed
    3. QC_AI_SUSPECT: Passed domain rules, flagged as subtle multi-variate anomaly by AI
    4. QC_EXPERT_REVIEW: Flagged by both physical rules and AI model
    """
    df_out = df.copy()
    statuses = []

    for _, row in df_out.iterrows():
        r_fail = bool(row.get(rule_col, False))
        ai_fail = bool(row.get(ai_col, False))

        if r_fail and ai_fail:
            statuses.append("QC_EXPERT_REVIEW")
        elif r_fail:
            statuses.append("QC_RULE_FAIL")
        elif ai_fail:
            statuses.append("QC_AI_SUSPECT")
        else:
            statuses.append("QC_PASS")

    df_out["qc_final_status"] = statuses
    return df_out
