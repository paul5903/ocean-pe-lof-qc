# -*- coding: utf-8 -*-
import pandas as pd
from src import qc_rules


def test_domain_rules_normal_row():
    normal_row = pd.Series({
        "lat": 15.0,
        "lon": 112.0,
        "temperature": 25.0,
        "salinity": 34.5,
        "depth": 100.0,
        "pressure": 102.0,
        "sound_vel_direct": 1530.0
    })
    is_fail, details = qc_rules.evaluate_domain_rules_row(normal_row)
    assert not is_fail
    assert details == "None"


def test_domain_rules_temperature_anomaly():
    anom_row = pd.Series({
        "lat": 15.0,
        "lon": 112.0,
        "temperature": 55.0,  # Physically impossible in open ocean
        "salinity": 34.5,
        "depth": 100.0,
        "pressure": 102.0,
        "sound_vel_direct": 1530.0
    })
    is_fail, details = qc_rules.evaluate_domain_rules_row(anom_row)
    assert is_fail
    assert "R2_Temperature_Out_Of_Range" in details


def test_assign_final_qc_status():
    df = pd.DataFrame({
        "qc_rule_fail": [False, True, False, True],
        "qc_ai_suspect": [False, False, True, True]
    })
    df_out = qc_rules.assign_final_qc_status(df)
    assert list(df_out["qc_final_status"]) == [
        "QC_PASS",
        "QC_RULE_FAIL",
        "QC_AI_SUSPECT",
        "QC_EXPERT_REVIEW"
    ]
