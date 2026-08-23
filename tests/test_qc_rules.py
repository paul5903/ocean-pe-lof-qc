import pandas as pd
from src.core import qc_rules

def test_normal_observation():
    row = pd.Series({
        "lat": 15.0, "lon": 112.0, "temperature": 25.0, "salinity": 34.5,
        "depth": 100.0, "pressure": 102.0, "sound_vel_direct": 1530.0
    })
    is_fail, details = qc_rules.evaluate_domain_rules_row(row)
    assert not is_fail
    assert details == "None"

def test_anomaly_observation():
    row = pd.Series({
        "lat": 15.0, "lon": 112.0, "temperature": 55.0, "salinity": 34.5,
        "depth": 100.0, "pressure": 102.0, "sound_vel_direct": 1530.0
    })
    is_fail, details = qc_rules.evaluate_domain_rules_row(row)
    assert is_fail
    assert "R2_Temperature_Out_Of_Range" in details
