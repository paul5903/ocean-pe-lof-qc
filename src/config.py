# -*- coding: utf-8 -*-
"""
Configuration and Path Management for Ocean PE-LOF QC.
Dynamically resolves paths relative to repository root.
"""
import os
from pathlib import Path
import yaml
from typing import Dict, Any

# Root & Subdirectory Paths
SRC_DIR = Path(__file__).resolve().parent
ROOT_DIR = SRC_DIR.parent
CONFIG_DIR = ROOT_DIR / "config"
DATA_DIR = ROOT_DIR / "data"
OUTPUTS_DIR = ROOT_DIR / "outputs"
REPORTS_DIR = ROOT_DIR / "reports"
MODELS_DIR = ROOT_DIR / "models"

# Ensure runtime directories exist
for p in [DATA_DIR, OUTPUTS_DIR, REPORTS_DIR, MODELS_DIR]:
    p.mkdir(parents=True, exist_ok=True)


def load_yaml(file_path: Path) -> Dict[str, Any]:
    """Safely loads YAML configuration files."""
    if not file_path.exists():
        raise FileNotFoundError(f"Configuration file not found at: {file_path}")
    with open(file_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


# Load Configuration Dictionaries
GLOBAL_CONFIG = load_yaml(CONFIG_DIR / "global_project.yaml") if (CONFIG_DIR / "global_project.yaml").exists() else load_yaml(CONFIG_DIR / "global_config.yaml")
FEATURE_CONFIG = load_yaml(CONFIG_DIR / "feature_schema.yaml")
MODEL_CONFIG = load_yaml(CONFIG_DIR / "model_parameters.yaml")

# Oceanographic Domain Rules
DOMAIN_RULES = GLOBAL_CONFIG.get("domain_rules", {})
TEMP_MIN = float(DOMAIN_RULES.get("temp_min", -2.5))
TEMP_MAX = float(DOMAIN_RULES.get("temp_max", 40.0))
SAL_MIN = float(DOMAIN_RULES.get("sal_min", 0.0))
SAL_MAX = float(DOMAIN_RULES.get("sal_max", 45.0))
DEPTH_MIN = float(DOMAIN_RULES.get("depth_min", 0.0))
DEPTH_MAX = float(DOMAIN_RULES.get("depth_max", 6000.0))
SOUND_VEL_MIN = float(DOMAIN_RULES.get("sound_vel_min", 1350.0))
SOUND_VEL_MAX = float(DOMAIN_RULES.get("sound_vel_max", 1700.0))
PRESSURE_RATIO_APPROX = float(DOMAIN_RULES.get("pressure_ratio_approx", 1.019716))
PRESSURE_TOL_ABS = float(DOMAIN_RULES.get("pressure_tol_abs", 20.0))
PRESSURE_TOL_REL = float(DOMAIN_RULES.get("pressure_tol_rel", 0.35))
SOUND_VEL_DIFF_MAX = float(DOMAIN_RULES.get("sound_vel_diff_max", 20.0))

# Model Hyperparameters
LOCKED_FEATURES = FEATURE_CONFIG.get("ordered_feature_names", [
    "lat", "lon", "depth", "temperature", "pressure", "sound_vel_direct",
    "salinity", "density", "sv_mackenzie_residual", "pressure_residual",
    "pressure_depth_ratio", "temp_gradient", "spatial_z_dist", "geo_out_of_bounds_dist"
])
N_NEIGHBORS = int(MODEL_CONFIG.get("n_neighbors", 35))
CONTAMINATION = float(MODEL_CONFIG.get("contamination", 0.08))
OPTIMAL_THRESHOLD = float(MODEL_CONFIG.get("optimal_threshold", 2.0748))
RANDOM_SEED = int(os.environ.get("OCEAN_QC_RANDOM_SEED", GLOBAL_CONFIG.get("execution", {}).get("random_seed", 42)))
DPI = int(os.environ.get("OCEAN_QC_DPI", GLOBAL_CONFIG.get("execution", {}).get("dpi", 300)))
