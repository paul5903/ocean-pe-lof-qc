# Physics-Embedded Local Outlier Factor (PE-LOF) for Oceanographic Quality Control

[![CI Pipeline](https://github.com/MeoU593/ocean-pe-lof-qc/actions/workflows/ci.yml/badge.svg)](https://github.com/MeoU593/ocean-pe-lof-qc/actions)
[![Python 3.9+](https://img.shields.io/badge/python-3.9%20%7C%203.10%20%7C%203.11%20%7C%203.12-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![DPI: 300](https://img.shields.io/badge/Figures-300%20DPI%20Ready-success.svg)](results/)

An open-source benchmark framework implementing **Physics-Embedded Local Outlier Factor (PE-LOF)** for automated Quality Control (QC) of oceanographic profile and CTD observations.

---

## 🌟 Key Features

- **Physics-Embedded Manifold Learning**: Incorporates Mackenzie (1981) acoustic speed, UNESCO EOS-80 density, hydrostatic pressure, and logarithmic vertical temperature decay gradient.
- **Strict Profile Partitioning (60:20:20)**: Train, Validation, and Test sets partitioned by whole profile units with zero data leakage.
- **Semi-Supervised Novelty Detection**: RobustScaler and LOF fitted strictly on clean observations ($QC \le 2$).
- **Precision-Recall Threshold Optimization**: Scientific threshold calibration maximizing safety-critical recall ($F_2$) or balanced performance ($F_1$).
- **Separation of Concerns**:
  - `src/core/`: Shared physics engine, feature extractor, QC rules, and evaluation metrics.
  - `src/international/`: Automated pipelines for global open-access datasets (Argo GDAC, WOCE CCHDO, NOAA WOA23).
  - `src/regional/`: Pipelines for regional/in-situ CTD survey data with synthetic anomaly generators.

---

## 📁 Repository Structure

```
ocean_pe_lof_qc/
├── .github/workflows/ci.yml         # Automated CI test matrix (Python 3.9-3.12)
├── config/                          # Declarative YAML configurations
│   ├── global_config.yaml           # Domain physical bounds & options
│   ├── feature_schema.yaml          # 14 physics-embedded feature definitions
│   └── model_parameters.yaml        # LOF hyperparameters & thresholds
├── results/                         # Pre-Generated Publication Assets
│   ├── international/               # Argo and WOCE validation figures & tables
│   └── regional_benchmark/          # 6-Model comparison figures & tables
├── src/
│   ├── config.py                    # Dynamic configuration and path manager
│   ├── core/                        # Shared algorithmic modules
│   │   ├── physics_engine.py        # Physical oceanography equations
│   │   ├── feature_builder.py       # 14-Feature vector extractor
│   │   ├── qc_rules.py              # IOC/WMO domain checks & 4-tier QC
│   │   ├── models.py                # PE-LOF model & Bootstrap CI
│   │   ├── schema.py                # Data schema normalization
│   │   ├── visualizer.py            # Publication plot generator
│   │   └── utils.py                 # Logging and file verification
│   ├── international/               # Global dataset pipelines
│   │   ├── download_real_data.py    # Downloader for GDAC, NOAA, CCHDO
│   │   └── run_all_international.py # International pipeline runner
│   └── regional/                    # Regional / in-situ CTD pipelines
│       ├── data_generator.py        # Synthetic profile & fault injector
│       └── run_regional_pipeline.py # Regional pipeline runner
├── tests/                           # Unit and integration test suite
│   ├── test_physics.py
│   ├── test_features.py
│   ├── test_qc_rules.py
│   └── test_pipelines.py
├── .gitignore
├── LICENSE
├── pyproject.toml
└── requirements.txt
```

---

## 🚀 Quickstart

### Installation

```bash
git clone https://github.com/MeoU593/ocean-pe-lof-qc.git
cd ocean-pe-lof-qc

python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### Running Regional / In-Situ Pipeline

```bash
# Run with synthetic profile generator (100% offline)
python -m src.regional.run_regional_pipeline --profiles 50

# Or run with custom survey data:
python -m src.regional.run_regional_pipeline --data-path data/custom_ctd_survey.xlsx
```

### Running International Pipelines (Argo & WOCE)

```bash
# 1. Download official GDAC Argo profiles
python -m src.international.download_real_data

# 2. Run independent international evaluation
python -m src.international.run_all_international --source argo
```

### Running Test Suite

```bash
pytest tests/ -v
```

---

## 📊 Benchmark Results Summary

| Dataset / Experiment | ROC-AUC (95% CI) | PR-AUC (95% CI) | F1-Score | Specificity | Sensitivity |
|---|---|---|---|---|---|
| **Regional Benchmark** | **0.9976** [0.9961 - 0.9988] | **0.9854** [0.9780 - 0.9912] | **95.00%** | **99.17%** | **99.13%** |
| **Argo Profiling Floats** | **0.9733** [0.9550 - 0.9944] | **0.8875** [0.8237 - 0.9694] | **80.72%** | **98.29%** | **79.37%** |
| **WOCE Deep-Sea CTD** | **0.9382** [0.8901 - 0.9966] | **0.1342** [0.0747 - 0.7973] | **32.35%** | **93.64%** | **84.62%** |

---

## 📜 License

This project is licensed under the [MIT License](LICENSE).
