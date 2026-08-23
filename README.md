# Physics-Embedded Local Outlier Factor (PE-LOF) for Oceanographic Quality Control

[![CI Pipeline](https://github.com/ocean-ai-research/ocean-pe-lof-qc/actions/workflows/ci.yml/badge.svg)](https://github.com/ocean-ai-research/ocean-pe-lof-qc/actions)
[![Python 3.8+](https://img.shields.io/badge/python-3.8%20%7C%203.9%20%7C%203.10%20%7C%203.11-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![DPI: 300](https://img.shields.io/badge/Figures-300%20DPI%20Publication%20Ready-success.svg)](results/)

An open-source research and operational benchmark framework implementing **Physics-Embedded Local Outlier Factor (PE-LOF)** for automated, high-precision Quality Control (QC) of oceanographic profile and CTD observations.

This repository packages:
1. **Core 6-Model Benchmark & Physics Ablation Suite**: Comparative evaluation of PE-LOF against Isolation Forest, One-Class SVM, Elliptic Envelope, DBSCAN, and Deep AutoEncoder.
2. **International Validation Pipelines**: Independent 60:20:20 validation on **Argo GDAC NetCDF** (3,240 profile obs) and **WOCE CCHDO CTD** (3,401 cruise obs) against NOAA **WOA23 Climatology**.
3. **Pre-Generated 300 DPI Figures & Tables**: Ready-to-publish scientific assets in `results/`.

---

## 🌟 Key Scientific Innovations

1. **Physics-Embedded Feature Representation (14 Dimensions)**:
   Embeds fundamental physical oceanography formulations into density-based manifold learning:
   - **Mackenzie (1981) Acoustic Speed Formulation**: 9-term equation mapping temperature, salinity, and depth into theoretical sound velocity.
   - **UNESCO EOS-80 Density Approximation**: Equation of state estimating in-situ density ($\rho$) and hydrostatic compressibility.
   - **Hydrostatic Pressure Consistency**: Linear depth-pressure ratio ($1.019716 \text{ dbar/m}$) and anomaly residuals.
   - **Logarithmic Vertical Temperature Decay Gradient**: $T / (\ln(1 + D) + 1)$ capturing thermocline stratifications.

2. **Profile-Aware 60:20:20 Partition (Zero Data Leakage)**:
   Data splits are strictly partitioned by whole `profile_id` / `station_id` units, ensuring that measurement levels from the same physical cast are never leaked between Train, Validation, and Test sets.

3. **Semi-Supervised Novelty Estimator**:
   `RobustScaler` and `LocalOutlierFactor` ($k=35$, $\text{contamination}=0.08$) are fitted **strictly on clean baseline samples** ($QC \le 2$) of the training split, enabling robust detection of novel, unseen sensor failure modes.

4. **Precision-Recall Threshold Calibration ($\tau^*$)**:
   Scientific determination of operating thresholds on validation curves balancing Safety-Critical Recall ($F_2$) and High-Precision Filtering ($F_1$).

5. **Statistical Rigor with Bootstrap 95% Confidence Intervals**:
   All evaluation metrics (ROC-AUC, PR-AUC, Sensitivity, Specificity, Precision, MCC) are reported with 1,000 non-parametric bootstrap iterations.

6. **4-Tier Hybrid Decision Engine**:
   Synthesizes physical domain rules (IOC/WMO GTSPP standards) with unsupervised AI anomaly detection:
   - `QC_PASS`: Passed physical rules and verified by PE-LOF.
   - `QC_RULE_FAIL`: Gross physical/range violation.
   - `QC_AI_SUSPECT`: Passed single-sensor rules but identified as a subtle multi-variate anomaly.
   - `QC_EXPERT_REVIEW`: Flagged by both physical rules and AI model.

---

## 📊 Comprehensive Experimental Results

### 1. Multi-Model Benchmark Comparison (Test Split)

| Model Architecture | ROC-AUC (95% CI) | PR-AUC (95% CI) | F1-Score | Recall (Sensitivity) | Specificity | MCC |
|---|---|---|---|---|---|---|
| **PE-LOF (Proposed)** | **0.9976** [0.9961 - 0.9988] | **0.9854** [0.9780 - 0.9912] | **95.00%** | **99.13%** | **99.17%** | **0.9458** |
| Isolation Forest | 0.9812 [0.9750 - 0.9870] | 0.9245 [0.9080 - 0.9410] | 89.20% | 91.50% | 98.40% | 0.8840 |
| One-Class SVM (RBF) | 0.9450 [0.9320 - 0.9570] | 0.8630 [0.8410 - 0.8850] | 82.10% | 85.30% | 96.80% | 0.8050 |
| Elliptic Envelope | 0.9210 [0.9050 - 0.9360] | 0.8120 [0.7890 - 0.8350] | 78.40% | 80.20% | 95.50% | 0.7620 |
| Deep AutoEncoder | 0.9680 [0.9580 - 0.9770] | 0.8990 [0.8800 - 0.9170] | 86.70% | 88.90% | 97.90% | 0.8560 |
| DBSCAN | 0.8830 [0.8650 - 0.9010] | 0.7450 [0.7180 - 0.7710] | 71.30% | 74.60% | 93.80% | 0.6890 |

### 2. Independent International Validation (Argo & WOCE)

| Experimental Source | Raw Input Format | Sample Size | Test ROC-AUC (95% CI) | Test PR-AUC (95% CI) | Test Specificity |
|---|---|---|---|---|---|
| **Argo Profiling Floats** | Ifremer GDAC NetCDF | 3,240 obs | **0.9733** [0.9550 - 0.9944] | **0.8875** [0.8237 - 0.9694] | **98.29%** |
| **WOCE Deep-Sea CTD** | UCSD CCHDO NetCDF | 3,401 obs | **0.9382** [0.8901 - 0.9966] | **0.1342** [0.0747 - 0.7973] | **93.64%** |
| **WOA23 Reference Grid** | NOAA NCEI NetCDF | 8,624 grids | Climatology Mean & Standard Deviation Reference Field | — | — |

---

## 📁 Repository Structure

```
ocean_pe_lof_qc/
├── .github/workflows/ci.yml         # Automated GitHub Actions CI pipeline
├── config/                          # Declarative YAML configurations
│   ├── global_config.yaml           # Domain physical bounds & runtime options
│   ├── feature_schema.yaml          # 14 physics-embedded feature definitions
│   └── model_parameters.yaml        # LOF hyperparameters & operating thresholds
├── results/                         # Pre-Generated Scientific Figures & Tables
│   ├── international_argo_woce/     # International Argo & WOCE evaluation
│   │   ├── figures_argo/            # 300 DPI Argo figures (ROC/PR, CM, GridSearch)
│   │   ├── figures_woce/            # 300 DPI WOCE figures (ROC/PR, CM, GridSearch)
│   │   ├── tables/                  # Consolidated benchmark Excel & CSV tables
│   │   └── reports/                 # Complete Markdown research articles
│   └── benchmark_6_models/          # Multi-model comparison suite
│       ├── figures_300dpi/          # 16 High-resolution publication figures (Fig1-Fig16)
│       └── tables_excel/            # Detailed performance & descriptive tables (Bang1-Bang7)
├── src/                             # Complete Python source code
│   ├── __init__.py
│   ├── config.py                    # Dynamic relative path loader
│   ├── physics_engine.py            # Mackenzie (1981), UNESCO EOS-80, Hydrostatics
│   ├── feature_builder.py           # 14-Feature vector engineering pipeline
│   ├── qc_rules.py                  # 7 IOC/WMO domain checks & 4-tier QC synthesis
│   ├── models.py                    # PhysicsEmbeddedLOF class & Bootstrap CI
│   ├── common_schema.py             # Heterogeneous data standardizer
│   ├── data_fetchers.py             # NetCDF & Tabular loaders
│   ├── download_all_real_data.py    # Downloader for real GDAC Argo, NOAA WOA23, CCHDO
│   ├── download_public_data.py      # Quick downloader for public sample suite
│   ├── data_generator.py            # Synthetic profile & sensor fault generator
│   ├── run_all_independent_pipelines.py # Full 8-step independent execution runner
│   ├── run_pipeline.py              # CLI benchmark orchestration entrypoint
│   ├── benchmark_all_models.py      # 6-Model benchmark runner
│   ├── deep_analysis.py             # Ablation study & physics contribution analysis
│   ├── generate_paper_report.py     # Markdown & LaTeX comparative table builder
│   ├── visualizer.py                # 300 DPI publication figure generator
│   └── utils.py                     # Logging and SHA-256 data integrity tracking
├── tests/                           # Unit and integration test suite
│   ├── test_physics_engine.py       # Physical equations unit tests
│   ├── test_feature_builder.py      # Feature extractor & NaN robustness tests
│   ├── test_qc_rules.py             # Physical rule validation tests
│   └── test_pe_lof_pipeline.py      # End-to-end integration test
├── .env.example                     # Environment configuration template
├── .gitignore                       # Clean gitignore (tracks results/, ignores raw caches)
├── CONTRIBUTING.md                  # Open-source contribution guidelines
├── LICENSE                          # MIT License
├── pyproject.toml                   # Standard packaging definition
└── requirements.txt                 # Pinned dependencies
```

---

## 🚀 Execution & Reproduction Guide

### 1. Installation

```bash
git clone https://github.com/ocean-ai-research/ocean-pe-lof-qc.git
cd ocean-pe-lof-qc

python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Run International Pipelines (Argo, WOCE & WOA23)

```bash
# 1. Download official NetCDF files from GDAC & CCHDO
python -m src.download_all_real_data

# 2. Run full 8-step independent 60:20:20 evaluation
python -m src.run_all_independent_pipelines

# 3. Compile comparative scientific report
python -m src.generate_paper_report
```

### 3. Run 6-Model Benchmark & Physics Ablation Study

```bash
# Run multi-model benchmark (LOF vs IF vs OCSVM vs EE vs DBSCAN vs AutoEncoder)
python -m src.benchmark_all_models

# Run deep physical ablation analysis
python -m src.deep_analysis
```

### 4. Run Automated Test Suite

```bash
pytest tests/ -v
```

---

## 🔒 Security & Data Integrity

- **Zero Information Leakage**: All absolute user paths, internal company identifiers, and raw private field files have been removed or replaced with dynamic generators.
- **Data Provenance**: All downloaded files are tracked via SHA-256 integrity hashes.
- **Self-Contained Execution**: Unit tests and demo runs work 100% offline out-of-the-box using the built-in physical profile generator.

---

## 📜 Citation & License

This project is licensed under the [MIT License](LICENSE).
