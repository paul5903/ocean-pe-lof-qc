# Contributing to Ocean PE-LOF QC

Thank you for your interest in contributing to the **Physics-Embedded Local Outlier Factor (PE-LOF)** project for oceanographic quality control!

## Code of Conduct
Please be respectful, constructive, and adhere to standard scientific integrity in all interactions and contributions.

## Development Workflow
1. **Fork the repository** on GitHub.
2. **Clone your fork**:
   ```bash
   git clone https://github.com/your-username/ocean-pe-lof-qc.git
   cd ocean-pe-lof-qc
   ```
3. **Create a virtual environment & install dependencies**:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   pip install -r requirements.txt
   ```
4. **Create a feature branch**:
   ```bash
   git checkout -b feature/your-feature-name
   ```
5. **Run test suite before submitting**:
   ```bash
   pytest tests/
   ```
6. **Submit a Pull Request (PR)** with a clear explanation of changes and benchmark impacts.

## Security & Sensitive Information Rule
- **NEVER commit private survey datasets, credentials, API keys, or raw confidential binaries.**
- Always use the provided synthetic generator (`python -m src.data.generate_mock_data`) or official open GDAC/NOAA links for reproducible test cases.
