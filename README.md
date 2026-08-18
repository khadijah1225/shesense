# SheSense: Female-Focused Stress Detection & Cycle Insights

**Build a female-focused stress detector from physiological signals, then relate stress load to menstrual-cycle regularity using temperature-based ovulation patterns.**

## 🎯 Project Goal

SheSense uses wearable sensor data to detect stress patterns in women and explores how chronic stress might be linked to menstrual cycle irregularities. The system trains a stress classifier from physiological signals (HR/HRV, EDA, skin temperature, accelerometry) and correlates "stress load" over time with temperature-based markers of ovulation (biphasic rise) and cycle logs.

## 🌟 Why It's Unique

- **Female-focused pipeline**: Supports female-only filtering where subject metadata is available
- **Stress-cycle connection**: Links stress patterns to ovulatory markers, not just single temperature cutoffs  
- **Pattern-level analysis**: Detects presence/absence of luteal temperature rise vs. Apple's retrospective estimation
- **Fairness-focused**: Ensures model performance across different female populations

## 📊 Datasets

- **Hongn et al. (Empatica E4)**: 18 female subjects (f01-f18) with BVP→HR/HRV, EDA, skin temperature, accelerometry
- **SWELL-KW**: Knowledge-work stress with ECG-derived HR/HRV + EDA from female participants

Notes:
- `scripts/extractdata.py` currently supports `wesad` and `swell` extraction.
- Hongn CSVs are expected to be prepared beforehand (or converted with your own preprocessing) and then passed to `scripts/build_dataset.py --hongn ...`.
- SWELL female-only filtering is not automatically enforced in `build_dataset.py` yet; you must provide female-only SWELL inputs upstream if that is required for your experiments.

## 🗂 Repository Structure

```
shesense/
├── .venv/                    # Python virtual environment  
├── backend/                  # API/model serving (planned)
├── data/
│   ├── raw/
│   │   ├── Wearable_Dataset/    # Hongn et al. Empatica E4 files
│   │   └── dataverse_files/     # SWELL-KW data
│   └── processed/               # Cleaned datasets
├── scripts/
│   ├── extractdata.py          # Raw data extraction utilities
│   ├── data_contracts.py       # Data standardization & validation
│   ├── features.py             # Feature extraction (HRV, EDA, temp)
│   ├── build_dataset.py        # Dataset harmonization & merging
│   ├── make_windows.py         # Windowing & train/val/test splits
│   └── train_baseline.py       # Baseline model training
├── tests/                      # Unit tests
├── simulator/                  # Mock streaming (planned)
└── requirements.txt
```

## 🚀 Quick Start

### 1. Setup Environment

```bash
# Python 3.8+ is required

# Create virtual environment
python -m venv .venv
.venv\Scripts\activate  # Windows
# source .venv/bin/activate  # Linux/Mac

# Install dependencies
pip install -r requirements.txt

# Optional sanity check
python setup.py
```

### 2. Prepare Input CSVs

```bash
# Extract WESAD data
python scripts/extractdata.py wesad data/raw/WESAD/S2.pkl data/processed/wesad_S2.csv

# Extract SWELL-KW data  
python scripts/extractdata.py swell data/raw/dataverse_files/swell_kw.zip data/processed/swell_kw.csv

# Hongn input should be an already prepared CSV, for example:
# data/processed/hongn.csv
```

### 3. Build Unified Dataset

```bash
# Harmonize and merge datasets
python scripts/build_dataset.py \
  --hongn data/processed/hongn.csv \
  --swell data/processed/swell_kw.csv \
  --output data/processed/combined_1hz.parquet \
  --female-only \
  --quality-report
```

### 4. Create Windowed Features

```bash
# Extract 60-second windows with 30-second hops
python scripts/make_windows.py \
  --input data/processed/combined_1hz.parquet \
  --output data/processed/windowed_features \
  --window-sec 60 \
  --hop-sec 30 \
  --female-only
```

### 5. Train Baseline Model

```bash
# Train Random Forest classifier
python scripts/train_baseline.py \
  --data data/processed/windowed_features \
  --model random_forest \
  --output artifacts/baseline_rf \
  --cross-val
```

## 🧪 Development Pipeline

### Current Implementation

- ✅ **Data Extraction**: `extractdata.py` handles WESAD pickle and SWELL-KW CSV extraction
- ✅ **Data Contracts**: Standardized schema across datasets with validation
- ✅ **Feature Engineering**: HRV (RMSSD/SDNN/LF/HF), EDA (tonic/phasic), temperature, accelerometer features
- ✅ **Windowing**: 60-second sliding windows with subject-aware train/val/test splits
- ✅ **Baseline Models**: RandomForest and XGBoost with comprehensive evaluation metrics

### Planned Features

- 🔄 **1D-CNN Model**: PyTorch-based deep learning for multichannel sequences
- 🔄 **Stress Load Aggregation**: Daily/weekly stress quantification from window predictions
- 🔄 **Cycle Detection**: Biphasic temperature rise detection for ovulation tracking
- 🔄 **Correlation Analysis**: Link stress patterns to cycle regularity
- 🔄 **MVP UI**: Simple web interface for cycle logging and visualization

## 📈 Target Metrics

- **Classification**: AUROC, AUPRC, F1, sensitivity/specificity
- **Fairness**: Performance consistency across female subgroups
- **Subject-wise**: Within-subject and cross-subject evaluation
- **Cycle Correlation**: Association between stress load and ovulation patterns

## 🧬 Research Focus

**Guardrails & Assumptions:**
- Research code prioritizing clarity & reproducibility
- Female-only filtering is automatic for Hongn-style `f*` subject IDs; SWELL requires explicit female subject selection upstream  
- Lightweight pytest tests for feature math and windowing
- Decision logging (dropped segments, subject exclusions) to artifacts/

## 📝 Example Commands

```bash
# Full pipeline example
python scripts/extractdata.py swell data/raw/dataverse_files.zip data/processed/swell.csv
python scripts/build_dataset.py --swell data/processed/swell.csv --output data/processed/female_1hz.parquet --female-only
python scripts/make_windows.py --input data/processed/female_1hz.parquet --output data/processed/features --window-sec 60
python scripts/train_baseline.py --data data/processed/features --model random_forest --output models/rf_baseline
```

## ✅ Testing

```bash
pytest -q

# Or run test files individually
pytest tests/test_data_contracts.py -q
pytest tests/test_features.py -q
```

## 📦 Expected Outputs

- `scripts/build_dataset.py`:
  - Output parquet at `--output`
  - Optional `data_quality_report.json` and `data_quality_summary.png` when `--quality-report` is set
- `scripts/make_windows.py`:
  - `train.parquet`, `test.parquet`, optional `val.parquet`, and `metadata.json` in `--output` directory
- `scripts/train_baseline.py`:
  - Evaluation artifacts such as `evaluation_results.json`, confusion matrix, ROC/PR curves, and feature importance plots

## 🚢 Pre-Push Checklist

```bash
# 1) Run tests
pytest -q

# 2) Review what will be committed
git status

# 3) Stage and commit
git add README.md
git commit -m "Align README with implemented CLI and workflow"

# 4) Push
git push origin main
```

## 🤝 Contributing

This is research code focused on advancing understanding of stress-cycle interactions in women's health. Contributions welcome for:

- Additional physiological feature extraction methods
- Improved cycle detection algorithms  
- Fairness and bias evaluation metrics
- Visualization and interpretation tools

## 📚 References

- Hongn et al.: Empatica E4 dataset for stress detection
- SWELL-KW: Knowledge work stress dataset
- Apple HealthKit: Temperature-based cycle tracking methods
