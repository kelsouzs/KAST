# Changelog

All notable changes to K-talysticFlow (KAST) will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

---

## [Unreleased]

### 🚧 In Development
- Additional evaluation metrics
- Enhanced visualization options
- Performance optimizations

---

## [1.0.0] - 2025-10-10

### 🎉 Initial Release

K-talysticFlow's first stable release! A complete deep learning pipeline for molecular activity prediction and virtual screening.

#### ✨ Core Features

**Pipeline Modules:**
- **Data Preparation** (`1_preparation.py`)
  - SMILES import and validation
  - Automatic train/test splitting
  - Balanced dataset generation
  - Activity labeling (active/inactive)

- **Featurization** (`2_featurization.py`)
  - ECFP/Morgan fingerprint generation (radius 2, 2048 bits)
  - DeepChem integration
  - Sparse matrix optimization
  - Batch processing for memory efficiency

- **Model Creation and Training** (`3_create_training.py`)
  - DeepChem MultitaskClassifier (neural network)
  - TensorFlow backend
  - Automatic checkpoint saving
  - Training metrics logging

- **Evaluation Suite** (4 modules)
  - `4_0_evaluation_main.py`: ROC/AUC, accuracy, precision, recall, F1-score
  - `4_1_cross_validation.py`: K-fold cross-validation with stratification
  - `4_2_enrichment_factor.py`: Enrichment factor calculation for screening validation
  - `4_3_tanimoto_similarity.py`: Chemical space analysis and diversity metrics
  - `4_4_learning_curve.py`: Training set size impact visualization

- **Prediction System** (2 modules)
  - `5_0_featurize_for_prediction.py`: New molecule featurization
  - `5_1_run_prediction.py`: Activity prediction with K-Prediction scoring

#### 🚀 Advanced Features

**Parallel Processing:**
- Multi-core support via joblib
- 5-10x speedup on large datasets (100K+ molecules)
- Automatic CPU detection and optimization
- Configurable worker count, batch size, and thresholds
- Memory-efficient batch processing
- Parallel-enabled scripts:
  - `2_featurization.py` (5-10x speedup)
  - `4_3_tanimoto_similarity.py` (3-5x speedup)
  - `4_4_learning_curve.py` (4-8x speedup)
  - `5_0_featurize_for_prediction.py` (5-10x speedup)

**Configuration System:**
- Centralized settings in `settings.py`
- Section 12: Parallel processing configuration
  - `ENABLE_PARALLEL_PROCESSING`: Master toggle
  - `N_WORKERS`: CPU core allocation (None=auto, -1=all, N=specific)
  - `PARALLEL_BATCH_SIZE`: Memory management (default: 100,000)
  - `PARALLEL_MIN_THRESHOLD`: Smart activation (default: 10,000)
- Easy customization for different hardware

**Interactive Control Panel:**
- `main.py`: User-friendly menu system
- Step-by-step workflow guidance
- Real-time parallel processing status display
- Advanced Options submenu ([8]):
  - Environment dependency checker
  - Parallel processing test suite
  - Runtime CPU core configuration
- Automatic screen clearing and formatting
- Progress indicators and colored output

**Logging & Error Handling:**
- File-based logging system (`results/logs/kast_YYYYMMDD.log`)
- Daily log rotation
- Error tracking with timestamps
- No console pollution (file-only logging)
- Comprehensive error messages

**Quality Assurance:**
- `check_env.py`: Dependency validation tool
- `test_parallel_compatibility.py`: 6-test comprehensive suite
  - Configuration validation
  - Import checking
  - Worker allocation logic
  - Threshold activation
  - Script integration
  - Performance validation

#### 🎨 User Experience

**Standardized Banners:**
- Consistent 70-character centered headers
- Script identification and descriptions
- Professional appearance across all modules

**Output Organization:**
- `results/` directory structure:
  - `featurized_datasets/`: Train/test features
  - `trained_model/`: Model checkpoints
  - `logs/`: Daily log files
  - Reports: Evaluation metrics and summaries
  - Plots: ROC curves, learning curves, visualizations

**K-Prediction Score:**
- Proprietary scoring system (0-1 range)
- Molecular activity ranking
- Confidence-based predictions
- Export to CSV with SMILES

#### 🛠️ Technical Stack

**Core Dependencies:**
- Python 3.9+
- DeepChem (deep learning framework)
- TensorFlow (neural network backend)
- RDKit (cheminformatics)
- scikit-learn (metrics and validation)
- joblib (parallel processing)
- pandas, numpy (data handling)
- matplotlib (visualization)
- tqdm (progress bars)

**Platform Support:**
- Windows (tested)
- Linux (compatible)
- macOS (compatible)

#### 📊 Performance

**Benchmarks on 100K molecule dataset:**
- Featurization: ~50 minutes → ~5 minutes (parallel)
- Tanimoto similarity: ~30 minutes → ~6 minutes (parallel)
- Learning curve: ~40 minutes → ~5 minutes (parallel)

**Memory Efficiency:**
- Sparse matrix compression
- Batch processing for large datasets
- Configurable memory footprint
- Disk-based storage for large features

#### 📚 Documentation

**User Documentation:**
- Comprehensive README.md
  - Installation guide
  - Quick start tutorial
  - Configuration instructions
  - FAQ section
  - Scientific workflow explanation
  - Usage examples
- Inline code comments
- Function docstrings

**Developer Documentation:**
- `PARALLEL_PROCESSING_GUIDE.md`: Detailed parallel implementation guide
- `COMPATIBILITY_CHECK.md`: Cross-platform compatibility notes
- `SPARSE_COMPRESSION_GUIDE.md`: Memory optimization strategies
- Test suite with examples

#### 🔧 Configuration Files

- `settings.py`: Central configuration (12 sections)
- `requirements.txt`: Python dependencies
- `.gitignore`: VCS exclusions
- `license.txt`: MIT license

#### 🎯 Use Cases

**Research Applications:**
- Virtual screening campaigns
- Lead compound identification
- Bioactivity prediction
- Chemical space exploration

**Educational Applications:**
- Deep learning in drug discovery
- Cheminformatics workflows
- Molecular fingerprinting
- Model evaluation techniques

#### ⚠️ Known Limitations

- Requires significant computational resources for large datasets
- GPU support not yet implemented

#### 💡 Future Roadmap

Planned for future releases:
- GPU acceleration for training
---


## Version Guidelines

This project follows [Semantic Versioning](https://semver.org/):
- **MAJOR** (X.0.0): Incompatible API changes
- **MINOR** (0.X.0): New functionality (backward-compatible)
- **PATCH** (0.0.X): Bug fixes (backward-compatible)

---

## Acknowledgments

**Development:**
- Késsia Souza Santos (Lead Developer)
- Prof. Dr. Manoelito Coelho dos Santos Junior (Supervisor)

**Institutional Support:**
- Laboratory of Molecular Modeling (LMM-UEFS)
- State University of Feira de Santana (UEFS)
- National Council for Scientific and Technological Development (CNPq)

**Open Source Community:**
- DeepChem contributors
- RDKit developers
- TensorFlow team
- All package maintainers

---

*For detailed information about each change, see commit history on GitHub.*
