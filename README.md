# 🚀 K-talysticFlow (KAST) — Deep Learning Molecular Screening Pipeline

![Python](https://img.shields.io/badge/Python-3.9%2B-blue.svg)
![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)
![Status](https://img.shields.io/badge/Status-In%20Development-orange.svg)
![DOI](https://img.shields.io/badge/DOI-coming%20soon-lightgrey.svg)
[![Documentation](https://img.shields.io/badge/Docs-here-8A2BE2)](https://lmm.uefs.br/wp-content/uploads/2025/10/KAST-Documentation.html)
[![LinkedIn](https://img.shields.io/badge/-LinkedIn-blue?style=flat-square&logo=linkedin&logoColor=white&link=https://www.linkedin.com/in/kelsouzs)](https://www.linkedin.com/in/kelsouzs)
[![GitHub](https://img.shields.io/badge/-GitHub-181717?style=flat-square&logo=github&logoColor=white&link=https://github.com/kelsouzs)](https://github.com/kelsouzs)

```
  __  __    _     ____  _____ 
  | |/ /   / \   / ___||_   _|
  | ' /   / _ \  \___ \  | |  
  | . \  / ___ \  ___) | | |  
  |_|\_\/_/   \_\|____/  |_|  K-talystic Automated Screening Taskflow
```

---

## 👨‍🔬 What is K-talysticFlow (KAST)?

K-talysticFlow is a fully automated, interactive pipeline for training, evaluating, and using Deep Learning models to predict molecular bioactivity.  
Built for virtual screening and rapid identification of promising chemical compounds — accelerating drug discovery.

**Current Version: 1.0.0** (Under Development)

Developed at the [Laboratory of Molecular Modeling (LMM-UEFS)](https://lmm.uefs.br/), funded by CNPq.

---

## ✨ Features at a Glance

- ⚡ **Automated CLI Workflow:** Interactive menu with progress bars and modular scripts
- 🚀 **Parallel Processing:** Multi-core support for 5-10x faster featurization and prediction
- 🧠 **Deep Learning Model:** Neural network (DeepChem/Tensorflow) learns molecular fingerprints (ECFP/Morgan)
- 🎯 **K-Prediction Score:** Proprietary scoring system for ranking molecular activity predictions
- 🧪 **Scientific Validation:** Full suite: ROC/AUC, Cross-Validation, Enrichment Factor, Tanimoto Similarity, Learning Curve
- 📊 **Rich Outputs:** Reports, CSVs, and publication-ready plots
- 🔄 **Reproducibility:** Environment checker, logging, config management
- 🖥️ **User-Friendly:** Step-by-step guidance, clean folder structure, error handling

---

## 📁 Project Structure

```
KAST/
├── 🧩 bin/                # Pipeline scripts
│   ├── ⚙️ 1_preparation.py           # Data split
│   ├── 🦾 2_featurization.py         # Fingerprinting
│   ├── 🤖 3_training.py              # Model training
│   ├── 📊 4_0_evaluation_main.py     # Main evaluation
│   ├── 🧪 ... (other evaluation scripts)
│   └── 🔮 5_1_run_prediction.py      # Prediction for new molecules
├── 🧬 data/              # Input data (.smi files)
│   └── 🧪 xx.smi
├── 📦 results/         # Outputs, logs, models, plots
├── 📝 settings.py           # Pipeline settings
├── 🛠️ utils.py            # Shared functions
├── 🖥️ main.py             # Interactive menu
├── 📄 requirements.txt    # Python requirements
├── � CHANGELOG.md        # Version history
├── �📜 LICENSE             # MIT License
├── 📚 README.md           # This file
```

---

## ⚙️ System Requirements

- **Python**: 3.9+ (tested)
- **Conda**: Recommended for RDKit and isolation
- **Main packages**: RDKit, DeepChem, Tensorflow, Scikit-learn, pandas, numpy, tqdm, matplotlib, joblib (for parallel processing)
- **CPU**: Multi-core processor recommended for parallel processing (optional but highly recommended)

---

## 🚀 Parallel Processing Configuration 

K-talysticFlow supports **multi-core parallel processing** for significant performance improvements on large datasets!

### ⚡ Performance Gains
- **5-10x faster** featurization on datasets with 100K+ molecules
- Automatic CPU detection and optimization
- Memory-efficient batch processing

### ⚙️ Configuration

Edit `settings.py` to customize parallel processing:

```python
# Section 12: PARALLEL PROCESSING CONFIGURATIONS
ENABLE_PARALLEL_PROCESSING = True    # Enable/disable parallelism
N_WORKERS = None                     # None = auto-detect (recommended)
PARALLEL_BATCH_SIZE = 100000         # Molecules per batch
PARALLEL_MIN_THRESHOLD = 10000       # Min dataset size for parallelism
```

**Configuration Options:**
- `N_WORKERS = None` → Auto-detect optimal cores (leaves 1 free) ✅ **RECOMMENDED**
- `N_WORKERS = -1` → Use all available cores
- `N_WORKERS = 4` → Use exactly 4 cores
- `N_WORKERS = 1` → Disable parallelism (sequential processing)

**Memory Recommendations:**
- **16GB+ RAM**: `PARALLEL_BATCH_SIZE = 100000` (default)
- **8GB RAM**: `PARALLEL_BATCH_SIZE = 50000`
- **4GB RAM**: `PARALLEL_BATCH_SIZE = 25000`

### 📊 Which Scripts Use Parallel Processing?

| Script | Parallel Support | Typical Speedup |
|--------|------------------|-----------------|
| `2_featurization.py` | ✅ Yes | 5-10x |
| `4_3_tanimoto_similarity.py` | ✅ Yes | 3-5x |
| `4_4_learning_curve.py` | ✅ Yes | 4-8x |
| `5_0_featurize_for_prediction.py` | ✅ Yes | 5-10x |
| Other scripts | Sequential | N/A |

**Note:** Parallel processing automatically activates only when dataset size exceeds `PARALLEL_MIN_THRESHOLD` (default: 10,000 molecules).

---

## 📖 Documentation

**[📘 Complete Documentation](https://lmm.uefs.br/wp-content/uploads/2025/10/KAST-Documentation.html)** — Full user guide with detailed explanations, examples, and troubleshooting.

---

## 🚀 Quick Start

| Task                | Command                        |
|---------------------|-------------------------------|
| Prepare data        | `python bin/1_preparation.py`  |
| Featurize           | `python bin/2_featurization.py`|
| Train model         | `python bin/3_training.py`     |
| Main evaluation     | `python bin/4_0_evaluation_main.py` |
| Predict new         | `python bin/5_1_run_prediction.py`|

Or use the interactive menu:
```bash
python main.py
```
And follow the on-screen instructions!

### 🎛️ Advanced Options Menu

Access testing and configuration tools via **option [8]** in the main menu:

| Option | Feature | Description |
|--------|---------|-------------|
| **[8] → [1]** | Check Environment | Verify all dependencies are installed |
| **[8] → [2]** | Test Parallel Processing | Run 6 automated tests for compatibility |
| **[8] → [3]** | Configure CPU Cores | Adjust parallel processing cores at runtime |

**Configure CPU cores without editing files:**
```bash
# In main menu, select [8] then [3]
[0] Auto-detect (recommended) → uses N-1 cores
[1-N] Use specific number of cores
[-1] Use ALL cores
```

---

## 👩‍🔬 Scientific Workflow

1. **Data Preparation**: Import SMILES, label, and split datasets
2. **Featurization**: Generate ECFP/Morgan fingerprints (QSAR standard)
3. **Training**: DeepChem MultitaskClassifier neural network
4. **Evaluation**: ROC/AUC, accuracy, precision, recall, F1, cross-validation, enrichment factor, Tanimoto similarity, learning curve
5. **Prediction**: Screen new molecules using K-Prediction Score, rank candidates, export CSV

---

## � Usage Examples

### Example 1: Complete Pipeline for New Project

```bash
# Start the interactive menu
python main.py

# Then follow this sequence:
# [1] Prepare and Split Data → Creates train/test sets
# [2] Generate Fingerprints → Featurizes molecules
# [3] Train the Model → Creates neural network
# [4] → [1] Run ALL evaluations → Comprehensive analysis
# [5] → [3] Featurize + Predict → Screen new molecules
```

### Example 2: Running Individual Scripts

```bash
# Prepare data
python bin/1_preparation.py

# Generate fingerprints (with parallel processing)
python bin/2_featurization.py

# Train model
python bin/3_training.py

# Evaluate with cross-validation only
python bin/4_1_cross_validation.py

# Predict new molecules
python bin/5_0_featurize_for_prediction.py
python bin/5_1_run_prediction.py
```

### Example 3: Using Parallel Processing

```python
# Option 1: Configure via interactive menu
python main.py
# Select [8] Advanced Options → [3] Configure CPU Cores
# Choose [0] for auto-detect (recommended)

# Option 2: Edit settings.py directly
# Open settings.py and modify Section 12:
ENABLE_PARALLEL_PROCESSING = True
N_WORKERS = None  # Auto-detect (uses N-1 cores)
PARALLEL_BATCH_SIZE = 100000
PARALLEL_MIN_THRESHOLD = 10000
```

### Example 4: Screening Large Compound Libraries

```bash
# Place your library in data/ folder
# Example: data/zinc_library.smi with 500K molecules

python main.py
# Select [5] → [1] Featurize for Prediction
# Select your library file
# Wait for parallel featurization (~5-10 minutes with parallel)

# Then predict
# Select [5] → [2] Only Predict
# Choose filtering: by cutoff or top-N
# Export results to CSV
```

### Example 5: Testing and Validation

```bash
# Check environment
python main.py
# Select [8] → [1] Check Environment

# Test parallel processing
python main.py
# Select [8] → [2] Test Parallel Processing

# Or run directly:
python bin/check_env.py
python bin/test_parallel_compatibility.py
```

### Example 6: Custom Configuration for Limited RAM

```python
# If you have 8GB RAM, edit settings.py:
ENABLE_PARALLEL_PROCESSING = True
N_WORKERS = 4  # Use 4 cores instead of all
PARALLEL_BATCH_SIZE = 50000  # Smaller batches
PARALLEL_MIN_THRESHOLD = 10000
```

### Example 7: Analyzing Results

```bash
# After pipeline completion, find results in:
results/
├── 01_train_set.csv              # Training molecules
├── 01_test_set.csv               # Test molecules  
├── 4_0_evaluation_report.txt    # Main metrics (AUC, accuracy, etc)
├── 4_1_cross_validation_results.txt
├── 4_2_enrichment_factor_results.txt
├── 05_new_molecule_predictions.csv  # Ranked predictions with K-scores
└── logs/kast_20251010.log       # Daily log file

# Open predictions in Excel/pandas:
import pandas as pd
df = pd.read_csv('results/05_new_molecule_predictions.csv')
top_candidates = df.head(100)  # Top 100 predictions
```

### Example 8: Troubleshooting

```bash
# If script fails, check the log:
cat results/logs/kast_YYYYMMDD.log  # Linux/Mac
type results\logs\kast_YYYYMMDD.log  # Windows

# Run dependency check:
python bin/check_env.py

# Test specific functionality:
python bin/test_parallel_compatibility.py
```

---

## �📈 Outputs

- `results/`: All logs, models, reports, plots (ROC, learning curve, etc)
- `05_new_molecule_predictions.csv`: Ranked predictions for new molecules (sorted by K-Prediction Score)

---

## ⭐ How to Cite

> **K-talysticFlow: A Deep Learning Pipeline for Virtual Screening of Bioactive Compounds**  
> Késsia S. Santos; Manoelito C. Santos Junior. (2025). Laboratory of Molecular Modeling (LMM), State University of Feira de Santana.

---

## 👥 Authors & Acknowledgments

- **Késsia Souza Santos**
    - Email: `kelsouzs.uefs@gmail.com`
    - [GitHub](https://github.com/kelsouzs)
    - [LinkedIn](https://www.linkedin.com/in/kelsouzs)
- **Advisor:** Prof. Dr. Manoelito Coelho dos Santos Junior
- **Lab:** [LMM-UEFS](https://lmm.uefs.br/)
- **Funding:** National Council for Scientific and Technological Development (CNPq)

---

## ❓ FAQ

**Q:** What Python version do I need?  
**A:** Python 3.9+ recommended.

**Q:** How do I install RDKit?  
**A:**  
```bash
conda install -c conda-forge rdkit
```

**Q:** Where do outputs go?  
**A:** All results are in the `results/` folder.

**Q:** How do I run on my own molecules?  
**A:** Place your `.smi` file in `dados/` and use `[5]` menu option.

**Q:** What is the K-Prediction Score?  
**A:** The K-Prediction Score is the proprietary scoring function used by K-talysticFlow to rank molecular activity predictions, with values ranging from 0 to 1 (higher scores indicate higher predicted activity).

**Q:** How do I test if parallel processing is working?  
**A:** Use the menu: `python main.py` → `[8] Advanced Options` → `[2] Test Parallel Processing`

**Q:** Can I change CPU cores without editing settings.py?  
**A:** Yes! Use menu option `[8]` → `[3] Configure CPU Cores` for interactive configuration.

---

## 🧪 For Developers

### Testing Parallel Processing

Validate the parallel processing implementation:

```bash
# Option 1: Via interactive menu
python main.py
# Select [8] → [2] Test Parallel Processing

# Option 2: Direct command
python bin/test_parallel_compatibility.py
```

The test suite runs 6 comprehensive tests:
1. ✅ Import settings.py
2. ✅ Verify configuration variables
3. ✅ Check required imports (joblib, multiprocessing)
4. ✅ Test get_optimal_workers() logic
5. ✅ Validate script compatibility
6. ✅ Test threshold activation logic

### Modifying Parallel Processing

When making changes to parallel processing code:

1. Edit the relevant script(s)
2. Update configuration in `settings.py` if needed
3. Run test suite to validate: `python bin/test_parallel_compatibility.py`
4. Test with real data: Use small dataset first
5. Update documentation (README.md)

---

## 🤝 Contributing

Pull requests welcome! For major changes, open an issue first to discuss.

---

## 📬 Contact

Questions or suggestions?  
Open an [issue](https://github.com/kelsouzs/KAST/issues) or email [lmm@uefs.br](mailto:lmm@uefs.br)


---


