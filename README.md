# 🚀 K-talysticFlow (KAST) — Deep Learning Molecular Screening Pipeline

![Python](https://img.shields.io/badge/Python-3.9%2B-blue.svg)
![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)
![Version](https://img.shields.io/badge/Version-1.0.0-green.svg)
![Status](https://img.shields.io/badge/Status-Stable-brightgreen.svg)
[![Documentation](https://img.shields.io/badge/Docs-here-8A2BE2)](https://kast.readthedocs.io/en/latest/)
[![LinkedIn](https://img.shields.io/badge/-LinkedIn-blue?style=flat-square&logo=linkedin&logoColor=white)](https://www.linkedin.com/in/kelsouzs)
[![GitHub](https://img.shields.io/badge/-GitHub-181717?style=flat-square&logo=github&logoColor=white)](https://github.com/kelsouzs)

```
  __  __    _     ____  _____ 
  | |/ /   / \   / ___||_   _|
  | ' /   / _ \  \___ \  | |  
  | . \  / ___ \  ___) | | |  
  |_|\_\/_/   \_\|____/  |_|  K-talystic Automated Screening Taskflow
```

---

## 👨‍🔬 What is K-talysticFlow?

K-talysticFlow (KAST) is an open-source virtual screening pipeline and computational chemistry tool designed to accelerate drug discovery. Built with Python, DeepChem, and TensorFlow, it automates the evaluation of antituberculosis agents and other molecular compounds using advanced Deep Learning models.

**Version:** 1.0.0 (Stable)  
**Developed at:** [Laboratory of Molecular Modeling (LMM-UEFS)](https://lmm.uefs.br/) — Funded by CNPq

---

## ✨ Features

- ⚡ **Interactive CLI Menu** — Easy step-by-step workflow
- 🚀 **Parallel Processing** — 5-10x faster with multi-core support
- 🧠 **Deep Learning** — DeepChem/TensorFlow neural networks
- 🎯 **K-Prediction Score** — Proprietary ranking for predictions
- 🧪 **Validation Suite** — ROC/AUC, Cross-Validation, Enrichment Factor, Similarity
- 🖥️ **One-Click Setup** — Automated environment creation for Windows and Linux

---

## 📁 Folder Structure

```
KAST/
├── bin/                    # Pipeline scripts (1-5)
├── data/                   # Input SMILES files (.smi)
├── results/                # Outputs (logs, models, reports)
├── settings.py             # Configuration & parallel processing
├── main.py                 # Interactive menu
├── setup.exe               # Windows automated setup
├── setup.sh                # Linux setup script
└── README.md               # This file
```

---

## ⚙️ Requirements

- **Python:** 3.9+
- **Conda:** Required for environment setup
- **Main packages:** RDKit, DeepChem, TensorFlow, scikit-learn, pandas, numpy, joblib
- **RAM:** 4GB+ (8GB+ recommended for parallel processing)

---

## 🚀 Installation

### 📦 Windows Users (Easiest Option!)

**Using `setup.exe` (Fully Automated)**

```
1. Download setup.exe from releases
2. Double-click setup.exe in the KAST folder
3. The installer will:
   ✅ Find your Conda installation automatically
   ✅ Create environment 'ktalysticflow' (or update if exists)
   ✅ Install all dependencies
   ✅ Create desktop + Start Menu shortcuts
   ✅ Generate run_kast.bat launcher
4. Click the desktop shortcut to launch KAST!
```

**What does `setup.exe` do?**
- **Locates Conda**: Searches standard installation paths (Anaconda3, miniconda3, mambaforge, Program Files, registry)
- **Environment Setup**: Creates or updates the `ktalysticflow` conda environment from `environment.yml`
- **Creates Launcher**: Generates `run_kast.bat` that automatically activates conda and launches KAST
- **Creates Shortcuts**: Desktop and Start Menu shortcuts that run KAST with one click
- **No Terminal Needed**: Runs KAST directly without opening Anaconda Prompt or PowerShell

**Quick Launch After Setup:**
- Click desktop shortcut "K-talysticFlow 1.0.0"
- Or: Double-click `run_kast.bat` in the folder
- Or: Start Menu → K-talysticFlow 1.0.0

---

### 🐧 Linux Users

**Using `setup.sh` (Fully Automated)**

```bash
# Make script executable
chmod +x setup.sh

# Run setup
./setup.sh
```

**What does `setup.sh` do?**
- **Checks Conda**: Verifies Conda is installed
- **Creates Environment**: Builds `ktalysticflow` environment from `environment.yml`
- **Installs Dependencies**: All required packages automatically

**After Setup:**
```bash
# Activate environment
conda activate ktalysticflow

# Launch KAST
python main.py
```

---

### 🛠️ Manual Setup (All Platforms)

```bash
# Create environment from file
conda env create -f environment.yml -y

# Activate
conda activate ktalysticflow

# Verify installation (optional)
python bin/check_env.py

# Launch KAST
python main.py
```

---

## ⚡ Parallel Processing

K-talysticFlow supports **multi-core parallel processing** for 5-10x faster performance!

### Quick Configuration

Edit `settings.py` (Section 12):

```python
ENABLE_PARALLEL_PROCESSING = True    # On/Off
N_WORKERS = None                     # None = auto-detect (RECOMMENDED)
PARALLEL_BATCH_SIZE = 100000         # Molecules per batch
PARALLEL_MIN_THRESHOLD = 10000       # Min dataset size for parallel
```

**Worker Options:**
- `N_WORKERS = None` → Auto-detect optimal cores ✅ **RECOMMENDED**
- `N_WORKERS = -1` → Use all cores
- `N_WORKERS = 4` → Use exactly 4 cores
- `N_WORKERS = 1` → Disable parallel (sequential only)

### Or Configure Interactively

```
python main.py
→ [8] Advanced Options
→ [3] Configure CPU Cores
→ Choose auto or specific number
```

### Scripts with Parallel Support

| Script | Speedup |
|--------|---------|
| `2_featurization.py` | 5-10x |
| `4_3_tanimoto_similarity.py` | 3-5x |
| `4_4_learning_curve.py` | 4-8x |
| `5_0_featurize_for_prediction.py` | 5-10x |

**Automatic activation:** Parallel mode only engages when dataset > 10,000 molecules.

---

## 🎯 Quick Start

```bash
# Launch interactive menu
python main.py
```

**Menu Options:**
1. Prepare & Split Data
2. Generate Fingerprints
3. Train Model
4. Evaluate (ROC/AUC, Cross-Val, etc.)
5. Predict New Molecules
6. View Results
7. Check Data Status
8. Advanced Tools (env check, parallel test, config)

**Or run individual scripts:**
```bash
python bin/1_preparation.py          # Prepare data
python bin/2_featurization.py        # Featurize molecules
python bin/3_create_training.py      # Train model
python bin/4_0_evaluation_main.py    # Evaluate
python bin/5_1_run_prediction.py     # Predict
```

---

## 📊 Outputs

All results saved to `results/` folder:

```
results/
├── 01_train_set.csv                 # Training data
├── 01_test_set.csv                  # Test data
├── 4_0_evaluation_report.txt        # Main metrics (AUC, accuracy)
├── 4_1_cross_validation_results.txt # Cross-validation scores
├── 4_2_enrichment_factor.txt        # Enrichment analysis
├── 4_3_tanimoto_similarity.txt      # Similarity analysis
├── 4_4_learning_curve.txt           # Model learning progression
├── 05_new_molecule_predictions.csv  # Predicted molecules (K-Score ranked)
├── plots/                           # ROC, Learning Curves (PNG/PDF)
└── logs/                            # kast_YYYYMMDD.log
```

---

## ❓ FAQ

**Q: Which setup should I use?**  
A: Windows → `setup.exe` (one-click). Linux → `./setup.sh`. Both handle everything automatically.

**Q: Do I need to type `conda activate` every time?**  
A: No! Shortcuts and `run_kast.bat` handle it automatically.

**Q: How much faster is parallel processing?**  
A: 5-10x faster for large datasets (100K+ molecules). Automatic on/off based on dataset size.

**Q: Where are my results?**  
A: All outputs in `results/` folder (logs, models, plots, CSVs).

**Q: My setup failed - what do I do?**  
A: Run `python bin/check_env.py` to diagnose. Check the logs in `results/logs/`.

**Q: Can I change settings after setup?**  
A: Yes! Edit `settings.py` anytime or use menu option `[8]→[3]` for interactive config.

---

## 🔗 Links

- 📘 **[Full Documentation](https://kast.readthedocs.io/en/latest/)**
- 👤 **[LinkedIn](https://www.linkedin.com/in/kelsouzs)**
- 💻 **[GitHub Profile](https://github.com/kelsouzs)**
- 🏫 **[LMM Laboratory](https://lmm.uefs.br/)**

---

## 👥 Authors

- **Késsia Souza Santos** — [GitHub](https://github.com/kelsouzs) | [LinkedIn](https://www.linkedin.com/in/kelsouzs)
- **Advisor:** Prof. Dr. Manoelito C. Santos Junior
- **Lab:** Laboratory of Molecular Modeling (LMM-UEFS)
- **Funding:** CNPq (Brazilian National Research Council)

---

## 📜 License

MIT License — See [LICENSE](LICENSE) file

---

**Questions or bugs?** Open an [issue](https://github.com/kelsouzs/KAST/issues) or email [kelsouzs.uefs@gmail.com](mailto:kelsouzs.uefs@gmail.com)
