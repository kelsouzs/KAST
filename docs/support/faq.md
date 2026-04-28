# FAQ — Frequently Asked Questions

Common questions about KAST answered.

---

## Installation & Setup

### Q: Do I need to install Anaconda?
**A:** Yes. Anaconda manages Python packages and environments. Download from [anaconda.com](https://www.anaconda.com/download).

### Q: Can I use Miniconda instead of Anaconda?
**A:** Yes! Both work. Miniconda is lighter. Both are supported on Windows and Linux.

### Q: setup.exe doesn't work on my computer
**A:** Try:
1. Right-click → Properties → "Unblock"
2. Ensure `setup.exe` is in same folder as `environment.yml`
3. Run as Administrator (right-click → Run as admin)
4. If still fails, use manual setup: `conda env create -f environment.yml -y`

### Q: What's the difference between setup.exe and setup.sh?
**A:** Both do the same thing:
- `setup.exe` → Windows automated installer
- `setup.sh` → Linux automated installer
Both create environments, install dependencies, and create shortcuts.

### Q: Does KAST work on Mac?
**A:** Not officially tested. Should work with Linux setup (`setup.sh`), but not guaranteed. File an issue if you try!

---

## Data & Preparation

### Q: What format should my SMILES files be?
**A:** Plain text, one SMILES per line:

```text
SMILES [space or tab] optional_name
CC(C)Cc1ccc(cc1)C(C)C(O)=O  ibuprofen
```

### Q: Does KAST normalize my data automatically?
**A:** Yes. During the `1_preparation.py` step, KAST automatically performs salt removal, charge neutralization, and canonicalization using RDKit. If you want to inspect how normalization affects your dataset, you can run `python bin/check_normalization.py` to generate a before-and-after report.

### Q: Can I use SMILES with salts or mixtures?
**A:** While KAST performs automatic normalization, it is still recommended to use canonical SMILES of pure compounds whenever possible. If your dataset contains salts, mixtures, or charged species, KAST will attempt to standardize them, but reviewing the normalization report is recommended.

### Q: Minimum number of molecules needed?
**A:** At least 50 active and 50 inactive compounds. For more reliable model performance, 100-1000+ molecules per class are recommended.

### Q: What's the maximum dataset size?
**A:** KAST works well with datasets containing 100K+ molecules. For very large datasets, enabling [Parallel Processing](../user-guide/parallel-processing.md) is strongly recommended.

### Q: Can I have imbalanced data (e.g., 10:1 inactive:active)?
**A:** Yes. KAST supports imbalanced datasets and is designed to handle this scenario during training, but model performance should still be evaluated carefully with metrics such as ROC-AUC, enrichment factor, and cross-validation.

### Q: How do I prepare data from a database?
**A:** Export the compounds as SMILES, remove obvious duplicates and invalid structures when possible, and place the files in the `data/` folder. KAST will then validate and standardize the molecules automatically during preprocessing. See [Data Preparation](../user-guide/data-preparation.md).

---

## Running KAST

### Q: How do I launch KAST?
**A:** 
- **Windows:** Click desktop shortcut or double-click `run_kast.bat`
- **Linux:** Click app menu shortcut or run `conda activate ktalysticflow && python main.py`

### Q: Can I run steps individually?
**A:** Yes! Choose any step from the menu. No need to run 1→2→3→4 in order (though recommended).

### Q: Can I skip Step 4 (Evaluation) and go straight to prediction?
**A:** No. You must train a model (Step 3) before you can predict (Steps 5-6).

### Q: How long does each step take?
**A:** Depends on dataset size:
- Step 1 (Prepare): seconds to minutes
- Step 2 (Featurize): minutes to hours (5-10x faster with parallel)
- Step 3 (Train): 5-30 minutes
- Step 4 (Evaluate): 1-10 minutes
- Steps 5-6 (Predict): seconds to minutes

### Q: Can I run KAST from the terminal instead of the menu?
**A:** Advanced users can run scripts directly:
```bash
python bin/1_preparation.py
python bin/2_featurization.py
python bin/3_create_training.py
```

---

## Performance & Optimization

### Q: How do I speed up KAST?
**A:** Enable [Parallel Processing](../user-guide/parallel-processing.md):
- `ENABLE_PARALLEL_PROCESSING = True`
- `N_WORKERS = None` (auto-detect)
- Gets 5-10x faster on large datasets!

### Q: My computer runs out of memory during featurization
**A:** Reduce `PARALLEL_BATCH_SIZE` in `settings.py`:
- 8GB RAM: Use 50,000
- 4GB RAM: Use 25,000

### Q: Should I use all CPU cores?
**A:** Usually no. Leave 1-2 free for OS:
- `N_WORKERS = None` (recommended) uses (total cores - 1)
- `N_WORKERS = -1` uses all cores (not recommended)

---

## Results & Interpretation

### Q: What do the K-Prediction scores mean?
**A:** The K-Prediction Score is the model's estimated probability for the active class (range 0 to 1). It is best used for ranking compounds by priority:
- `0.9 - 1.0` → Very likely active (high confidence)
- `0.7 - 0.9` → Likely active (medium-high confidence)
- `0.5 - 0.7` → Possibly active (medium confidence)
- `0.0 - 0.5` → Likely inactive

### Q: My model's test AUC is much higher than cross-validation
**A:** This suggests potential overfitting. Check:
- **Data Quality:** Are there duplicates or mislabeled compounds?
- **Dataset Size:** Is the training set large enough?
- **Analysis:** Use the [Learning Curve](../user-guide/step-by-step.md) evaluation to visualize how your model learns and identify if it needs more data.

### Q: Cross-validation AUC varies a lot between folds
**A:** This indicates the model may be unstable or the dataset is heterogeneous:
- **Data Volume:** Try increasing the size of your training set.
- **Consistency:** Verify if there are significant outliers or anomalous clusters in the data.
- **Feature Quality:** Check for structural biases in the scaffold split.

### Q: How do I know if predictions are trustworthy?
**A:** Trustworthiness is determined by evaluating the pipeline's overall rigor:
- **Model Metrics:** Consistent AUC > 0.80 is generally a strong indicator.
- **Enrichment:** High enrichment factors (e.g., > 2x at 10%) indicate the model is correctly ranking the top compounds.
- **Stability:** Low variance in cross-validation results across folds.

---

## Troubleshooting

### Q: "ImportError: No module named 'tensorflow'"
**A:** Dependencies not installed. Re-run:
```bash
# Windows
setup.exe

# Linux
./setup.sh

# Or manual
conda env create -f environment.yml -y
```

### Q: "SMILES validation failed"
**A:** Some molecules have invalid SMILES. Check file format in `data/` folder.

### Q: Predictions take too long
**A:** Enable parallel processing (see [Parallel Processing](../user-guide/parallel-processing.md)).

### Q: Where are my results?
**A:** All in `results/` folder. Check:
- `results/4_0_evaluation_report.txt` → metrics
- `results/05_new_molecule_predictions.csv` → predictions
- `results/plots/` → visualizations
- `results/logs/` → detailed logs

---

## Windows vs Linux

### Q: I have Windows 11, will KAST work?
**A:** Yes! Tested and working on Windows 11 with Anaconda.

### Q: Can I run KAST on Ubuntu?
**A:** Yes! Tested on Ubuntu 20.04 LTS and newer.

### Q: Same training data, different results on Windows vs Linux?
**A:** Shouldn't happen. KAST uses fixed seeds for reproducibility. If it does, check [settings.py](../support/configuration.md).

---

## Getting Help

### Still have questions?
→ Check **[Troubleshooting](troubleshooting.md)** guide  
→ Open issue on [GitHub](https://github.com/kelsouzs/KAST/issues)  
→ Email: lmm@uefs.br

