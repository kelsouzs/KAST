# 🛠️ Troubleshooting Guide

Solutions to common issues and errors in K-talysticFlow.

---

## 📋 Table of Contents

1. [Installation Issues](#-installation-issues)
2. [Data Preparation Errors](#-data-preparation-errors)
3. [Featurization Problems](#-featurization-problems)
4. [Training Issues](#-training-issues)
5. [Memory Errors](#-memory-errors)
6. [Parallel Processing Problems](#-parallel-processing-problems)
7. [Prediction Errors](#-prediction-errors)
8. [Performance Issues](#-performance-issues)
9. [General Errors](#-general-errors)

---

## 🚨 Installation Issues

### Issue: `ModuleNotFoundError: No module named 'rdkit'`

**Cause:** RDKit not installed

**Solution:**
```bash
# Using Conda (RECOMMENDED)
conda install -c conda-forge rdkit

# OR using pip (may fail on some systems)
pip install rdkit-pypi
```

---

### Issue: `ImportError: DLL load failed` (Windows)

**Cause:** Missing Visual C++ redistributables

**Solution:**
1. Download [Microsoft Visual C++ Redistributable](https://aka.ms/vs/17/release/vc_redist.x64.exe)
2. Install and restart
3. Reinstall Python packages

---

### Issue: TensorFlow installation fails

**Solution:**
```bash
# Install specific version
pip install tensorflow==2.15.0

# If still fails (CPU only)
pip install tensorflow-cpu
```

---

### Issue: `ImportError: cannot import name 'DeepChem'`

**Solution:**
```bash
pip uninstall deepchem
pip install deepchem==2.7.1
```

---

### Issue: Permission denied when installing

**Solution (Linux/Mac):**
```bash
pip install -r requirements.txt --user
```

**Solution (Windows):**
Run command prompt as Administrator

---

## 📂 Data Preparation Errors

### Issue: `FileNotFoundError: 'actives.smi' not found`

**Cause:** SMILES files not in `data/` folder

**Solution:**
1. Create `data/` folder in project root
2. Place `actives.smi` and `inactives.smi` there
3. Check file names (case-sensitive on Linux)

```bash
# Check structure
ls data/
# Should show:
# actives.smi
# inactives.smi
```

---

### Issue: `ValueError: Invalid SMILES: XYZ`

**Cause:** Malformed SMILES string

**Solution:**
1. Validate SMILES using RDKit:
```python
from rdkit import Chem
mol = Chem.MolFromSmiles('YOUR_SMILES')
if mol is None:
    print("Invalid SMILES")
```

2. Remove invalid entries from `.smi` files
3. Use canonicalized SMILES

---

### Issue: `Error: Insufficient data (< 50 molecules per class)`

**Cause:** Too few compounds

**Solution:**
- **Minimum:** 50 actives + 50 inactives
- Add more compounds to dataset
- Or adjust threshold in `settings.py`:
```python
MIN_MOLECULES_PER_CLASS = 20  # Lower threshold
```

---

### Issue: Train/test split fails

**Cause:** Scaffold splitting issues

**Solution:**
Try random splitting instead:

Edit `bin/1_preparation.py`:
```python
# Find this line:
splitter = dc.splits.ScaffoldSplitter()

# Change to:
splitter = dc.splits.RandomSplitter()
```

---

## 🧬 Featurization Problems

### Issue: `MemoryError during featurization`

**Cause:** Dataset too large for available RAM

**Solution:**

1. **Reduce batch size:**
```python
PARALLEL_BATCH_SIZE = 50000  # Reduce from 100000
```

2. **Disable parallelism:**
```python
ENABLE_PARALLEL_PROCESSING = False
```

3. **Process in chunks manually**

---

### Issue: Featurization extremely slow

**Cause:** Large dataset with sequential processing

**Solution:**

1. **Enable parallel processing:**
```python
ENABLE_PARALLEL_PROCESSING = True
N_WORKERS = None  # Auto mode
```

2. **Increase workers:**
```python
N_WORKERS = 6  # Or your CPU core count - 1
```

---

### Issue: `ValueError: Fingerprint size must be > 0`

**Cause:** Invalid fingerprint configuration

**Solution:**
Check `settings.py`:
```python
FP_SIZE = 2048      # Must be > 0
FP_RADIUS = 3       # Must be > 0
```

---

### Issue: `RDKit WARNING: not removing hydrogen atom`

**Cause:** RDKit warnings (usually harmless)

**Solution:**
These are warnings, not errors. To suppress:
```python
from rdkit import RDLogger
RDLogger.DisableLog('rdApp.*')
```

Already handled in K-talysticFlow code.

---

## 🤖 Training Issues

### Issue: Training stuck at 0% for long time

**Cause:** Very large dataset or slow initialization

**Solution:**
- ✅ **Normal behavior** for first epoch (TensorFlow initialization)
- Wait 2-5 minutes before concluding it's stuck
- Check CPU/GPU usage in Task Manager

---

### Issue: `ValueError: No training data found`

**Cause:** Featurization step not completed

**Solution:**
1. Run featurization first:
```bash
python main.py
# Select [2] Featurize Molecules
```

2. Check `results/featurized_datasets/train/` exists

---

### Issue: Training loss not decreasing

**Causes & Solutions:**

**1. Learning rate too high:**
```python
'learning_rate': 0.0001  # Reduce from 0.001
```

**2. Random labels (data quality issue):**
- Verify label correctness
- Check if actives and inactives are truly different

**3. Model too simple:**
```python
'layer_sizes': [2048, 1024, 512]  # Increase complexity
```

---

### Issue: `CUDA out of memory` (GPU)

**Solution:**

1. **Reduce batch size** (TensorFlow internal):
   - Edit DeepChem model parameters (advanced)

2. **Switch to CPU:**
```bash
export CUDA_VISIBLE_DEVICES=""  # Linux/Mac
set CUDA_VISIBLE_DEVICES=  # Windows
```

3. **Use smaller model:**
```python
'layer_sizes': [512, 256]
```

---

### Issue: Training finishes but no model file

**Cause:** Error during checkpoint saving

**Solution:**
1. Check `results/trained_model/` folder exists
2. Check write permissions
3. Review `results/03_training_log.txt` for errors

---

## 💾 Memory Errors

### Issue: `MemoryError: Unable to allocate array`

**Cause:** Insufficient RAM

**Solutions:**

**1. Reduce batch size:**
```python
PARALLEL_BATCH_SIZE = 25000  # Reduce significantly
```

**2. Disable parallelism:**
```python
ENABLE_PARALLEL_PROCESSING = False
```

**3. Reduce workers:**
```python
N_WORKERS = 2  # Use fewer cores
```

**4. Close background programs**

**5. Use swap/pagefile** (slower but works)

**6. Upgrade RAM** (8GB → 16GB)

---

### Issue: Python process killed suddenly

**Cause:** Out-of-memory (OOM) killer (Linux)

**Solution:**
```bash
# Check logs
dmesg | grep -i "killed process"

# If OOM killer was triggered:
# 1. Reduce memory usage (see above)
# 2. Add swap space
sudo fallocate -l 8G /swapfile
sudo chmod 600 /swapfile
sudo mkswap /swapfile
sudo swapon /swapfile
```

---

## ⚡ Parallel Processing Problems

### Issue: No speedup from parallelism

**Causes:**

**1. Dataset too small** (< 10K):
```python
PARALLEL_MIN_THRESHOLD = 1000  # Lower threshold
```

**2. Only 1-2 cores available:**
```bash
# Check cores
python -c "import os; print(os.cpu_count())"
```

**3. I/O bottleneck** (slow disk):
- Use SSD instead of HDD
- Can't fix with parallelism

**4. Script doesn't support parallelism:**
- See [Parallel Processing Guide](Parallel-Processing) for supported scripts

---

### Issue: `joblib` errors

**Error:**
```
AttributeError: module 'joblib' has no attribute 'Parallel'
```

**Solution:**
```bash
pip uninstall joblib
pip install joblib==1.3.0
```

---

### Issue: Parallel test suite fails

**Solution:**
```bash
# Run test suite
python bin/test_parallel_compatibility.py

# If Test 1 fails: joblib issue
pip install --upgrade joblib

# If Test 2 fails: reduce workers
N_WORKERS = 2

# If Test 3 fails: memory issue
PARALLEL_BATCH_SIZE = 25000
```

---

### Issue: Slower with parallelism enabled

**Cause:** Overhead exceeds benefit (small dataset)

**Solution:**
Disable for small datasets:
```python
PARALLEL_MIN_THRESHOLD = 50000  # Only parallelize large datasets
```

---

## 🔮 Prediction Errors

### Issue: `FileNotFoundError: Model checkpoint not found`

**Cause:** Model not trained yet

**Solution:**
1. Train model first:
```bash
python main.py
# Select [3] Train Model
```

2. Verify `results/trained_model/checkpoint1.pt` exists

---

### Issue: Predictions all the same value

**Causes:**

**1. Poor model (AUC < 0.6):**
- Retrain with better data
- Check evaluation metrics

**2. Invalid input data:**
- Verify SMILES are correct
- Check featurization completed

**3. Wrong model loaded:**
- Check `training_metadata.json` matches current settings

---

### Issue: `ValueError: Feature mismatch`

**Cause:** Prediction fingerprints don't match training

**Solution:**
Ensure same parameters for prediction:
```python
# settings.py must match training config
FP_SIZE = 2048      # Same as training
FP_RADIUS = 3       # Same as training
```

**If changed after training:**
Re-featurize and re-predict:
```bash
python main.py
# [5] Featurize for Prediction
# [6] Run Prediction
```

---

### Issue: Predictions take too long

**Solution:**

1. **Enable parallelism:**
```python
ENABLE_PARALLEL_PROCESSING = True
```

2. **Increase workers:**
```python
N_WORKERS = -1
```

3. **Check dataset size:**
```bash
wc -l data/zinc_library.smi
# If > 100K, expect 10-30 min even with parallelism
```

---

## 🚀 Performance Issues

### Issue: Pipeline very slow overall

**Checklist:**

✅ **Parallel processing enabled?**
```python
ENABLE_PARALLEL_PROCESSING = True
```

✅ **Using SSD or HDD?**
- HDD: 5-10x slower I/O
- Solution: Use SSD

✅ **Sufficient RAM?**
- Check usage in Task Manager
- Close background programs

✅ **CPU usage low?**
- May indicate I/O bottleneck
- Can't fix with CPU optimization

✅ **Antivirus scanning files?**
- Exclude K-talysticFlow folder

---

### Issue: Specific script very slow

**Script-specific solutions:**

**1_preparation.py (slow):**
- Normal for large datasets
- Scaffold splitting is intensive
- Expected: 1-5 min for 10K molecules

**2_featurization.py (slow):**
- Enable parallelism (5-10x faster)
- Use more workers

**3_create_training.py (slow):**
- Reduce epochs
- Use GPU
- Simplify model

**4_4_learning_curve.py (slow):**
- Most time-consuming evaluation
- Enable parallelism
- Reduce training sizes in script (advanced)

---

## 🔧 General Errors

### Issue: `ImportError: cannot import name 'MultitaskClassifier'`

**Cause:** DeepChem version mismatch

**Solution:**
```bash
pip install deepchem==2.7.1
```

---

### Issue: Control panel menu not displaying correctly

**Cause:** Terminal encoding issues

**Solution:**

**Windows:**
```powershell
chcp 65001  # Set UTF-8 encoding
```

**Linux/Mac:**
```bash
export LANG=en_US.UTF-8
```

---

### Issue: Logs not generated

**Cause:** Logging directory doesn't exist

**Solution:**
```bash
mkdir -p results/logs
```

Or let scripts create automatically (check write permissions).

---

### Issue: `PermissionError: [Errno 13]`

**Cause:** Insufficient write permissions

**Solution:**

**Linux/Mac:**
```bash
chmod -R 755 results/
```

**Windows:**
Right-click folder → Properties → Security → Edit → Grant Full Control

---

### Issue: Conflicting package versions

**Error:**
```
ERROR: pip's dependency resolver does not currently take into account all the packages that are installed.
```

**Solution:**
```bash
# Create fresh environment
conda create -n kast_clean python=3.10 -y
conda activate kast_clean
conda install -c conda-forge rdkit -y
pip install -r requirements.txt
```

---

### Issue: Script exits without error message

**Solution:**
Check log files:
```bash
# Main log
cat results/logs/kast_YYYYMMDD.log

# Script-specific logs
cat results/02_featurization_log.txt
cat results/03_training_log.txt
# etc.
```

---

### Issue: Results folder messy/corrupted

**Solution:**
```bash
# Backup current results
mv results results_backup_$(date +%Y%m%d)

# Create fresh results folder
mkdir results
mkdir results/logs
mkdir results/featurized_datasets
mkdir results/trained_model

# Re-run pipeline
python main.py
```

---

## 🆘 Getting More Help

### Step 1: Check Logs

```bash
# Recent errors
tail -n 50 results/logs/kast_*.log

# Specific script log
cat results/03_training_log.txt
```

---

### Step 2: Enable Debugging

Edit `settings.py`:
```python
DEBUG = True
VERBOSE = True
```

---

### Step 3: Run Dependency Checker

```bash
python main.py
# [8] Advanced Options → [1] Check Dependencies
```

---

### Step 4: Test Parallel Processing

```bash
python main.py
# [8] Advanced Options → [2] Test Parallel Processing
```

---

### Step 5: Minimal Reproducible Example

Create small test dataset:
```bash
# Create minimal data
head -n 100 data/actives.smi > data/actives_test.smi
head -n 100 data/inactives.smi > data/inactives_test.smi

# Update settings.py temporarily
ACTIVE_SMILES_FILE = 'data/actives_test.smi'
INACTIVE_SMILES_FILE = 'data/inactives_test.smi'

# Run pipeline
python main.py
```

---

### Step 6: Report Issue

If problem persists, open a GitHub issue with:

1. **Error message** (full traceback)
2. **Log files** (attach relevant logs)
3. **System info:**
```bash
python --version
pip list | grep -E 'deepchem|rdkit|tensorflow'
uname -a  # Linux/Mac
systeminfo  # Windows
```
4. **Steps to reproduce**

**GitHub Issues:** https://github.com/kelsouzs/kast/issues

---

## 📚 Related Resources

- [FAQ](FAQ) - Common questions
- [Installation](Installation) - Setup guide
- [User Manual](User-Manual) - Usage instructions
- [Parallel Processing](Parallel-Processing) - Performance optimization

---

<div align="center">
<p>← <a href="Home">Back to Wiki Home</a></p>
<p><strong>Still stuck?</strong> Contact: kelsouzs.uefs@gmail.com</p>
</div>
