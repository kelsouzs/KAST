# Configuration

Advanced settings explained.

---

## Where to Configure

Edit `settings.py` in the main KAST folder:

```python
# settings.py

# Section 1: Reproducibility
RANDOM_SEED = 42
NUMPY_SEED = 42
TENSORFLOW_SEED = 42
PYTHONHASHSEED = 0  # Environment variable

# Section 2: Data Processing
TRAIN_TEST_SPLIT = 0.80          # 80% train, 20% test
VALIDATION_SPLIT = 0.20          # Of training data

# ... (more sections)

# Section 12: PARALLEL PROCESSING
ENABLE_PARALLEL_PROCESSING = True
N_WORKERS = None
PARALLEL_BATCH_SIZE = 100000
PARALLEL_MIN_THRESHOLD = 10000
```

---

## All Settings Explained

### Reproducibility

```python
RANDOM_SEED = 42
NUMPY_SEED = 42
TENSORFLOW_SEED = 42
PYTHONHASHSEED = 0
```

**What it does:** Ensures identical results on every run

**Change if:** You want different random initialization (not recommended for reproducibility)

---

### Data Processing

```python
TRAIN_TEST_SPLIT = 0.80  # 80/20 split
VALIDATION_SPLIT = 0.20  # 20% of training used for validation
```

**What it does:** Controls how data is divided

---

### Model Architecture

```python
FINGERPRINT_RADIUS = 2             # Morgan fingerprint radius
FINGERPRINT_LENGTH = 2048          # Feature vector size
```

**What it does:** Molecular fingerprint generation

**Notes:**
- Radius 2 = standard (ECFP4)
- 2048 bits = standard feature size
- Rarely need to change these

---

### Neural Network

```python
# In settings.py
MODEL_PARAMS = {
    'n_tasks': 1,
    'layer_sizes': [1000, 500],
    'dropouts': 0.25,
    'learning_rate': 0.001,
    'mode': 'classification',
    'nb_epoch': 50
}

# Training configurations
NB_EPOCH_TRAIN = 50    # Epochs for final model
NB_EPOCH_CV = 30       # Epochs for cross-validation
NB_EPOCH_LC = 20       # Epochs for learning curve
CLASSIFICATION_THRESHOLD = 0.5
```

**What each parameter does:**

#### `layer_sizes: [1000, 500]`
Defines the neural network **architecture** — number of neurons in each hidden layer.

**Selection guideline** (based on dataset size):
- **Small dataset** (<1K molecules): `[256, 128]` — fewer parameters, less overfitting risk
- **Medium dataset** (1K-10K molecules): `[1000, 500]` ← **recommended**
- **Large dataset** (>10K molecules): `[2048, 1024, 512]` — can learn more complex patterns

**Decision rule:** Use a **pyramid shape** (each layer smaller than previous) to force the network to compress information into abstract representations.

**Adjust if:**
- **Underfitting** (low AUC on both train & test): **increase** layer sizes
- **Overfitting** (high train AUC, low test AUC): **decrease** layer sizes

#### `dropouts: 0.25`
**Random neuron deactivation** during training (25% of neurons turn off each step). Prevents overfitting by forcing the network to learn redundant representations.

**Recommended values:**
- `0.1` → minimal regularization (large datasets >50K molecules)
- `0.25` → moderate ← **recommended**
- `0.5` → strong regularization (severe overfitting observed)

**Adjust if:** `AUC_train - AUC_test > 0.15` (overfitting) → increase to 0.5

#### `learning_rate: 0.001`
Controls the "step size" the Adam optimizer takes when adjusting weights. Too high = overshoots. Too low = learns slowly.

**Recommended values:**
- `0.01` → large steps, fast but unstable
- `0.001` → default Adam ← **recommended**
- `0.0001` → small steps, stable but slow

**Adjust if:**
- Loss oscillates wildly: **decrease** to 0.0001
- Loss plateau too early: **increase** to 0.01 (with caution)

#### `nb_epoch` (different for different uses)
Number of complete passes over the dataset during training.

**Why different values?**
- `NB_EPOCH_TRAIN = 50`: Final model needs full training
- `NB_EPOCH_CV = 30`: Cross-validation runs 5× anyway
- `NB_EPOCH_LC = 20`: Learning curve runs many times, so faster iterations

**Decide by:** Run learning curve (option 4→5 in main menu)
- If loss still decreasing at epoch 50 → increase to 100
- If loss plateaued by epoch 30 → reduce to 35

#### `CLASSIFICATION_THRESHOLD: 0.5`
**Decision boundary** for binary classification. Prediction probability ≥ threshold = "active".

**Recommended values by use case:**
- `0.5` → balanced (equal importance to false positives/negatives)
- `0.3` → **sensitive mode** (capture more actives, accept more false positives)
  - Use when: "Don't miss any active compound"
  - Context: Initial virtual screening, discovery phase
- `0.7` → **specific mode** (only predict when very confident)
  - Use when: "Can't afford false positives"
  - Context: Final selection for synthesis/testing

---

**Quick Decision Table:**

| Observable symptom | Adjustment needed |
|:---|:---|
| High train AUC, low test AUC (overfitting) | Increase dropout, decrease layer_sizes or epochs |
| Low train AUC, low test AUC (underfitting) | Increase layer_sizes or epochs, decrease dropout |
| Loss doesn't converge smoothly | Decrease learning_rate to 0.0001 |
| Loss converges before epoch 50 | Reduce nb_epoch to avoid wasting time |
| Many false negatives (missing actives) | Decrease CLASSIFICATION_THRESHOLD to 0.3 |
| Many false positives (wrong predictions) | Increase CLASSIFICATION_THRESHOLD to 0.7 |

---

**Safe defaults:**
- ✅ Start with values above — they're optimized for 1K-10K molecular datasets
- ✅ Only adjust if model performance is poor
- ✅ Always validate changes with learning curve analysis

---

### Parallel Processing

```python
ENABLE_PARALLEL_PROCESSING = True
N_WORKERS = None                    # Auto-detect
PARALLEL_BATCH_SIZE = 100000        # Molecules per batch
PARALLEL_MIN_THRESHOLD = 10000      # Min size to activate
```

**What it does:** Multi-core processing configuration

**See [Parallel Processing](../user-guide/parallel-processing.md) guide** for detailed info

**Quick settings:**

| RAM | N_WORKERS | BATCH_SIZE |
|-----|-----------|-----------|
| 4GB | 2 | 25,000 |
| 8GB | 4 | 50,000 |
| 16GB | 6 | 100,000 |
| 32GB+ | -1 | 200,000 |

---

### Output

```python
OUTPUT_FOLDER = 'results'
PLOT_DPI = 300                      # Plot resolution
SAVE_PLOTS = True
SAVE_CSV = True
VERBOSE_LOGGING = True
```

**What it does:** Output file organization and detail level

**Common adjustments:**
- `PLOT_DPI = 300` → Publication quality
- `PLOT_DPI = 72` → Web quality (smaller files)
- `VERBOSE_LOGGING = False` → Less log file output

---

## Configure Interactively

Without editing `settings.py`, use the menu:

```bash
python main.py
→ [8] Advanced Options
→ [3] Configure CPU Cores
```

**Interactive options:**
- [0] Auto-detect workers
- [1-N] Specific number of workers
- [-1] Use all cores

---

## Reset to Defaults

If you mess up settings, revert:

```bash
# Backup your changes
cp settings.py settings.py.backup

# Download original from GitHub or reinstall
conda env create -f environment.yml -y --force-reinstall
```

---

## Environment Variables

Some settings can be overridden via environment:

```bash
# Linux/Mac
export KAST_N_WORKERS=4
export KAST_BATCH_SIZE=50000
python bin/2_featurization.py

# Windows PowerShell
$env:KAST_N_WORKERS=4
$env:KAST_BATCH_SIZE=50000
python bin\2_featurization.py

# Windows Command Prompt
set KAST_N_WORKERS=4
set KAST_BATCH_SIZE=50000
python bin\2_featurization.py
```

---

## Common Configurations

### Scenario 1: Small Dataset (< 1K molecules)
```python
TRAIN_SET_FRACTION = 0.8        # 80/20 split
MODEL_PARAMS = {
    'layer_sizes': [256, 128],
    'dropouts': 0.3,
    'learning_rate': 0.001,
    'nb_epoch': 50
}
CLASSIFICATION_THRESHOLD = 0.4  # More sensitive
```

### Scenario 2: Standard/Medium Dataset (1K-10K molecules) ← **RECOMMENDED**
```python
TRAIN_SET_FRACTION = 0.7        # 70/30 split
MODEL_PARAMS = {
    'layer_sizes': [1000, 500],
    'dropouts': 0.25,
    'learning_rate': 0.001,
    'nb_epoch': 50
}
CLASSIFICATION_THRESHOLD = 0.5  # Balanced
```

### Scenario 3: Large Dataset (> 10K molecules)
```python
TRAIN_SET_FRACTION = 0.8        # 80/20 split
MODEL_PARAMS = {
    'layer_sizes': [2048, 1024, 512],
    'dropouts': 0.2,
    'learning_rate': 0.001,
    'nb_epoch': 50
}
CLASSIFICATION_THRESHOLD = 0.5
```

### Scenario 4: Overfitting Detected
```python
# Observable: High train AUC, low test AUC
MODEL_PARAMS = {
    'layer_sizes': [512, 256],      # Reduce complexity
    'dropouts': 0.5,                # Increase regularization
    'learning_rate': 0.001,
    'nb_epoch': 30                  # Shorter training
}
TRAIN_SET_FRACTION = 0.9            # More test data
```

### Scenario 5: Underfitting Detected  
```python
# Observable: Low AUC on both train and test
MODEL_PARAMS = {
    'layer_sizes': [2048, 1024],    # More capacity
    'dropouts': 0.15,               # Less regularization
    'learning_rate': 0.001,
    'nb_epoch': 100                 # Longer training
}
TRAIN_SET_FRACTION = 0.8            # Less test data
```

---

## How to Choose Hyperparameters

### Step 1: Know Your Dataset Size
```
Molecules count: ___ (from [1] Prepare Data output)

IF < 1K        → Use Scenario 1 (small dataset)
IF 1K - 10K    → Use Scenario 2 (medium) ← most common
IF > 10K       → Use Scenario 3 (large)
```

### Step 2: Train and Evaluate
```bash
python main.py
→ [1] Prepare Data
→ [2] Generate Fingerprints  
→ [3] Create and Train Model
→ [4] Model Evaluation → [1] Run ALL evaluations
```

### Step 3: Review Learning Curve
```bash
python main.py → [4] → [1] (Main Report)
```

Look at plots in `results/plots/`:
- `learning_curve.png` — shows if you're overfitting or underfitting
- `confusion_matrix.png` — shows false positives/negatives

### Step 4: Adjust Based on Signals

**IF you see overfitting (train AUC high, test AUC low):**
- Increase `dropouts` from 0.25 → 0.5
- Decrease `layer_sizes` from [1000, 500] → [512, 256]
- Reduce `NB_EPOCH_TRAIN` from 50 → 35

**IF you see underfitting (both AUC low):**
- Increase `layer_sizes` from [1000, 500] → [2048, 1024]
- Decrease `dropouts` from 0.25 → 0.1
- Increase `NB_EPOCH_TRAIN` from 50 → 100

**IF you have many false negatives (missing actives):**
- Decrease `CLASSIFICATION_THRESHOLD` from 0.5 → 0.3

**IF you have many false positives (wrong predictions):**
- Increase `CLASSIFICATION_THRESHOLD` from 0.5 → 0.7

### Step 5: Repeat
Make one change at a time, retrain, and compare. Document what worked!

---

## Parameter Interdependencies

⚠️ **These interact:**

| If you... | Then typically... |
|:---|:---|
| Increase layer_sizes | ...may need lower dropout |
| Increase dropout | ...model may underfit, need more epochs |
| Decrease learning_rate | ...training slower, may need more epochs |
| Use smaller TRAIN_SET_FRACTION | ...model sees less training data, may underfit |

**Pro tip:** Change **one** parameter at a time and measure impact!

---

## Why These Defaults?

The **default parameters** (`layer_sizes=[1000, 500]`, `dropout=0.25`, `learning_rate=0.001`) are:

✅ **Empirically proven** for molecular ML on 1K-10K compounds
✅ **Balanced** between underfitting and overfitting  
✅ **Efficient** (converge in ~50 epochs without wasting time)
✅ **Reproducible** (documented in literature)

**For publication:** You can justify using defaults by citing:
> *"Hyperparameters were set to established defaults for medium-sized molecular datasets: two hidden layers (1000, 500 neurons), dropout rate of 0.25, Adam optimizer with learning rate 0.001, trained for 50 epochs. Configuration was validated via learning curve analysis to confirm convergence without overfitting."*

This shows you're **informed**, not just guessing!

---

## Best Practices

### ✅ Do
- Keep `RANDOM_SEED` fixed for reproducibility
- Use `N_WORKERS = None` (auto-detect) unless you know better
- Increase `PARALLEL_BATCH_SIZE` only if you have plenty of RAM
- Start with defaults and adjust only if needed
- Change **one parameter at a time** and measure impact
- **Document** what changes you make and why

### ❌ Don't
- Change `TENSORFLOW_SEED` unless you know why
- Set `N_WORKERS` higher than your CPU core count
- Make `BATCH_SIZE` too large (> 64 usually unnecessary)
- Modify neural network layers without testing
- Change multiple hyperparameters simultaneously (can't tell what worked)
- Use different hyperparameters on train vs test sets

---

## Verify Configuration

Check current settings:

```bash
python -c "import settings as cfg; print(cfg.__dict__)"
```

Or create test script:

```bash
# settings_check.py
import settings as cfg

print("Parallel Processing:")
print(f"  Enabled: {cfg.ENABLE_PARALLEL_PROCESSING}")
print(f"  Workers: {cfg.N_WORKERS}")
print(f"  Batch Size: {cfg.PARALLEL_BATCH_SIZE}")

print("\nModel:")
print(f"  Hidden Layers: {cfg.HIDDEN_LAYERS}")
print(f"  Epochs: {cfg.EPOCHS}")
print(f"  Learning Rate: {cfg.LEARNING_RATE}")
```

---

## Further Reading & Foundations

For deeper understanding of hyperparameter tuning, neural network architecture, and molecular machine learning:

### Molecular Machine Learning

1. **Ramsundar, B.** et al. Massively Multitask Networks for Drug Discovery.
   - *arXiv:1502.02072* (2015)
   - Link: https://arxiv.org/abs/1502.02072

2. **Ma, J.** et al. Deep Neural Nets as a Method for Quantitative Structure-Activity Relationships.
   - *J. Chem. Inf. Model.* 2015, 55, 263–274
   - DOI: 10.1021/ci500747n

3. **Wu, Z.** et al. MoleculeNet: A Benchmark for Molecular Machine Learning.
   - *Chem. Sci.* 2018, 9, 513–530
   - DOI: 10.1039/c7sc02664a
   - PubMed: 29629118

### Deep Learning Theory

4. **Goodfellow, I., Bengio, Y., Courville, A.** Deep Learning.
   - MIT Press, 2016
   - Chapter 5.2: "Capacity, Overfitting and Underfitting"
   - Link: https://www.deeplearningbook.org/

5. **Bengio, Y.** et al. Representation Learning: A Review and New Perspectives.
   - *IEEE Trans. Pattern Anal. Mach. Intell.* 2013, 35, 1798–1828
   - DOI: 10.1109/TPAMI.2013.50

---

## Key Concepts Summary

| Concept | Where it appears | Relevant reading |
|:---|:---|:---|
| **Overfitting prevention via dropout** | `dropouts` parameter | Goodfellow et al., Ch. 5.2 |
| **Model capacity trade-off** | `layer_sizes` tuning | Goodfellow et al., Ch. 5.2 |
| **Molecular fingerprints** | `FINGERPRINT_RADIUS`, `FINGERPRINT_LENGTH` | Wu et al., MoleculeNet |
| **Multitask learning** | Model architecture | Ramsundar et al. |
| **QSAR via deep learning** | Classification threshold, evaluation metrics | Ma et al. |

---

