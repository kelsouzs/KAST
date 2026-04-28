# ⚙️ Configuration Guide

Complete guide to customizing K-talysticFlow settings.

---

## 📁 Configuration File: `settings.py`

All K-talysticFlow configurations are centralized in `settings.py` at the project root.

```python
# settings.py
import os
from pathlib import Path

# ... configuration sections ...
```

---

## 📋 Configuration Sections

### Section 1: Main Paths

```python
PROJECT_ROOT = Path(__file__).parent.resolve()
DATA_RAW_DIR = PROJECT_ROOT / 'data'
RESULTS_DIR = PROJECT_ROOT / 'results'
ACTIVE_SMILES_FILE = DATA_RAW_DIR / 'actives.smi'
INACTIVE_SMILES_FILE = DATA_RAW_DIR / 'inactives.smi'
```

**When to modify:**
- Using different folder structure
- Files named differently

**Example:**
```python
ACTIVE_SMILES_FILE = DATA_RAW_DIR / 'my_actives.smiles'
INACTIVE_SMILES_FILE = DATA_RAW_DIR / 'my_inactives.smiles'
```

---

### Section 2: Basic Configurations

```python
TEST_SET_FRACTION = 0.3
RANDOM_STATE = 42
FP_SIZE = 2048
FP_RADIUS = 3
```

#### TEST_SET_FRACTION
**Interactive Selection:** When running `1_preparation.py`, you'll be prompted to choose the split ratio.

**Available Options:**
- `0.2` → 80/20 split ✅ **RECOMMENDED for small datasets**
- `0.3` → 70/30 split (more test data)
- `0.1` → 90/10 split (maximum training data)
- Custom → Enter your preferred ratio (5-50%)

**When to change:**
- Small dataset → Use 0.2 (more training data)
- Large dataset → Use 0.3-0.4 (better validation)

---

#### RANDOM_STATE
**Default:** `42`

**Purpose:** Reproducibility (same splits every time)

**Options:**
- Any integer (e.g., 0, 123, 999)
- `None` → Different splits each run

**When to change:**
- Want different train/test splits
- Testing model robustness

---

#### FP_SIZE (Fingerprint Size)
**Default:** `2048` bits

**Options:**
- `512` → Smaller, faster, less info
- `1024` → Balanced
- `2048` → Standard ✅ **RECOMMENDED**
- `4096` → Larger, more info, slower

**Impact:**
- Larger → More information but slower and more memory
- Smaller → Faster but may lose information

**When to change:**
- Memory constraints → Use 512 or 1024
- Large dataset → Try 4096 for better performance

---

#### FP_RADIUS
**Default:** `3`

**Options:**
- `2` → ECFP4 (smaller substructures)
- `3` → ECFP6 ✅ **RECOMMENDED**
- `4` → ECFP8 (larger substructures)

**Impact:**
- Larger radius → Captures larger molecular patterns
- Smaller radius → More focused on local features

**Recommendation:** Start with 3, adjust based on molecule size

---

### Section 5: Model Parameters

```python
MODEL_PARAMS = {
    'n_tasks': 1,
    'layer_sizes': [1000, 500],
    'dropouts': 0.25,
    'learning_rate': 0.001,
    'mode': 'classification',
    'nb_epoch': 50
}
```

#### layer_sizes (Neural Network Architecture)
**Default:** `[1000, 500]` (2 hidden layers)

**Options:**
```python
# Smaller/faster model
'layer_sizes': [512, 256]

# Larger/more complex model
'layer_sizes': [2048, 1024, 512]

# Very deep model
'layer_sizes': [1024, 512, 256, 128]
```

**Guidelines:**
- **Small dataset (< 1K):** `[512, 256]`
- **Medium dataset (1K-10K):** `[1000, 500]` ✅
- **Large dataset (> 10K):** `[2048, 1024, 512]`

---

#### dropouts (Regularization)
**Default:** `0.25` (25% dropout)

**Options:**
- `0.1` → Light regularization
- `0.25` → Moderate ✅ **RECOMMENDED**
- `0.5` → Strong regularization (prevents overfitting)

**When to change:**
- **Overfitting** (train AUC >> test AUC) → Increase to 0.5
- **Underfitting** (both low) → Decrease to 0.1

---

#### learning_rate
**Default:** `0.001`

**Options:**
- `0.0001` → Slow, stable learning
- `0.001` → Standard ✅ **RECOMMENDED**
- `0.01` → Fast, may be unstable

**When to change:**
- **Loss not decreasing** → Reduce to 0.0001
- **Very slow training** → Increase to 0.01 (with caution)

---

#### nb_epoch (Training Epochs)
**Default:** `50`

**Options:**
- `30` → Faster training
- `50` → Standard ✅ **RECOMMENDED**
- `100` → More training (may overfit)

**When to change:**
- **Quick testing** → 20-30 epochs
- **Production model** → 50-100 epochs
- **Overfitting** → Reduce to 30

---

### Section 6: Training Configurations

```python
NB_EPOCH_TRAIN = 50
NB_EPOCH_CV = 30
NB_EPOCH_LC = 20
CLASSIFICATION_THRESHOLD = 0.5
```

#### NB_EPOCH_* (Epochs for Different Stages)
- **NB_EPOCH_TRAIN**: Main training (50)
- **NB_EPOCH_CV**: Cross-validation (30) - faster
- **NB_EPOCH_LC**: Learning curve (20) - even faster

**Why different values?**
- CV and LC run multiple times → Use fewer epochs to save time
- Still get valid comparisons

---

#### CLASSIFICATION_THRESHOLD
**Default:** `0.5` (50% probability cutoff)

**Options:**
- `0.3` → More predictions as "active" (higher recall, lower precision)
- `0.5` → Balanced ✅ **RECOMMENDED**
- `0.7` → Fewer predictions as "active" (lower recall, higher precision)

**When to change:**
- **Need high recall** (don't miss actives) → 0.3-0.4
- **Need high precision** (few false positives) → 0.6-0.7

---

### Section 7: Validation Configurations

```python
N_FOLDS_CV = 5
EF_FRACTIONS_PERCENT = [1.0, 2.0, 5.0, 10.0]
ENRICHMENT_FACTORS = [0.01, 0.05, 0.1]
TANIMOTO_SAMPLE_SIZE = 1000
```

#### N_FOLDS_CV (Cross-Validation Folds)
**Default:** `5`

**Options:**
- `3` → Faster, less reliable
- `5` → Balanced ✅ **RECOMMENDED**
- `10` → More reliable, slower

**When to change:**
- **Quick testing** → 3 folds
- **Publication** → 10 folds for robustness

---

#### EF_FRACTIONS_PERCENT (Enrichment Factor Cutoffs)
**Default:** `[1.0, 2.0, 5.0, 10.0]` (top 1%, 2%, 5%, 10%)

**When to change:**
- **Large library** → Add 0.5% or 0.1%
- **Small library** → Use only [5.0, 10.0]

---

#### TANIMOTO_SAMPLE_SIZE
**Default:** `1000` (sample 1000 molecules)

**Options:**
- `500` → Faster
- `1000` → Balanced ✅
- `5000` → More accurate, slower

**When to change:**
- **Large dataset (> 50K)** → Use 5000 for better statistics
- **Quick analysis** → Use 500

---

### Section 8: Data Validation Configurations

```python
MIN_MOLECULES_PER_CLASS = 50
MAX_MOLECULES_TOTAL = 100000
MIN_SMILES_LENGTH = 5
MAX_SMILES_LENGTH = 200
```

#### MIN_MOLECULES_PER_CLASS
**Default:** `50` (minimum 50 actives and 50 inactives)

**Options:**
- `20` → Lower threshold (less reliable)
- `50` → Recommended minimum ✅
- `100` → Better for robust models

---

#### MAX_MOLECULES_TOTAL
**Default:** `100000` (100K molecules max)

**Purpose:** Memory protection

**When to change:**
- **More RAM (32GB+)** → Increase to 500000
- **Less RAM (8GB)** → Decrease to 50000

---

#### MIN/MAX_SMILES_LENGTH
**Default:** `5` to `200` characters

**Purpose:** Filter out very small or very large molecules

**When to change:**
- **Peptides/polymers** → Increase MAX to 500
- **Fragments only** → Decrease MIN to 3

---

### Section 12: Parallel Processing Configurations

```python
ENABLE_PARALLEL_PROCESSING = True
N_WORKERS = None
PARALLEL_BATCH_SIZE = 100000
PARALLEL_MIN_THRESHOLD = 10000
```

**See [Parallel Processing Guide](Parallel-Processing.md) for complete documentation.**

---

## 🎯 Configuration Recipes

### Recipe 1: Fast Testing

```python
# Quick runs for testing
TEST_SET_FRACTION = 0.2
FP_SIZE = 1024
FP_RADIUS = 2
MODEL_PARAMS = {
    'layer_sizes': [512, 256],
    'nb_epoch': 20
}
N_FOLDS_CV = 3
```

---

### Recipe 2: Production Model

```python
# High-quality model for publication
TEST_SET_FRACTION = 0.3
FP_SIZE = 2048
FP_RADIUS = 3
MODEL_PARAMS = {
    'layer_sizes': [1000, 500],
    'nb_epoch': 100,
    'dropouts': 0.3
}
N_FOLDS_CV = 10
TANIMOTO_SAMPLE_SIZE = 5000
```

---

### Recipe 3: Large Dataset (> 50K)

```python
# Optimized for large datasets
FP_SIZE = 2048
MODEL_PARAMS = {
    'layer_sizes': [2048, 1024, 512],
    'nb_epoch': 50
}
ENABLE_PARALLEL_PROCESSING = True
N_WORKERS = -1
PARALLEL_BATCH_SIZE = 200000
```

---

### Recipe 4: Low Memory (8GB RAM)

```python
# Minimize memory usage
FP_SIZE = 1024
FP_RADIUS = 2
MODEL_PARAMS = {
    'layer_sizes': [512, 256],
}
ENABLE_PARALLEL_PROCESSING = False
PARALLEL_BATCH_SIZE = 50000
MAX_MOLECULES_TOTAL = 50000
```

---

## 🔧 Advanced Customization

### Modifying Scripts

For advanced users who want to modify pipeline behavior:

**Edit individual scripts in `bin/` folder:**

```python
# bin/2_featurization.py
# Find and modify featurization parameters

# bin/3_create_training.py
# Customize training loop, callbacks, etc.
```

**Recommendation:** Create backup before modifying:
```bash
cp bin/3_create_training.py bin/3_create_training.py.backup
```

---

### Custom Loss Functions

Edit `bin/3_create_training.py` to use custom loss:

```python
# Find MultitaskClassifier initialization
model = dc.models.MultitaskClassifier(
    # ... existing params ...
    # Add custom loss (advanced)
)
```

---

### Custom Metrics

Add to evaluation scripts:

```python
# bin/4_0_evaluation_main.py
from sklearn.metrics import matthews_corrcoef

# Add after existing metrics
mcc = matthews_corrcoef(y_true, y_pred)
print(f"Matthews Correlation Coefficient: {mcc:.4f}")
```

---

## 📝 Configuration Best Practices

### ✅ Do's

1. **Start with defaults** before customizing
2. **Document changes** in comments
3. **Backup settings.py** before major changes
4. **Test on small dataset** first
5. **Keep RANDOM_STATE consistent** for reproducibility
6. **Match FP parameters** between training and prediction

### ❌ Don'ts

1. **Don't change settings mid-pipeline** (except N_WORKERS)
2. **Don't use very small epochs** (< 10) for final models
3. **Don't set N_WORKERS too high** (leaves no CPU for system)
4. **Don't ignore memory errors** (reduce batch size instead)
5. **Don't modify settings during training** (restart pipeline)

---

## 🔄 Applying Configuration Changes

### Changes requiring re-run:

| Changed Setting | Re-run From Step |
|----------------|------------------|
| FP_SIZE, FP_RADIUS | [2] Featurization |
| TEST_SET_FRACTION | [1] Preparation |
| MODEL_PARAMS | [3] Training |
| N_FOLDS_CV | [4] Cross-validation |
| N_WORKERS | No re-run needed |

---

## 📊 Monitoring Configuration Impact

### Compare Model Versions

```bash
# Save results with descriptive names
mv results results_config_v1
# Modify settings
# Re-run pipeline
mv results results_config_v2
# Compare metrics
diff results_config_v1/4_0_evaluation_report.txt \
     results_config_v2/4_0_evaluation_report.txt
```
---