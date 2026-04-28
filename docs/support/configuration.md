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
HIDDEN_LAYERS = [512, 256, 128]
DROPOUT_RATE = 0.5
LEARNING_RATE = 0.001
EPOCHS = 50
BATCH_SIZE = 32
```

**What it does:** Model structure and training parameters

**Tuning guide:**
| Setting | Increase | Decrease |
|---------|----------|----------|
| `HIDDEN_LAYERS` | If underfitting | If overfitting |
| `DROPOUT_RATE` | If overfitting | If underfitting |
| `LEARNING_RATE` | Training too slow | May diverge |
| `EPOCHS` | Not converging | Wastes time |
| `BATCH_SIZE` | If OOM errors | Slower training |

**Safe defaults:**
- Start with defaults above
- Only adjust if model performance is poor

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

### Scenario 1: Low RAM (4GB)
```python
ENABLE_PARALLEL_PROCESSING = True
N_WORKERS = 2
PARALLEL_BATCH_SIZE = 25000
PARALLEL_MIN_THRESHOLD = 5000
HIDDEN_LAYERS = [256, 128]
BATCH_SIZE = 16
```

### Scenario 2: Standard Setup (8GB RAM)
```python
ENABLE_PARALLEL_PROCESSING = True
N_WORKERS = None                # Auto-detect
PARALLEL_BATCH_SIZE = 50000
PARALLEL_MIN_THRESHOLD = 10000
HIDDEN_LAYERS = [512, 256, 128]
BATCH_SIZE = 32
```

### Scenario 3: High-Performance (16GB+ RAM)
```python
ENABLE_PARALLEL_PROCESSING = True
N_WORKERS = -1                  # Use all cores
PARALLEL_BATCH_SIZE = 200000
PARALLEL_MIN_THRESHOLD = 5000
HIDDEN_LAYERS = [1024, 512, 256]
BATCH_SIZE = 64
```

### Scenario 4: Debugging/Single-Core
```python
ENABLE_PARALLEL_PROCESSING = False
N_WORKERS = 1
HIDDEN_LAYERS = [256, 128]
BATCH_SIZE = 16
VERBOSE_LOGGING = True
PLOT_DPI = 150
```

---

## Best Practices

### ✅ Do
- Keep `RANDOM_SEED` fixed for reproducibility
- Use `N_WORKERS = None` (auto-detect) unless you know better
- Increase `PARALLEL_BATCH_SIZE` only if you have plenty of RAM
- Start with defaults and adjust only if needed

### ❌ Don't
- Change `TENSORFLOW_SEED` unless you know why
- Set `N_WORKERS` higher than your CPU core count
- Make `BATCH_SIZE` too large (> 64 usually unnecessary)
- Modify neural network layers without testing

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

