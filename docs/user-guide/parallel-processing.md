# Parallel Processing

Speed up KAST with multi-core parallel processing — 5-10x faster on large datasets!

---

## What Is It?

Parallel processing distributes work across multiple CPU cores simultaneously, instead of using just one core.

**Performance Gains:**
- **Small datasets (< 10K molecules):** Minimal improvement
- **Medium datasets (10K-100K):** 3-5x faster
- **Large datasets (100K+):** 5-10x faster

---

## Quick Enable

**Edit `settings.py` (Section 12):**

```python
# Section 12: PARALLEL PROCESSING CONFIGURATIONS

ENABLE_PARALLEL_PROCESSING = True    # On/Off
N_WORKERS = None                     # None = auto-detect (RECOMMENDED)
PARALLEL_BATCH_SIZE = 100000         # Molecules per batch
PARALLEL_MIN_THRESHOLD = 10000       # Min dataset size for parallel
```

**Or configure interactively:**
```bash
python main.py
→ [8] Advanced Options
→ [3] Configure CPU Cores
→ Choose [0] auto-detect or specific number
```

---

## N_WORKERS Options

| Setting | What Happens | Best For |
|---------|-------------|----------|
| `N_WORKERS = None` | Auto-detect optimal cores (N-1) | ✅ **RECOMMENDED** |
| `N_WORKERS = -1` | Use ALL available cores | High-performance systems |
| `N_WORKERS = 4` | Use exactly 4 cores | Limited RAM (8GB) |
| `N_WORKERS = 1` | Single-core only (disable parallel) | Debugging |

---

## Memory Settings

Adjust `PARALLEL_BATCH_SIZE` based on available RAM:

| RAM Available | Batch Size | N_WORKERS |
|---------------|-----------|-----------|
| 4GB | 25,000 | 2 |
| 8GB | 50,000 | 4 |
| 16GB | 100,000 | 6 |
| 32GB+ | 200,000 | 8+ |

**Formula:** (Available RAM in GB - 2) × 10,000 ≈ good batch size

---

## Scripts That Support Parallel

| Script | Speedup | When Enabled |
|--------|---------|-------------|
| `2_featurization.py` | 5-10x | Dataset > 10K molecules |
| `4_3_tanimoto_similarity.py` | 3-5x | Always (if enabled) |
| `4_4_learning_curve.py` | 4-8x | Always (if enabled) |
| `5_0_featurize_for_prediction.py` | 5-10x | Dataset > 10K molecules |

---

## Configuration Examples

### Example 1: Small Dataset (8GB RAM)
```python
ENABLE_PARALLEL_PROCESSING = True
N_WORKERS = 4                       # Use 4 cores
PARALLEL_BATCH_SIZE = 50000
PARALLEL_MIN_THRESHOLD = 10000      # Activate for 10K+ molecules
```

### Example 2: Large Dataset (16GB+ RAM)
```python
ENABLE_PARALLEL_PROCESSING = True
N_WORKERS = None                    # Auto-detect
PARALLEL_BATCH_SIZE = 100000        # Default, good for 16GB+
PARALLEL_MIN_THRESHOLD = 10000
```

### Example 3: High-Performance (32GB+ RAM)
```python
ENABLE_PARALLEL_PROCESSING = True
N_WORKERS = -1                      # Use ALL cores
PARALLEL_BATCH_SIZE = 200000
PARALLEL_MIN_THRESHOLD = 5000       # Activate earlier
```

### Example 4: Debugging (Single Core)
```python
ENABLE_PARALLEL_PROCESSING = False  # Or N_WORKERS = 1
PARALLEL_BATCH_SIZE = 100000
PARALLEL_MIN_THRESHOLD = 10000
```

---

## Monitor Performance

During execution, KAST shows:
```
Parallel Processing ENABLED (6 workers)
Processing 150,000 molecules in batches of 100,000...
Batch 1/2: [████████████████] 100% 2m 15s
Batch 2/2: [████████████████] 100% 2m 10s
Total time: 4m 25s (would be 30m single-core!)
```

---

## Troubleshooting

### "Out of Memory" error
- **Cause:** Batch size too large for your RAM
- **Fix:** Reduce `PARALLEL_BATCH_SIZE` (try 50,000 or 25,000)

### "Only using 1 core"
- **Cause:** Dataset smaller than `PARALLEL_MIN_THRESHOLD`
- **Fix:** Either use larger dataset or lower `PARALLEL_MIN_THRESHOLD`

### "Slower than expected"
- **Cause:** Too many workers for available cores
- **Fix:** Set `N_WORKERS` to your actual core count (not more)

### "Can't find joblib"
- **Cause:** Dependency not installed
- **Fix:** `pip install joblib` or re-run setup.exe/setup.sh

---

## Test Parallel Setup

Verify parallel processing works:

```bash
python main.py
→ [8] Advanced Options
→ [2] Test Parallel Processing
```

Should show 6 tests completing successfully:
```
TEST 1: Import settings.py... [OK]
TEST 2: Configuration Variables... [OK]
TEST 3: Required Imports... [OK]
TEST 4: Worker Optimization Logic... [OK]
TEST 5: Script Compatibility... [OK]
TEST 6: Parallel Threshold Logic... [OK]
✅ ALL TESTS COMPLETED SUCCESSFULLY!
```

---


