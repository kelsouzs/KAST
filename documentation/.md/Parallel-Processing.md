# ⚡ Parallel Processing Guide

Complete guide to configuring and optimizing parallel processing in K-talysticFlow.

---

## 🚀 What is Parallel Processing?

Parallel processing allows K-talysticFlow to use multiple CPU cores simultaneously, dramatically reducing computation time for intensive tasks like:

- 🧬 **Featurization** (5-10x faster)
- 🔬 **Tanimoto Similarity** (3-5x faster)
- 📊 **Learning Curves** (4-8x faster)
- 🔮 **Predictions** (5-10x faster)

---

## 💡 Performance Gains

### Real-World Benchmarks

| Dataset Size | Sequential Time | Parallel Time (6 cores) | Speedup |
|--------------|----------------|------------------------|---------|
| 1,000 molecules | 2 min | 1 min | 2x |
| 10,000 molecules | 20 min | 4 min | 5x |
| 50,000 molecules | 90 min | 12 min | 7.5x |
| 100,000 molecules | 180 min | 20 min | 9x |

*Benchmarks: Intel i7-8700 (6 cores, 12 threads), 16GB RAM*

---

## ⚙️ Configuration

### Method 1: settings.py (Permanent)

Edit `settings.py` - Section 12:

```python
# Section 12: PARALLEL PROCESSING CONFIGURATIONS
# ============================================================================

# Enable/disable parallelism globally
ENABLE_PARALLEL_PROCESSING = True

# Number of CPU cores to use
# Options:
#   None  = auto-detect (cpu_count - 1) ✅ RECOMMENDED
#   -1    = use ALL cores
#   1     = disable parallelism
#   N     = use exactly N cores (e.g., 4, 6, 8)
N_WORKERS = None

# Batch size for memory-efficient processing
# Larger = faster but more RAM
PARALLEL_BATCH_SIZE = 100000

# Minimum dataset size to trigger parallelism
# Below this threshold, runs sequentially
PARALLEL_MIN_THRESHOLD = 10000
```

---

### Method 2: Runtime Configuration (Temporary)

Use the control panel for on-the-fly changes:

```bash
python main.py
# Select [8] Advanced Options
# Select [3] Configure Parallel Processing Workers
```

**Example:**
```
Current Configuration:
  ENABLE_PARALLEL_PROCESSING = True
  N_WORKERS = None (Auto: 7 cores detected)
  PARALLEL_BATCH_SIZE = 100000
  PARALLEL_MIN_THRESHOLD = 10000

Enter new N_WORKERS value: 4
✅ Runtime configuration updated: N_WORKERS = 4
```

**Note:** Runtime changes are temporary (session only)

---

## 🎯 Optimal Configuration Guide

### Auto Mode (Recommended)

```python
N_WORKERS = None
```

**Pros:**
- ✅ Automatically detects optimal cores
- ✅ Leaves 1 core free for system
- ✅ Safe for all hardware
- ✅ Best for general use

**When to use:** Default for most users

---

### Fixed Core Count

```python
N_WORKERS = 4  # Use exactly 4 cores
```

**Pros:**
- ✅ Predictable resource usage
- ✅ Good for shared systems
- ✅ Consistent performance

**When to use:**
- Shared workstations
- Need consistent resource allocation
- Troubleshooting performance

---

### Maximum Performance

```python
N_WORKERS = -1  # Use ALL cores
```

**Pros:**
- ✅ Maximum speed
- ✅ Best for dedicated machines

**Cons:**
- ⚠️ May slow down system responsiveness
- ⚠️ Not recommended during multitasking

**When to use:**
- Dedicated analysis machine
- Batch processing overnight
- Maximum speed priority

---

### Sequential Processing

```python
ENABLE_PARALLEL_PROCESSING = False
# OR
N_WORKERS = 1
```

**Pros:**
- ✅ Lowest memory usage
- ✅ Easiest debugging
- ✅ Compatible with all systems

**Cons:**
- ⚠️ 5-10x slower

**When to use:**
- Low-end hardware
- Memory constraints
- Debugging issues
- Very small datasets (< 1,000 molecules)

---

## 📊 Hardware-Specific Recommendations

### Low-End Systems
*Dual-core CPU, 8GB RAM*

```python
ENABLE_PARALLEL_PROCESSING = False
N_WORKERS = 1
PARALLEL_BATCH_SIZE = 50000
```

---

### Mid-Range Systems
*Quad-core CPU, 16GB RAM*

```python
ENABLE_PARALLEL_PROCESSING = True
N_WORKERS = None  # Auto: will use 3 cores
PARALLEL_BATCH_SIZE = 100000
```

---

### High-End Systems
*8+ core CPU, 32GB+ RAM*

```python
ENABLE_PARALLEL_PROCESSING = True
N_WORKERS = None  # Auto: will use cpu_count-1
PARALLEL_BATCH_SIZE = 200000
```

---

### Server/HPC Systems
*16+ cores, 64GB+ RAM*

```python
ENABLE_PARALLEL_PROCESSING = True
N_WORKERS = -1  # Use all cores
PARALLEL_BATCH_SIZE = 500000
```

---

## 🧪 Testing Your Configuration

### Run the Test Suite

```bash
python main.py
# [8] Advanced Options → [2] Test Parallel Processing
```

**The suite runs 6 tests:**

#### Test 1: Basic Parallel Execution
✅ Verifies joblib Parallel functionality

#### Test 2: Large Array Processing
✅ Tests 1M element computation (real-world scale)

#### Test 3: Memory Efficiency
✅ Ensures batch processing works correctly

#### Test 4: Error Handling
✅ Tests recovery from worker failures

#### Test 5: Performance Benchmark
✅ Compares sequential vs parallel (10K operations)

#### Test 6: Worker Scaling
✅ Tests performance with 1, 2, 4, and max workers

**Interpretation:**
- **All tests PASSED** ✅ - Optimal configuration
- **Some tests FAILED** ⚠️ - Reduce workers or batch size
- **Test 5 speedup < 2x** - Consider disabling parallelism

---

## 📈 Memory Considerations

### PARALLEL_BATCH_SIZE

Controls how many molecules are processed in each batch.

**Trade-off:**
- **Larger batches** = Faster but more RAM
- **Smaller batches** = Slower but safer

**Guidelines:**

| Available RAM | Recommended Batch Size | Max Dataset |
|---------------|------------------------|-------------|
| 8 GB | 50,000 | ~100K molecules |
| 16 GB | 100,000 | ~500K molecules |
| 32 GB | 200,000 | ~1M molecules |
| 64 GB+ | 500,000 | Unlimited |

**Signs batch size is too large:**
- 💥 Out of memory errors
- 💥 System freezing
- 💥 Swap usage spikes

**Solution:** Reduce by 50%

---

## 🔍 Which Scripts Use Parallelism?

| Script | Parallelism | Speedup | Bottleneck |
|--------|-------------|---------|------------|
| `1_preparation.py` | ❌ No | N/A | I/O bound |
| `2_featurization.py` | ✅ Yes | 5-10x | CPU bound |
| `3_create_training.py` | ⚠️ Partial* | Varies | GPU/CPU bound |
| `4_0_evaluation_main.py` | ❌ No | N/A | Fast already |
| `4_1_cross_validation.py` | ❌ No | N/A | Model overhead |
| `4_2_enrichment_factor.py` | ❌ No | N/A | Fast already |
| `4_3_tanimoto_similarity.py` | ✅ Yes | 3-5x | CPU bound |
| `4_4_learning_curve.py` | ✅ Yes | 4-8x | CPU bound |
| `5_0_featurize_for_prediction.py` | ✅ Yes | 5-10x | CPU bound |
| `5_1_run_prediction.py` | ⚠️ Partial* | Varies | Model overhead |

*TensorFlow uses internal parallelism (separate from joblib)

---

## 🛠️ Troubleshooting

### Issue: No speedup observed

**Possible causes:**
1. Dataset too small (< 10,000 molecules)
2. `PARALLEL_MIN_THRESHOLD` not reached
3. I/O bottleneck (slow disk)
4. Only 1-2 cores available

**Solutions:**
- Check dataset size
- Lower `PARALLEL_MIN_THRESHOLD`
- Use SSD storage
- Verify CPU core count

---

### Issue: System becomes unresponsive

**Cause:** Too many workers consuming all CPU

**Solution:**
```python
N_WORKERS = None  # Auto mode (leaves 1 core free)
# OR
N_WORKERS = cpu_count // 2  # Use half of cores
```

---

### Issue: Out of memory errors

**Cause:** Batch size too large

**Solution:**
```python
PARALLEL_BATCH_SIZE = 50000  # Reduce by 50%
# OR
N_WORKERS = 2  # Reduce workers
```

---

### Issue: "joblib" import error

**Solution:**
```bash
pip install joblib
```

---

### Issue: Slower than expected

**Check:**
1. Disk speed (use SSD if possible)
2. RAM usage (swap = slow)
3. Background processes
4. CPU temperature (throttling?)

**Benchmark:**
```bash
python bin/test_parallel_compatibility.py
```

---

## 📊 Performance Monitoring

### View Real-Time Status

The control panel shows current configuration:

```
⚡ Parallel Processing: ENABLED (6 workers)
```

### Check Logs

```bash
# View featurization log
cat results/02_featurization_log.txt

# Look for:
# "Using parallel processing with X workers"
# "Processed Y molecules in Z seconds"
```

---

## 🎓 Advanced: Custom Parallelization

For developers extending K-talysticFlow:

```python
from joblib import Parallel, delayed
import settings as cfg

def process_molecule(smiles):
    # Your processing logic
    return result

def parallel_process(smiles_list):
    if cfg.ENABLE_PARALLEL_PROCESSING and len(smiles_list) >= cfg.PARALLEL_MIN_THRESHOLD:
        # Parallel mode
        results = Parallel(n_jobs=cfg.N_WORKERS)(
            delayed(process_molecule)(smiles) 
            for smiles in smiles_list
        )
    else:
        # Sequential mode
        results = [process_molecule(smiles) for smiles in smiles_list]
    
    return results
```

---

## 📝 Best Practices

### ✅ Do's

1. **Use auto mode** (`N_WORKERS = None`) for most cases
2. **Test your configuration** before big runs
3. **Monitor memory usage** during first runs
4. **Adjust batch size** based on RAM
5. **Run overnight** for maximum performance (use -1 workers)
6. **Benchmark** with `test_parallel_compatibility.py`

### ❌ Don'ts

1. **Don't use all cores** during multitasking
2. **Don't set batch size too high** (RAM limits)
3. **Don't parallelize small datasets** (< 1,000 molecules)
4. **Don't forget to save** settings after optimization
5. **Don't ignore memory warnings**

---
