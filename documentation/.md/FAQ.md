# ❓ Frequently Asked Questions (FAQ)

Common questions and answers about K-talysticFlow.

---

## 🎯 General Questions

### What is K-talysticFlow?

K-talysticFlow (KAST) is an automated deep learning pipeline for predicting molecular bioactivity. It helps researchers identify promising drug candidates through virtual screening using neural networks trained on molecular fingerprints.

---

### Who should use K-talysticFlow?

**Ideal for:**
- 🧬 Computational chemists
- 💊 Drug discovery researchers
- 🎓 Graduate students in cheminformatics
- 🔬 Medicinal chemists performing virtual screening
- 📊 Data scientists working with molecular data

**Not suitable for:**
- Complete beginners without chemistry background
- Projects requiring quantum mechanics (use DFT software instead)
- Protein-ligand docking (use AutoDock, Vina, etc.)

---

### Is K-talysticFlow free?

Yes! K-talysticFlow is **open-source** and licensed under the **MIT License**. You can use it freely for:
- ✅ Academic research
- ✅ Commercial projects
- ✅ Educational purposes

---

### What makes K-talysticFlow different?

1. **Fully automated** - No coding required for standard workflows
2. **Interactive menu** - User-friendly control panel
3. **Comprehensive validation** - 5 evaluation modules built-in
4. **K-Prediction Score** - Proprietary ranking system
5. **Parallel processing** - 5-10x faster than sequential
6. **Production-ready** - Logging, error handling, reproducibility

---

## 📊 Data & Inputs

### What input format does K-talysticFlow accept?

**SMILES format** (`.smi` or `.smiles` files)

**Format:**
```
SMILES [space] optional_name
```

**Example:**
```
CC(C)Cc1ccc(cc1)C(C)C(O)=O  ibuprofen
CN1C=NC2=C1C(=O)N(C)C(=O)N2C  caffeine
```

---

### How much data do I need?

**Minimum:**
- 50 active compounds
- 50 inactive compounds
- Total: 100+ molecules

**Recommended:**
- 500+ actives
- 500+ inactives
- Total: 1,000-10,000 molecules

**Optimal:**
- 5,000+ actives
- 5,000+ inactives
- Total: 10,000+ molecules

**Rule of thumb:** More data = better model (up to ~100K molecules)

---

### What is a good active/inactive ratio?

**Best:** 1:1 (balanced)
```
3,000 actives : 3,000 inactives ✅
```

**Acceptable:** 1:2 to 1:10
```
1,000 actives : 5,000 inactives ⚠️
```

**Poor:** > 1:10
```
100 actives : 5,000 inactives ❌
```

**Note:** K-talysticFlow automatically balances datasets in `1_preparation.py`

---

### Can I use my own molecular descriptors?

Currently, K-talysticFlow uses **Morgan Fingerprints (ECFP)** exclusively. 

**Customization available:**
- Radius (default: 3)
- Size (default: 2048 bits)

Edit in `settings.py`:
```python
FP_SIZE = 2048
FP_RADIUS = 3
```

**Future versions** may support custom descriptors.

---

### What if I only have active compounds?

You need **both actives and inactives** to train a binary classifier.

**Options:**
1. **Generate decoys** using tools like:
   - [DUD-E](http://dude.docking.org/) (Drug-like decoys)
   - [NRLiSt BDB](https://nrlist.bdb.tools/)
   - Random compounds from ZINC
2. **Use negative examples** from literature
3. **Experimental inactives** from screening data

---

### Can I predict multiple targets at once?

Currently, K-talysticFlow supports **single-target** prediction (binary classification: active/inactive for one target).

**Workaround:** Train separate models for each target.

---

## 🤖 Model & Training

### How long does training take?

**Typical times:**

| Dataset Size | Training Time |
|--------------|---------------|
| 1,000 molecules | 2-5 minutes |
| 10,000 molecules | 5-10 minutes |
| 50,000 molecules | 10-20 minutes |
| 100,000 molecules | 20-40 minutes |

*Times for default 50 epochs on mid-range CPU*

---

### Can I use my own neural network architecture?

Yes! Edit `settings.py`:

```python
MODEL_PARAMS = {
    'n_tasks': 1,
    'layer_sizes': [1000, 500],  # Change architecture here
    'dropouts': 0.25,
    'learning_rate': 0.001,
    'mode': 'classification',
    'nb_epoch': 50
}
```

**Example - Deeper network:**
```python
'layer_sizes': [2048, 1024, 512, 256]
```

**Example - Smaller network:**
```python
'layer_sizes': [512, 256]
```

---

### What is a good ROC-AUC score?

| AUC Score | Performance |
|-----------|-------------|
| 0.9 - 1.0 | Excellent ⭐⭐⭐⭐⭐ |
| 0.8 - 0.9 | Very Good ⭐⭐⭐⭐ |
| 0.7 - 0.8 | Good ⭐⭐⭐ |
| 0.6 - 0.7 | Fair ⭐⭐ |
| < 0.6 | Poor ⭐ (retrain) |

**For publication:** AUC > 0.75 is generally acceptable.

---

### My model overfits. What should I do?

**Overfitting signs:**
- Training AUC >> Test AUC (gap > 0.15)
- High training accuracy, low test accuracy

**Solutions:**

1. **Increase dropout:**
```python
'dropouts': 0.5  # Increase from 0.25
```

2. **Reduce epochs:**
```python
'nb_epoch': 30  # Decrease from 50
```

3. **Get more training data**

4. **Simplify architecture:**
```python
'layer_sizes': [512, 256]  # Simpler than [1000, 500]
```

5. **Use data augmentation** (future feature)

---

### Can I use a GPU?

Yes! TensorFlow automatically uses GPU if available.

**To enable:**
```bash
pip install tensorflow[and-cuda]
```

**Verify:**
```python
import tensorflow as tf
print(tf.config.list_physical_devices('GPU'))
```

**Note:** GPU helps most for very large datasets (>50K molecules)

---

## ⚡ Performance & Speed

### How can I make it faster?

1. **Enable parallel processing:**
```python
ENABLE_PARALLEL_PROCESSING = True
N_WORKERS = None  # Auto mode
```

2. **Increase batch size** (if you have RAM):
```python
PARALLEL_BATCH_SIZE = 200000
```

3. **Use SSD storage** instead of HDD

4. **Close background programs**

5. **Use GPU** for training (TensorFlow)

---

### Why is parallel processing not helping?

**Possible reasons:**

1. **Dataset too small** (< 10,000 molecules)
   - Below `PARALLEL_MIN_THRESHOLD`
   - Solution: Lower threshold or disable parallelism

2. **Only 1-2 CPU cores**
   - Not enough workers for speedup
   - Solution: Upgrade hardware or use cloud computing

3. **I/O bottleneck** (slow disk)
   - Reading/writing is the slowest part
   - Solution: Use SSD

4. **Script doesn't support parallelism**
   - Some scripts are sequential by design
   - See [Parallel Processing Guide](Parallel-Processing)

---

### How much RAM do I need?

**Minimum:** 8 GB (small datasets < 10K)

**Recommended:** 16 GB (datasets 10K-50K)

**Optimal:** 32 GB+ (datasets > 50K)

**RAM usage formula (rough):**
```
RAM (GB) ≈ (Dataset size × Fingerprint size × 8) / 1e9
           + (Batch size × 2) / 1e6

Example: 50K molecules, 2048-bit FP, 100K batch
       ≈ (50000 × 2048 × 8) / 1e9 + (100000 × 2) / 1e6
       ≈ 0.82 GB + 0.2 GB ≈ 1 GB
```

*Actual usage varies with parallel processing overhead*

---

## 🔮 Predictions & Screening

### What is the K-Prediction Score?

**K-Prediction Score = Probability × 100**

It's a 0-100 scale for easy interpretation:
- **90-100:** Very likely active (high priority)
- **70-89:** Likely active (good candidates)
- **50-69:** Possibly active (medium priority)
- **Below 50:** Likely inactive

See [Output Analysis](Output-Analysis) for details.

---

### How many compounds should I test experimentally?

**Depends on your budget and model quality:**

**High-quality model (AUC > 0.85, EF@1% > 10):**
- Test **top 1-2%** of predictions
- K-Score > 80

**Good model (AUC 0.75-0.85, EF@1% = 5-10):**
- Test **top 5%** of predictions
- K-Score > 60

**Fair model (AUC 0.65-0.75, EF@1% < 5):**
- Test **top 10%** or validate with secondary assay
- K-Score > 50

---

### Can I screen a million compounds?

**Yes!** K-talysticFlow can handle large libraries.

**Tips for huge libraries:**

1. **Enable parallel processing:**
```python
ENABLE_PARALLEL_PROCESSING = True
N_WORKERS = -1  # Use all cores
```

2. **Increase batch size:**
```python
PARALLEL_BATCH_SIZE = 500000
```

3. **Run overnight** or on HPC cluster

4. **Split library** into chunks if memory issues

**Time estimate:**
- 1M molecules ≈ 1-3 hours (parallel, 8 cores)

---

### Are predictions reliable for molecules very different from training set?

**No!** Predictions are most reliable for molecules **similar to training data**.

**Check Tanimoto similarity:**
- **Tanimoto > 0.5** to training set → Reliable predictions ✅
- **Tanimoto 0.3-0.5** → Moderate confidence ⚠️
- **Tanimoto < 0.3** → Unreliable (out of applicability domain) ❌

**Recommendation:**
Run `4_3_tanimoto_similarity.py` to assess chemical space coverage.

---

## 🛠️ Technical Questions

### What Python version do I need?

**Required:** Python 3.9+

**Recommended:** Python 3.10 or 3.11

**Not supported:** Python 3.8 or earlier

---

### Can I run K-talysticFlow on Windows?

**Yes!** K-talysticFlow is fully compatible with:
- ✅ Windows 10/11
- ✅ Linux (Ubuntu, CentOS, etc.)
- ✅ macOS

---

### Do I need Conda or can I use pip?

**Recommended:** Conda (for RDKit)

**Alternative:** pip + system RDKit

**Why Conda?**
- RDKit installation is much easier
- Better dependency management
- Isolated environments

---

### Can I use K-talysticFlow in Jupyter Notebooks?

**Yes!** You can import and use modules:

```python
import sys
sys.path.append('/path/to/kast')

from bin import preparation, featurization
import settings as cfg

# Run pipeline steps programmatically
```

**But:** The interactive menu (`main.py`) is designed for terminal use.

---

### Is there a Docker image?

**Not yet**, but coming soon!

**Current workaround:**
```dockerfile
FROM continuumio/miniconda3
RUN conda install -c conda-forge rdkit
RUN pip install deepchem tensorflow scikit-learn
# ... etc
```

---

### Can I use K-talysticFlow in a web application?

**Yes!** You can integrate it as a backend:

```python
from bin.run_prediction import predict_activity

# API endpoint
@app.route('/predict', methods=['POST'])
def predict():
    smiles_list = request.json['smiles']
    predictions = predict_activity(smiles_list)
    return jsonify(predictions)
```

**Note:** Requires additional web framework (Flask, FastAPI, etc.)

---

## 📚 Scientific Questions

### What type of molecular activity can I predict?

K-talysticFlow is designed for **binary classification:**
- Active vs Inactive
- Toxic vs Non-toxic
- Binder vs Non-binder
- Hit vs Non-hit

**Examples:**
- Enzyme inhibitors
- Receptor agonists/antagonists
- Antimicrobial agents
- Cytotoxicity
- Blood-brain barrier permeability (BBB+/BBB-)

**Not suitable for:**
- Regression (e.g., IC50, Ki values) - use different tools
- Multi-class classification (A vs B vs C)
- Time-series predictions

---

### How does K-talysticFlow compare to other tools?

| Feature | KAST | DeepChem | AutoML | Commercial Tools |
|---------|------|----------|--------|------------------|
| **Ease of use** | ⭐⭐⭐⭐⭐ | ⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐⭐ |
| **Automation** | Full | Partial | Full | Full |
| **Validation suite** | Comprehensive | Basic | Good | Excellent |
| **Parallel processing** | ✅ | ❌ | ✅ | ✅ |
| **Cost** | Free | Free | Varies | $$$$ |
| **Customization** | High | Very High | Low | Medium |
| **Support** | Community | Community | Vendor | Vendor |

**When to use KAST:**
- Need full automation + flexibility
- Academic research
- Limited budget
- Want comprehensive validation

**When to use alternatives:**
- Need advanced features (e.g., 3D descriptors)
- Large-scale enterprise deployment
- Prefer GUI over CLI

---

### Can I publish results from K-talysticFlow?

**Yes!** K-talysticFlow is suitable for academic publication.

**Please cite:**
```bibtex
@software{kast2025,
  author = {Santos, Késsia Souza},
  title = {K-talysticFlow: Automated Deep Learning Pipeline for Molecular Screening},
  year = {2025},
  version = {1.0.0},
  url = {https://github.com/kelsouzs/kast}
}
```

---

### What are the limitations of K-talysticFlow?

**Current limitations:**

1. **Binary classification only** (not regression)
2. **2D fingerprints only** (no 3D descriptors yet)
3. **Single target** (not multi-task learning)
4. **No explicit chemistry rules** (not rule-based)
5. **Requires both active and inactive data**

**Future enhancements planned:**
- Regression models
- 3D descriptors
- Multi-target prediction
- Explainability (SHAP, attention)

---
**Still have questions?**
- 📧 Email: kelsouzs.uefs@gmail.com
- 🐛 GitHub Issues: [Report a problem](https://github.com/kelsouzs/kast/issues)

---

