# 📖 User Manual

Complete guide to using K-talysticFlow for molecular activity prediction.

---

## 📋 Table of Contents

1. [Quick Start](#-quick-start)
2. [Preparing Your Data](#-preparing-your-data)
3. [Running the Pipeline](#-running-the-pipeline)
4. [Menu Options](#-menu-options)
5. [Understanding Outputs](#-understanding-outputs)
6. [Best Practices](#-best-practices)

---

## 🚀 Quick Start

After completing [Installation](Installation), launch the control panel:

```bash
# From the project directory:
python main.py
```

**You'll see the main menu:**
```
=========================================================
    🧬 K-talysticFlow (KAST) Control Panel v1.0.0
=========================================================
  K-atalystic Automated Screening Taskflow
  Automated Deep Learning for Molecular Screening
---------------------------------------------------------

[1] 📋 Data Preparation (Split Train/Test)
[2] 🧬 Featurize Molecules (Generate Fingerprints)
[3] 🤖 Train Model
[4] 📊 Evaluate Model (Multiple Options)
[5] 🔮 Featurize New Molecules for Prediction
[6] 🎯 Run Predictions on New Molecules
[7] 🚀 Run Complete Pipeline (Steps 1-4)
[8] ⚙️  Advanced Options
[0] ❌ Exit

⚡ Parallel Processing: ENABLED (6 workers)
---------------------------------------------------------
Enter your choice: 
```

👉 **Next Section:** [📂 Preparing Your Data](#-preparing-your-data)

---

## 📂 Preparing Your Data

### Data Format

K-talysticFlow requires SMILES format files:

#### Active Compounds (`actives.smi`)
```
CC(C)Cc1ccc(cc1)C(C)C(O)=O  ibuprofen
CN1C=NC2=C1C(=O)N(C(=O)N2C)C  caffeine
```

#### Inactive Compounds (`inactives.smi`)
```
CC(=O)OC1=CC=CC=C1C(=O)O  aspirin
CCCCCCCCCCCCCCCC  hexadecane
```

### File Requirements

- **Format:** `.smi` or `.smiles`
- **Structure:** `SMILES [space] optional_name`
- **Location:** Place files in `data/` folder
- **Names:** 
  - `actives.smi` for active compounds
  - `inactives.smi` for inactive compounds
  - `zinc_library.smi` (or any name) for prediction

### Data Quality Guidelines

✅ **Good practices:**
- Use canonicalized SMILES
- Remove duplicates
- Validate structures
- Balance active/inactive ratio (1:1 to 1:10)
- Minimum 50 molecules per class
- Maximum 100,000 molecules total

❌ **Avoid:**
- Invalid SMILES
- Salts/mixtures (unless intended)
- Very small molecules (< 5 atoms)
- Very large molecules (> 200 atoms)

---

## 🔄 Running the Pipeline

### Option 1: Complete Pipeline (Recommended for First Use)

**Menu Option [7]** - Runs all steps automatically:

1. Data Preparation
2. Featurization
3. Model Training
4. Full Evaluation Suite

```bash
Enter your choice: 7
```

**What happens:**
- ✅ Splits data into train/test sets (you'll choose the ratio interactively)
- ✅ Generates molecular fingerprints
- ✅ Trains neural network model
- ✅ Runs all validation tests
- ⏱️ Time: ~10-30 minutes (depends on dataset size)

---

### Option 2: Step-by-Step Workflow

For more control, run each step individually:

#### Step 1: Data Preparation [1]

```bash
Enter your choice: 1
```

**What it does:**
- Reads `actives.smi` and `inactives.smi`
- Validates SMILES structures
- Balances dataset
- Splits using Scaffold Splitting (70/30)
- Labels: active=1, inactive=0

**Outputs:**
- `results/01_train_set.csv`
- `results/01_test_set.csv`

**⏱️ Time:** 1-5 minutes

---

#### Step 2: Featurization [2]

```bash
Enter your choice: 2
```

**What it does:**
- Converts SMILES to Morgan Fingerprints (ECFP)
- Radius: 3, Size: 2048 bits
- Uses parallel processing (5-10x faster)
- Creates DeepChem datasets

**Outputs:**
- `results/featurized_datasets/train/`
- `results/featurized_datasets/test/`
- `results/02_featurization_log.txt`

**⏱️ Time:** 2-10 minutes (parallel) | 10-60 minutes (sequential)

---

#### Step 3: Model Training [3]

```bash
Enter your choice: 3
```

**What it does:**
- Trains Multi-Layer Perceptron (MLP)
- Architecture: Input → 1000 → 500 → Output
- 50 epochs, dropout 0.25
- TensorFlow backend

**Outputs:**
- `results/trained_model/checkpoint1.pt`
- `results/training_metadata.json`
- `results/03_training_log.txt`

**⏱️ Time:** 5-20 minutes

---

#### Step 4: Model Evaluation [4]

```bash
Enter your choice: 4
```

**Submenu appears:**
```
[1] Main Evaluation Report (AUC, Accuracy, etc.)
[2] Cross-Validation
[3] Enrichment Factor
[4] Tanimoto Similarity Analysis
[5] Learning Curve Generation
[6] Run All Evaluation Scripts
[0] Back to Main Menu
```

**Recommended:** Choose [6] to run all evaluations

**Outputs:**
- `results/4_0_evaluation_report.txt`
- `results/4_1_cross_validation_results.txt`
- `results/4_2_enrichment_factor_results.txt`
- `results/4_3_tanimoto_similarity_results.txt`
- `results/4_4_learning_curve_results.txt`
- Various plots in `results/`

**⏱️ Time:** 10-40 minutes

---

#### Step 5-6: Predictions on New Molecules

##### Step 5: Featurize New Molecules [5]

```bash
Enter your choice: 5
```

**Requirements:**
- Place your SMILES file in `data/` folder
- Update `PREDICTION_SMILES_FILE` in `settings.py` if needed

**What it does:**
- Reads new molecules
- Generates fingerprints (same parameters as training)
- Prepares for prediction

**Outputs:**
- `results/prediction_featurized/`
- `results/5_0_featurization_report.txt`

**⏱️ Time:** 1-15 minutes (depends on dataset size)

---

##### Step 6: Run Predictions [6]

```bash
Enter your choice: 6
```

**What it does:**
- Loads trained model
- Predicts activity for each molecule
- Calculates **K-Prediction Score**
- Ranks molecules by score

**Outputs:**
- `results/5_1_new_molecule_predictions.csv`

**⏱️ Time:** 1-5 minutes

**CSV Format:**
```csv
SMILES,Probability,K_Prediction_Score,Rank
CC(C)Cc1ccc(cc1)C(C)C,0.89,89.0,1
CN1C=NC2=C1C(=O)N,0.45,45.0,2
...
```

---

## ⚙️ Advanced Options [8]

```bash
Enter your choice: 8
```

**Submenu:**
```
[1] 🔍 Check Environment & Dependencies
[2] 🧪 Test Parallel Processing Compatibility
[3] ⚡ Configure Parallel Processing Workers
[0] ⬅️  Back to Main Menu
```

### [1] Check Dependencies

Verifies all required packages are installed correctly.

**Use when:**
- First installation
- After updating packages
- Troubleshooting errors

---

### [2] Test Parallel Processing

Runs 6 comprehensive tests:
1. ✅ Basic parallel execution
2. ✅ Large array processing
3. ✅ Memory efficiency
4. ✅ Error handling
5. ✅ Performance benchmark
6. ✅ Worker scaling

**Use when:**
- Configuring optimal workers
- Diagnosing performance issues
- Verifying multi-core support

---

### [3] Configure Workers

Adjust CPU core allocation on-the-fly without editing files.

**Example:**
```
Current: N_WORKERS = 6
Enter new value (None/auto, -1/all, or number): 4
✅ Updated to 4 workers
```

---

## 📊 Understanding Outputs

### Key Output Files

| File | Description |
|------|-------------|
| `01_train_set.csv` | Training molecules with labels |
| `01_test_set.csv` | Test molecules with labels |
| `4_0_evaluation_report.txt` | Main performance metrics |
| `4_0_test_predictions.csv` | Test set predictions |
| `5_1_new_molecule_predictions.csv` | **Final ranked predictions** |

### Understanding K-Prediction Score

**K-Prediction Score = Probability × 100**

- **Score 90-100:** Very likely active (high confidence)
- **Score 70-89:** Likely active (medium-high confidence)
- **Score 50-69:** Possibly active (medium confidence)
- **Score 30-49:** Possibly inactive (medium-low confidence)
- **Score 0-29:** Likely inactive (low confidence)

**Interpretation:**
- Focus on top-ranked molecules (highest scores)
- Scores > 70 are good candidates for experimental validation
- Consider enrichment factor when prioritizing hits

**See [Output Analysis](Output-Analysis) for detailed interpretation.**

---

## 🎯 Best Practices

### ✅ Do's

1. **Always run validation suite** before predictions
2. **Check evaluation metrics** (AUC > 0.7 is good)
3. **Use balanced datasets** (similar active/inactive counts)
4. **Enable parallel processing** for large datasets
5. **Review logs** in `results/logs/` for errors
6. **Backup your model** in `results/trained_model/`
7. **Document your workflow** and parameter changes

### ❌ Don'ts

1. **Don't skip validation steps** - you need to know model quality
2. **Don't use very imbalanced data** (e.g., 1:100 ratio)
3. **Don't ignore low AUC scores** (< 0.6 = poor model)
4. **Don't modify trained model files** manually
5. **Don't delete featurized datasets** if retraining
6. **Don't run multiple instances** on same results folder

---

## 🔄 Typical Workflows

### Workflow A: New Project

```
1. Prepare data files → Place in data/
2. Run Option [7] → Complete Pipeline
3. Check results → Review metrics
4. If AUC > 0.7 → Proceed to predictions
5. Run Options [5] + [6] → Predict new molecules
6. Analyze outputs → Select top hits
```

---

### Workflow B: Model Optimization

```
1. Run initial pipeline → Option [7]
2. Check learning curve → Option [4][5]
3. Adjust parameters in settings.py → If needed
4. Retrain → Option [3]
5. Re-evaluate → Option [4][6]
6. Compare metrics → Iterate if necessary
```

---

### Workflow C: Batch Predictions

```
1. Train model once → Options [1][2][3]
2. Validate thoroughly → Option [4][6]
3. For each new library:
   a. Place .smi file in data/
   b. Update settings.py
   c. Run Options [5] + [6]
   d. Collect predictions
```

---

## 🕒 Time Estimates

| Task | Small Dataset | Medium Dataset | Large Dataset |
|------|--------------|---------------|---------------|
|  | (< 1K molecules) | (1K-10K) | (10K-100K) |
| Preparation | < 1 min | 1-2 min | 2-5 min |
| Featurization | 1-2 min | 5-10 min | 10-30 min |
| Training | 2-5 min | 5-10 min | 10-20 min |
| Evaluation | 5-10 min | 10-20 min | 20-40 min |
| Prediction | < 1 min | 1-5 min | 5-15 min |

*Times assume parallel processing enabled with 4-8 cores*

---

## 📞 Need Help?

- **[FAQ](FAQ)** - Frequently asked questions
- **[Troubleshooting](Troubleshooting)** - Common issues
- **[Configuration Guide](Configuration)** - Customize settings
- **GitHub Issues** - Report bugs
- **Email:** kelsouzs.uefs@gmail.com

---

<div align="center">
<p>← <a href="Home">Back to Wiki Home</a> | <a href="Pipeline-Steps">Next: Pipeline Steps →</a></p>
</div>
