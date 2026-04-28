# Step-by-Step Guide

Detailed walkthrough of each pipeline step.

---

## Before You Start

✅ Have you completed [Installation](../getting-started/installation.md)?  
✅ Do you have `data/actives.smi` and `data/inactives.smi` ready?  
✅ Check [Data Preparation](data-preparation.md) for file format

---

## Step 1️⃣: Data Preparation

**Menu Option:** `[1]`  
**Script:** `bin/1_preparation.py`

### What Happens
- Reads active and inactive molecule files
- Validates SMILES structures
- Removes invalid/malformed entries
- Splits into train/test sets

### Input
```
data/actives.smi
data/inactives.smi
```

### Menu Interaction
```
[1] 📋 Data Preparation (Split Train/Test)

Select the train/test split ratio (0-100, default 80): 80
Loading actives.smi... ✅ Loaded 500 active compounds
Loading inactives.smi... ✅ Loaded 500 inactive compounds

Validating SMILES...
✅ 998 valid molecules retained
❌ 2 invalid molecules removed

Splitting into train (80%) and test (20%)...
Train set: 798 molecules (399 active, 399 inactive)
Test set: 200 molecules (100 active, 100 inactive)

Saving to results/...
✅ Complete!
```

### Output Files
```
results/
├── 01_train_set.csv      # 798 molecules for training
└── 01_test_set.csv       # 200 molecules for testing
```

---

## Step 2️⃣: Featurization

**Menu Option:** `[2]`  
**Script:** `bin/2_featurization.py`

### What Happens
- Converts SMILES into Morgan fingerprints
- Generates 2048-dimensional binary vectors
- Creates ML-ready feature representations
- **Supports parallel processing** (5-10x faster!)

### Menu Interaction
```
[2] 🧬 Featurize Molecules (Generate Fingerprints)

Featurizing training set...
Processing 798 molecules...
Parallel Processing ENABLED (6 workers)
Progress: [████████████████] 100% 2m 15s

Featurizing test set...
Processing 200 molecules...
Progress: [████████████████] 100% 0m 45s

✅ Complete! Features saved.
```

### Output
Fingerprint data saved internally (featurized numpy arrays)

### Pro Tips
- Enable [Parallel Processing](parallel-processing.md) for datasets > 10K molecules
- This step is usually the bottleneck for large datasets

---

## Step 3️⃣: Model Training

**Menu Option:** `[3]`  
**Script:** `bin/3_create_training.py`

### What Happens
- Builds a neural network using DeepChem + TensorFlow
- Trains on featurized training data
- Learns patterns in active/inactive molecules
- Ensures reproducibility with fixed seeds

### Menu Interaction
```
[3] 🤖 Train Model

Building neural network...
- Input: 2048 features (Morgan fingerprints)
- Hidden layers: [512, 256, 128]
- Output: Binary (active/inactive)

Training reproducibility configured (seeds: Python, NumPy, TensorFlow)

Starting training...
Epoch 1/50: loss=0.65, val_loss=0.62
Epoch 10/50: loss=0.42, val_loss=0.40
Epoch 25/50: loss=0.28, val_loss=0.31
Epoch 50/50: loss=0.18, val_loss=0.22

✅ Model trained! Saved to results/model.pkl
```

### Output Files
```
results/
├── model.pkl             # Trained neural network
└── training_metrics.txt  # Training history
```

---

## Step 4️⃣: Model Evaluation

**Menu Option:** `[4]` → Choose evaluation type

### 4.1: Main Evaluation
**Script:** `4_0_evaluation_main.py`

```
Evaluating on test set (200 molecules)...

Results:
  ROC-AUC Score: 0.87
  Accuracy: 0.85
  Sensitivity (TPR): 0.82
  Specificity (TNR): 0.88
  Precision: 0.86
  Recall: 0.82
  F1-Score: 0.84

✅ Report saved: 4_0_evaluation_report.txt
```

### 4.2: Cross-Validation
**Script:** `4_1_cross_validation.py`

```
Running 5-fold cross-validation...

Fold 1/5: AUC=0.85, Accuracy=0.84
Fold 2/5: AUC=0.86, Accuracy=0.85
Fold 3/5: AUC=0.87, Accuracy=0.86
Fold 4/5: AUC=0.88, Accuracy=0.87
Fold 5/5: AUC=0.86, Accuracy=0.85

Mean AUC: 0.864 ± 0.011
Mean Accuracy: 0.854 ± 0.011

✅ Results saved
```

### 4.3: Enrichment Factor
**Script:** `4_2_enrichment_factor.py`

```
Calculating enrichment factor...
- 10% screened: 3.2x better than random
- 20% screened: 2.1x better than random
- 50% screened: 1.5x better than random

✅ Plot saved: enrichment_curve.png
```

### 4.4: Tanimoto Similarity
**Script:** `4_3_tanimoto_similarity.py`

```
Computing molecular similarity...
Processing 200 test molecules...
Parallel Processing ENABLED (6 workers)

Mean Similarity: 0.45
Min Similarity: 0.12
Max Similarity: 0.89

✅ Similarity matrix saved
```

### 4.5: Learning Curve
**Script:** `4_4_learning_curve.py`

```
Generating learning curves...
Training sizes: [50, 100, 200, 400, 798]

Training AUC:  [0.70, 0.78, 0.82, 0.85, 0.88]
Validation AUC: [0.68, 0.76, 0.81, 0.84, 0.87]

Plot saved: learning_curve.png
```

---

## Step 5️⃣: Featurize New Molecules

**Menu Option:** `[5]`  
**Script:** `bin/5_0_featurize_for_prediction.py`

### What Happens
- Converts your screening library into fingerprints
- Uses same parameters as training
- Prepares data for prediction

### Menu Interaction
```
[5] 🔮 Featurize New Molecules for Prediction

Select library file to featurize
Available files in data/:
  1. my_library.smi (1,500 molecules)
  2. zinc_subset.smi (5,000 molecules)

Enter choice: 1

Processing 1,500 molecules...
Parallel Processing ENABLED (6 workers)
Progress: [████████████████] 100% 1m 30s

✅ Featurized! Ready for prediction.
```

### Output
Featurized library saved internally

---

## Step 6️⃣: Run Predictions

**Menu Option:** `[6]`  
**Script:** `bin/5_1_run_prediction.py`

### What Happens
- Runs trained model on new molecules
- Generates K-Prediction Score (0-1)
- Ranks results by predicted activity
- Exports CSV

### Menu Interaction
```
[6] 🎯 Run Predictions on New Molecules

Select featurized library: my_library_featurized.pkl
Running predictions on 1,500 molecules...
Progress: [████████████████] 100% 0m 45s

Prediction Results:
  Total: 1,500 molecules
  Predicted Active (score > 0.5): 437 molecules
  Predicted Inactive (score ≤ 0.5): 1,063 molecules

Ranking by K-Prediction Score...

Export options:
  [1] Top 100 predictions
  [2] All predictions
  [3] Custom threshold

Enter choice: 1

Exporting top 100 to CSV...
✅ Saved: results/05_new_molecule_predictions.csv
```

### Output Files
```
results/
└── 05_new_molecule_predictions.csv

Contents:
SMILES                              K-Score  Predicted_Class
CCc1ccc(cc1)O                       0.94     Active
Cc1ccccc1C                          0.92     Active
CCCc1ccccc1O                        0.87     Active
```

---

## Understanding Your Results

✅ **Good Model:** AUC > 0.8, Accuracy > 80%  
⚠️ **Decent Model:** AUC 0.7-0.8, Accuracy 70-80%  
❌ **Poor Model:** AUC < 0.7, Accuracy < 70%

If results are poor:
→ Check [Troubleshooting](../support/troubleshooting.md)  
→ Review [Data Preparation](data-preparation.md)

---

## Next Steps

→ **[Understanding Outputs](outputs.md)** — Interpret results  
→ **[Parallel Processing](parallel-processing.md)** — Speed things up
