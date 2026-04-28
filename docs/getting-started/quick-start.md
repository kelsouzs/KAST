# Quick Start

Get KAST running and complete your first analysis in **5 minutes**.

---

## Step 1: Launch KAST

**Windows:**
- Click desktop shortcut **"K-talysticFlow 1.0.0"**
- Or double-click `run_kast.bat`

**Linux:**
- Search for **"K-talysticFlow"** in your app menu
- Or run: `conda activate ktalysticflow && python main.py`

---

## Step 2: The Main Menu

You'll see:
```
=========================================================
    🧬 K-talysticFlow (KAST) Control Panel v1.0.0
=========================================================

[1] 📋 Data Preparation (Split Train/Test)
[2] 🧬 Featurize Molecules (Generate Fingerprints)
[3] 🤖 Train Model
[4] 📊 Evaluate Model
[5] 🔮 Featurize New Molecules for Prediction
[6] 🎯 Run Predictions on New Molecules
[7] 📊 View Results & Statistics
[8] ⚙️  Advanced Options (Check env, test parallel, etc)
[0] ❌ Exit

Enter your choice: 
```

---

## Step 3: Prepare Your Data

**Create two SMILES files in the `data/` folder:**

`data/actives.smi`:
```
CC(C)Cc1ccc(cc1)C(C)C(O)=O  ibuprofen
CN1C=NC2=C1C(=O)N(C(=O)N2C)C  caffeine
CC(C)CC1=CC(=C(C=C1)C(C)C)O  ibuprofen
```

`data/inactives.smi`:
```
CCCCCCCCCCCCCCCC  hexadecane
CC(=O)OC1=CC=CC=C1C(=O)O  aspirin
CCC(C)C(O)=O  2-methylbutanoic acid
```

**Format:** `SMILES [space] optional_name`

---

## Step 4: Run the Pipeline

**In the menu, press `[1]`** to start:

```
[1] 📋 Data Preparation
Enter percentage for training set (default 80): 80
Loading actives.smi... ✅ Loaded 500 active compounds
Loading inactives.smi... ✅ Loaded 500 inactive compounds
Creating train/test split...
```

**Then press `[2]`** to featurize:
```
[2] 🧬 Featurize Molecules
Processing molecules with RDKit...
Generating Morgan fingerprints (radius=2)...
```

**Then press `[3]`** to train the model:
```
[3] 🤖 Train Model
Building neural network...
Training on 800 molecules...
Epochs: [████████████████] 100%
Training complete! ✅
```

**Then press `[4]`** to evaluate:
```
[4] 📊 Evaluate Model
ROC-AUC Score: 0.85
Accuracy: 0.82
Cross-Validation F1: 0.83
...
Report saved to results/4_0_evaluation_report.txt
```

---

## Step 5: Predict New Molecules

**Place your library in `data/my_library.smi`:**
```
CCC1=CC=CC=C1  ethylbenzene
CC1=CC=CC=C1  toluene
CCc1ccccc1O  o-ethylphenol
```

**In the menu, press `[5]`** to prepare new data:
```
[5] 🔮 Featurize New Molecules
Select file to featurize: my_library.smi
Processing 3 new molecules...
```

**Then press `[6]`** to predict:
```
[6] 🎯 Run Predictions
Select featurized file: my_library_featurized.pkl
Running predictions on 3 molecules...
Results saved to results/05_new_molecule_predictions.csv
```

---

## Step 6: Check Your Results

All outputs saved to `results/` folder:

```
results/
├── 01_train_set.csv                 # Training data
├── 01_test_set.csv                  # Test data
├── 4_0_evaluation_report.txt        # Metrics (AUC, accuracy, etc)
├── 4_1_cross_validation_results.txt # Cross-val scores
├── 05_new_molecule_predictions.csv  # Predicted molecules + K-scores
├── plots/
│   ├── roc_curve.png               # ROC plot
│   └── learning_curve.png          # Model learning progression
└── logs/
    └── kast_YYYYMMDD.log           # Detailed log file
```

**Open predictions CSV:**
```
SMILES                              K-Score  Predicted_Class
CCC1=CC=CC=C1                       0.92     Active
CC1=CC=CC=C1                        0.45     Inactive
CCc1ccccc1O                         0.78     Active
```

## 🎯 You're Done!

You've completed:
- ✅ Data preparation
- ✅ Featurization
- ✅ Model training
- ✅ Evaluation
- ✅ Predictions


