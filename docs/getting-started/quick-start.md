# Quick Start

Get KAST running and complete your first analysis in **5 minutes**.

---

## Step 1: Launch KAST

### **Windows:**

**Option A - Desktop Shortcut (Recommended):**
- Click the **"K-talysticFlow"** shortcut on your desktop

**Option B - Command Line:**
```bash
conda activate ktalysticflow
python main.py
```

### **Linux/macOS:**

**Option A - Command Line (Recommended & Easiest):**
```bash
kast
```
This command works from **any terminal, any directory**.

**Option B - Desktop Menu:**
- Search for **"K-talysticFlow"** in your application menu
- Click to launch

**Option C - Manual:**
```bash
conda activate ktalysticflow
cd /path/to/KAST
python main.py
```

---

## Step 2: The Main Menu

You'll see:
```
╔════════════════════════════════════════╗
║  Control Panel - KAST                 ║
╚════════════════════════════════════════╝

  [1] Prepare and Split Data
  [2] Generate Fingerprints
  [3] Create and Train the Model
  [4] Evaluate the Model
  [5] Predict Activity of New Molecules

  [8] Advanced Options (Testing & Configuration)
  [9] About & How to Cite
  [0] Exit Program

Enter your choice number: 
```

---

## Step 3: Prepare Your Data

**Create two SMILES files in the `data/` folder:**

`data/actives.smi`:
```
CC(C)Cc1ccc(cc1)C(C)C(O)=O  ibuprofen
CN1C=NC2=C1C(=O)N(C(=O)N2C)C  caffeine
```

`data/inactives.smi`:
```
CCCCCCCCCCCCCCCC  hexadecane
CC(=O)OC1=CC=CC=C1C(=O)O  aspirin
```

**Format:** `SMILES [space] optional_name`

---

## Step 4: Run the Pipeline (Sequential Steps)

**Press `[1]`** → Prepare and split data
**Press `[2]`** → Generate molecular fingerprints  
**Press `[3]`** → Create and train the model  
**Press `[4]`** → Evaluate model performance  
**Press `[5]`** → Make predictions on new molecules

---

## Next Steps

- **Check environment:** `[8]` → `[1]`
- **View results:** Check the `results/` folder
- **Help:** `[9]` for credits and documentation

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


