# 🔬 Pipeline Steps - Detailed Documentation

Complete technical documentation of each K-talysticFlow pipeline step.

---

## 📋 Table of Contents

1. [Overview](#overview)
2. [Step 1: Data Preparation](#step-1--data-preparation)
3. [Step 2: Featurization](#step-2--featurization)
4. [Step 3: Model Training](#step-3--model-training)
5. [Step 4: Model Evaluation](#step-4--model-evaluation)
6. [Step 5-6: Prediction](#step-5-6--prediction)

---

## Step 1: 📋 Data Preparation

**Script:** `bin/1_preparation.py`  
**Menu Option:** `[1]`  
**Purpose:** Clean, validate, and split molecular data

### Input

- `data/actives.smi` - Active compounds
- `data/inactives.smi` - Inactive compounds

**Format:**
```
SMILES [space] optional_name
CC(C)Cc1ccc(cc1)C(C)C(O)=O  ibuprofen
```

---

### Process

#### 1.1 Data Loading

```python
# Read SMILES files
actives = pd.read_csv('actives.smi', sep='\t', header=None)
inactives = pd.read_csv('inactives.smi', sep='\t', header=None)
```

---

#### 1.2 SMILES Validation

**Checks performed:**
- Valid SMILES syntax (RDKit parsing)
- Length constraints (5-200 characters)
- Duplicate removal
- Sanitization

**Example validation:**
```python
from rdkit import Chem

def validate_smiles(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return False  # Invalid
    return True
```

---

#### 1.3 Data Balancing

Balances active/inactive ratio to prevent class imbalance:

**Strategies:**
- **If actives > inactives:** Undersample actives
- **If inactives > actives:** Undersample inactives
- **Target:** 1:1 ratio (configurable)

---

#### 1.4 Labeling

Assigns binary labels:
- **Actives:** `active = 1`
- **Inactives:** `active = 0`

**Output DataFrame:**
```
     smiles                          active
0    CC(C)Cc1ccc(cc1)C(C)C(O)=O     1
1    CCCCCCCCCCCCCCCC                0
```

---

#### 1.5 Scaffold Splitting

**Algorithm:** DeepChem's ScaffoldSplitter

**Purpose:** 
- Ensures train/test sets have different molecular scaffolds
- Better test of generalization
- More realistic than random splitting

**How it works:**
1. Generate Bemis-Murcko scaffold for each molecule
2. Group molecules by scaffold
3. Split scaffolds (not individual molecules) into train/test

**Visual:**
```
Training Set Scaffolds: [A, B, C]
Test Set Scaffolds:     [D, E]
```

**Parameters:**
- **Split ratio:** Selected interactively when running the script (80/20 recommended, or custom)
- **Random state:** 42 (for reproducibility)

---

### Output

**Files created:**
- `results/01_train_set.csv`
- `results/01_test_set.csv`

**Format:**
```csv
smiles,active
CC(C)Cc1ccc(cc1)C(C)C(O)=O,1
CN1C=NC2=C1C(=O)N(C)C(=O)N2C,1
CCCCCCCCCCCCCCCC,0
```

**Statistics logged:**
```
Total molecules: 10,000
Active compounds: 5,000
Inactive compounds: 5,000
Training set: 7,000 (3,500 active, 3,500 inactive)
Test set: 3,000 (1,500 active, 1,500 inactive)
```

---

### Configuration

Key settings in `settings.py`:

```python
# Train/test split: Selected interactively during script execution
RANDOM_STATE = 42
MIN_MOLECULES_PER_CLASS = 50
MAX_MOLECULES_TOTAL = 100000
MIN_SMILES_LENGTH = 5
MAX_SMILES_LENGTH = 200
```

---

### Troubleshooting

**Issue:** "Insufficient molecules"
```python
MIN_MOLECULES_PER_CLASS = 20  # Lower threshold
```

**Issue:** "Invalid SMILES"
- Check SMILES syntax
- Remove invalid entries manually

---

## Step 2: 🧬 Featurization

**Script:** `bin/2_featurization.py`  
**Menu Option:** `[2]`  
**Purpose:** Convert SMILES to numerical fingerprints

### Input

- `results/01_train_set.csv`
- `results/01_test_set.csv`

---

### Process

#### 2.1 Morgan Fingerprint Generation

**Algorithm:** Extended Connectivity Fingerprints (ECFP)

**Parameters:**
```python
FP_RADIUS = 3     # ECFP6 (radius × 2)
FP_SIZE = 2048    # Number of bits
```

**How it works:**
1. For each atom, identify circular substructures up to radius
2. Hash each substructure to a bit position
3. Set corresponding bits to 1 in 2048-bit vector

**Example:**
```
SMILES: CCO (ethanol)
Substructures at radius 3:
  - [CH3]-C
  - C-[CH2]-O
  - [CH2]-[OH]
→ Hashed to bits: [45, 234, 567, ...]
→ Fingerprint: [0,0,0,...,1(bit 45),...,1(bit 234),...,1(bit 567),...]
```

---

#### 2.2 Parallel Processing

**If enabled** (`ENABLE_PARALLEL_PROCESSING = True`):

```python
from joblib import Parallel, delayed

def generate_fp(smiles):
    mol = Chem.MolFromSmiles(smiles)
    fp = AllChem.GetMorganFingerprintAsBitVect(
        mol, radius=FP_RADIUS, nBits=FP_SIZE
    )
    return np.array(fp)

# Parallel execution
fingerprints = Parallel(n_jobs=N_WORKERS)(
    delayed(generate_fp)(smiles) for smiles in smiles_list
)
```

**Speedup:** 5-10x faster for large datasets

---

#### 2.3 DeepChem Dataset Creation

Converts numpy arrays to DeepChem format:

```python
dataset = dc.data.NumpyDataset(
    X=fingerprints,      # Feature matrix (N × 2048)
    y=labels,            # Labels (N × 1)
    ids=smiles_list      # Molecule IDs
)
```

---

#### 2.4 Sparse Matrix Optimization

For memory efficiency:

```python
from scipy.sparse import csr_matrix

# Convert to sparse matrix (most bits are 0)
X_sparse = csr_matrix(fingerprints)
```

**Memory saved:** ~80-90% for typical fingerprints

---

### Output

**Directory structure:**
```
results/featurized_datasets/
├── train/
│   ├── metadata.csv.gzip
│   ├── shard-0-ids.npy
│   ├── shard-0-w.npy
│   ├── shard-0-X.npy         # Fingerprints
│   ├── shard-0-y.npy         # Labels
│   └── tasks.json
└── test/
    └── (same structure)
```

**Log file:** `results/02_featurization_log.txt`

**Content:**
```
Featurization Log
=================
Date: 2025-10-10 14:30:22
Fingerprint Type: Morgan (ECFP)
Radius: 3
Size: 2048 bits

Training Set:
  Molecules: 7,000
  Time: 45.3 seconds
  Parallel: Yes (6 workers)

Test Set:
  Molecules: 3,000
  Time: 19.2 seconds
```

---

### Configuration

```python
FP_SIZE = 2048
FP_RADIUS = 3
ENABLE_PARALLEL_PROCESSING = True
N_WORKERS = None
PARALLEL_BATCH_SIZE = 100000
```

---

### Troubleshooting

**Issue:** Memory error
```python
PARALLEL_BATCH_SIZE = 50000
ENABLE_PARALLEL_PROCESSING = False
```

**Issue:** Too slow
```python
ENABLE_PARALLEL_PROCESSING = True
N_WORKERS = -1
```

---

## Step 3: 🤖 Model Training

**Script:** `bin/3_create_training.py`  
**Menu Option:** `[3]`  
**Purpose:** Train deep neural network classifier

### Input

- `results/featurized_datasets/train/`
- `results/featurized_datasets/test/` (for validation)

---

### Process

#### 3.1 Model Architecture

**Type:** Multi-Layer Perceptron (MLP)

**Default architecture:**
```
Input (2048) → Dense(1000) → Dropout(0.25) → Dense(500) → Dropout(0.25) → Output(1)
```

**Visual:**
```
[2048 bits] 
    ↓
[1000 neurons, ReLU] 
    ↓
[Dropout 25%]
    ↓
[500 neurons, ReLU]
    ↓
[Dropout 25%]
    ↓
[1 neuron, Sigmoid] → Probability
```

---

#### 3.2 Model Configuration

```python
MODEL_PARAMS = {
    'n_tasks': 1,                  # Binary classification
    'layer_sizes': [1000, 500],    # Hidden layer sizes
    'dropouts': 0.25,              # Regularization
    'learning_rate': 0.001,        # Adam optimizer
    'mode': 'classification',
    'nb_epoch': 50                 # Training epochs
}
```

---

#### 3.3 Loss Function

**Binary Cross-Entropy:**

$$\mathcal{L} = -\frac{1}{N}\sum_{i=1}^{N} [y_i \log(\hat{y}_i) + (1-y_i)\log(1-\hat{y}_i)]$$

Where:
- $y_i$ = true label (0 or 1)
- $\hat{y}_i$ = predicted probability
- $N$ = number of samples

---

#### 3.4 Training Loop

```python
model = dc.models.MultitaskClassifier(**MODEL_PARAMS)

for epoch in range(NB_EPOCH):
    # Train on batches
    loss = model.fit(train_dataset, nb_epoch=1)
    
    # Validate
    val_loss = model.evaluate(test_dataset)
    
    # Log progress
    print(f"Epoch {epoch+1}/{NB_EPOCH}: Loss={loss:.4f}")
```

---

#### 3.5 Optimization

**Optimizer:** Adam
- Adaptive learning rate
- Momentum-based
- Good for noisy gradients

**Regularization:**
- **Dropout:** Randomly disables 25% of neurons during training
- **Early stopping:** (optional, not default)

---

#### 3.6 Checkpointing

Saves model after training:

```python
model.save_checkpoint(model_dir='results/trained_model/')
```

---

### Output

**Files created:**

1. **Model checkpoint:**
   - `results/trained_model/checkpoint1.pt`
   - Contains trained weights

2. **Metadata:**
   - `results/training_metadata.json`
   ```json
   {
     "model_type": "MultitaskClassifier",
     "layer_sizes": [1000, 500],
     "dropouts": 0.25,
     "learning_rate": 0.001,
     "nb_epoch": 50,
     "training_date": "2025-10-10",
     "fingerprint_size": 2048,
     "fingerprint_radius": 3
   }
   ```

3. **Training log:**
   - `results/03_training_log.txt`
   ```
   Training Log
   ============
   Architecture: [1000, 500]
   Epochs: 50
   
   Epoch 1/50: Loss=0.6234
   Epoch 2/50: Loss=0.5423
   ...
   Epoch 50/50: Loss=0.1245
   
   Training completed in 8.3 minutes
   ```

---

### Configuration

```python
MODEL_PARAMS = {
    'layer_sizes': [1000, 500],
    'dropouts': 0.25,
    'learning_rate': 0.001,
    'nb_epoch': 50
}
```

**Customization examples:**

**Deeper network:**
```python
'layer_sizes': [2048, 1024, 512, 256]
```

**Prevent overfitting:**
```python
'dropouts': 0.5,
'nb_epoch': 30
```

---

### Troubleshooting

**Issue:** Loss not decreasing
```python
'learning_rate': 0.0001  # Reduce
```

**Issue:** Overfitting
```python
'dropouts': 0.5,         # Increase
'nb_epoch': 30           # Reduce
```

**Issue:** Training too slow
```python
'nb_epoch': 30           # Reduce epochs
'layer_sizes': [512, 256]  # Simpler model
```

---

## Step 4: 📊 Model Evaluation

**Scripts:** `bin/4_*.py`  
**Menu Option:** `[4]`  
**Purpose:** Comprehensive model validation

### 4.0 Main Evaluation

**Script:** `bin/4_0_evaluation_main.py`

#### Metrics Calculated

1. **ROC-AUC** (Receiver Operating Characteristic)
2. **Accuracy**
3. **Precision** (Positive Predictive Value)
4. **Recall** (Sensitivity)
5. **F1-Score**
6. **Confusion Matrix**

#### Output

`results/4_0_evaluation_report.txt`:
```
Model Evaluation Report
=======================
Test Set Performance:

ROC-AUC: 0.8523
Accuracy: 82.34%
Precision: 0.78
Recall: 0.85
F1-Score: 0.81

Confusion Matrix:
                Predicted Negative  Predicted Positive
Actual Negative        850                120
Actual Positive         80                450
```

---

### 4.1 Cross-Validation

**Script:** `bin/4_1_cross_validation.py`

#### Process

k-fold stratified cross-validation (default k=5):

```python
for fold in range(N_FOLDS):
    # Split data
    train_fold, val_fold = split(dataset, fold)
    
    # Train model
    model.fit(train_fold)
    
    # Evaluate
    auc_fold = evaluate(val_fold)
```

#### Output

`results/4_1_cross_validation_results.txt`:
```
5-Fold Cross-Validation Results
================================

Fold 1: AUC=0.83, Acc=0.78
Fold 2: AUC=0.85, Acc=0.81
Fold 3: AUC=0.82, Acc=0.79
Fold 4: AUC=0.84, Acc=0.80
Fold 5: AUC=0.86, Acc=0.82

Mean AUC: 0.84 ± 0.015
Mean Accuracy: 0.80 ± 0.015
```

---

### 4.2 Enrichment Factor

**Script:** `bin/4_2_enrichment_factor.py`

#### Formula

$$EF_x\% = \frac{\text{Actives in top } x\%}{\text{Total actives} \times x\%}$$

#### Output

`results/4_2_enrichment_factor_results.txt`:
```
Enrichment Factor Analysis
==========================

EF @ 1%:  15.2
EF @ 2%:  12.8
EF @ 5%:   8.5
EF @ 10%:  5.2

Interpretation:
- Testing top 1% yields 15.2× more actives than random
```

---

### 4.3 Tanimoto Similarity

**Script:** `bin/4_3_tanimoto_similarity.py`

#### Calculation

$$Tanimoto(A, B) = \frac{|A \cap B|}{|A \cup B|}$$

#### Output

- `results/4_3_tanimoto_similarity_results.txt`
- Similarity distribution plots

---

### 4.4 Learning Curve

**Script:** `bin/4_4_learning_curve.py`

#### Process

Train models with increasing data sizes:

```python
sizes = [500, 1000, 2000, 5000, 10000]
for size in sizes:
    train_subset = dataset[:size]
    model.fit(train_subset)
    auc = evaluate(test_set)
```

#### Output

- `results/4_4_learning_curve_results.txt`
- Learning curve plot (`learning_curve.png`)

---

## Step 5-6: 🔮 Prediction

### Step 5: Featurization for Prediction

**Script:** `bin/5_0_featurize_for_prediction.py`  
**Menu Option:** `[5]`

**Process:** Same as Step 2, but for new molecules

**Input:** `data/zinc_library.smi` (or other library)

**Output:** `results/prediction_featurized/`

---

### Step 6: Run Prediction

**Script:** `bin/5_1_run_prediction.py`  
**Menu Option:** `[6]`

#### Process

```python
# Load model
model = dc.models.MultitaskClassifier()
model.restore(checkpoint='results/trained_model/')

# Predict
predictions = model.predict(new_dataset)

# Calculate K-Prediction Score
k_scores = predictions[:, 1] * 100

# Rank
ranked = sort_by_score(k_scores)
```

#### Output

`results/5_1_new_molecule_predictions.csv`:
```csv
SMILES,Probability,K_Prediction_Score,Rank
CC(C)Cc1ccc(cc1)C(C)C(O)=O,0.9523,95.23,1
CN1C=NC2=C1C(=O)N(C)C(=O)N2C,0.8845,88.45,2
CCCCCCCCCCCCCCCC,0.1523,15.23,150
```

---

## 🎯 Complete Pipeline Flow

```
Raw Data (SMILES)
    ↓ [1_preparation.py]
Train/Test Split (CSV)
    ↓ [2_featurization.py]
Fingerprints (DeepChem format)
    ↓ [3_training.py]
Trained Model (checkpoint)
    ↓ [4_*.py]
Validation Reports
    ↓ [5_0_featurize_for_prediction.py]
New Molecule Fingerprints
    ↓ [5_1_run_prediction.py]
Ranked Predictions (CSV)
```

