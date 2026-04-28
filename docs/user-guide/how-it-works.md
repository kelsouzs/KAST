# Pipeline Overview

K-talysticFlow is a **5-step automated pipeline** for molecular screening. Understand what each step does.

---

## The 5 Steps

### Step 1: 📋 Data Preparation
**File:** `bin/1_preparation.py`

**What it does:**
- Reads active and inactive molecule files (SMILES format)
- Validates SMILES structures
- Removes invalid molecules
- Splits into train/test sets (e.g., 80/20)

**Input:** `data/actives.smi` + `data/inactives.smi`  
**Output:** `train_set.csv` + `test_set.csv`

---

### Step 2: 🧬 Featurization
**File:** `bin/2_featurization.py`

**What it does:**
- Converts SMILES into machine learning features (fingerprints)
- Uses Morgan fingerprints (ECFP4, radius=2)
- Generates numerical vectors from molecular structures
- **Supports parallel processing** (5-10x faster on large datasets)

**Input:** Train/test CSV files  
**Output:** Featurized numpy arrays

---

### Step 3: 🤖 Model Training
**File:** `bin/3_create_training.py`

**What it does:**
- Builds a neural network using DeepChem + TensorFlow
- Trains on fingerprint features
- Learns to predict molecular activity
- Ensures reproducibility with fixed random seeds

**Input:** Featurized training data  
**Output:** Trained model file + training metrics

---

### Step 4: 📊 Model Evaluation
**Menu Option:** `[4]` → choose evaluation type

**Available evaluations:**

| Evaluation | File | Purpose |
|-----------|------|---------|
| **Main Evaluation** | `4_0_evaluation_main.py` | ROC/AUC, accuracy, precision, recall, F1 |
| **Cross-Validation** | `4_1_cross_validation.py` | Verify model generalization (5-fold CV) |
| **Enrichment Factor** | `4_2_enrichment_factor.py` | How much better than random screening |
| **Tanimoto Similarity** | `4_3_tanimoto_similarity.py` | Structural diversity of predictions |
| **Learning Curve** | `4_4_learning_curve.py` | Model performance vs. training size |

**Input:** Trained model + test data  
**Output:** Plots + detailed metrics

---

### Steps 5: 🔮 Prediction

**Files:** `bin/5_0_featurize_for_prediction.py` + `bin/5_1_run_prediction.py`

---

#### Step 5.0 — Featurize for Prediction

New molecules are converted to the same **2048-bit sparse binary vector** representation
used during training (ECFP/Morgan fingerprints, radius=3). The resulting feature matrix
is stored in HDF5 format (`.h5`) as a **CSR (Compressed Sparse Row)** sparse matrix to
minimize memory footprint, since fingerprint vectors are typically over 95% sparse
(Rogers & Hahn, 2010).

Each molecule is represented as an input vector $\mathbf{x} \in \{0, 1\}^{2048}$,
where each bit encodes the presence or absence of a circular structural fragment.

---

#### Step 5.1 — Inference via the Trained MLP

KAST is **not a fine-tuning or transfer learning workflow**. The `MultitaskClassifier`
loaded here is the exact model trained from scratch in Step 3, restored from its saved
checkpoint via DeepChem's TensorFlow backend (Ramsundar et al., 2019).

**How the MLP works — in plain terms:**

Think of the network as a series of filters. Each hidden layer receives the output of
the previous one, applies a set of learned weights and biases, and passes only the
"activated" signals forward. This allows the model to learn increasingly abstract
patterns from the molecular fingerprint (Goodfellow, Bengio & Courville, 2016).

**Hidden layer computation:**

For each hidden layer $l$:

$$
\mathbf{h}^{(l)} = \text{ReLU}\!\left(\mathbf{W}^{(l)}\,\mathbf{h}^{(l-1)} + \mathbf{b}^{(l)}\right)
$$

where:
- $\mathbf{W}^{(l)}$ — weight matrix learned during training
- $\mathbf{b}^{(l)}$ — bias vector (shifts the activation threshold)
- $\text{ReLU}(z) = \max(0,\, z)$ — keeps only positive signals, introduces non-linearity

**Output layer — Softmax:**

The final layer converts raw scores into probabilities:

$$
\hat{\mathbf{y}} = \text{Softmax}\!\left(\mathbf{W}^{(L)}\,\mathbf{h}^{(L-1)} + \mathbf{b}^{(L)}\right)
$$

$$
\text{Softmax}(z_i) = \frac{e^{z_i}}{\displaystyle\sum_{j} e^{z_j}}, \quad j \in \{\text{inactive},\, \text{active}\}
$$

This ensures that $\hat{y}_0 + \hat{y}_1 = 1$, producing a valid probability distribution
over both classes (Bishop, 2006).

**K-Prediction Score:**

$$
\text{K-Score} = \hat{y}_1 = P(\text{active} \mid \mathbf{x})
$$

The score is the probability assigned to the **active class** (index 1 of the Softmax
output), and all molecules are ranked in descending order of this value.

Predictions are computed in **batches of 512 molecules**, with optional parallel dispatch
across CPU cores via `joblib`. Each worker loads its own model instance to avoid
TensorFlow serialization issues.

#### Output — User-Controlled Export

After all predictions are computed, the full results DataFrame is generated and the user
interactively selects how to export:

| Export Option | Description |
|---|---|
| All molecules | Complete ranked CSV |
| High activity only (score ≥ 0.7) | Confident hits only |
| Medium to high (score ≥ 0.5) | Broader candidate set |
| Custom score cutoff | User-defined threshold |
| Top N molecules | Fixed number of top candidates |

**Three output files are saved atomically** (via temporary files + safe move):

| File | Contents |
|---|---|
| `<name>.csv` | Full ranked predictions (SMILES, molecule name, K-Score) |
| `<name>_report.txt` | Score distribution summary + top 20 candidates |
| `<name>.smi` | SMILES file of selected molecules for downstream use |

**Input:** `data/my_library.smi`  
**Output:** user-named files in `results/`

---

## Data Flow

```
Active & Inactive          
    ↓
[1] Data Prep
    ↓
Train Set | Test Set
    ↓
[2] Featurize
    ↓
Feature Vectors
    ↓
[3] Train Model
    ↓
Neural Network
    ↓
[4] Evaluate
    ↓
Metrics & Plots
    ↓
        ↓ New Molecules
        ↓
    [5] Featurize New
        ↓
    [6] Predict
        ↓
    Ranked Results (CSV)
```
---

## Further Reading & Foundations

 - Rogers, D. & Hahn, M. (2010). Extended-connectivity fingerprints. *J. Chem. Inf. Model.*, 50(5), 742–754. https://doi.org/10.1021/ci100050t
 - Ramsundar, B., Eastman, P., Walters, P., Pande, V., Leswing, K. & Wu, Z. (2019). *Deep Learning for the Life Sciences*. O'Reilly Media. — DeepChem framework & `MultitaskClassifier`
 - Wu, Z., Ramsundar, B., Feinberg, E. N., Gomes, J., Geniesse, C., Pappu, A. S., Leswing, K. & Pande, V. (2018). MoleculeNet: a benchmark for molecular machine learning. *Chemical Science*, 9(2), 513–530. https://doi.org/10.1039/C7SC02664A
 - Goodfellow, I., Bengio, Y. & Courville, A. (2016). *Deep Learning*. MIT Press. https://www.deeplearningbook.org
 - Bishop, C. M. (2006). *Pattern Recognition and Machine Learning*. Springer.