# Overview

**K-talysticFlow** (KAST) is an automated pipeline for training, evaluating, and applying deep learning models to predict molecular bioactivity. It is designed for **virtual screening**, helping identify promising compounds from large chemical libraries in a fast, reproducible, and user-friendly workflow. 

---

## What Does KAST Do?

KAST automates a complete end-to-end workflow for molecular activity prediction:

1. **Prepare Data** — Clean, validate, and organize user-provided molecules in **SMILES** format, separating active and inactive compounds into training and test sets. 
2. **Generate Features** — Convert molecules into machine-learning-ready **ECFP/Morgan fingerprints**. 
3. **Train Model** — Build and train a deep learning model from scratch using the user’s featurized training data. KAST uses a **DeepChem-based neural network implemented with `MultitaskClassifier`**, with configurable hidden layers, dropout, and learning rate. 
4. **Evaluate** — Assess model quality using metrics and analyses such as **ROC-AUC, cross-validation, enrichment factor, Tanimoto similarity, and learning curves**. 
5. **Predict** — Screen new molecules and rank them according to their predicted probability of activity and the **K-Prediction Score**. 

---

## How the Learning Works

KAST does not rely on a fixed pretrained predictor. Instead, it **creates and trains a new model based on the user’s own dataset**, using labeled active/inactive compounds as supervision. This makes the workflow suitable for target-specific virtual screening, where model performance depends on the quality and representativeness of the user’s dataset. 

## How It Is Scored

The **K-Prediction Score** is derived from the output of the trained `MultitaskClassifier`. During the prediction phase, the network processes molecular fingerprints and generates a probability distribution across classes. The K-Prediction Score is defined as the scalar value representing the **predicted probability of the active class** ( \(P(active)\) ). These scores are utilized primarily for **ranking and prioritization** in virtual screening workflows, where the relative ordering of candidates provides a robust metric for experimental validation, even if absolute probability calibration is subject to the specific training dataset distribution. 

## How It Works (Simple Version)

```
Your Data (SMILES)
       ↓
   [STEP 1] Prepare & Split
       ↓
   [STEP 2] Featurize
       ↓
   [STEP 3] Train Model
       ↓
   [STEP 4] Evaluate
       ↓
   [STEP 5-6] Predict New
       ↓
   Results (CSV + Plots)
```

Each step is interactive — you choose options at each stage. No coding needed!

---

## Key Capabilities

| Feature | Benefit |
|---------|---------|
| **Interactive Menu** | Click through the workflow step-by-step |
| **Parallel Processing** | 5-10x faster on large datasets (100K+ molecules) |
| **Full Validation** | ROC/AUC, Cross-Validation, Enrichment Factor, Similarity |
| **K-Prediction Score** | Proprietary ranking score for molecules (0-1) |
| **One-Click Setup** | `setup.exe` (Windows) or `setup.sh` (Linux) handles everything |
| **Automatic Shortcuts** | Desktop shortcut to launch KAST without terminal |

---

## Who Should Use KAST?

✅ **Researchers in drug discovery**  
✅ **Computational chemists**  
✅ **Anyone screening chemical libraries**  
✅ **Students learning ML + chemistry**  

---

## What You'll Need

- ✅ SMILES file with active molecules (e.g., `actives.smi`)
- ✅ SMILES file with inactive molecules (e.g., `inactives.smi`)
- ✅ (Optional) New molecules to predict (e.g., `library.smi`)

**Format example:**
```
CC(C)Cc1ccc(cc1)C(C)C(O)=O  ibuprofen
CN1C=NC2=C1C(=O)N(C(=O)N2C)C  caffeine
```

---

## Tested Platforms

- ✅ **Windows 11** with Anaconda
- ✅ **Ubuntu 20.04 LTS** with Conda

Both platforms create desktop shortcuts and handle all conda setup automatically.

---

### Further Reading & Foundations
KAST is built upon foundational principles of chemoinformatics and machine learning. For deeper technical insights into the methodologies used:

- **Molecular Fingerprints (ECFP):** Rogers, D., & Hahn, M. (2010). Extended-Connectivity Fingerprints. *Journal of Chemical Information and Modeling*, 50(5), 742-754. [doi:10.1021/ci100050t](https://doi.org/10.1021/ci100050t)
- **Deep Learning in Drug Discovery:** Ramsundar, B., et al. (2019). *Deep Learning for the Life Sciences: Applying Deep Learning to Genomics, Microscopy, Drug Discovery, and More*. O'Reilly Media.
- **Imbalanced Learning in Chemistry:** Jiang, J., et al. (2025). A review of machine learning methods for imbalanced data challenges in chemistry. *Chemical Science*, 16, 7637–7658. [doi:10.1039/D5SC00270B](https://doi.org/10.1039/D5SC00270B)