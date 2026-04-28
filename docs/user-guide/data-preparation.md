# Data Preparation

How to format and prepare your molecular data for KAST.

---

## File Format: SMILES

SMILES (Simplified Molecular Input Line Entry System) is the standard text format for molecules.

**Format:**
```
SMILES [space/tab] optional_name
```

**Example:**
```
CC(C)Cc1ccc(cc1)C(C)C(O)=O  ibuprofen
CN1C=NC2=C1C(=O)N(C(=O)N2C)C  caffeine
CC(C)CC1=CC(=C(C=C1)C(C)C)O  another_compound
```

---

## File Organization

Place SMILES files in the `data/` folder:

```
KAST/
├── data/
│   ├── actives.smi        ← Active compounds (required)
│   ├── inactives.smi      ← Inactive compounds (required)
│   └── my_library.smi     ← (Optional) New molecules to predict
├── results/               ← Outputs will go here
├── main.py
└── ...
```

---

## Required Files

### data/actives.smi
Active (bioactive) compounds that show the desired property.

**Minimum:** 50 molecules (use with caution; see Data Quality Guidelines)
**Recommended:** 500+ molecules for Deep Learning accuracy

**Format:** SMILES string followed by a space or tab and the molecule name.
```text
CC(C)Cc1ccc(cc1)C(C)C(O)=O  ibuprofen
CN1C=NC2=C1C(=O)N(C(=O)N2C)C  caffeine
CC(C)CC1=CC(=C(C=C1)C(C)C)O  compound3
```

---

### data/inactives.smi
Inactive (non-bioactive) compounds that do not show the desired property.

**Minimum:** 50 molecules (use with caution; see Data Quality Guidelines)
**Recommended:** 500+ molecules for Deep Learning accuracy

**Format:** SMILES string followed by a space or tab and the molecule name.
```text
CCCCCCCCCCCCCCCC  hexadecane
CC(=O)OC1=CC=CC=C1C(=O)O  aspirin
CCC(C)C(O)=O  carboxylic_acid
```
---

## Optional File

### `data/my_library.smi`
New molecules to screen (predict activity).

Can have any name, used in Step 5-6.

Example:
```
CCc1ccccc1  ethylbenzene
Cc1ccccc1  toluene
CCC1=CC=CC=C1O  ethylphenol
```

---

## Data Quality Guidelines

### Good Practices
- Use **canonical SMILES** (standardized form)
- Remove **duplicates** before starting
- Validate SMILES structures with RDKit
- Balance active/inactive ratio (aim for 1:1 to 1:10)
- Aim for **500+ molecules per class** for reliable Deep Learning performance
- Maximum **100,000 molecules** per file

### ML/DL Warning (Overfitting Risk)
- **Minimum Threshold (50 molecules):** Technically accepted by the pipeline, but **strongly discouraged** for Deep Learning. With small datasets, models will likely **overfit**, memorizing the training data rather than learning chemical features.
- **Data Scaling:** If you have < 200 molecules per class, consider using simpler machine learning models (e.g., Random Forest or SVM) instead of deep neural networks to maintain predictive stability.

### Avoid
- Invalid SMILES (won't parse)
- Salts/mixtures (unless intended)
- Very small molecules (< 5 heavy atoms)
- Very large molecules (> 200 heavy atoms)
- Incomplete or corrupted files

---

## Quality Check

Before running KAST, validate your SMILES:

```bash
# Option 1: Use KAST's built-in check
python main.py
[1] Data Preparation
# It validates during preparation

# Option 2: Manual check with RDKit
python -c "
from rdkit import Chem
with open('data/actives.smi') as f:
    for line in f:
        smiles = line.split()[0]
        if Chem.MolFromSmiles(smiles) is None:
            print(f'Invalid SMILES: {smiles}')
"
```

---

## Balance Active/Inactive

**Recommended ratios:**
- 1:1 (50 active, 50 inactive) — Balanced (optimal for training)
- 1:5 (100 active, 500 inactive) — Realistic for many screening libraries
- 1:10 (100 active, 1000 inactive) — Highly imbalanced (requires more attention to metrics)

**Handling Imbalance:**
KAST incorporates strategies to manage imbalanced datasets within the pipeline. While the system can process significantly imbalanced data, maintaining a ratio between 1:1 and 1:10 is recommended for better model generalization and to avoid bias toward the majority class (Banerjee et al., 2018; Jiang et al., 2025).

---

## Large Datasets (100K+ molecules)

For very large libraries:
1. **Enable parallel processing** (see [Parallel Processing](parallel-processing.md))
2. **Increase `PARALLEL_BATCH_SIZE`** if you have 16GB+ RAM
3. **Use filtering** to reduce library size first
4. **Monitor RAM** during featurization

---

## Canonicalization

**Note:** KAST automatically canonicalizes all your SMILES during the `1_preparation.py` step. You do **not** need to canonicalize your files manually for the pipeline to work.

If you prefer to pre-process your files externally to ensure consistency before importing them into KAST, you can use the following RDKit snippet:

```bash
# Optional: Manual canonicalization using RDKit
python -c "
from rdkit import Chem
import sys

infile, outfile = sys.argv, sys.argv[1][2]
with open(infile) as f, open(outfile, 'w') as out:
    for line in f:
        parts = line.strip().split()
        smiles = parts
        name = parts if len(parts) > 1 else ''[1]
        mol = Chem.MolFromSmiles(smiles)
        if mol:
            canonical = Chem.MolToSmiles(mol)
            out.write(f'{canonical}  {name}\n' if name else f'{canonical}\n')
" data/actives.smi data/actives_canonical.smi
```

---

## References

1. **Banerjee et al. (2018).** [Prediction Is a Balancing Act...](https://doi.org/10.3389/fchem.2018.00362). *Frontiers in Chemistry*.
2. **Jiang et al. (2025).** [A review of machine learning methods for imbalanced data challenges in chemistry](https://doi.org/10.1039/D5SC00270B). *Chemical Science*.
---