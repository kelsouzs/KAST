# Understanding Outputs

How to read and interpret KAST results.

---

## Output Folder Structure

All results saved to `results/` folder:

```
results/
├── 01_train_set.csv                     # Training data
├── 01_test_set.csv                      # Test data
├── model.pkl                            # Trained neural network
├── 4_0_evaluation_report.txt           # Main metrics
├── 4_1_cross_validation_results.txt    # Cross-val scores
├── 4_2_enrichment_factor_results.txt   # Enrichment analysis
├── 4_3_tanimoto_similarity_results.txt # Similarity analysis
├── 4_4_learning_curve_results.txt      # Learning curve data
├── 05_new_molecule_predictions.csv     # Prediction results
├── plots/
│   ├── roc_curve.png
│   ├── learning_curve.png
│   ├── enrichment_curve.png
│   └── similarity_heatmap.png
└── logs/
    └── kast_20251028.log               # Detailed execution log
```

---

## CSV Files

### `01_train_set.csv` & `01_test_set.csv`

**Columns:**
- `SMILES` — Molecular structure
- `Label` — 1 (active) or 0 (inactive)
- `Name` — Compound name (if provided)

**Example:**
```
SMILES,Label,Name
CC(C)Cc1ccc(cc1)C(C)C(O)=O,1,ibuprofen
CN1C=NC2=C1C(=O)N(C(=O)N2C)C,1,caffeine
CCCCCCCCCCCCCCCC,0,hexadecane
```

### `05_new_molecule_predictions.csv`

**Columns:**
- `SMILES` — Molecular structure
- `K-Score` — Prediction score (0.0-1.0)
- `Predicted_Class` — "Active" or "Inactive"

**Example:**
```
SMILES,K-Score,Predicted_Class
CCc1ccccc1O,0.94,Active
Cc1ccccc1C,0.92,Active
CCCc1ccccc1O,0.87,Active
Cc1ccc(cc1)C,0.45,Inactive
CCc1ccccc1,0.22,Inactive
```

**Interpretation:**
- **K-Score 0.9-1.0** → Very likely active 
- **K-Score 0.7-0.9** → Likely active 
- **K-Score 0.5-0.7** → Uncertain 
- **K-Score 0.0-0.5** → Likely inactive

The K-Prediction Score represents the predicted probability of the active class P(active).
In virtual screening workflows, probability-based scores are primarily used for ranking
and prioritization rather than as absolute estimates, as discriminative power is generally
more relevant than probability calibration for hit selection (Truchon & Bayly, 2007).
---

## Metrics Files

### `4_0_evaluation_report.txt`

Main evaluation metrics on test set:

```
ROC-AUC Score: 0.87
Accuracy: 0.85
Sensitivity (Recall): 0.82
Specificity: 0.88
Precision: 0.86
F1-Score: 0.84
```

**Interpretation:**
| Metric | What It Means | Good Value |
|--------|-------------|-----------|
| **ROC-AUC** | Overall model performance (0-1) | > 0.8 |
| **Accuracy** | % correct predictions | > 80% |
| **Sensitivity** | % of actives found | > 80% |
| **Specificity** | % of inactives correctly rejected | > 80% |
| **Precision** | % of predictions that are correct | > 80% |
| **F1-Score** | Balance between precision & recall | > 0.8 |

The ROC-AUC is the recommended primary metric for evaluating binary classifiers in
bioactivity prediction, as it is threshold-independent and robust to class imbalance
(Hanley & McNeil, 1982). For imbalanced chemical datasets, F1-Score and Sensitivity
are particularly important as complementary metrics (Jiang et al., 2025).

---

### `4_1_cross_validation_results.txt`

5-fold cross-validation scores:

```
Fold 1: AUC=0.85, Accuracy=0.84
Fold 2: AUC=0.86, Accuracy=0.85
Fold 3: AUC=0.87, Accuracy=0.86
Fold 4: AUC=0.88, Accuracy=0.87
Fold 5: AUC=0.86, Accuracy=0.85

Mean AUC: 0.864 ± 0.011
Mean Accuracy: 0.854 ± 0.011
```

**Interpretation:**
- Low variation (±0.01) → Model is stable
- High variation (±0.1) → Model is unstable
- If CV score << test score → Overfitting

Cross-validation provides a less biased estimate of generalization performance than
a single train/test split. In QSAR modeling, k-fold cross-validation is considered
essential to assess model robustness and detect overfitting (Tropsha, 2010).

---

### `4_2_enrichment_factor_results.txt`

How much better than random screening:

```
Enrichment Factor at 10%: 3.2x
Enrichment Factor at 20%: 2.1x
Enrichment Factor at 50%: 1.5x
```

**Interpretation:**
- **EF = 3.2x** → By screening top 10%, you find 3.2x more actives than random
- Higher EF = better virtual screening tool

The Enrichment Factor (EF) at a given percentage quantifies the ability of a model
to concentrate actives in the top-ranked fraction of a screened library relative to
random selection. It is one of the most widely used metrics to evaluate practical
virtual screening performance (Truchon & Bayly, 2007).


---

### `4_3_tanimoto_similarity_results.txt`

Molecular diversity metrics:

```
Mean Similarity: 0.45
Min Similarity: 0.12
Max Similarity: 0.89
```

**Interpretation:**
- **Mean < 0.5** → Diverse library 
- **Mean > 0.7** → Redundant/similar structures 

Molecular similarity is computed using the Tanimoto coefficient over binary molecular
fingerprints. Values above 0.7 are commonly interpreted as structurally similar compounds,
while values below 0.3 indicate high structural diversity (Willett, Barnard & Downs, 1998).

---

### `4_4_learning_curve_results.txt`

Model improvement with more data:

```
Training Size | Train AUC | Val AUC
50           | 0.70     | 0.68
100          | 0.78     | 0.76
200          | 0.82     | 0.81
400          | 0.85     | 0.84
798          | 0.88     | 0.87
```

**Interpretation:**
- **Val AUC increasing** → Model improves with more data 
- **Val AUC plateauing** → More data won't help much 

Learning curves are a standard diagnostic tool in machine learning to evaluate
whether a model would benefit from additional training data or requires architectural
changes (Ramsundar et al., 2019).

---

## Plots

### `roc_curve.png`
ROC (Receiver Operating Characteristic) curve showing model discrimination ability.

**Interpretation:**
- Curve closer to top-left → Better model
- Diagonal line → Random classifier
- Area under curve (AUC) > 0.8 → Good

### `learning_curve.png`
How accuracy improves as training set grows.

**Interpretation:**
- Curves converging → Model has learned most patterns
- Curves still diverging → More data would help

### `enrichment_curve.png`
Virtual screening performance across different screening percentages.

**Interpretation:**
- Steep initial slope → Model finds actives early
- Steep = good for virtual screening 

### `similarity_heatmap.png`
Molecular diversity visualization.

**Interpretation:**
- Dark blue (high similarity) = similar molecules
- Light colors = diverse library

---

## Log File

### `logs/kast_YYYYMMDD.log`

Detailed execution log with timestamps and debug info.

**Check log if:**
- Something fails
- You need execution details
- Debugging issues

---

## Quality Assessment

### Good Results
- AUC > 0.85
- Accuracy > 85%
- CV stability ± < 0.05
- Learning curve converges
- Clear enrichment factor (> 2x at 10%)

### Acceptable Results
- AUC 0.75-0.85
- Accuracy 75-85%
- CV stability ± 0.05-0.10
- Model still learning with more data

### Poor Results
- AUC < 0.70
- Accuracy < 70%
- High CV variation (± > 0.15)
- Enrichment factor < 1.5x
- Check: data quality, balance, duplicate molecules

---

## Exporting for Publication

### CSV Export
```bash
# All results already in CSV format
# Open in Excel or Python:
import pandas as pd
results = pd.read_csv('results/05_new_molecule_predictions.csv')
top_100 = results.head(100)
top_100.to_csv('top_100_predicted_actives.csv')
```

### Plot Export
Plots automatically saved as PNG (high resolution for publications).

### Report Generation
```bash
# Combine all results
cat results/4_0_evaluation_report.txt \
    results/4_1_cross_validation_results.txt \
    > publication_report.txt
```

---

## Further Reading & Foundations

- **ROC-AUC:** Hanley, J.A., & McNeil, B.J. (1982). The meaning and use of the area under a receiver operating characteristic (ROC) curve. *Radiology*, 143(1), 29-36. [doi:10.1148/radiology.143.1.7063747](https://doi.org/10.1148/radiology.143.1.7063747)
- **Enrichment Factor:** Truchon, J.F., & Bayly, C.I. (2007). Evaluating virtual screening methods: good and bad metrics for the "early recognition" problem. *Journal of Chemical Information and Modeling*, 47(2), 488-508. [doi:10.1021/ci600426e](https://doi.org/10.1021/ci600426e)
- **Tanimoto Similarity:** Willett, P., Barnard, J.M., & Downs, G.M. (1998). Chemical Similarity Searching. *Journal of Chemical Information and Computer Sciences*, 38(6), 983-996.
- **Cross-Validation in QSAR:** Tropsha, A. (2010). Best Practices for QSAR Model Development, Validation, and Exploitation. *Molecular Informatics*, 29(6-7), 476-488.
- **Imbalanced Learning:** Jiang, J., et al. (2025). A review of machine learning methods for imbalanced data challenges in chemistry. *Chemical Science*, 16, 7637-7658. [doi:10.1039/D5SC00270B](https://doi.org/10.1039/D5SC00270B)
- **Deep Learning Pipeline:** Ramsundar, B., et al. (2019). *Deep Learning for the Life Sciences*. O'Reilly Media.
