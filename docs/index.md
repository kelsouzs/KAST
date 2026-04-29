# K-talysticFlow (KAST) Documentation
```{raw} html
<div style="text-align: center; padding: 2.5rem 0 1.5rem 0;">
  <h1 style="font-size: 2.2rem; margin: 0;">K-talysticFlow (KAST)</h1>
  <p style="font-size: 1.1rem; color: #000000; margin-top: 0.5rem;">
    <strong>K</strong>-atalystic <strong>A</strong>utomated <strong>S</strong>creening
    <strong>T</strong>askflow — Automated Deep Learning for Molecular Bioactivity Prediction
  </p>
</div>
```

***

## What is KAST?

K-talysticFlow (KAST) is an open-source pipeline that democratizes the use of deep learning
for molecular bioactivity prediction in drug discovery and virtual screening workflows.
KAST was developed at the Laboratory of Molecular Modeling (LMM-UEFS) to provide
researchers with a reproducible, end-to-end solution — from data preparation to prediction —
without requiring deep expertise in machine learning infrastructure.

The pipeline is built on [DeepChem](https://github.com/deepchem/deepchem) and TensorFlow,
using Morgan/ECFP fingerprints as molecular descriptors and a MultitaskClassifier neural
network trained from scratch on user-provided bioactivity data.

What can you use KAST for? Here are some examples:

- Predict the bioactivity of small drug-like molecules against a biological target
- Rank large compound libraries by predicted probability of activity
- Train a custom deep learning model using your own active/inactive dataset
- Evaluate model quality with ROC-AUC, enrichment factor, and cross-validation
- Export ranked candidate lists for downstream experimental validation

KAST is a machine learning training and inference tool — it learns from your data and
builds a target-specific model. It does not ship with pre-trained models for arbitrary targets.

***

## Quick Start

The fastest way to get started is to set up the Conda environment and launch the interactive menu:

```bash
conda env create -f environment.yml
conda activate ktalysticflow
python main.py
```

Then follow the step-by-step pipeline:

```
[1] Prepare Data       → Clean and organize your SMILES dataset
[2] Featurize          → Generate Morgan/ECFP fingerprints
[3] Train Model        → Build your deep learning model from scratch
[4] Evaluate           → ROC-AUC, cross-validation, enrichment factor
[5] Predict            → Screen new molecules and export ranked results
```

***

## About

KAST is developed and maintained at the
[Laboratory of Molecular Modeling (LMM-UEFS)](https://lmm.uefs.br/) by Késsia Souza Santos.
Contributions, issues, and suggestions are welcome via the
[GitHub repository](https://github.com/kelsouzs/KAST).

**Funding:** This project was developed with support from
[CNPq](https://www.gov.br/cnpq/) (undergraduate research scholarship, PIBIC/IC)
and is currently continued under a
[CAPES](https://www.gov.br/capes/) graduate research scholarship (MSc).

***

```{toctree}
:maxdepth: 1
:caption: Getting Started
:hidden:

getting-started/overview
getting-started/installation
getting-started/quick-start
```

```{toctree}
:maxdepth: 1
:caption: User Guide
:hidden:

user-guide/pipeline
user-guide/step-by-step
user-guide/data-preparation
user-guide/parallel-processing
user-guide/outputs
```

```{toctree}
:maxdepth: 1
:caption: Support
:hidden:

support/faq
support/troubleshooting
support/configuration
```