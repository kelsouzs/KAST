# 🚀 K-talysticFlow (KAST) — Deep Learning Molecular Screening Pipeline

![Python](https://img.shields.io/badge/Python-3.9%2B-blue.svg)
![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)
![Status](https://img.shields.io/badge/Status-In%20Development-orange.svg)
[![LinkedIn](https://img.shields.io/badge/-LinkedIn-blue?style=flat-square&logo=linkedin&logoColor=white&link=https://www.linkedin.com/in/kelsouzs)](https://www.linkedin.com/in/kelsouzs)
[![GitHub](https://img.shields.io/badge/-GitHub-181717?style=flat-square&logo=github&logoColor=white&link=https://github.com/kelsouzs)](https://github.com/kelsouzs)

```
  __  __    _     ____  _____ 
  | |/ /   / \   / ___||_   _|
  | ' /   / _ \  \___ \  | |  
  | . \  / ___ \  ___) | | |  
  |_|\_\/_/   \_\|____/  |_|  K-talystic Automated Screening Taskflow
```

---

## 👨‍🔬 What is K-talysticFlow (KAST)?

K-talysticFlow is a fully automated, interactive pipeline for training, evaluating, and using Deep Learning models to predict molecular bioactivity.  
Built for virtual screening and rapid identification of promising chemical compounds — accelerating drug discovery.

Developed at the [Laboratory of Molecular Modeling (LMM-UEFS)](https://lmm.uefs.br/), funded by CNPq.

---

## ✨ Features at a Glance

- ⚡ **Automated CLI Workflow** 
- 🧠 **Deep Learning Model**
- 🎯 **K-Prediction Score** 
- 🧪 **Scientific Validation** 
- 📊 **Rich Outputs** 
- 🖥️ **User-Friendly** 

---

## 📁 Project Structure

```
KAST/
├── 🧩 bin/                # Pipeline scripts
│   ├── ⚙️ 1_preparation.py           # Data split
│   ├── 🦾 2_featurization.py         # Fingerprinting
│   ├── 🤖 3_training.py              # Model training
│   ├── 📊 4_0_evaluation_main.py     # Main evaluation
│   ├── 🧪 ... (other evaluation scripts)
│   └── 🔮 5_1_run_prediction.py      # Prediction for new molecules
├── 🧬 data/              # Input data (.smi files)
│   └── 🧪 xx.smi
├── 📦 results/         # Outputs, logs, models, plots
├── 📝 settings.py           # Pipeline settings
├── 🛠️ utils.py            # Shared functions
├── 🖥️ main.py             # Interactive menu
├── 📄 requirements.txt    # Python requirements
├── 📜 LICENSE             # MIT License
├── 📚 README.md           # This file
```

---

## ⚙️ System Requirements

- **Python**: 3.9+ (tested)
- **Conda**: Recommended for RDKit and isolation
- **Main packages**: RDKit, DeepChem, Tensorflow, Scikit-learn, pandas, numpy, tqdm, matplotlib

---

## 🚀 Quick Start

| Task                | Command                        |
|---------------------|-------------------------------|
| Prepare data        | `python bin/1_preparation.py`  |
| Featurize           | `python bin/2_featurization.py`|
| Train model         | `python bin/3_training.py`     |
| Main evaluation     | `python bin/4_0_evaluation_main.py` |
| Predict new         | `python bin/5_1_run_prediction.py`|

Or use the interactive menu:
```bash
python main.py
```
And follow the on-screen instructions!

---

## 👩‍🔬 Scientific Workflow

1. **Data Preparation**: Import SMILES, label, and split datasets
2. **Featurization**: Generate ECFP/Morgan fingerprints (QSAR standard)
3. **Training**: DeepChem MultitaskClassifier neural network
4. **Evaluation**: ROC/AUC, accuracy, precision, recall, F1, cross-validation, enrichment factor, Tanimoto similarity, learning curve
5. **Prediction**: Screen new molecules using K-Prediction Score, rank candidates, export CSV

---

## 📈 Outputs

- `results/`: All logs, models, reports, plots (ROC, learning curve, etc)
- `05_new_molecule_predictions.csv`: Ranked predictions for new molecules (sorted by K-Prediction Score)

---

## ⭐ How to Cite

> **K-talysticFlow: A Deep Learning Pipeline for Virtual Screening of Bioactive Compounds**  
> Késsia S. Santos; Manoelito C. Santos Junior. (2025). Laboratory of Molecular Modeling (LMM), State University of Feira de Santana.

---

## 👥 Authors & Acknowledgments

- **Késsia Souza Santos**
    - Email: `kelsouzs.uefs@gmail.com`
    - [GitHub](https://github.com/kelsouzs)
    - [LinkedIn](https://www.linkedin.com/in/kelsouzs)
- **Advisor:** Prof. Dr. Manoelito Coelho dos Santos Junior
- **Lab:** [LMM-UEFS](https://lmm.uefs.br/)
- **Funding:** National Council for Scientific and Technological Development (CNPq)

---

## ❓ FAQ

**Q:** What Python version do I need?  
**A:** Python 3.9+ recommended.

**Q:** How do I install RDKit?  
**A:**  
```bash
conda install -c conda-forge rdkit
```

**Q:** Where do outputs go?  
**A:** All results are in the `results/` folder.

**Q:** How do I run on my own molecules?  
**A:** Place your `.smi` file in `data/` and use `[5]` menu option.

**Q:** What is the K-Prediction Score?  
**A:** The K-Prediction Score is the proprietary scoring function used by K-talysticFlow to rank molecular activity predictions, with values ranging from 0 to 1 (higher scores indicate higher predicted activity).

---

## 🤝 Contributing

Pull requests welcome! For major changes, open an issue first to discuss.

---

## 📬 Contact

Questions or suggestions?  
Open an [issue](https://github.com/kelsouzs/KAST/issues) or email [kelsouzs.uefs@gmail.com](mailto:kelsouzs.uefs@gmail.com)


---



