# KAST Wiki

<div align="center">
<h2>🧬 K-atalystic Automated Screening Taskflow</h2>
<p><em>Automated Deep Learning Pipeline for Molecular Bioactivity Prediction</em></p>
<p>A comprehensive, user-friendly solution for training and deploying Machine Learning models in drug discovery</p>

![Python](https://img.shields.io/badge/Python-3.9%2B-blue.svg)
![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)
![Version](https://img.shields.io/badge/Version-1.0.0-green.svg)
![Status](https://img.shields.io/badge/Status-Stable-brightgreen.svg)

</div>

---

## 🚀 What is K-talysticFlow?

K-talysticFlow or **K**-atalystic **A**utomated **S**creening **T**askflow (KAST) is a fully automated, interactive pipeline designed to streamline the process of training, evaluating, and using Deep Learning models for predicting molecular bioactivity. Built on a robust stack including **DeepChem**, **RDKit**, and **TensorFlow**, it provides an end-to-end solution for computational drug discovery and virtual screening.

**Developed at:** [Laboratory of Molecular Modeling (LMM-UEFS)](https://lmm.uefs.br/)  
**Funding:** CNPq  
**Current Version:** 1.0.0 (Stable Release - October 10, 2025)

---

## ✨ Key Features

### 🎯 Core Capabilities
- 🤖 **Fully Automated**: Interactive menu-driven interface for a seamless workflow
- 🧪 **Deep Learning Model**: Multi-Layer Perceptron (MLP) trained on Morgan Fingerprints (ECFP)
- 📊 **Comprehensive Validation Suite**: Rigorous model assessment including:
  - ROC/AUC Analysis
  - Enrichment Factor Calculation
  - k-fold Cross-Validation with Scaffold Splitting
  - Tanimoto Similarity Analysis
  - Learning Curve Generation
- 🔄 **Complete End-to-End Pipeline**: From raw SMILES data to actionable predictions
- 🐧 **Cross-Platform**: Compatible with Windows and Linux
- 📈 **Analysis-Ready Outputs**: Clear reports, graphs, and CSV files

### ⚡ Advanced Features
- 🚀 **Parallel Processing**: Multi-core support for 5-10x faster performance
  - Automatic CPU detection
  - Memory-efficient batch processing
  - Configurable worker allocation
- 🎯 **K-Prediction Score**: Proprietary scoring system for ranking molecular activity
- 📝 **Comprehensive Logging**: Daily log rotation with detailed error tracking
- 🔬 **Quality Assurance**: Built-in dependency checker and test suite
- ⚙️ **Flexible Configuration**: Centralized settings management in `settings.py`

---

## 📚 Quick Navigation

| Section | Description |
| --- | --- |
| [🚀 Installation Guide](Installation) | Complete setup instructions and requirements |
| [📖 User Manual](User-Manual) | Step-by-step usage guide with examples |
| [🔬 Pipeline Steps](Pipeline-Steps) | Detailed documentation of each script |
| [⚡ Parallel Processing](Parallel-Processing) | Configuration and optimization guide |
| [📊 Output Analysis](Output-Analysis) | How to interpret results and K-Prediction scores |
| [⚙️ Configuration Guide](Configuration) | Customize pipeline settings |
| [❓ FAQ](FAQ) | Frequently asked questions |
| [🛠️ Troubleshooting](Troubleshooting) | Common issues and solutions |
| [📝 API Reference](API-Reference) | Function and module documentation |

---

```

**📖 For detailed technical documentation of each step, see [Pipeline Steps Guide →](Pipeline-Steps)**

---

## 🎯 Getting Started

### Quick Start Guide

1. **[Install K-talysticFlow](Installation)** - Set up your environment
2. **Prepare your data** - Place SMILES files in `data/` folder
3. **Run the pipeline** - Execute `python main.py` and follow the menu
4. **[Analyze results](Output-Analysis)** - Interpret your predictions

### Recommended Workflow

```bash
# 1. Check environment
python main.py → Option [8] → [1] Check Dependencies

# 2. Run full pipeline
python main.py → Option [7] Run Complete Pipeline

# 3. Analyze outputs
Check results/ folder for reports, plots, and predictions
```

---

## 📊 System Requirements

### Minimum Requirements
- **Python**: 3.9+
- **RAM**: 8 GB
- **CPU**: Dual-core processor
- **Disk**: 2 GB free space

### Recommended Requirements
- **Python**: 3.10+
- **RAM**: 16 GB+
- **CPU**: Quad-core or better (for parallel processing)
- **Disk**: 5 GB+ free space
- **GPU**: Optional (TensorFlow GPU support)

### Key Dependencies
- RDKit
- DeepChem
- TensorFlow
- Scikit-learn
- Pandas, NumPy
- Matplotlib, Seaborn
- Joblib (parallel processing)
- TQDM (progress bars)

---

## 🎓 Citation

If you use K-talysticFlow in your research, please cite:

```bibtex
@software{kast2025,
  author = {Santos, Késsia Souza},
  title = {K-talysticFlow: Automated Deep Learning Pipeline for Molecular Screening},
  year = {2025},
  version = {1.0.0},
  url = {https://github.com/kelsouzs/kast},
  institution = {Laboratory of Molecular Modeling, UEFS}
}
```

---

## 📞 Support & Contact

- **GitHub Issues**: [Report bugs or request features](https://github.com/kelsouzs/kast/issues)
- **Email**: lmm@uefs.br
- **LinkedIn**: [@kelsouzs](https://www.linkedin.com/in/kelsouzs)
- **GitHub**: [@kelsouzs](https://github.com/kelsouzs)
- **Wiki**: Browse this documentation for detailed guides

---

## 📜 License

This project is licensed under the **MIT License** - see the [LICENSE](../license.txt) file for details.

---

## 🙏 Acknowledgments

- **Funding**: CNPq (Conselho Nacional de Desenvolvimento Científico e Tecnológico)
- **Institution**: Laboratory of Molecular Modeling (LMM-UEFS)
- **Community**: DeepChem, RDKit, and TensorFlow teams

---

<div align="center">
<p><strong>Version:</strong> 1.0.0 | <strong>Last Updated:</strong> October 10, 2025</p>
<p><em>Made with ❤️ for the computational chemistry community</em></p>
<p><strong>Developer:</strong> Késsia Souza Santos (@kelsouzs)</p>
</div>
