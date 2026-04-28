# 🚀 Installation Guide

This guide will walk you through setting up K-talysticFlow on your system.

---

## 📋 Prerequisites

Before installing K-talysticFlow, ensure you have:

- **Python 3.9 or higher** installed
- **Conda** (recommended for managing dependencies)
- **Git** (for cloning the repository)
- At least **8 GB RAM** (16 GB+ recommended)
- **2-5 GB** free disk space

---

## ⚙️ Step-by-Step Installation

### 1. Clone the Repository

```bash
# Clone from GitHub
git clone https://github.com/kelsouzs/kast.git
cd kast
```

Or download and extract the ZIP file from GitHub.

---

### 2. Option 1: Create Conda Environment from environment.yml (Recommended)

The easiest way using the provided configuration file:

```bash
# Create environment from environment.yml
conda env create -f environment.yml

# Activate the environment
conda activate ktalysticflow

# Test installation
python bin/check_env.py
```

### 3. Option 2: Create Conda Environment Manually

If you prefer to create the environment manually:

```bash
# Create environment with Python 3.9
conda create -n ktalysticflow python=3.9 -y

# Activate the environment
conda activate ktalysticflow

# Install RDKit from Conda
conda install -c conda-forge rdkit -y

# Install other dependencies from requirements.txt
pip install -r requirements.txt
```

---

### 5. Verify Installation

Run the built-in environment checker:

```bash
python main.py
```

Then select:
- **Option [8]** - Advanced Options
- **Option [1]** - Check Environment & Dependencies

The checker will verify:
- ✅ All required packages are installed
- ✅ Correct versions
- ✅ Import functionality
- ✅ System compatibility

**Expected Output:**
```
========================================================
         K-talysticFlow Dependency Checker
========================================================

✅ Python version: 3.10.12 (OK)
✅ RDKit: 2023.09.1 (OK)
✅ DeepChem: 2.7.1 (OK)
✅ TensorFlow: 2.15.0 (OK)
...
========================================================
✅ All dependencies are correctly installed!
========================================================
```

---

## 🔧 Platform-Specific Instructions

### 🐧 Linux

```bash
# Install system dependencies (Ubuntu/Debian)
sudo apt-get update
sudo apt-get install python3-dev build-essential

# Then follow steps 2-5 above
```

### 🪟 Windows

```powershell
# Use Anaconda Prompt or PowerShell
# Ensure conda is in PATH

# Then follow steps 2-5 above
```

### 🍎 macOS

```bash
# Install Xcode command line tools
xcode-select --install

# Then follow steps 2-5 above
```

---

## 🧪 Test Your Installation

### Quick Test

Run the parallel processing test suite:

```bash
python main.py
# Select [8] Advanced Options → [2] Test Parallel Processing
```

This runs 6 comprehensive tests to verify:
- ✅ Basic parallelism
- ✅ Large dataset handling
- ✅ Memory efficiency
- ✅ Error handling
- ✅ Performance benchmarks

---

## 📦 requirements.txt Contents

```txt
deepchem>=2.7.0
rdkit>=2022.9.5
tensorflow>=2.10.0
scikit-learn>=1.2.0
pandas>=1.5.0
numpy>=1.23.0
matplotlib>=3.6.0
seaborn>=0.12.0
joblib>=1.2.0
tqdm>=4.65.0
```

---

## 🔄 Updating K-talysticFlow

To update to the latest version:

```bash
# Pull latest changes
git pull origin main

# Update dependencies
pip install -r requirements.txt --upgrade
```

---

## ❌ Uninstallation

To remove K-talysticFlow:

```bash
# Deactivate environment
conda deactivate

# Remove conda environment
conda env remove -n kast

# Delete repository folder
cd ..
rm -rf kast  # Linux/Mac
rmdir /s kast  # Windows
```

---

## 🛠️ Troubleshooting Installation

### Issue: RDKit installation fails

**Solution:** Use Conda instead of pip:
```bash
conda install -c conda-forge rdkit
```

### Issue: TensorFlow errors

**Solution:** Install specific version:
```bash
pip install tensorflow==2.15.0
```

### Issue: Memory errors during installation

**Solution:** Install packages one by one:
```bash
pip install deepchem
pip install tensorflow
# ... etc
```

### Issue: Permission denied (Linux/Mac)

**Solution:** Use `--user` flag:
```bash
pip install -r requirements.txt --user
```

---

## ✅ Next Steps

Once installation is complete:

1. **[Read the User Manual]()** - Learn how to use K-talysticFlow
2. **[Configure Settings]()** - Customize for your needs
3. **[Run Your First Analysis]()** - Get started!

---

## 📞 Need Help?

- **Check [FAQ]()** for common questions
- **See [Troubleshooting]()** for known issues
- **Open an issue** on GitHub
- **Contact:** lmm@uefs.br

---