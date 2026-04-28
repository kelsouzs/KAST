# Installation

Two installation methods: **Automated** (recommended) or **Manual**.

---

## 🪟 Windows Installation

### Option 1: Automated Setup (Recommended!) ⚡

**Best for:** Everyone — no terminal needed!

1. **Download `setup.exe`** from the KAST releases page
2. **Double-click `setup.exe`** in the KAST folder
3. Wait for the installer to finish (2-3 minutes)

**What `setup.exe` does automatically:**
```
✅ Finds your Anaconda/Miniconda installation
✅ Creates or updates 'ktalysticflow' conda environment
✅ Installs all Python dependencies
✅ Creates desktop shortcut "K-talysticFlow 1.0.0"
✅ Generates run_kast.bat launcher script
✅ Creates Start Menu shortcut
```

**After setup, launch KAST:**
- 🖱️ Click desktop shortcut **"K-talysticFlow 1.0.0"**, or
- 🖱️ Find in **Start Menu → K-talysticFlow 1.0.0**, or
- 🖱️ Double-click `run_kast.bat` in the KAST folder

**That's it!** KAST runs directly — no terminal, no `conda activate` needed.

---

### Option 2: Manual Setup

If `setup.exe` doesn't work, use the command line:

```powershell
# Navigate to KAST folder
cd path\to\KAST

# Create environment
conda env create -f environment.yml -y

# Activate
conda activate ktalysticflow

# Run KAST
python main.py
```

---

## 🐧 Linux Installation

### Option 1: Automated Setup (Recommended!) ⚡

**Tested on:** Ubuntu 20.04 LTS and newer

1. **Make setup script executable:**
   ```bash
   chmod +x setup.sh
   ```

2. **Run setup:**
   ```bash
   ./setup.sh
   ```

3. Wait for environment to be created (3-5 minutes)

**What `setup.sh` does automatically:**
```
✅ Checks if Conda is installed
✅ Creates 'ktalysticflow' conda environment
✅ Installs all Python dependencies
✅ Creates .desktop shortcut in apps menu
```

**After setup, launch KAST:**
- **Option A:** Search for **"K-talysticFlow"** in your app menu and click
- **Option B:** From terminal:
  ```bash
  conda activate ktalysticflow
  python main.py
  ```

---

### Option 2: Manual Setup

If `setup.sh` doesn't work:

```bash
# Navigate to KAST folder
cd /path/to/KAST

# Create environment
conda env create -f environment.yml -y

# Activate
conda activate ktalysticflow

# Run KAST
python main.py
```

---

## ⚙️ System Requirements

| Requirement | Details |
|-------------|---------|
| **Python** | 3.9 or higher |
| **Conda** | Anaconda or Miniconda |
| **RAM** | 4GB minimum, 8GB+ recommended |
| **Disk** | 2-5GB free space |
| **OS** | Windows 11+ or Ubuntu 20.04+ |

---

## ✅ Verify Installation

After setup completes, check that everything works:

```bash
# Activate environment (if needed)
conda activate ktalysticflow

# Check dependencies
python bin/check_env.py
```

You should see:
```
🔍 Checking environment...
Importing 'tensorflow'... [OK]
Importing 'rdkit'... [OK]
...
Success! All dependencies are installed.
```

---

## 🛠️ Troubleshooting Installation

### "Conda not found"
- **Cause:** Conda isn't installed or not in PATH
- **Fix:** Install [Anaconda](https://www.anaconda.com/download) or [Miniconda](https://docs.conda.io/projects/miniconda/en/latest/)

### "setup.exe doesn't run"
- **Cause:** Windows Defender blocking or wrong location
- **Fix:** 
  - Ensure `setup.exe` is in the same folder as `environment.yml`
  - Right-click → Properties → "Unblock" if prompted

### "setup.sh permission denied"
- **Cause:** Script not marked as executable
- **Fix:** `chmod +x setup.sh` then run again

### "Environment creation failed"
- **Cause:** Network issue or conflicting packages
- **Fix:** Try manual setup and check [Troubleshooting](../support/troubleshooting.md)

---

## Next Steps

→ **[Quick Start](quick-start.md)** — Launch KAST and run your first analysis
