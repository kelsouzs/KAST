# Troubleshooting

Common issues and solutions.

---

## Installation Issues

### Setup fails with "Conda not found"

**Cause:** Anaconda/Miniconda not installed or not in PATH

**Solution:**
1. Install Anaconda: [anaconda.com/download](https://www.anaconda.com/download)
2. Make sure to check "Add Anaconda to PATH" during installation
3. Restart computer
4. Try setup again

---

### setup.exe won't run (Windows)

**Cause:** Blocked by Windows Defender or permissions

**Solutions:**
1. **Try as Administrator:**
   - Right-click `setup.exe` → "Run as administrator"

2. **Unblock file:**
   - Right-click `setup.exe` → Properties
   - Check "Unblock" at bottom → Apply → OK

3. **Verify location:**
   - Ensure `setup.exe` is in same folder as `environment.yml`
   - Move to a simple path (avoid spaces): `C:\KAST\`

4. **Use manual setup:**
   ```bash
   cd path\to\KAST
   conda env create -f environment.yml -y
   ```

---

### setup.sh won't run (Linux)

**Cause:** Script not executable

**Solution:**
```bash
chmod +x setup.sh
./setup.sh
```

---

### "Cannot find conda.exe" during setup

**Cause:** Conda not in standard location

**Solution:**
1. Find Conda:
   ```bash
   where conda    # Windows
   which conda    # Linux/Mac
   ```

2. If not found, manually specify path in setup or use manual setup

---

## Dependency Issues

### "ImportError: No module named 'tensorflow'"

**Cause:** Dependencies not installed or environment not activated

**Solution (Windows):**
1. Click desktop shortcut "K-talysticFlow 1.0.0" (handles activation)
2. Or activate manually:
   ```bash
   conda activate ktalysticflow
   python bin/check_env.py
   ```

**Solution (Linux):**
```bash
conda activate ktalysticflow
python bin/check_env.py
```

If still fails:
```bash
python bin/check_env.py  # See which packages are missing
conda env remove -n ktalysticflow -y
conda env create -f environment.yml -y
```

---

### "ImportError: No module named 'rdkit'"

**Cause:** RDKit not installed (common on some systems)

**Solution:**
```bash
conda activate ktalysticflow
conda install -c conda-forge rdkit
```

---

### "Cannot allocate memory" during featurization

**Cause:** Dataset too large or batch size too large

**Solutions:**
1. **Reduce batch size** in `settings.py`:
   ```python
   PARALLEL_BATCH_SIZE = 25000  # Reduce from 100000
   ```

2. **Use fewer workers** in `settings.py`:
   ```python
   N_WORKERS = 2  # Reduce from auto
   ```

3. **Disable parallel processing**:
   ```python
   ENABLE_PARALLEL_PROCESSING = False
   ```

4. **Use subset of data:**
   - Test with first 10K molecules
   - Check if it's a data quality issue

---

## Data Issues

### "Invalid SMILES in file"

**Cause:** Malformed SMILES structures

**Solution:**
1. Validate SMILES using RDKit:
   ```bash
   python -c "
   from rdkit import Chem
   with open('data/actives.smi') as f:
       for i, line in enumerate(f):
           smiles = line.split()[0]
           if Chem.MolFromSmiles(smiles) is None:
               print(f'Line {i+1}: Invalid SMILES: {smiles}')
   "
   ```

2. Clean your SMILES file and try again

3. Use online tool: [SMILES validation](https://www.chemspider.com/StructureSearch.aspx)

---

### "No molecules loaded from file"

**Cause:** File format wrong or empty file

**Solution:**
1. Check file exists: `data/actives.smi` and `data/inactives.smi`
2. Check format: One SMILES per line, not Excel format
3. Ensure file encoding is UTF-8 (not Unicode)
4. Try file with known-good SMILES and verify it works

---

### "Data imbalance too large"

**Not an error**, but might affect model. KAST handles it automatically.

**If model performs poorly:**
1. Try balancing active/inactive ratio closer to 1:1 or 1:5
2. Check data quality (duplicates, mislabeling)
3. Consider using different actives/inactives source

---

## Runtime Issues

### Pipeline crashes with "Out of Memory"

**Solutions:**
1. Reduce parallel batch size (see above)
2. Use fewer cores: `N_WORKERS = 2`
3. Disable parallel: `ENABLE_PARALLEL_PROCESSING = False`
4. Use smaller dataset (test with 10K molecules first)
5. Close other applications to free RAM

---

### "GPU/CUDA not found" warning

**This is normal!** KAST works fine on CPU only.

**If you want GPU support** (advanced):
```bash
conda install tensorflow-gpu=2.13
```

---

### Featurization is very slow

**Solutions:**
1. **Enable parallel processing** (see [Parallel Processing](../user-guide/parallel-processing.md)):
   ```python
   ENABLE_PARALLEL_PROCESSING = True
   N_WORKERS = None  # Auto-detect
   ```

2. **Check CPU usage:**
   - Windows: Task Manager → Performance tab
   - Linux: `top` or `htop` command
   - If not using all cores, verify parallel is enabled

3. **Reduce dataset size** for testing

---

### "Process finished with exit code 1"

**Generic error** — check full output for details.

**Solutions:**
1. Scroll up in terminal to see actual error message
2. Check log file: `results/logs/kast_YYYYMMDD.log`
3. Run `python bin/check_env.py` to verify dependencies
4. Try step individually: `python bin/1_preparation.py`

---

## Results Issues

### Model performance is terrible (AUC < 0.60)

**Possible causes & solutions:**

| Problem | Check | Solution |
|---------|-------|----------|
| Bad data quality | Look for duplicates, mislabeled molecules | Clean data, remove duplicates |
| Too much class imbalance | Active:Inactive ratio | Try 1:1 or 1:5 ratio |
| Insufficient data | < 100 molecules per class | Get more compounds |
| Wrong SMILES | Invalid structures | Validate SMILES with RDKit |
| Random seed issue | Different results each run | Check seed settings |

---

### Training seems stuck (no output for 10 minutes)

**This can be normal for large datasets!**

**Check if process is alive:**
- Watch CPU usage (should be active)
- Check memory usage (shouldn't max out)

**If truly stuck:**
- Ctrl+C to cancel
- Reduce dataset size and try again
- Check logs: `results/logs/kast_*.log`

---

### Cross-validation scores very different from test AUC

**Possible signs of:** Overfitting or data issues

**Solutions:**
1. Check for duplicate molecules across folds
2. Verify data quality
3. Try with more training data
4. Check Learning Curve (Step 4.5)

---

## Platform-Specific Issues

### Windows: Shortcuts don't work

**Solution:**
1. Delete broken shortcut
2. Re-run `setup.exe`
3. Or manually create: `run_kast.bat` should be in KAST folder
4. Double-click `run_kast.bat`

---

### Linux: App menu shortcut missing

**Solution:**
```bash
./setup.sh  # Re-run to create shortcut

# Or manually create:
mkdir -p ~/.local/share/applications
cat > ~/.local/share/applications/kast.desktop << EOF
[Desktop Entry]
Type=Application
Name=K-talysticFlow
Icon=python
Exec=bash -c "cd $(pwd) && conda activate ktalysticflow && python main.py"
Terminal=true
EOF
```

---

## Getting Help

### Can't find answer here?

1. **Check logs:** `results/logs/kast_YYYYMMDD.log`
2. **Check [FAQ](faq.md)** for common questions
3. **Verify environment:** `python bin/check_env.py`
4. **Test parallel setup:** `python bin/test_parallel_compatibility.py`

### Report issue on GitHub:
- [github.com/kelsouzs/KAST/issues](https://github.com/kelsouzs/KAST/issues)
- Include: OS, error message, steps to reproduce

### Email support:
- kelsouzs.uefs@gmail.com
- Include: Full error output, log file, dataset info if possible

---

## Still Having Issues?

**Provide this information:**
- OS and version (Windows 11, Ubuntu 20.04, etc)
- Anaconda or Miniconda version
- Full error message (copy-paste from terminal)
- Log file content: `cat results/logs/kast_*.log`
- Dataset size and approximate molecule count

