# -----------------------------------------------------------------------------
# UTILITIES MODULE - K-talysticFlow (KAST)
# -----------------------------------------------------------------------------
# Contains functions shared across different scripts in the project to
# avoid code repetition and maintain organization.
# -----------------------------------------------------------------------------

import os
import sys
import logging
from typing import List, Optional
from datetime import datetime

import numpy as np
import deepchem as dc
from rdkit import Chem
from rdkit.Chem import AllChem, DataStructs
from tqdm import tqdm

current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

import settings as cfg

def load_smiles_from_file(filepath: str) -> List[str]:
    """
    Loads a list of SMILES from a text file (.smi, .txt).
    Assumes one SMILES per line, ignoring empty lines.
    """
    if not os.path.exists(filepath):
        print(f"\nERROR: SMILES file '{filepath}' not found.")
        return []
    try:
        with open(filepath, 'r') as f:
            smiles_list = [line.strip() for line in f if line.strip()]
        print(f"  -> Loaded {len(smiles_list)} SMILES from '{os.path.basename(filepath)}'")
        return smiles_list
    except Exception as e:
        print(f"\nERROR reading file '{filepath}': {e}")
        return []

def get_morgan_fp(smiles_str: str, radius: int, nBits: int) -> Optional[DataStructs.ExplicitBitVect]:
    """
    Calculates the Morgan Fingerprint for a single SMILES using the new 'Generator' standard.
    Returns the fingerprint or None if the SMILES is invalid.
    """
    mol = Chem.MolFromSmiles(smiles_str)
    if mol:
        fpgen = AllChem.GetMorganGenerator(radius=radius, fpSize=nBits)
        return fpgen.GetFingerprint(mol)
    return None

def ensure_dir_exists(dir_path: str):
    """
    Ensures that a directory exists. Creates it if not present.
    """
    if not os.path.exists(dir_path):
        print(f"\nINFO: Creating missing directory: {dir_path}")
        os.makedirs(dir_path)

def validate_smiles(smiles_list: List[str]) -> List[str]:
    valid_smiles = []
    invalid_count = 0

    for smi in smiles_list:
        mol = Chem.MolFromSmiles(smi)
        if mol is not None:
            canonical_smi = Chem.MolToSmiles(mol)
            valid_smiles.append(canonical_smi)
        else:
            invalid_count += 1
    
    if invalid_count > 0:
        print(f"⚠️  {invalid_count} invalid SMILES were removed")
    
    return valid_smiles

def load_and_featurize_full_dataset() -> Optional[dc.data.Dataset]:

    actives_raw = load_smiles_from_file(cfg.ACTIVE_SMILES_FILE)
    inactives_raw = load_smiles_from_file(cfg.INACTIVE_SMILES_FILE)
    
    if not actives_raw or not inactives_raw:
        print("\nERROR: Failed to load SMILES files.")
        return None

    print("Validating SMILES...")
    actives = validate_smiles(actives_raw)
    inactives = validate_smiles(inactives_raw)
    
    if not actives or not inactives:
        print("\nERROR: No valid SMILES found after validation.")
        return None

    print(f"Loaded dataset: {len(actives)} actives, {len(inactives)} inactives")

    if len(actives) < cfg.MIN_MOLECULES_PER_CLASS or len(inactives) < cfg.MIN_MOLECULES_PER_CLASS:
        print(f"⚠️ Dataset does not meet minimum criteria ({cfg.MIN_MOLECULES_PER_CLASS} per class)")
        return None

    all_smiles = np.array(actives + inactives)
    all_labels = np.array([1] * len(actives) + [0] * len(inactives))

    featurizer = dc.feat.CircularFingerprint(size=cfg.FP_SIZE, radius=cfg.FP_RADIUS)
    
    features = []
    failed_count = 0
    
    for smi in tqdm(all_smiles, desc="Featurizing Full Dataset"):
        try:
            feat = featurizer.featurize([smi])[0]
            if feat is not None and feat.size > 0:
                features.append(feat)
            else:
                features.append(None)
                failed_count += 1
        except Exception:
            features.append(None)
            failed_count += 1

    valid_indices = [i for i, x in enumerate(features) if x is not None]
    
    if not valid_indices:
        print("\nERROR: No molecule was successfully featurized.")
        return None
    
    features_valid = np.vstack([features[i] for i in valid_indices])
    labels_valid = all_labels[valid_indices]
    smiles_valid = all_smiles[valid_indices]
    
    print(f"✅ {len(features_valid)} out of {len(all_smiles)} molecules featurized successfully.")
    if failed_count > 0:
        print(f"⚠️ {failed_count} molecules failed featurization")
    
    return dc.data.NumpyDataset(X=features_valid, y=labels_valid, ids=smiles_valid)


# ============================================================================
# BANNER/HEADER FORMATTING
# ============================================================================

def print_script_banner(title: str, description: str = ""):
    """
    Prints a standardized banner for script headers with centered text.
    
    Args:
        title (str): Main title of the script
            Example: "K-talysticFlow | Step 1: Preparing and Splitting Data"
        description (str, optional): Subtitle or additional description
    
    Output Format:
        ======================================================================
                    K-talysticFlow | Step 1: Data Preparation
                              Optional description here
        ======================================================================
    
    Banner width: 70 characters
    
    Usage:
        >>> print_script_banner("K-talysticFlow | Step 2: Featurization")
        >>> print_script_banner("Main Evaluation", "ROC/AUC Analysis")
    """
    width = 70
    separator = "=" * width
    
    print(f"\n{separator}")
    print(title.center(width))
    if description:
        print(description.center(width))
    print(f"{separator}\n")


# ============================================================================
# LOGGING UTILITIES
# ============================================================================

def setup_script_logging(script_name: str):
    """
    Configures logging for individual scripts to save errors to log files.
    Each script logs to the main KAST log file in results/logs/.
    
    Args:
        script_name (str): Name of the script (e.g., "1_preparation", "2_featurization")
    
    Returns:
        logging.Logger: Configured logger instance for the script
    
    Log Format:
        "YYYY-MM-DD HH:MM:SS - script_name - LEVEL - message"
    
    Log File:
        results/logs/kast_YYYYMMDD.log (daily rotation)
    
    Features:
        - File-only logging (no console output)
        - Automatic directory creation
        - Duplicate handler prevention
        - UTF-8 encoding support
    
    Example:
        >>> logger = setup_script_logging("2_featurization")
        >>> logger.info("Starting featurization")
        >>> logger.error("Failed to process molecule")
    """
    log_dir = os.path.join(os.path.dirname(__file__), 'results', 'logs')
    os.makedirs(log_dir, exist_ok=True)
    
    # Configure logging to append to today's log file
    log_filename = f"kast_{datetime.now().strftime('%Y%m%d')}.log"
    log_path = os.path.join(log_dir, log_filename)
    
    # Configure logger for this script
    logger = logging.getLogger(script_name)
    logger.setLevel(logging.INFO)
    
    # Avoid duplicate handlers
    if not logger.handlers:
        handler = logging.FileHandler(log_path, encoding='utf-8')
        handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
        logger.addHandler(handler)
    
    return logger


def log_error(logger, error_msg: str, exception: Exception = None):
    """
    Logs an error message and optionally the exception details with traceback.
    
    Args:
        logger (logging.Logger): Logger instance from setup_script_logging()
        error_msg (str): Human-readable error message to log
        exception (Exception, optional): Exception object to include traceback
    
    Side Effects:
        Writes error message to log file with:
            - Full traceback if exception provided (exc_info=True)
            - Plain error message if no exception
    
    Example:
        >>> try:
        ...     risky_operation()
        ... except Exception as e:
        ...     log_error(logger, "Operation failed", e)
    """
    if exception:
        logger.error(f"{error_msg}: {str(exception)}", exc_info=True)
    else:
        logger.error(error_msg)