# -----------------------------------------------------------------------------
# UTILITIES MODULE - K-talysticFlow (KAST)
# -----------------------------------------------------------------------------
# Contains functions shared across different scripts in the project to
# avoid code repetition and maintain organization.
# -----------------------------------------------------------------------------

# ============================================================================
# CRITICAL: Suppress warnings BEFORE imports
# ============================================================================
import os
import sys

# Set environment variables to suppress TensorFlow/Protobuf warnings
os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '3')
os.environ.setdefault('TF_ENABLE_ONEDNN_OPTS', '0')
os.environ.setdefault('PYTHONWARNINGS', 'ignore')
os.environ.setdefault('PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION', 'python')

import warnings
warnings.filterwarnings('ignore')

import logging

# Redirect stderr/stdout during DeepChem import to suppress dependency warnings
_stderr = sys.stderr
_stdout = sys.stdout
sys.stderr = open(os.devnull, 'w')
sys.stdout = open(os.devnull, 'w')

# Import DeepChem (will try to import optional dependencies)
import deepchem as dc

# Restore stderr/stdout
sys.stderr.close()
sys.stderr = _stderr
sys.stdout.close()
sys.stdout = _stdout

# Suppress DeepChem logging
logging.getLogger('deepchem').setLevel(logging.CRITICAL)
logging.getLogger('pytorch_lightning').setLevel(logging.CRITICAL)
logging.getLogger('jax').setLevel(logging.CRITICAL)

from typing import List, Optional
from datetime import datetime

import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem, DataStructs
from tqdm import tqdm

current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

import settings as cfg

def load_smiles_from_file(filepath: str, verbose: bool = True) -> List[str]:
    """
    Loads a list of SMILES from a text file (.smi, .txt).
    Assumes one SMILES per line, ignoring empty lines.
    
    Args:
        filepath: Path to the SMILES file
        verbose: If True, prints loading messages
    """
    if not os.path.exists(filepath):
        print(f"\n❌ ERROR: SMILES file '{filepath}' not found.")
        return []
    try:
        with open(filepath, 'r') as f:
            smiles_list = [line.strip() for line in f if line.strip()]
        if verbose:
            print(f"   ✓ Loaded {len(smiles_list)} SMILES from '{os.path.basename(filepath)}'")
        return smiles_list
    except Exception as e:
        print(f"\n❌ ERROR reading file '{filepath}': {e}")
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

def normalize_molecule(mol: Chem.Mol) -> Optional[Chem.Mol]:
    """
    Normalizes a molecule using RDKit MolStandardize.
    
    Steps:
    1. Remove salts/solvents (keep largest fragment)
    2. Neutralize charges (if possible)
    3. Canonicalize tautomers (optional, more intensive)
    
    Returns:
        Normalized molecule or None if fails
    """
    if mol is None:
        return None
    
    try:
        from rdkit.Chem.MolStandardize import rdMolStandardize
        
        # 1. Fragment parent (remove salts, keep largest fragment)
        #    Example: "CCO.Cl" -> "CCO"
        uncharger = rdMolStandardize.Uncharger()
        lfc = rdMolStandardize.LargestFragmentChooser()
        mol = lfc.choose(mol)
        
        # 2. Neutralize charges when possible
        #    Example: "CC(=O)[O-]" -> "CC(=O)O"
        mol = uncharger.uncharge(mol)
        
        # 3. Canonicalize tautomers (optional - can be slow)
        #    Example: "CC(O)=CC" -> "CC(=O)CC" (prefer keto form)
        # Uncomment if needed:
        # te = rdMolStandardize.TautomerEnumerator()
        # mol = te.Canonicalize(mol)
        
        # Sanitize final molecule
        Chem.SanitizeMol(mol)
        
        return mol
        
    except Exception as e:
        # If normalization fails, return None
        return None


def validate_smiles(smiles_list: List[str], normalize: bool = True, verbose: bool = True, show_progress: bool = False) -> List[str]:
    """
    Validates and canonicalizes SMILES strings.
    
    Args:
        smiles_list: List of SMILES strings
        normalize: If True, applies robust normalization (salts, charges, tautomers)
        verbose: If True, prints validation statistics
        show_progress: If True, shows progress bar (recommended for > 1000 molecules)
    
    Returns:
        List of valid canonical SMILES
    """
    from tqdm import tqdm
    
    valid_smiles = []
    invalid_count = 0
    normalized_count = 0

    # Use tqdm if requested
    iterator = tqdm(smiles_list, desc="  Validating", unit="mol") if show_progress else smiles_list

    for smi in iterator:
        mol = Chem.MolFromSmiles(smi)
        
        if mol is not None:
            # Apply normalization if requested
            if normalize:
                mol_normalized = normalize_molecule(mol)
                if mol_normalized is not None:
                    mol = mol_normalized
                    # Check if molecule changed after normalization
                    smi_normalized = Chem.MolToSmiles(mol)
                    if smi != smi_normalized:
                        normalized_count += 1
                else:
                    # Normalization failed, use original
                    pass
            
            # Canonicalize
            canonical_smi = Chem.MolToSmiles(mol)
            valid_smiles.append(canonical_smi)
        else:
            invalid_count += 1
    
    if verbose:
        if invalid_count > 0:
            print(f"    ⚠️  {invalid_count} invalid SMILES removed")
    
    return valid_smiles


def validate_smiles_chunked(smiles_list: List[str], chunk_size: int = 100000, 
                           normalize: bool = True, n_jobs: int = -1) -> List[str]:
    """
    Validates SMILES in chunks with parallel processing for HUGE datasets (millions+).
    
    Strategy for 1 billion molecules:
    1. Process in chunks (avoid memory overflow)
    2. Each chunk is parallelized internally
    3. Progress bar shows overall progress
    
    Performance estimate:
    - 1M molecules: ~30 seconds (8 cores)
    - 10M molecules: ~5 minutes (8 cores)
    - 100M molecules: ~50 minutes (8 cores)
    - 1B molecules: ~8 hours (8 cores)
    
    Args:
        smiles_list: List of SMILES strings
        chunk_size: Size of each chunk (default 100k = good balance)
        normalize: Apply normalization
        n_jobs: Number of parallel jobs (-1 = all cores)
    
    Returns:
        List of valid canonical SMILES
    """
    from joblib import Parallel, delayed
    from tqdm import tqdm
    import numpy as np
    
    def _process_single_smiles(smi: str, normalize: bool) -> tuple:
        """Process one SMILES (called in parallel) with warnings suppressed"""
        import warnings
        import os
        import sys
        
        # CRITICAL: Set environment BEFORE any imports
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
        os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
        os.environ['PYTHONWARNINGS'] = 'ignore'
        os.environ['PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION'] = 'python'
        
        # Suppress all warnings in worker processes
        warnings.filterwarnings('ignore')
        warnings.simplefilter('ignore')
        
        # Suppress TensorFlow logging
        import logging
        logging.getLogger('tensorflow').setLevel(logging.ERROR)
        logging.getLogger('tensorflow').propagate = False
        
        # Monkey-patch TensorFlow deprecation warnings
        try:
            import tensorflow as tf
            # Disable deprecation warnings by replacing the function
            tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
        except:
            pass
        
        # Redirect stderr temporarily during imports to suppress protobuf warnings
        stderr_backup = sys.stderr
        stdout_backup = sys.stdout
        try:
            sys.stderr = open(os.devnull, 'w')
            sys.stdout = open(os.devnull, 'w')
            
            # Suppress RDKit warnings (must be done per-process)
            from rdkit import RDLogger
            RDLogger.DisableLog('rdApp.*')
        finally:
            # Restore stderr/stdout after critical imports
            if sys.stderr != stderr_backup:
                sys.stderr.close()
                sys.stderr = stderr_backup
            if sys.stdout != stdout_backup:
                sys.stdout.close()
                sys.stdout = stdout_backup
        
        try:
            mol = Chem.MolFromSmiles(smi)
            if mol is None:
                return None, False
            
            if normalize:
                mol_normalized = normalize_molecule(mol)
                if mol_normalized is not None:
                    original_smi = Chem.MolToSmiles(mol)
                    mol = mol_normalized
                    canonical_smi = Chem.MolToSmiles(mol)
                    was_normalized = (original_smi != canonical_smi)
                    return canonical_smi, was_normalized
            
            canonical_smi = Chem.MolToSmiles(mol)
            return canonical_smi, False
        except:
            return None, False
    
    total = len(smiles_list)
    num_chunks = (total + chunk_size - 1) // chunk_size
    
    all_valid_smiles = []
    total_invalid = 0
    total_normalized = 0
    
    # Process chunks with progress bar (showing molecules, not chunks!)
    print()
    pbar = tqdm(total=total, desc="Validating", unit="mol", unit_scale=True)
    
    for chunk_idx in range(num_chunks):
        start_idx = chunk_idx * chunk_size
        end_idx = min(start_idx + chunk_size, total)
        chunk = smiles_list[start_idx:end_idx]
        
        # Parallel process this chunk
        results = Parallel(n_jobs=n_jobs, backend='loky')(
            delayed(_process_single_smiles)(smi, normalize) for smi in chunk
        )
        
        # Collect results and update progress by molecules processed
        for canonical_smi, was_normalized in results:
            if canonical_smi is not None:
                all_valid_smiles.append(canonical_smi)
                if was_normalized:
                    total_normalized += 1
            else:
                total_invalid += 1
            pbar.update(1)  # Update for each molecule processed
    
    pbar.close()
    
    print(f"   ✓ Processed {len(all_valid_smiles):,}/{total:,} valid molecules")

    # Return the list of valid canonical SMILES so callers receive the results
    return all_valid_smiles


def load_and_featurize_full_dataset() -> Optional[dc.data.Dataset]:

    actives_raw = load_smiles_from_file(cfg.ACTIVE_SMILES_FILE)
    inactives_raw = load_smiles_from_file(cfg.INACTIVE_SMILES_FILE)
    
    if not actives_raw or not inactives_raw:
        print("\nERROR: Failed to load SMILES files.")
        return None

    print("\nValidating SMILES...")
    actives = validate_smiles(actives_raw)
    inactives = validate_smiles(inactives_raw)
    
    if not actives or not inactives:
        print("\nERROR: No valid SMILES found after validation.")
        return None

    print(f"\nLoaded dataset: {len(actives)} actives, {len(inactives)} inactives")

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