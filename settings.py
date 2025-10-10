import os
from pathlib import Path

# --- 1. MAIN PATHS ---
PROJECT_ROOT = Path(__file__).parent.resolve()
DATA_RAW_DIR = PROJECT_ROOT / 'data'
RESULTS_DIR = PROJECT_ROOT / 'results'
ACTIVE_SMILES_FILE = DATA_RAW_DIR / 'actives.smi'
INACTIVE_SMILES_FILE = DATA_RAW_DIR / 'inactives.smi'

# --- 2. BASIC CONFIGURATIONS ---
TEST_SET_FRACTION = 0.3
RANDOM_STATE = 42
FP_SIZE = 2048
FP_RADIUS = 3

# --- 3. OUTPUT FILES FROM SCRIPT 1 ---
OUTPUT_TRAIN_CSV = '01_train_set.csv'
OUTPUT_TEST_CSV = '01_test_set.csv'

# --- 5. PROCESSING DIRECTORIES ---
PREPARED_DATA_DIR = RESULTS_DIR / 'prepared_data'        
FEATURIZED_DATA_DIR = RESULTS_DIR / 'featurized_datasets'   

TRAIN_DATA_DIR = os.path.join(str(FEATURIZED_DATA_DIR), 'train')              
TEST_DATA_DIR = os.path.join(str(FEATURIZED_DATA_DIR), 'test')                

MODEL_DIR = RESULTS_DIR / 'trained_model'
PREDICTION_FEATURIZED_BASE_DIR = RESULTS_DIR / 'prediction_featurized'  # Base directory for all prediction datasets
PREDICTION_FEATURIZED_DIR = PREDICTION_FEATURIZED_BASE_DIR  # Default (will be updated dynamically)

# --- 5. MODEL PARAMETERS ---
MODEL_PARAMS = {
    'n_tasks': 1,
    'layer_sizes': [1000, 500],
    'dropouts': 0.25,
    'learning_rate': 0.001,
    'mode': 'classification',
    'nb_epoch': 50  
}

# --- 6. TRAINING CONFIGURATIONS ---
NB_EPOCH_TRAIN = 50
NB_EPOCH_CV = 30
NB_EPOCH_LC = 20
CLASSIFICATION_THRESHOLD = 0.5

# --- 7. VALIDATION CONFIGURATIONS ---
N_FOLDS_CV = 5                                            
EF_FRACTIONS_PERCENT = [1.0, 2.0, 5.0, 10.0]              
ENRICHMENT_FACTORS = [0.01, 0.05, 0.1] 
TANIMOTO_SAMPLE_SIZE = 1000             

# --- 8. DATA VALIDATION CONFIGURATIONS ---
MIN_MOLECULES_PER_CLASS = 50
MAX_MOLECULES_TOTAL = 100000
MIN_SMILES_LENGTH = 5
MAX_SMILES_LENGTH = 200

# --- 9. COLUMN NAMES ---
SMILES_COL = 'smiles'
LABEL_COL = 'active'  

# --- 10. FINAL OUTPUT FILES ---
OUTPUT_PREDICTIONS_CSV = '5_1_new_molecule_predictions.csv'

# --- 11. EXTRA CONFIGURATIONS (FOR COMPATIBILITY) ---
N_JOBS = -1          # ✅ For parallelization
VERBOSE = True       # ✅ For verbose logs
DEBUG = False        # ✅ For debugging

# --- 12. PARALLEL PROCESSING CONFIGURATIONS ---
# ============================================================================
# Global configurations for parallel processing across ALL scripts
# These settings control how the pipeline uses multiprocessing to speed up
# computation-intensive tasks like featurization, prediction, and analysis.
# ============================================================================

# Enable/disable parallel processing globally for ALL scripts
# Set to False if you experience issues or want sequential processing
ENABLE_PARALLEL_PROCESSING = True

# Number of CPU cores to use (applies to ALL parallelized scripts)
# Options:
#   None  = auto-detect optimal (cpu_count - 1) ✅ RECOMMENDED
#           This leaves one core free for system operations
#   -1    = use ALL available cores (may slow down system responsiveness)
#   1     = disable parallelism completely (sequential processing only)
#   N     = use exactly N cores (e.g., 2, 4, 8, 16)
#
# Examples:
#   N_WORKERS = None   # Auto: uses 7 cores on 8-core CPU
#   N_WORKERS = 4      # Fixed: always uses 4 cores
#   N_WORKERS = 1      # Sequential: no parallelism
N_WORKERS = 10

# Batch size for memory-efficient parallel processing
# Larger values = faster but use more RAM
# Smaller values = slower but safer for limited RAM
# Recommended: 100000 for systems with 16GB+ RAM
#              50000 for systems with 8GB RAM
#              25000 for systems with 4GB RAM
PARALLEL_BATCH_SIZE = 100000  # molecules per batch

# Minimum dataset size to trigger parallel processing
# Below this threshold, sequential processing is actually faster
# due to multiprocessing overhead (process spawning, data serialization)
# Recommended: Keep at 10000 unless you have specific reasons to change
PARALLEL_MIN_THRESHOLD = 10000  # molecules

# ============================================================================
# Scripts that use these configurations:
#   - 2_featurization.py        (train/test featurization)
#   - 5_0_featurize_for_prediction.py  (prediction featurization)
#   - 5_1_run_prediction.py     (batch predictions)
#   - 4_1_cross_validation.py   (k-fold CV)
#   - 4_3_tanimoto_similarity.py (similarity calculations)
# ============================================================================