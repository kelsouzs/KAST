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
PREDICTION_FEATURIZED_DIR = RESULTS_DIR / 'prediction_featurized'

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
N_JOBS = -1         
VERBOSE = True      
DEBUG = False    
