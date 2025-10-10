"""
K-talysticFlow - Step 4.1: Cross-Validation

This script evaluates the robustness and generalization ability of the model
using k-fold Cross-Validation.

Technique Used:
- Scaffold Split: This splitting method groups molecules with the same
  "scaffold" (core molecular skeleton) into the same set. This forces the
  model to learn to generalize to new chemical structures.
"""

import os
import warnings
import logging

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
warnings.filterwarnings('ignore', category=DeprecationWarning)
warnings.filterwarnings('ignore', category=UserWarning)
warnings.filterwarnings('ignore', category=FutureWarning)

import tensorflow as tf
tf.get_logger().setLevel('ERROR')
logging.getLogger('deepchem').setLevel('ERROR')

import sys
import numpy as np
import deepchem as dc
from sklearn.metrics import roc_auc_score
from multiprocessing import cpu_count
from joblib import Parallel, delayed
from typing import List, Tuple
from tqdm import tqdm
import platform

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import settings as cfg
from utils import ensure_dir_exists, load_smiles_from_file


# ============================================================================
# PARALLEL FEATURIZATION FOR CROSS-VALIDATION
# ============================================================================

def get_optimal_workers() -> int:
    """Get optimal number of workers from settings.py configuration."""
    if cfg.N_WORKERS is not None:
        if cfg.N_WORKERS == -1:
            return cpu_count() or 4
        elif cfg.N_WORKERS >= 1:
            return cfg.N_WORKERS
    n_cpus = cpu_count() or 4
    return max(1, n_cpus - 1)


def featurize_chunk(smiles_chunk: List[str], fp_size: int, fp_radius: int) -> List:
    """Featurize a chunk of SMILES in parallel worker."""
    try:
        import warnings
        warnings.filterwarnings('ignore')
        import os
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
        
        from rdkit import RDLogger
        RDLogger.DisableLog('rdApp.*')
        
        import deepchem as dc
        featurizer = dc.feat.CircularFingerprint(size=fp_size, radius=fp_radius)
        features = featurizer.featurize(smiles_chunk)
        return features
    except Exception as e:
        print(f"⚠️ Worker error: {e}")
        return [None] * len(smiles_chunk)


def featurize_parallel(smiles_list: List[str], fp_size: int, fp_radius: int) -> np.ndarray:
    """Featurize SMILES using parallel processing."""
    n_workers = get_optimal_workers()
    
    print(f"⚡ Parallel featurization: {n_workers} workers")
    
    # Split into chunks
    chunk_size = len(smiles_list) // n_workers
    if chunk_size == 0:
        chunk_size = len(smiles_list)
    
    chunks = [smiles_list[i:i + chunk_size] for i in range(0, len(smiles_list), chunk_size)]
    
    # Process in parallel
    results = Parallel(n_jobs=n_workers, verbose=5, backend='loky')(
        delayed(featurize_chunk)(chunk, fp_size, fp_radius)
        for chunk in chunks
    )
    
    # Flatten results
    features = []
    for chunk_features in results:
        features.extend(chunk_features)
    
    return np.array(features)


# ============================================================================


def ensure_cv_reproducibility(seed=42): 
    import random
    import numpy as np
    import tensorflow as tf
    
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)
    
    os.environ['TF_DETERMINISTIC_OPS'] = '1'
    os.environ['PYTHONHASHSEED'] = str(seed)
    
    try:
        tf.config.experimental.enable_op_determinism()
    except:
        pass
    
    try:
        import deepchem as dc
        dc.utils.set_random_seed(seed)
    except:
        pass
    
    print("✅ Cross-validation reproducibility configured")

def load_and_featurize_all_data():
    from rdkit import RDLogger
    RDLogger.DisableLog('rdApp.*')
    
    print("\nLoading all data for Cross-Validation...")
    
    try:
        actives = load_smiles_from_file(cfg.ACTIVE_SMILES_FILE)
        inactives = load_smiles_from_file(cfg.INACTIVE_SMILES_FILE)
        
        if not actives or not inactives:
            print("\nERROR: Failed to load SMILES files.")
            print("➡️ Check that the files exist and contain valid data.")
            return None

        all_smiles = np.array(actives + inactives)
        all_labels = np.array([1] * len(actives) + [0] * len(inactives))

        print(f"  -> Loaded active molecules: {len(actives)}")
        print(f"  -> Loaded inactive molecules: {len(inactives)}")
        print("  -> Featurizing entire dataset...")
        
        # Use parallel featurization if enabled and dataset is large enough
        use_parallel = (cfg.ENABLE_PARALLEL_PROCESSING and 
                        len(all_smiles) >= cfg.PARALLEL_MIN_THRESHOLD)
        
        if use_parallel:
            print(f"  -> Using parallel featurization ({platform.system()})")
            features = featurize_parallel(list(all_smiles), cfg.FP_SIZE, cfg.FP_RADIUS)
        else:
            if not cfg.ENABLE_PARALLEL_PROCESSING:
                print(f"  -> Using sequential featurization (parallel disabled)")
            else:
                print(f"  -> Using sequential featurization (dataset < {cfg.PARALLEL_MIN_THRESHOLD:,})")
            featurizer = dc.feat.CircularFingerprint(size=cfg.FP_SIZE, radius=cfg.FP_RADIUS)
            features = featurizer.featurize(all_smiles)
        
        valid_indices = [i for i, x in enumerate(features) if x is not None and x.size > 0]
        features_valid = features[valid_indices]
        labels_valid = all_labels[valid_indices]
        smiles_valid = all_smiles[valid_indices]
        
        print(f"\n  -> Successfully featurized {len(features_valid)} out of {len(all_smiles)} molecules.")
        
        return dc.data.NumpyDataset(X=features_valid, y=labels_valid, ids=smiles_valid)
        
    except Exception as e:
        print(f"\nFeaturization error: {e}")
        return None

def run_cross_validation(dataset):
    """Performs k-fold cross-validation on the dataset."""
    print(f"\nStarting {cfg.N_FOLDS_CV}-fold Cross-Validation with ScaffoldSplitter...")

    try:
        splitter = dc.splits.ScaffoldSplitter()
        folds = splitter.k_fold_split(dataset, k=cfg.N_FOLDS_CV)
    except Exception as e:
        print(f"\n ⚠️ WARNING: ScaffoldSplitter failed ({e})")
        print("  -> Using RandomStratifiedSplitter as fallback...")
        splitter = dc.splits.RandomStratifiedSplitter()
        folds = splitter.k_fold_split(dataset, k=cfg.N_FOLDS_CV)

    cv_auc_scores = []
    
    for i, (train_fold, valid_fold) in enumerate(folds):
        print(f" Fold {i+1}/{cfg.N_FOLDS_CV}: Training...")
        
        try:
            model_cv = dc.models.MultitaskClassifier(
                n_tasks=cfg.MODEL_PARAMS['n_tasks'],
                n_features=cfg.FP_SIZE,
                layer_sizes=cfg.MODEL_PARAMS['layer_sizes'],
                dropouts=cfg.MODEL_PARAMS['dropouts'],
                mode=cfg.MODEL_PARAMS['mode'],
                learning_rate=cfg.MODEL_PARAMS['learning_rate']
            )
            
            model_cv.fit(train_fold, nb_epoch=cfg.MODEL_PARAMS['nb_epoch'])
            
            y_true_cv = valid_fold.y.flatten()
            y_pred_proba_raw = model_cv.predict(valid_fold)
            
            if y_pred_proba_raw.ndim == 3:
                y_pred_proba_cv = y_pred_proba_raw[:, 0, 1]
            else:
                y_pred_proba_cv = y_pred_proba_raw[:, 1]
            
            if len(np.unique(y_true_cv)) < 2:
                fold_auc = np.nan
                print(f"\n WARNING: Only one class in fold {i+1}. AUC not computed.")
            else:
                fold_auc = roc_auc_score(y_true_cv, y_pred_proba_cv)
                print(f"\n Fold validation AUC: {fold_auc:.4f}")
            
            cv_auc_scores.append(fold_auc)
            
        except Exception as e:
            print(f"\n❌ ERROR in fold {i+1}: {e}")
            cv_auc_scores.append(np.nan)

    return cv_auc_scores

def display_summary(mean_auc, std_auc):
    """Displays a formatted summary of cross-validation results."""
    print("\n" + "-"*57)
    print("CROSS-VALIDATION SUMMARY".center(57))
    print("-"*57)
    print(f"  Mean AUC (average of folds) : {mean_auc:.4f}")
    print(f"  Std Dev (AUC)               : {std_auc:.4f}")
    print("-"*57)

def save_cv_results(scores):
    """Calculates statistics, saves, and returns cross-validation results."""
    valid_scores = [s for s in scores if not np.isnan(s)]
    if not valid_scores:
        print("\nERROR: No AUC scores could be calculated.")
        return None, None

    mean_auc = np.mean(valid_scores)
    std_auc = np.std(valid_scores)

    try:
        results_path = os.path.join(cfg.RESULTS_DIR, '4_1_cross_validation_results.txt')
        ensure_dir_exists(cfg.RESULTS_DIR)
        
        report = (
            f"=== CROSS-VALIDATION RESULTS ===\n"
            f"Configuration: {cfg.N_FOLDS_CV} folds\n"
            f"Split method: ScaffoldSplitter\n\n"
            f"AUC scores per fold:\n"
        )
        
        for i, score in enumerate(valid_scores):
            report += f"  Fold {i+1}: {score:.4f}\n"
        
        report += f"\n=== FINAL STATISTICS ===\n"
        report += f"Mean AUC: {mean_auc:.4f}\n"
        report += f"Std Dev AUC: {std_auc:.4f}\n"
        report += f"Valid folds: {len(valid_scores)}/{len(scores)}\n"
        
        with open(results_path, "w") as f:
            f.write(report)
        
        print(f"\n✅ Full report saved at: {results_path}")
        return mean_auc, std_auc
        
    except Exception as e:
        print(f"\n ⚠️ ERROR saving results: {e}")
        return mean_auc, std_auc


def main():
    ensure_cv_reproducibility(seed=42)
    
    from utils import print_script_banner, setup_script_logging
    logger = setup_script_logging("4_1_cross_validation")
    
    print_script_banner("K-talysticFlow | Step 4.1: Cross-Validation")
    logger.info("Starting cross-validation")

    full_dataset = load_and_featurize_all_data()
    if full_dataset is None:
        logger.error("Failed to load and featurize data")
        sys.exit(1)

    cv_scores = run_cross_validation(full_dataset)


    mean_auc, std_auc = save_cv_results(cv_scores)


    if mean_auc is not None:
        display_summary(mean_auc, std_auc)
        print("\n✅ Cross-Validation completed successfully!")
        print("\n➡️ Next step: run the remaining detailed evaluation scripts.")
        logger.info("Cross-validation completed successfully")
    else:
        print("\n❌ Cross-Validation failed!")
        logger.error("Cross-validation failed")
        sys.exit(1)

if __name__ == '__main__':
    main()
