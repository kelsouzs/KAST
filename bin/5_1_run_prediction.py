"""
K-talysticFlow - Step 5.1: Running Predictions

This script loads the featurized database (prepared in step 5.0)
and the trained model (generated in step 3) to perform activity
predictions on new molecules.

REQUIREMENTS:
- Featurized dataset in cfg.PREDICTION_FEATURIZED_DIR (created by step 5.0)
- Trained model in cfg.MODEL_DIR (created by step 3)

OUTPUT:
- CSV file with predictions sorted by descending probability
- TXT report with prediction statistics and top candidates
- SMI file with SMILES and molecule names for top candidates
"""

import sys
import os
import logging
import warnings
from typing import Tuple, Optional
import random

# Suppress all warnings and TensorFlow/DeepChem messages BEFORE imports
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TF_CPP_MIN_VLOG_LEVEL'] = '3'
warnings.filterwarnings('ignore')
import logging
logging.getLogger().setLevel(logging.ERROR)

import tensorflow as tf
tf.get_logger().setLevel('ERROR')
logging.getLogger('tensorflow').setLevel('ERROR')
logging.getLogger('deepchem').setLevel('ERROR')
warnings.filterwarnings('ignore', category=UserWarning)
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=DeprecationWarning)

import pandas as pd
import deepchem as dc
import numpy as np
from tqdm import tqdm
import h5py
from scipy.sparse import csr_matrix
from multiprocessing import cpu_count
from joblib import Parallel, delayed
from typing import List
import platform

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import settings as cfg
from utils import ensure_dir_exists


# ============================================================================
# PARALLEL PREDICTION FUNCTIONS
# ============================================================================

def get_optimal_workers() -> int:
    """Get optimal number of workers from settings.py configuration."""
    if cfg.N_WORKERS is not None:
        if cfg.N_WORKERS == -1:
            return cpu_count() or 4
        elif cfg.N_WORKERS >= 1:
            return cfg.N_WORKERS
    
    # Auto-detect (N_WORKERS = None)
    n_cpus = cpu_count() or 4
    return max(1, n_cpus - 1)


def predict_batch_worker(batch_data: Tuple[int, csr_matrix], model_dir: str, model_params: dict) -> Tuple[int, np.ndarray]:
    """
    Worker function to predict on a single batch.
    Each worker loads its own model instance to avoid pickling issues.
    
    Args:
        batch_data: Tuple of (batch_index, sparse_batch)
        model_dir: Path to model directory
        model_params: Model configuration parameters
    
    Returns:
        Tuple of (batch_index, predictions)
    """
    try:
        # CRITICAL: Suppress ALL warnings/logs in worker processes
        import warnings
        import logging
        import os
        import sys
        
        warnings.filterwarnings('ignore')
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
        os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
        logging.getLogger().setLevel(logging.ERROR)
        
        # Redirect stderr to null to suppress DeepChem import warnings
        stderr_backup = sys.stderr
        sys.stderr = open(os.devnull, 'w')
        
        import deepchem as dc
        import numpy as np
        
        # Restore stderr after import
        sys.stderr.close()
        sys.stderr = stderr_backup
        
        batch_idx, batch_sparse = batch_data
        
        # Each worker needs its own model instance
        n_features = batch_sparse.shape[1]
        model = dc.models.MultitaskClassifier(
            n_tasks=model_params['n_tasks'],
            n_features=n_features,
            layer_sizes=model_params['layer_sizes'],
            dropouts=model_params['dropouts'],
            mode=model_params['mode'],
            learning_rate=model_params['learning_rate'],
            model_dir=model_dir
        )
        model.restore()
        
        # Convert batch to dense and predict
        batch_dense = batch_sparse.toarray()
        batch_preds = model.predict_on_batch(batch_dense)
        
        return batch_idx, batch_preds
        
    except Exception as e:
        print(f"⚠️ Worker error {batch_idx}: {e}")
        # Return empty predictions on error
        return batch_idx, np.array([])


# ============================================================================


def ensure_reproducibility(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)
    
    os.environ['TF_DETERMINISTIC_OPS'] = '1'
    os.environ['PYTHONHASHSEED'] = str(seed)

    try:
        tf.config.experimental.enable_op_determinism()
        print("✅ TensorFlow deterministic operations enabled")
    except Exception:
        print("⚠️ TensorFlow deterministic operations not available (older TF version)")


def generate_molecule_names(smiles_list, prefix="KAST"):
    total_molecules = len(smiles_list)
    
    if total_molecules < 1000:
        padding = 3  
    elif total_molecules < 10000:
        padding = 4  
    elif total_molecules < 100000:
        padding = 5  
    else:
        padding = 6 
    
    molecule_names = []
    for i, smiles in enumerate(smiles_list, 1):
        name = f"{prefix}_{i:0{padding}d}"
        molecule_names.append(name)
    
    return molecule_names


def load_sparse_hdf5_data() -> Optional[Tuple[csr_matrix, list]]:
    """Load sparse featurized data from HDF5 file."""
    from rdkit import RDLogger
    RDLogger.DisableLog('rdApp.*')

    print("📦 Loading featurized dataset...")

    if not os.path.exists(cfg.PREDICTION_FEATURIZED_DIR):
        print(f"\nERROR: Directory '{cfg.PREDICTION_FEATURIZED_DIR}' not found.")
        print("➡️ Run '[5] -> [1] Prepare database' first.")
        return None, None
    
    hdf5_path = os.path.join(cfg.PREDICTION_FEATURIZED_DIR, 'featurized_data.h5')
    
    if not os.path.exists(hdf5_path):
        print(f"\nERROR: file not found at '{hdf5_path}'")
        print("➡️ Run step 5.0 to featurize your data first.")
        return None, None
    
    try:
        with h5py.File(hdf5_path, 'r') as h5f:
            # Load sparse matrix components
            data = h5f['data'][:]
            indices = h5f['indices'][:]
            indptr = h5f['indptr'][:]
            shape = tuple(h5f['shape'][:])
            
            # Reconstruct sparse matrix
            features_sparse = csr_matrix((data, indices, indptr), shape=shape)
            
            # Load SMILES
            smiles_list = [s.decode('utf-8') if isinstance(s, bytes) else s 
                          for s in h5f['smiles'][:]]
            
            # Load metadata
            total_molecules = h5f.attrs['total_molecules']
            space_saved = h5f.attrs.get('space_saved_percent', 0)
            
            file_size_mb = os.path.getsize(hdf5_path) / (1024**2)
            
            print(f"\n✅ Dataset loaded successfully:")
            print(f"  • Molecules: {total_molecules:,}")
            
            return features_sparse, smiles_list
            
    except Exception as e:
        print(f"\n⚠️ ERROR loading dataset: {e}")
        return None, None


def load_data_and_model() -> Optional[Tuple]:
    """Load sparse data and trained model."""
    # Load sparse features
    features_sparse, smiles_list = load_sparse_hdf5_data()
    if features_sparse is None:
        return None, None, None
    
    print(f"\n🤖 Loading trained model...")

    if not os.path.exists(cfg.MODEL_DIR):
        print(f"\n ⚠️ ERROR: Model '{cfg.MODEL_DIR}' not found.")
        print("➡️ Run '[3] Train Model' first.")
        return None, None, None
        
    try:
        n_features = features_sparse.shape[1]
        
        model = dc.models.MultitaskClassifier(
            n_tasks=cfg.MODEL_PARAMS['n_tasks'],
            n_features=n_features,
            layer_sizes=cfg.MODEL_PARAMS['layer_sizes'],
            dropouts=cfg.MODEL_PARAMS['dropouts'],
            mode=cfg.MODEL_PARAMS['mode'],
            learning_rate=cfg.MODEL_PARAMS['learning_rate'],
            model_dir=cfg.MODEL_DIR
        )
        
        model.restore()
        print("✅ Model loaded successfully")
        
        return features_sparse, smiles_list, model
        
    except Exception as e:
        print(f"\n⚠️ ERROR loading model: {e}")
        return None, None, None


def run_prediction(model: dc.models.Model, features_sparse: csr_matrix, smiles_list: list) -> pd.DataFrame:
    """
    Run predictions on sparse feature matrix.
    Uses parallel processing if enabled in settings.py.
    """
    print(f"\n🔮 Running predictions for {len(smiles_list):,} molecules...")
    
    try:
        batch_size = 512
        total_molecules = features_sparse.shape[0]
        total_batches = (total_molecules + batch_size - 1) // batch_size
        
        # Check if parallel processing should be used
        use_parallel = (cfg.ENABLE_PARALLEL_PROCESSING and 
                        total_molecules >= cfg.PARALLEL_MIN_THRESHOLD)
        
        if use_parallel:
            # PARALLEL PREDICTION
            n_workers = get_optimal_workers()
            print(f"\n⚡ Using parallel prediction:")
            
            # Prepare batches
            batches = []
            for i in range(0, total_molecules, batch_size):
                end_idx = min(i + batch_size, total_molecules)
                batch_sparse = features_sparse[i:end_idx]
                batches.append((i // batch_size, batch_sparse))
            
            # Process batches in parallel with tqdm progress bar
            print(f"\n🚀 Processing {total_batches} batches with {n_workers} workers...")
            
            from tqdm import tqdm
            results = Parallel(n_jobs=n_workers, verbose=0, backend='loky')(
                delayed(predict_batch_worker)(batch, cfg.MODEL_DIR, cfg.MODEL_PARAMS)
                for batch in tqdm(batches, desc="Predicting", unit="batch")
            )
            
            # Sort results by batch index and concatenate
            results.sort(key=lambda x: x[0])
            y_pred_proba_list = [pred for _, pred in results if len(pred) > 0]
            
            if not y_pred_proba_list:
                print("\n❌ ERROR: No predictions generated")
                return pd.DataFrame()
            
            y_pred_proba_raw = np.concatenate(y_pred_proba_list, axis=0)
            print(f"\n✅ Prediction complete!")
            
        else:
            # SEQUENTIAL PREDICTION (fallback)
            print(f"\n📝 Using sequential prediction")
            if not cfg.ENABLE_PARALLEL_PROCESSING:
                print(f"   • Reason: Parallel processing disabled in settings.py")
            else:
                print(f"   • Reason: Dataset < {cfg.PARALLEL_MIN_THRESHOLD:,} molecules")
            
            y_pred_proba_list = []
            
            print(f"\nProcessing {total_batches} batches...")
            for i in tqdm(range(0, total_molecules, batch_size), 
                          total=total_batches, 
                          desc="Prediction Progress"):
                end_idx = min(i + batch_size, total_molecules)
                batch_sparse = features_sparse[i:end_idx]
                batch_dense = batch_sparse.toarray()
                batch_preds = model.predict_on_batch(batch_dense)
                y_pred_proba_list.append(batch_preds)
                del batch_sparse, batch_dense

            y_pred_proba_raw = np.concatenate(y_pred_proba_list, axis=0)

        if y_pred_proba_raw.ndim == 3:
            y_pred_proba_active = y_pred_proba_raw[:, 0, 1]
        elif y_pred_proba_raw.ndim == 2 and y_pred_proba_raw.shape[1] >= 2:
            y_pred_proba_active = y_pred_proba_raw[:, 1]
        else:
            print(f"\n⚠️ WARNING: Unexpected prediction shape: {y_pred_proba_raw.shape}")
            y_pred_proba_active = y_pred_proba_raw.flatten()

        molecule_names = generate_molecule_names(smiles_list, prefix="KAST")
        
        results_df = pd.DataFrame({
            'molecule_name': molecule_names,
            'smiles': smiles_list,
            'K-prediction Score': y_pred_proba_active
        })

        results_df = results_df.sort_values(
            by='K-prediction Score', 
            ascending=False
        ).reset_index(drop=True)
        
        return results_df
        
    except Exception as e:
        print(f"\n⚠️ ERROR during prediction: {e}")
        import traceback
        traceback.print_exc()
        return pd.DataFrame()


def display_prediction_statistics(results_df: pd.DataFrame):
    total_molecules = len(results_df)
    
    very_high = len(results_df[results_df['K-prediction Score'] > 0.9])
    high_prob = len(results_df[(results_df['K-prediction Score'] > 0.7) & 
                              (results_df['K-prediction Score'] <= 0.9)])
    medium_prob = len(results_df[(results_df['K-prediction Score'] > 0.5) & 
                                 (results_df['K-prediction Score'] <= 0.7)])
    low_medium = len(results_df[(results_df['K-prediction Score'] > 0.3) & 
                               (results_df['K-prediction Score'] <= 0.5)])
    low_prob = len(results_df[results_df['K-prediction Score'] <= 0.3])
    
    print("\n" + "="*70)
    print("                  PREDICTION STATISTICS SUMMARY")
    print("="*70)
    print(f"\n               📊 Total molecules analyzed: {total_molecules}")
    print("\n  🎯 K-Prediction Score Distribution:")
    print(f"\n  🔥 Very High (> 0.9)    : {very_high:6d} molecules ({very_high/total_molecules*100:5.1f}%)")
    print(f"  ⭐ High (0.7-0.9)       : {high_prob:6d} molecules ({high_prob/total_molecules*100:5.1f}%)")
    print(f"  💛 Medium (0.5-0.7)     : {medium_prob:6d} molecules ({medium_prob/total_molecules*100:5.1f}%)")
    print(f"  🟡 Low-Medium (0.3-0.5) : {low_medium:6d} molecules ({low_medium/total_molecules*100:5.1f}%)")
    print(f"  ⚪ Low (≤ 0.3)          : {low_prob:6d} molecules ({low_prob/total_molecules*100:5.1f}%)")
    print("-"*70)
    
    return {
        'total': total_molecules,
        'very_high': very_high,
        'high': high_prob,
        'medium': medium_prob,
        'low_medium': low_medium,
        'low': low_prob
    }


def get_custom_filename():
    """Ask user for custom filename prefix."""
    print("\n               📝 Custom File Naming")
    
    while True:
        try:
            filename = input("\nEnter a custom name for your output files (or press Enter for default): ").strip()
            
            if not filename:
                return "5_1_predictions"
            
            invalid_chars = ['<', '>', ':', '"', '/', '\\', '|', '?', '*']
            for char in invalid_chars:
                filename = filename.replace(char, '_')
            
            if len(filename) > 50:
                print("⚠️ Filename too long. Please use less than 50 characters.")
                print("-"*70)
                continue
                
            print(f"\n✅ Files will be saved as: {filename}.csv, {filename}_report.txt, {filename}.smi")
            print("-"*70)
            return filename
            
        except KeyboardInterrupt:
            print("\nOperation cancelled by user.")
            print("-"*70)
            sys.exit(0)


def get_smi_export_preference(stats):
    """Ask user about SMI file export preferences."""
    print("\n               🧬 SMI File Export Options")
    print("\nChoose which molecules to export to .smi file:")
    print()
    print("  1. All molecules ({:,} total)".format(stats['total']))
    print("  2. High activity candidates only (score > 0.7) - {:,} molecules".format(stats['very_high'] + stats['high']))
    print("  3. Medium to high activity (score > 0.5) - {:,} molecules".format(stats['very_high'] + stats['high'] + stats['medium']))
    print("  4. Custom cutoff (you define the minimum score)")
    print("  5. Top N molecules (you define the number)")
    print()
    
    while True:
        try:
            choice = input("Select option (1-5): ").strip()
            
            if choice == '1':
                return 'all', None
            elif choice == '2':
                return 'cutoff', 0.7
            elif choice == '3':
                return 'cutoff', 0.5
            elif choice == '4':
                while True:
                    try:
                        cutoff = float(input("Enter minimum K-prediction score (0.0-1.0): "))
                        if 0.0 <= cutoff <= 1.0:
                            return 'cutoff', cutoff
                        else:
                            print("⚠️ Please enter a value between 0.0 and 1.0")
                    except ValueError:
                        print("⚠️ Please enter a valid number")
            elif choice == '5':
                while True:
                    try:
                        top_n = int(input(f"Enter number of top molecules (1-{stats['total']}): "))
                        if 1 <= top_n <= stats['total']:
                            return 'top_n', top_n
                        else:
                            print(f"⚠️ Please enter a number between 1 and {stats['total']}")
                    except ValueError:
                        print("⚠️ Please enter a valid integer")
            else:
                print("⚠️ Invalid option. Please select 1-5.")
                
        except KeyboardInterrupt:
            print("\nOperation cancelled by user.")
            sys.exit(0)


def save_smi_file(results_df: pd.DataFrame, output_path: str, export_type: str, value: float = None):
    """Save SMI file based on user preferences."""
    try:
        if export_type == 'all':
            df_to_save = results_df
            description = "all molecules"
        elif export_type == 'cutoff':
            df_to_save = results_df[results_df['K-prediction Score'] > value]
            description = f"molecules with score > {value}"
        elif export_type == 'top_n':
            df_to_save = results_df.head(int(value))
            description = f"top {int(value)} molecules"
        else:
            df_to_save = results_df
            description = "all molecules"
        
        with open(output_path, 'w') as f:
            for _, row in df_to_save.iterrows():
                f.write(f"{row['smiles']}\t{row['molecule_name']}\n")
        
        count = len(df_to_save)
        print(f"\n📄 SMI file saved: {output_path}")
        print(f"   🧬 Contains {count:,} molecules ({description})")
        
        if count == 0:
            print("⚠️ Warning: No molecules met the specified criteria!")
        
    except Exception as e:
        print(f"⚠️ ERROR saving SMI file: {e}")


def display_and_save_results(results_df: pd.DataFrame, custom_filename: str, smi_export_type: str, smi_value: float = None):
    """
    Save results with custom filename and SMI preferences.
    Uses temporary files and atomic operations to prevent partial/corrupted outputs.
    """
    if results_df.empty:
        print("\n⚠️ ERROR: No results to save.")
        return
    
    # Define final output paths
    output_csv = os.path.join(cfg.RESULTS_DIR, f"{custom_filename}.csv")
    output_report = os.path.join(cfg.RESULTS_DIR, f"{custom_filename}_report.txt")
    output_smi = os.path.join(cfg.RESULTS_DIR, f"{custom_filename}.smi")
    
    # Define temporary file paths
    temp_csv = output_csv + ".tmp"
    temp_report = output_report + ".tmp"
    temp_smi = output_smi + ".tmp"
    
    temp_files = [temp_csv, temp_report, temp_smi]
    
    try:
        ensure_dir_exists(cfg.RESULTS_DIR)
        
        # Step 1: Save CSV to temporary file
        print("\n💾 Saving results...")
        results_df.to_csv(temp_csv, index=False)

        total_molecules = len(results_df)
        very_high = len(results_df[results_df['K-prediction Score'] > 0.9])
        high_prob = len(results_df[(results_df['K-prediction Score'] > 0.7) & 
                                  (results_df['K-prediction Score'] <= 0.9)])
        medium_prob = len(results_df[(results_df['K-prediction Score'] > 0.5) & 
                                    (results_df['K-prediction Score'] <= 0.7)])
        low_medium = len(results_df[(results_df['K-prediction Score'] > 0.3) & 
                                   (results_df['K-prediction Score'] <= 0.5)])
        low_prob = len(results_df[results_df['K-prediction Score'] <= 0.3])

        # Step 2: Generate report to temporary file
        with open(temp_report, "w", encoding="utf-8") as rep:
            rep.write("="*70 + "\n")
            rep.write("                    PREDICTION RESULTS REPORT\n")
            rep.write("="*70 + "\n")
            rep.write(f"Generated on: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            rep.write(f"Total molecules analyzed: {total_molecules:,}\n\n")
            
            rep.write("K-PREDICTION SCORE DISTRIBUTION:\n")
            rep.write("-"*70 + "\n")
            rep.write(f"Very High (> 0.9)    : {very_high:6,} molecules ({very_high/total_molecules*100:5.1f}%)\n")
            rep.write(f"High (0.7-0.9)       : {high_prob:6,} molecules ({high_prob/total_molecules*100:5.1f}%)\n")
            rep.write(f"Medium (0.5-0.7)     : {medium_prob:6,} molecules ({medium_prob/total_molecules*100:5.1f}%)\n")
            rep.write(f"Low-Medium (0.3-0.5) : {low_medium:6,} molecules ({low_medium/total_molecules*100:5.1f}%)\n")
            rep.write(f"Low (≤ 0.3)          : {low_prob:6,} molecules ({low_prob/total_molecules*100:5.1f}%)\n\n")
            
            rep.write("TOP 20 MOST PROMISING CANDIDATES:\n")
            rep.write("-"*70 + "\n")
            for i, (_, row) in enumerate(results_df.head(20).iterrows(), 1):
                prob = row['K-prediction Score']
                name = row['molecule_name']
                smi = row['smiles']
                rep.write(f"{i:2d}. {name} | {prob:.4f} | {smi}\n")
            rep.write("="*70 + "\n")

        # Step 3: Generate SMI file to temporary location
        save_smi_file(results_df, temp_smi, smi_export_type, smi_value)
        
        # Step 4: All files generated successfully - now move them atomically
        print("   🔄 Finalizing files...")
        import shutil
        import time
        
        def safe_move_with_retry(src, dst, max_retries=3):
            """
            Safely move file with retry logic for Windows 'file in use' errors.
            If move fails, tries copy+delete as fallback.
            """
            for attempt in range(max_retries):
                try:
                    # Try atomic move first
                    if os.path.exists(dst):
                        os.remove(dst)  # Remove old file if exists
                    shutil.move(src, dst)
                    return True
                except (PermissionError, OSError) as e:
                    if attempt < max_retries - 1:
                        # Wait a bit and retry (Windows file lock issue)
                        time.sleep(0.5)
                        continue
                    else:
                        # Last resort: copy + delete
                        try:
                            shutil.copy2(src, dst)
                            os.remove(src)
                            return True
                        except Exception as fallback_error:
                            raise Exception(f"Failed to move {src} to {dst}: {fallback_error}")
            return False
        
        safe_move_with_retry(temp_csv, output_csv)
        safe_move_with_retry(temp_report, output_report)
        safe_move_with_retry(temp_smi, output_smi)
        
        print(f"\n✅ All files saved successfully:")
        print(f"   📊 CSV file: {output_csv}")
        print(f"   📝 Report: {output_report}")
        print(f"   🧬 SMI file: {output_smi}")
        
    except Exception as e:
        print(f"\n❌ ERROR saving results: {e}")
        print("   🧹 Cleaning up temporary files...")
        
        # Cleanup: Remove any temporary files that may have been created
        for temp_file in temp_files:
            try:
                if os.path.exists(temp_file):
                    os.remove(temp_file)
                    print(f"   ✓ Removed: {os.path.basename(temp_file)}")
            except Exception as cleanup_error:
                print(f"   ⚠️ Could not remove {os.path.basename(temp_file)}: {cleanup_error}")
        
        print("\n⚠️ No output files were created due to the error.")
        print("   Please check the error message above and try again.")


def main():
    ensure_reproducibility(seed=42)
    
    from utils import print_script_banner, setup_script_logging
    logger = setup_script_logging("5_1_run_prediction")
    
    print_script_banner("K-talysticFlow | Step 5.1: Running Predictions")
    logger.info("Starting prediction on new molecules")
    
    loaded_assets = load_data_and_model()
    if loaded_assets[0] is None:
        print("\n❌ Failed to load data or model.")
        logger.error("Failed to load data or model")
        sys.exit(1)
        
    features_sparse, smiles_list, model = loaded_assets

    results_df = run_prediction(model, features_sparse, smiles_list)
    if results_df.empty:
        print("\n❌ Prediction execution failed.")
        logger.error("Prediction execution failed - no results generated")
        sys.exit(1)
    stats = display_prediction_statistics(results_df)
    custom_filename = get_custom_filename()
    smi_export_type, smi_value = get_smi_export_preference(stats)
    display_and_save_results(results_df, custom_filename, smi_export_type, smi_value)
    
    print("\n✅ Prediction completed successfully!")
    logger.info("Prediction completed successfully")


if __name__ == '__main__':
    main()