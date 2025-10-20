"""
K-talysticFlow - Step 5.0: Featurization for Prediction

This script loads a file containing molecules in SMILES format,
featurizes each molecule using Morgan Fingerprints, and saves
the featurized dataset directly to disk for later use in step 5.1.

INPUT:
- SMILES file selected interactively from the /data folder

OUTPUT:
- Featurized dataset saved in cfg.PREDICTION_FEATURIZED_DIR
- Featurization report with statistics

NEXT STEP:
- Run step 5.1 to perform predictions
"""

# ============================================================================
# CRITICAL: Suppress ALL warnings BEFORE any imports
# This must be THE FIRST code executed to catch Protobuf/TensorFlow warnings
# ============================================================================
import os
import sys

# Set environment variables BEFORE importing anything
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Suppress TensorFlow C++ warnings
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'  # Disable oneDNN warnings
os.environ['PYTHONWARNINGS'] = 'ignore'     # Suppress Python warnings
os.environ['PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION'] = 'python'  # Use pure Python protobuf

# Redirect stderr AND stdout to devnull to catch early import warnings
_stderr = sys.stderr
_stdout = sys.stdout
sys.stderr = open(os.devnull, 'w')
sys.stdout = open(os.devnull, 'w')

import warnings
import logging

# Configure warnings and logging
warnings.filterwarnings('ignore')
warnings.simplefilter('ignore')
logging.getLogger().setLevel(logging.CRITICAL)

# Import TensorFlow and DeepChem silently (suppress lightning/jax warnings)
import tensorflow as tf
import deepchem as dc

# Restore stderr/stdout after imports
sys.stderr.close()
sys.stderr = _stderr
sys.stdout.close()
sys.stdout = _stdout

# Now configure logging
tf.get_logger().setLevel('ERROR')
logging.getLogger('deepchem').setLevel(logging.CRITICAL)
logging.getLogger('pytorch_lightning').setLevel(logging.CRITICAL)
logging.getLogger('jax').setLevel(logging.CRITICAL)

# Now safe to import other modules
from typing import List, Tuple, Optional
import gc
import shutil

import sys
import numpy as np
import pandas as pd
from tqdm import tqdm
from datetime import datetime
import h5py
from scipy.sparse import csr_matrix, vstack as sparse_vstack
from multiprocessing import cpu_count
from joblib import Parallel, delayed
import platform

from rdkit import RDLogger
RDLogger.DisableLog('rdApp.*')

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import settings as cfg
from utils import ensure_dir_exists, load_smiles_from_file, validate_smiles, validate_smiles_chunked
from deepchem.data import DiskDataset  

# ============================================================================
# PARALLEL FEATURIZATION FUNCTIONS (Cross-platform: Windows + Linux)
# ============================================================================

def featurize_smiles_chunk(smiles_chunk: List[str], fp_size: int, fp_radius: int) -> Tuple[List, List, int, int]:
    """
    Featurizes a chunk of SMILES in a separate process.
    Joblib handles data transfer efficiently (shared memory when possible).
    
    Args:
        smiles_chunk: List of SMILES strings to featurize
        fp_size: Fingerprint size (e.g., 2048)
        fp_radius: Fingerprint radius (e.g., 2)
    
    Returns:
        Tuple of (features_list, valid_smiles, successful_count, failed_count)
    """
    try:
        # CRITICAL: Suppress ALL warnings/logs in worker processes
        import warnings
        import logging
        import os
        import sys
        
        # Set environment BEFORE any imports
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
        os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
        os.environ['PYTHONWARNINGS'] = 'ignore'
        os.environ['PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION'] = 'python'
        
        # Suppress warnings
        warnings.filterwarnings('ignore')
        warnings.simplefilter('ignore')
        
        # CRITICAL: Disable ALL logging at WARNING level
        logging.getLogger().setLevel(logging.CRITICAL)
        logging.getLogger('tensorflow').setLevel(logging.CRITICAL)
        logging.getLogger('tensorflow.python').setLevel(logging.CRITICAL)
        logging.getLogger('tensorflow.python.util.deprecation').setLevel(logging.CRITICAL)
        
        # Redirect stderr to null to suppress Protobuf/TF import warnings
        stderr_backup = sys.stderr
        stdout_backup = sys.stdout
        try:
            sys.stderr = open(os.devnull, 'w')
            sys.stdout = open(os.devnull, 'w')
            
            # Suppress RDKit warnings
            from rdkit import RDLogger
            RDLogger.DisableLog('rdApp.*')
            
            # Import DeepChem (may trigger TensorFlow/Protobuf imports)
            import deepchem as dc
            import numpy as np
            
            # Create featurizer (may trigger TF deprecation warnings)
            featurizer = dc.feat.CircularFingerprint(size=fp_size, radius=fp_radius)
        finally:
            # Restore stderr/stdout after import and featurizer creation
            if sys.stderr != stderr_backup:
                sys.stderr.close()
                sys.stderr = stderr_backup
            if sys.stdout != stdout_backup:
                sys.stdout.close()
                sys.stdout = stdout_backup
        
        features_list = []
        valid_smiles = []
        successful = 0
        failed = 0
        
        for smiles in smiles_chunk:
            try:
                features = featurizer.featurize([smiles])
                if features is not None and len(features) > 0 and features[0] is not None:
                    if features[0].size > 0:
                        features_list.append(features[0].astype(np.float32))
                        valid_smiles.append(smiles)
                        successful += 1
                    else:
                        failed += 1
                else:
                    failed += 1
            except Exception:
                failed += 1
        
        return features_list, valid_smiles, successful, failed
    except Exception as e:
        # If worker crashes, return empty results
        print(f"⚠️ Worker error: {e}")
        return [], [], 0, len(smiles_chunk)


def get_optimal_workers() -> int:
    """
    Determines optimal number of workers based on settings.py configuration.
    Falls back to auto-detection if N_WORKERS is None.
    Cross-platform: works on Windows, Linux, macOS.
    """
    # Use configuration from settings.py
    if cfg.N_WORKERS is not None:
        if cfg.N_WORKERS == -1:
            # Use all cores
            n_cpus = cpu_count() or 4
            return n_cpus
        elif cfg.N_WORKERS >= 1:
            # Use specified number
            return cfg.N_WORKERS
    
    # Auto-detect (N_WORKERS = None)
    try:
        n_cpus = cpu_count()
        if n_cpus is None:
            n_cpus = 4  # Fallback
        
        # Use n-1 cores to keep system responsive, minimum 1
        optimal = max(1, n_cpus - 1)
        return optimal
    except Exception as e:
        print(f"\n⚠️  Could not detect CPU count: {e}")
        print(f"Using 4 workers")
        return 4


def split_into_chunks(smiles_list: List[str], n_chunks: int) -> List[List[str]]:
    """
    Splits SMILES list into roughly equal chunks for parallel processing.
    """
    chunk_size = len(smiles_list) // n_chunks
    remainder = len(smiles_list) % n_chunks
    
    chunks = []
    start = 0
    
    for i in range(n_chunks):
        # Distribute remainder across first chunks
        extra = 1 if i < remainder else 0
        end = start + chunk_size + extra
        chunks.append(smiles_list[start:end])
        start = end
    
    return chunks


# ============================================================================

def check_disk_space(required_gb: float) -> bool:
    """Check if there's enough disk space available."""
    try:
        import psutil
        disk_usage = psutil.disk_usage(os.path.dirname(cfg.PREDICTION_FEATURIZED_DIR))
        available_gb = disk_usage.free / (1024**3)
        
        print(f"\n💾 Disk Space Check:")
        print(f"Available: {available_gb:.1f} GB")
        print(f"Required:  ~{required_gb:.1f} GB")
        
        if available_gb < required_gb:
            print(f"\n⚠️ WARNING: Insufficient disk space!")
            print(f"You need at least {required_gb:.1f} GB free")
            print(f"Current free space: {available_gb:.1f} GB")
            return False
        elif available_gb < required_gb * 1.5:
            print(f"⚠️ Space is tight! Recommend having {required_gb*1.5:.1f} GB free")
        else:
            print(f"✅ Sufficient space available")
        
        return True
    except ImportError:
        print("\n⚠️ Cannot check disk space (psutil not installed)")
        print("   Proceeding anyway...")
        return True
    except Exception as e:
        print(f"\n⚠️ Error checking disk space: {e}")
        return True

def select_input_file() -> Optional[str]:
    """Allows the user to interactively select a SMILES file from the /data folder."""
    data_dir = os.path.join(project_root, 'data')

    print(f"📁 Searching for SMILES files in folder: {data_dir}")
    
    if not os.path.exists(data_dir):
        print(f"\n❌ ERROR: '/data' folder not found at: {data_dir}")
        print("➡️ Create the '/data' folder and place your SMILES files there.")
        return None
    
    valid_extensions = ['.smi']
    available_files = [f for f in os.listdir(data_dir) if any(f.lower().endswith(ext) for ext in valid_extensions)]
    
    if not available_files:
        print(f"\n❌ ERROR: No valid files found in '/data'")
        print(f"➡️ Accepted extensions: {', '.join(valid_extensions)}")
        return None
    
    print(f"\n📋 Available files ({len(available_files)} found):")
    print("-" * 60)
    for i, file in enumerate(available_files, 1):
        file_path = os.path.join(data_dir, file)
        size_mb = os.path.getsize(file_path) / 1024 / 1024
        print(f"  {i:2}. {file:<30} ({size_mb:.2f} MB)")
    print("-" * 60)
    print(f"  [0] Cancel and return to menu")
    
    invalid_attempts = 0
    max_attempts = 3
    
    while True:
        try:
            choice = input(f"\nSelect file (1-{len(available_files)}): ").strip()
            if not choice:
                print("❌ Empty input. Enter a number.")
                invalid_attempts += 1
                if invalid_attempts >= max_attempts:
                    print(f"\n⚠️ Too many invalid attempts. Returning to menu...")
                    return None
                continue
            
            choice_num = int(choice)
            
            # Allow user to cancel with 0
            if choice_num == 0:
                print("\n⚠️ Operation cancelled by user.")
                return None
            
            if 1 <= choice_num <= len(available_files):
                selected_file = available_files[choice_num - 1]
                selected_path = os.path.join(data_dir, selected_file)
                print(f"\n✅ Selected file: {selected_file}")
                return selected_path
            else:
                print(f"❌ Invalid number. Enter 0 to cancel or 1-{len(available_files)} to select.")
                invalid_attempts += 1
                if invalid_attempts >= max_attempts:
                    print(f"\n⚠️ Too many invalid attempts. Returning to menu...")
                    return None
        except ValueError:
            print("❌ Invalid input. Enter numbers only.")
            invalid_attempts += 1
            if invalid_attempts >= max_attempts:
                print(f"\n⚠️ Too many invalid attempts. Returning to menu...")
                return None
        except KeyboardInterrupt:
            print("\n\nOperation cancelled by user.")
            return None

def get_custom_output_name(default_name: str) -> str:
    """
    Ask user for custom output folder name to allow multiple featurized datasets.
    Returns the custom name or default if user presses Enter.
    """
    print(f"\n📝 Output Folder Naming")
    print(f"   Default: {default_name}")
    print(f"   You can specify a custom name to keep multiple featurized datasets.")
    
    invalid_attempts = 0
    max_attempts = 3
    
    while True:
        try:
            custom_name = input(f"\nEnter custom folder name (or press Enter for default): ").strip()
            
            if not custom_name:
                return default_name
            
            # Sanitize filename (remove invalid characters)
            invalid_chars = ['<', '>', ':', '"', '/', '\\', '|', '?', '*']
            for char in invalid_chars:
                custom_name = custom_name.replace(char, '_')
            
            if len(custom_name) > 50:
                print("⚠️ Name too long. Please use less than 50 characters.")
                invalid_attempts += 1
                if invalid_attempts >= max_attempts:
                    print(f"\n⚠️ Too many invalid attempts. Using default: {default_name}")
                    return default_name
                continue
            
            print(f"\n✅ Output folder will be: {custom_name}")
            return custom_name
            
        except KeyboardInterrupt:
            print(f"\n\n⚠️ Operation cancelled. Using default: {default_name}")
            return default_name


def check_and_handle_existing_data(output_dir: str) -> bool:
    """
    Check if featurized data already exists and ask user what to do.
    Returns True to proceed, False to cancel.
    """
    if not os.path.exists(output_dir):
        return True

    print(f"\n\033[93m⚠️ WARNING: Featurized data already exists at:\033[0m")
    print(f"  {output_dir}")
    print(f"\nWhat do you want to do?")
    print(f"  [1] Overwrite (delete existing data)")
    print(f"  [2] Keep existing and use different name")
    print(f"  [3] Cancel operation")
    
    invalid_attempts = 0
    max_attempts = 3
    
    while True:
        choice = input("\nSelect option (1-3): ").strip()
        
        if choice == '1':
            print("🗑️ Removing existing data...")
            try:
                shutil.rmtree(output_dir)
                print("✅ Existing data removed")
                return True
            except Exception as e:
                print(f"\n⚠️ Error removing data: {e}")
                return False
                
        elif choice == '2':
            print("\n💡 Tip: You'll be asked for a different folder name")
            return True
            
        elif choice == '3':
            print("\n⚠️ Operation cancelled by user.")
            return False
            
        else:
            print("❌ Invalid option. Select 1, 2, or 3.")
            invalid_attempts += 1
            if invalid_attempts >= max_attempts:
                print(f"\n⚠️ Too many invalid attempts. Operation cancelled.")
                return False

def load_input_smiles(input_file_path: str) -> Optional[List[str]]:
    """Loads SMILES molecules from the specified input file."""
    print(f"\n🔄 Loading molecules from: {input_file_path}")
    
    if not os.path.exists(input_file_path):
        print(f"\n⚠️ ERROR: Input file '{input_file_path}' not found.")
        print("➡️ Verify that the file exists in the correct location.")
        print("➡️ File should contain one SMILES per line.")
        return None
    
    try:
        smiles_list_raw = load_smiles_from_file(input_file_path)
        if not smiles_list_raw:
            print("\n⚠️ ERROR: No SMILES molecules found in file.")
            return None

        print(f"\n✅ Found {len(smiles_list_raw):,} SMILES.")
        
        # CRITICAL: Validate and canonicalize SMILES (same as training pipeline)
        print(" \n🔄 Validating and canonicalizing SMILES...")
        
        # Strategy based on dataset size:
        # < 1K: Simple processing (no progress)
        # 1K - 10K: Progress bar (sequential)
        # > 10K: Chunked + parallel processing (much faster!)
        if len(smiles_list_raw) < 1000:
            smiles_list = validate_smiles(smiles_list_raw, show_progress=False)
        elif len(smiles_list_raw) < 10000:
            smiles_list = validate_smiles(smiles_list_raw, show_progress=True)
        else:
            # For large datasets (10K+), use chunked parallel processing
            print(f"  ⚡ Large dataset detected - using optimized parallel processing")
            # Use settings from advanced menu (respect user configuration)
            n_jobs = get_optimal_workers() if cfg.ENABLE_PARALLEL_PROCESSING else 1
            chunk_size = min(cfg.PARALLEL_BATCH_SIZE, max(5000, len(smiles_list_raw) // 20))
            smiles_list = validate_smiles_chunked(smiles_list_raw, chunk_size=chunk_size, n_jobs=n_jobs)
        
        if not smiles_list:
            print("\n⚠️ ERROR: No valid SMILES after canonicalization.")
            return None
        
        invalid_count = len(smiles_list_raw) - len(smiles_list)
        if invalid_count > 0:
            print(f" ⚠️ {invalid_count} invalid SMILES removed")
        
        print(f" ✅ {len(smiles_list)} valid canonical SMILES")

        unique_smiles = list(set(smiles_list))
        duplicates = len(smiles_list) - len(unique_smiles)
        
        if duplicates > 0:
            print(f"\n ⚠️ {duplicates} duplicate SMILES found (after canonicalization)")
            print(f" \nUsing {len(unique_smiles)} unique SMILES")
            final_count = len(unique_smiles)
        else:
            final_count = len(smiles_list)
        
        # Estimate disk space needed
        # Each molecule: 2048 features × 4 bytes (float32) = 8 KB
        # Plus overhead for DiskDataset structure ≈ 2x
        estimated_gb = (final_count * 2048 * 4 * 2) / (1024**3)
        
        # Check disk space
        if not check_disk_space(estimated_gb):
            print("\n❌ Cannot proceed due to insufficient disk space!")
            print("\n💡 Solutions:")
            print("1. Free up disk space")
            print("2. Process fewer molecules (sample the dataset)")
            print("3. Change output directory to a drive with more space")
            return None
        
        if duplicates > 0:
            return unique_smiles
        
        return smiles_list
        
    except Exception as e:
        print(f"\n⚠️ ERROR loading SMILES file: {e}")
        return None

def cleanup_temp_files(temp_dir: str, partial_output_dir: str = None):
    """Emergency cleanup function to free disk space."""
    print("\n🧹 Cleaning up temporary files...")
    cleaned_size = 0
    
    try:
        # Clean temp directory
        if temp_dir and os.path.exists(temp_dir):
            for root, dirs, files in os.walk(temp_dir):
                for file in files:
                    file_path = os.path.join(root, file)
                    try:
                        size = os.path.getsize(file_path)
                        os.remove(file_path)
                        cleaned_size += size
                    except:
                        pass
            try:
                shutil.rmtree(temp_dir)
            except:
                pass
        
        # Clean partial output if requested
        if partial_output_dir and os.path.exists(partial_output_dir):
            try:
                shutil.rmtree(partial_output_dir)
                print(f"Removed incomplete output: {partial_output_dir}")
            except:
                pass
        
        if cleaned_size > 0:
            size_mb = cleaned_size / (1024 * 1024)
            print(f"✅ Freed ~{size_mb:.1f} MB of disk space")
        
        # Force garbage collection
        gc.collect()
        
    except Exception as e:
        print(f"⚠️ Error during cleanup: {e}")

def featurize_and_save_to_disk(smiles_list: List[str]) -> Tuple[bool, dict]:
    """
    Featurizes molecules and saves them in Sparse + HDF5 format.
    Uses settings from settings.py (cfg.ENABLE_PARALLEL_PROCESSING, cfg.N_WORKERS).
    
    Args:
        smiles_list: List of SMILES strings
    
    Returns:
        Tuple of (success_bool, stats_dict)
    """
    print(f"\n🚀 Starting featurization of {len(smiles_list):,} molecules...")
    
    # Determine if parallel processing should be used based on settings
    use_parallel = (cfg.ENABLE_PARALLEL_PROCESSING and 
                    len(smiles_list) >= cfg.PARALLEL_MIN_THRESHOLD)
    
    stats = {
        'total_input': len(smiles_list),
        'successful': 0,
        'failed': 0,
        'invalid_smiles': [],
        'start_time': datetime.now(),
        'space_saved_percent': 0,
        'parallel_used': use_parallel
    }
    
    try:
        # ============ PARALLEL FEATURIZATION ============
        if use_parallel:
            # Get number of workers from settings
            n_workers = get_optimal_workers()
            
            system = platform.system()
            print(f"⚡ Parallel featurization enabled")
            print("\n⚠️ If process fails, data will be automatically cleaned up.")
            
            # Use batch size from settings
            batch_size = cfg.PARALLEL_BATCH_SIZE
            n_batches = (len(smiles_list) + batch_size - 1) // batch_size
            
            # We'll write incremental outputs to HDF5 to avoid holding everything in RAM
            # Do NOT accumulate all_features/all_valid_smiles here (OOM risk)
            all_valid_smiles = []  # used only for temporary batch appends
            
            print(f"\n📊 Processing {len(smiles_list):,} molecules in {n_batches} batches...")
            print(f"  • Batch size: {batch_size:,} molecules")
            print(f"  • Workers per batch: {n_workers}")
            
            # Prepare HDF5 file for incremental append (create dir BEFORE progress bar!)
            ensure_dir_exists(cfg.PREDICTION_FEATURIZED_DIR)
            hdf5_path = os.path.join(cfg.PREDICTION_FEATURIZED_DIR, 'featurized_data.h5')
            print()
            
            # Process batches with tqdm progress bar (by total molecules)
            # `tqdm` already imported at module level; avoid local import here
            pbar = tqdm(total=len(smiles_list), desc="Featurizing", unit="mol", unit_scale=True)

            # Initialize counters for incremental storage
            total_rows = 0
            total_nnz = 0

            # Open file once and append per batch
            h5f = h5py.File(hdf5_path, 'w')
            # Create extendable datasets (1D arrays for CSR components)
            maxshape_none = (None,)
            h5_data = h5f.create_dataset('data', shape=(0,), maxshape=maxshape_none, dtype=np.float32, compression='gzip')
            h5_indices = h5f.create_dataset('indices', shape=(0,), maxshape=maxshape_none, dtype=np.int32, compression='gzip')
            h5_indptr = h5f.create_dataset('indptr', shape=(1,), maxshape=maxshape_none, dtype=np.int64, compression='gzip')
            # start indptr with 0
            h5_indptr[:] = np.array([0], dtype=np.int64)
            h5_smiles = h5f.create_dataset('smiles', shape=(0,), maxshape=maxshape_none, dtype=h5py.string_dtype(encoding='utf-8'), compression='gzip')

            for batch_idx in range(n_batches):
                start_idx = batch_idx * batch_size
                end_idx = min(start_idx + batch_size, len(smiles_list))
                batch_smiles = smiles_list[start_idx:end_idx]
                
                # Split batch into chunks for parallel processing
                smiles_chunks = split_into_chunks(batch_smiles, n_workers)
                
                # Process chunks with granular progress updates
                # Using return_as='generator' allows us to update as each chunk completes
                from joblib import Parallel, delayed
                
                results = []
                for result in Parallel(n_jobs=n_workers, verbose=0, backend='loky', return_as='generator')(
                    delayed(featurize_smiles_chunk)(chunk, cfg.FP_SIZE, cfg.FP_RADIUS)
                    for chunk in smiles_chunks
                ):
                    results.append(result)
                    # Update progress CORRECTLY: by number of successful+failed molecules processed
                    features_list, valid_smiles, successful, failed = result
                    pbar.update(successful + failed)  # Total molecules processed in this chunk

                # For this batch: aggregate only batch-level features and write to disk
                batch_features = []
                batch_valid_smiles = []
                for features_list, valid_smiles, successful, failed in results:
                    batch_features.extend(features_list)
                    batch_valid_smiles.extend(valid_smiles)
                    stats['successful'] += successful
                    stats['failed'] += failed

                # If no successful features in batch, continue
                if len(batch_features) == 0:
                    # Clean up and continue
                    del results, batch_features, batch_valid_smiles
                    gc.collect()
                    continue

                # Convert batch to dense then to sparse (batch-level)
                try:
                    features_dense_batch = np.vstack(batch_features)
                    batch_sparse = csr_matrix(features_dense_batch)
                except Exception as e:
                    print(f"\n⚠️ Error: {e}")
                    del results, batch_features, batch_valid_smiles
                    gc.collect()
                    continue

                # Append CSR components to HDF5 datasets
                # Append data
                data_to_append = batch_sparse.data.astype(np.float32)
                old_len = h5_data.shape[0]
                new_len = old_len + data_to_append.shape[0]
                h5_data.resize((new_len,))
                h5_data[old_len:new_len] = data_to_append

                # Append indices
                inds_to_append = batch_sparse.indices.astype(np.int32)
                old_len_i = h5_indices.shape[0]
                new_len_i = old_len_i + inds_to_append.shape[0]
                h5_indices.resize((new_len_i,))
                h5_indices[old_len_i:new_len_i] = inds_to_append

                # Update indptr: append batch indptr[1:] + total_nnz
                batch_indptr = batch_sparse.indptr.astype(np.int64)
                to_append_indptr = batch_indptr[1:] + total_nnz
                old_len_p = h5_indptr.shape[0]
                new_len_p = old_len_p + to_append_indptr.shape[0]
                h5_indptr.resize((new_len_p,))
                h5_indptr[old_len_p:new_len_p] = to_append_indptr

                # Append smiles
                old_smiles_len = h5_smiles.shape[0]
                add_smiles = np.array(batch_valid_smiles, dtype='S')
                h5_smiles.resize((old_smiles_len + add_smiles.shape[0],))
                # Write as utf-8 strings
                h5_smiles[old_smiles_len:old_smiles_len + add_smiles.shape[0]] = [s.decode('utf-8') if isinstance(s, bytes) else s for s in add_smiles]

                # Update counters
                total_rows += batch_sparse.shape[0]
                total_nnz += batch_sparse.data.shape[0]

                # Free memory for this batch
                del results, batch_features, batch_valid_smiles, features_dense_batch, batch_sparse, data_to_append, inds_to_append, batch_indptr, to_append_indptr, add_smiles
                gc.collect()
            
            pbar.close()

            # Finalize HDF5: write metadata and compute space saved
            try:
                # h5_indptr last element is already set; shape for CSR
                final_shape = (total_rows, cfg.FP_SIZE)

                # CRITICAL: Save shape as dataset (required for loading!)
                h5f.create_dataset('shape', data=final_shape)

                # Compute sparse size bytes from datasets
                sparse_bytes = h5_data.size * h5_data.dtype.itemsize + h5_indices.size * h5_indices.dtype.itemsize + h5_indptr.size * h5_indptr.dtype.itemsize
                sparse_size_gb = sparse_bytes / (1024**3)

                dense_size_gb = (stats['successful'] * cfg.FP_SIZE * 4) / (1024**3)
                stats['space_saved_percent'] = ((dense_size_gb - sparse_size_gb) / dense_size_gb) * 100 if dense_size_gb > 0 else 0.0

                # Update metadata attributes
                h5f.attrs['total_molecules'] = stats['successful']
                h5f.attrs['fp_size'] = cfg.FP_SIZE
                h5f.attrs['fp_radius'] = cfg.FP_RADIUS
                h5f.attrs['format'] = 'sparse_csr'
                h5f.attrs['space_saved_percent'] = stats['space_saved_percent']
                h5f.attrs['parallel_used'] = stats['parallel_used']

                # Close HDF5 file
                h5f.close()
                print(f"\n✅ All done!")
                print(f"   Total processed: {stats['successful'] + stats['failed']:,}")
                print(f"   Successful: {stats['successful']:,}")
                print(f"   Failed: {stats['failed']:,}")
            except Exception as e:
                # Ensure file is closed on error
                try:
                    h5f.close()
                except:
                    pass
                raise
            
        # ============ SEQUENTIAL FEATURIZATION (FALLBACK) ============
        else:
            print(f"\n📝 Sequential featurization (single-threaded)")
            print("⚠️ If process fails, temporary files will be automatically cleaned up.")
            
            featurizer = dc.feat.CircularFingerprint(size=cfg.FP_SIZE, radius=cfg.FP_RADIUS)
            all_features = []
            all_valid_smiles = []
            
            for smiles in tqdm(smiles_list, desc="Featurizing (sequential)"):
                try:
                    features = featurizer.featurize([smiles])
                    if features is not None and len(features) > 0 and features[0] is not None:
                        if features[0].size > 0:
                            all_features.append(features[0].astype(np.float32))
                            all_valid_smiles.append(smiles)
                            stats['successful'] += 1
                        else:
                            stats['failed'] += 1
                    else:
                        stats['failed'] += 1
                except Exception:
                    stats['failed'] += 1

            # ============ VALIDATION (SEQUENTIAL MODE) ============
            if stats['successful'] == 0:
                print("\n❌ ERROR: No molecule could be featurized.")
                cleanup_temp_files(None, cfg.PREDICTION_FEATURIZED_DIR)
                return False, stats
            
            # ============ CONVERT TO SPARSE FORMAT (BATCH-WISE TO AVOID MEMORY ERROR) ============
            stats['end_time'] = datetime.now()
            stats['duration'] = stats['end_time'] - stats['start_time']
            
            print("\n💾 Saving to disk...")
            ensure_dir_exists(cfg.PREDICTION_FEATURIZED_DIR)
            hdf5_path = os.path.join(cfg.PREDICTION_FEATURIZED_DIR, 'featurized_data.h5')
            
            
            try:
                # Calculate space savings estimate (before processing)
                dense_size_gb = (stats['successful'] * cfg.FP_SIZE * 4) / (1024**3)  # float32
                
                # Process in chunks of max 10K molecules at a time to limit RAM usage
                SAVE_CHUNK_SIZE = 10000
                n_save_chunks = (len(all_features) + SAVE_CHUNK_SIZE - 1) // SAVE_CHUNK_SIZE
                
                sparse_chunks = []
                total_sparse_bytes = 0
                
                for chunk_idx in range(n_save_chunks):
                    start_idx = chunk_idx * SAVE_CHUNK_SIZE
                    end_idx = min(start_idx + SAVE_CHUNK_SIZE, len(all_features))
                    
                    # Process this chunk
                    chunk_features = all_features[start_idx:end_idx]
                    features_dense_chunk = np.vstack(chunk_features)
                    sparse_chunk = csr_matrix(features_dense_chunk)
                    sparse_chunks.append(sparse_chunk)
                    
                    total_sparse_bytes += (sparse_chunk.data.nbytes + 
                                           sparse_chunk.indices.nbytes + 
                                           sparse_chunk.indptr.nbytes)
                    
                    # Free memory immediately
                    del chunk_features, features_dense_chunk
                    gc.collect()
                
                # Stack all sparse chunks into final sparse matrix
                all_features_sparse = sparse_vstack(sparse_chunks)
                del sparse_chunks
                gc.collect()
                
                # Calculate actual space savings
                sparse_size_gb = total_sparse_bytes / (1024**3)
                stats['space_saved_percent'] = ((dense_size_gb - sparse_size_gb) / dense_size_gb) * 100
                
                
                # ============ SAVE TO HDF5 WITH COMPRESSION ============
                with h5py.File(hdf5_path, 'w') as h5f:
                    # Save sparse matrix components with compression
                    h5f.create_dataset('data', data=all_features_sparse.data, 
                                       compression='gzip', compression_opts=9)
                    h5f.create_dataset('indices', data=all_features_sparse.indices, 
                                       compression='gzip', compression_opts=9)
                    h5f.create_dataset('indptr', data=all_features_sparse.indptr, 
                                       compression='gzip', compression_opts=9)
                    h5f.create_dataset('shape', data=all_features_sparse.shape)
                    
                    # Save SMILES as strings
                    dt = h5py.string_dtype(encoding='utf-8')
                    h5f.create_dataset('smiles', data=all_valid_smiles, dtype=dt,
                                       compression='gzip', compression_opts=9)
                    
                    # Save metadata
                    h5f.attrs['total_molecules'] = stats['successful']
                    h5f.attrs['fp_size'] = cfg.FP_SIZE
                    h5f.attrs['fp_radius'] = cfg.FP_RADIUS
                    h5f.attrs['format'] = 'sparse_csr'
                    h5f.attrs['space_saved_percent'] = stats['space_saved_percent']
                    h5f.attrs['parallel_used'] = stats['parallel_used']
                
                
                print(f"\n✅ Featurization completed:")
                print(f" • Successes: {stats['successful']}")
                print(f" • Failures: {stats['failed']}")
                    
            except MemoryError as mem_err:
                print(f"\n💥 MEMORY ERROR during save!")
                print(f"Error: {mem_err}")
                raise
            except OSError as disk_err:
                print(f"\n💾 DISK ERROR during save!")
                print(f"Error: {disk_err}")
                print(f"Possible causes: Out of disk space, disk write failure")
                raise

            # Successful completion (sequential mode)
            del all_features_sparse, all_valid_smiles
            gc.collect()
        
        # Common validation for both parallel and sequential modes
        if stats['successful'] == 0:
            print("\n❌ ERROR: No molecule could be featurized.")
            cleanup_temp_files(None, cfg.PREDICTION_FEATURIZED_DIR)
            return False, stats
            
        # Set end time if not already set
        if 'end_time' not in stats:
            stats['end_time'] = datetime.now()
            stats['duration'] = stats['end_time'] - stats['start_time']
        
        return True, stats
        
    except MemoryError as mem_err:
        print(f"\n💥 CRITICAL: Out of Memory!")
        print(f"Attempted to allocate memory but failed: {mem_err}")
        print(f"Recommendation: Process fewer molecules or increase system RAM")
        cleanup_temp_files(None, cfg.PREDICTION_FEATURIZED_DIR)
        stats['end_time'] = datetime.now()
        stats['duration'] = stats.get('end_time', datetime.now()) - stats['start_time']
        return False, stats
        
    except OSError as disk_err:
        print(f"\n💾 CRITICAL: Disk Error!")
        print(f"{disk_err}")
        if "space" in str(disk_err).lower() or "102400000" in str(disk_err):
            print(f"Likely cause: Insufficient disk space")
            print(f"Recommendation: Free up disk space or change output directory")
        cleanup_temp_files(None, cfg.PREDICTION_FEATURIZED_DIR)
        stats['end_time'] = datetime.now()
        stats['duration'] = stats.get('end_time', datetime.now()) - stats['start_time']
        return False, stats
        
    except KeyboardInterrupt:
        print(f"\n\n⚠️ Process interrupted by user!")
        cleanup_temp_files(None, cfg.PREDICTION_FEATURIZED_DIR)
        stats['end_time'] = datetime.now()
        stats['duration'] = stats.get('end_time', datetime.now()) - stats['start_time']
        return False, stats
        
    except Exception as e:
        print(f"\n ⚠️ ERROR during featurization: {e}")
        print(f"Error type: {type(e).__name__}")
        cleanup_temp_files(None, cfg.PREDICTION_FEATURIZED_DIR)
        stats['end_time'] = datetime.now()
        stats['duration'] = stats.get('end_time', datetime.now()) - stats['start_time']
        return False, stats
        
    finally:
        # Final garbage collection
        gc.collect()

def generate_featurization_report(stats: dict, success: bool) -> str:
    status_text = "SUCCESS" if success else "FAILURE"
    duration_str = str(stats.get('duration', 'N/A')).split('.')[0]
    
    report = f"""=== FEATURIZATION REPORT FOR PREDICTION ===
Date/Time: {stats['start_time'].strftime('%Y-%m-%d %H:%M:%S')}
Duration: {duration_str}
Status: {status_text}

=== GENERAL STATISTICS ===
Total input molecules: {stats['total_input']}
Successful featurizations: {stats['successful']}
Failed featurizations: {stats['failed']}
Success rate: {stats['successful']/stats['total_input']*100:.2f}%

=== FEATURIZATION SETTINGS ===
Type: Morgan Circular Fingerprints
Radius: {cfg.FP_RADIUS}
Size: {cfg.FP_SIZE} bits
Save format: Sparse CSR + HDF5 with gzip compression
Space saved: {stats.get('space_saved_percent', 0):.1f}%
Parallel processing: {'Yes' if stats.get('parallel_used', False) else 'No'}
System: {platform.system()} {platform.release()}

=== OUTPUT DATA ===
Output directory: {cfg.PREDICTION_FEATURIZED_DIR}
Valid molecules saved: {stats['successful']}
File: featurized_data.h5

=== ERROR HANDLING ===
Automatic cleanup on failure: ENABLED
Memory error detection: ENABLED
Disk error detection: ENABLED
Interrupt handling (Ctrl+C): ENABLED
"""
    if stats['failed'] > 0:
        report += f"\n=== FAILED MOLECULES ({min(stats['failed'], 10)} first) ===\n"
        for invalid in stats['invalid_smiles'][:10]:
            report += f"{invalid}\n"
        if stats['failed'] > 10:
            report += f"... and {stats['failed'] - 10} more failed molecules.\n"
    if success:
        report += f"\n=== NEXT STEPS ===\n"
        report += f"1. Run step 5.1 to make predictions on this dataset\n"
    return report

def save_featurization_report(stats: dict, success: bool):
    try:
        report_content = generate_featurization_report(stats, success)
        report_path = os.path.join(cfg.RESULTS_DIR, '5_0_featurization_report.txt')
        ensure_dir_exists(cfg.RESULTS_DIR)
        with open(report_path, 'w') as f:
            f.write(report_content)
        print(f"\n✅ Report saved: {report_path}")
    except Exception as e:
        print(f"\n⚠️ Error saving report: {e}")

def display_summary(stats: dict, success: bool):
    if success:
        print("\n=========================================================")
        print("==           FEATURIZATION SUMMARY                     ==")
        print("=========================================================")
        duration_str = str(stats.get('duration', 'N/A')).split('.')[0]
        print(f"  Status                     : ✅ SUCCESS")
        print(f"  Molecules processed        : {stats['total_input']}")
        print(f"  Successful featurizations  : {stats['successful']}")
        print(f"  Failed featurizations      : {stats['failed']}")
        print(f"  Success rate               : {stats['successful']/stats['total_input']*100:.1f}%")
        print(f"  Processing time            : {duration_str}")
        print("=========================================================")

def display_parallel_config(n_molecules: int):
    """Display parallel processing configuration from settings.py."""
    if cfg.ENABLE_PARALLEL_PROCESSING and n_molecules >= cfg.PARALLEL_MIN_THRESHOLD:
        n_workers = get_optimal_workers()
        print(f"✅ Parallel processing will be used")
    elif cfg.ENABLE_PARALLEL_PROCESSING:
        print(f"⚠️ Dataset too small, using sequential processing")
    else:
        print(f"⚠️ Parallel processing disabled in settings.py")
    

def main():
    from utils import print_script_banner, setup_script_logging
    logger = setup_script_logging("5_0_featurize_prediction")
    

    from main import display_splash_screen
    display_splash_screen()
    print_script_banner("K-talysticFlow | Step 5.0: Featurization for Prediction")
    logger.info("Starting featurization for prediction")
    
    # Select input file
    input_file_path = select_input_file()
    if input_file_path is None:
        logger.info("Operation cancelled by user")
        sys.exit(0)  # Exit gracefully without error
    
    # Get custom output folder name (allows multiple datasets)
    input_filename = os.path.splitext(os.path.basename(input_file_path))[0]
    default_folder_name = input_filename  # Just the filename, no prefix
    
    # Create base directory if doesn't exist
    ensure_dir_exists(cfg.PREDICTION_FEATURIZED_BASE_DIR)
    
    # Check if default exists and handle
    default_full_path = os.path.join(cfg.PREDICTION_FEATURIZED_BASE_DIR, default_folder_name)
    if not check_and_handle_existing_data(default_full_path):
        logger.info("Operation cancelled by user")
        sys.exit(0)
    
    # Ask for custom name
    custom_folder_name = get_custom_output_name(default_folder_name)
    custom_output_dir = os.path.join(cfg.PREDICTION_FEATURIZED_BASE_DIR, custom_folder_name)
    
    # Update config to use custom directory
    original_dir = cfg.PREDICTION_FEATURIZED_DIR
    cfg.PREDICTION_FEATURIZED_DIR = custom_output_dir
    
    print(f"\n📂 Output directory: {cfg.PREDICTION_FEATURIZED_DIR}")
    
    # Load SMILES
    smiles_list = load_input_smiles(input_file_path)
    if smiles_list is None:
        cfg.PREDICTION_FEATURIZED_DIR = original_dir  # Restore
        logger.error("Failed to load SMILES from input file")
        sys.exit(1)
    
    # Display parallel processing configuration
    display_parallel_config(len(smiles_list))
    
    success, stats = featurize_and_save_to_disk(smiles_list)
    save_featurization_report(stats, success)
    display_summary(stats, success)
    
    if success:
        print("\n✅ Featurization completed successfully!")
        if stats.get('parallel_used'):
            print(f"⚡ Parallel processing used ({platform.system()})")
        print("\n➡️ Next Step: '[2] Only Predict'.")
        logger.info("Featurization for prediction completed successfully")
    else:
        print("\n❌ Featurization failed, check the report file!")
        logger.error("Featurization for prediction failed")
        sys.exit(1)

if __name__ == '__main__':
    main()
