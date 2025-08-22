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

import os
import warnings
import logging
from typing import List, Tuple, Optional
import gc
import tempfile
import shutil

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
warnings.filterwarnings('ignore', category=UserWarning)
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=DeprecationWarning)

import tensorflow as tf
tf.get_logger().setLevel('ERROR')
logging.getLogger('deepchem').setLevel('ERROR')

import sys
import numpy as np
import deepchem as dc
import pandas as pd
from tqdm import tqdm
from datetime import datetime
import pickle

from rdkit import RDLogger
RDLogger.DisableLog('rdApp.*')

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import settings as cfg
from utils import ensure_dir_exists, load_smiles_from_file
from deepchem.data import DiskDataset  

def select_input_file() -> Optional[str]:
    """Allows the user to interactively select a SMILES file from the /data folder."""
    data_dir = os.path.join(project_root, 'data')

    print(f"\n📁 Searching for SMILES files in folder: {data_dir}")
    
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
    
    while True:
        try:
            choice = input(f"\nSelect file (1-{len(available_files)}): ").strip()
            if not choice:
                print("❌ Empty input. Enter a number.")
                continue
            choice_num = int(choice)
            if 1 <= choice_num <= len(available_files):
                selected_file = available_files[choice_num - 1]
                selected_path = os.path.join(data_dir, selected_file)
                print(f"\n✅ Selected file: {selected_file}")
                return selected_path
            else:
                print(f"❌ Invalid number. Enter between 1 and {len(available_files)}.")
        except ValueError:
            print("❌ Invalid input. Enter numbers only.")
        except KeyboardInterrupt:
            print("\n\nOperation cancelled by user.")
            return None

def clean_previous_featurized_data():
    if not os.path.exists(cfg.PREDICTION_FEATURIZED_DIR):
        return

    print(f"\n\033[93m ⚠️ WARNING: Previous featurized data found at:\033[0m")
    print(f"  {cfg.PREDICTION_FEATURIZED_DIR}")
    
    while True:
        choice = input("\nDo you want to overwrite with new featurization? (y/n): ").lower()
        if choice in ['y', 'yes']:
            print("Clearing...")
            try:
                shutil.rmtree(cfg.PREDICTION_FEATURIZED_DIR)
                print("\n✅ Previous data removed")
            except Exception as e:
                print(f"\n⚠️ Warning: Error removing previous data: {e}")
            break
        elif choice in ['n', 'no']:
            print("\nOperation cancelled.")
            sys.exit(0)
        else:
            print("Invalid option. Type 'y' or 'n'.")

def load_input_smiles(input_file_path: str) -> Optional[List[str]]:
    """Loads SMILES molecules from the specified input file."""
    print(f"\n Loading molecules from: {input_file_path}")
    
    if not os.path.exists(input_file_path):
        print(f"\n⚠️ ERROR: Input file '{input_file_path}' not found.")
        print("➡️   Verify that the file exists in the correct location.")
        print("➡️   File should contain one SMILES per line.")
        return None
    
    try:
        smiles_list = load_smiles_from_file(input_file_path)
        if not smiles_list:
            print("\n⚠️ ERROR: No SMILES molecules found in file.")
            return None

        print(f"\n ✅ Found {len(smiles_list)} SMILES.")

        unique_smiles = list(set(smiles_list))
        duplicates = len(smiles_list) - len(unique_smiles)
        
        if duplicates > 0:
            print(f"\n⚠️ {duplicates} duplicate SMILES found")
            print(f" Using {len(unique_smiles)} unique SMILES")
            return unique_smiles
        
        return smiles_list
        
    except Exception as e:
        print(f"\n⚠️ ERROR loading SMILES file: {e}")
        return None

def featurize_and_save_to_disk(smiles_list: List[str]) -> Tuple[bool, dict]:
    """Featurizes molecules in batches and saves them as a DeepChem DiskDataset."""
    print(f"\nStarting featurization of {len(smiles_list)} molecules...")
    stats = {
        'total_input': len(smiles_list),
        'successful': 0,
        'failed': 0,
        'invalid_smiles': [],
        'start_time': datetime.now()
    }
    
    try:
        temp_dir = tempfile.mkdtemp(prefix='kast_featurization_')
        featurizer = dc.feat.CircularFingerprint(size=cfg.FP_SIZE, radius=cfg.FP_RADIUS)
        
        valid_smiles = []
        batch_size = 5000  
        print("\nStarting featurization...")
        progress_bar = tqdm(total=len(smiles_list), desc="Featurizing")
        batch_files = []

        for batch_idx, start_idx in enumerate(range(0, len(smiles_list), batch_size)):
            end_idx = min(start_idx + batch_size, len(smiles_list))
            batch_smiles = smiles_list[start_idx:end_idx]
            batch_features = []
            batch_valid_smiles = []
            
            for i, smiles in enumerate(batch_smiles):
                global_idx = start_idx + i
                try:
                    features = featurizer.featurize([smiles])
                    if features is not None and len(features) > 0 and features[0] is not None:
                        if features[0].size > 0:
                            batch_features.append(features[0].astype(np.float32))
                            batch_valid_smiles.append(smiles)
                            stats['successful'] += 1
                        else:
                            stats['failed'] += 1
                            if len(stats['invalid_smiles']) < 100:
                                stats['invalid_smiles'].append(f"Line {global_idx+1}: {smiles[:50]}...")
                    else:
                        stats['failed'] += 1
                        if len(stats['invalid_smiles']) < 100:
                            stats['invalid_smiles'].append(f"Line {global_idx+1}: {smiles[:50]}...")
                    progress_bar.update(1)
                except Exception as e:
                    stats['failed'] += 1
                    if len(stats['invalid_smiles']) < 100:
                        stats['invalid_smiles'].append(f"Line {global_idx+1}: {smiles[:50]}... (Error: {str(e)[:30]})")
                    progress_bar.update(1)

            if batch_features:
                batch_data = {
                    'features': np.vstack(batch_features),
                    'smiles': batch_valid_smiles
                }
                batch_file = os.path.join(temp_dir, f'batch_{batch_idx:04d}.pkl')
                with open(batch_file, 'wb') as f:
                    pickle.dump(batch_data, f)
                batch_files.append(batch_file)
                valid_smiles.extend(batch_valid_smiles)
                del batch_features, batch_valid_smiles, batch_data
                gc.collect()
        
        progress_bar.close()
        stats['end_time'] = datetime.now()
        stats['duration'] = stats['end_time'] - stats['start_time']
        
        if stats['successful'] == 0:
            print("\n❌ ERROR: No molecule could be featurized.")
            shutil.rmtree(temp_dir)
            return False, stats

        chunk_size = 10  
        all_features_chunks = []
        all_smiles_final = []
        for i in range(0, len(batch_files), chunk_size):
            chunk_files = batch_files[i:i+chunk_size]
            chunk_features = []
            for batch_file in chunk_files:
                with open(batch_file, 'rb') as f:
                    batch_data = pickle.load(f)
                    chunk_features.append(batch_data['features'])
                    all_smiles_final.extend(batch_data['smiles'])
                os.remove(batch_file)
            if chunk_features:
                chunk_combined = np.vstack(chunk_features)
                chunk_file = os.path.join(temp_dir, f'chunk_{i//chunk_size:04d}.npy')
                np.save(chunk_file, chunk_combined)
                all_features_chunks.append(chunk_file)
                del chunk_features, chunk_combined
                gc.collect()

        print("Saving...")
        ensure_dir_exists(cfg.PREDICTION_FEATURIZED_DIR)
        feature_chunks = []
        for chunk_file in all_features_chunks:
            chunk_data = np.load(chunk_file)
            feature_chunks.append(chunk_data)
            os.remove(chunk_file)
        
        final_features = np.vstack(feature_chunks)
        dummy_labels = np.zeros(len(all_smiles_final), dtype=np.float32)

        dataset = DiskDataset.from_numpy(
            X=final_features,
            y=dummy_labels,
            ids=all_smiles_final,
            data_dir=cfg.PREDICTION_FEATURIZED_DIR
        )

        print(f"\n✅ Featurization completed:")
        print(f"     • Successes: {stats['successful']}")
        print(f"     • Failures: {stats['failed']}")
        print(f"     • Success rate: {stats['successful']/stats['total_input']*100:.1f}%")
        print(f"     • Dataset saved: {cfg.PREDICTION_FEATURIZED_DIR}")

        shutil.rmtree(temp_dir)
        del final_features, dataset, dummy_labels, feature_chunks
        gc.collect()
        
        return True, stats
        
    except Exception as e:
        print(f"\n ⚠️ ERROR during featurization: {e}")
        try:
            if 'temp_dir' in locals():
                shutil.rmtree(temp_dir)
        except:
            pass
        return False, stats

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
Save method: Temporary files (Windows-compatible)

=== OUTPUT DATA ===
Output directory: {cfg.PREDICTION_FEATURIZED_DIR}
Valid molecules saved: {stats['successful']}
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

def main():
    print("\n--- K-talysticFlow | Step 5.0: Featurization for Prediction ---")
    clean_previous_featurized_data()
    
    input_file_path = select_input_file()
    if input_file_path is None:
        sys.exit(1)
    
    smiles_list = load_input_smiles(input_file_path)
    if smiles_list is None:
        sys.exit(1)
    
    success, stats = featurize_and_save_to_disk(smiles_list)
    save_featurization_report(stats, success)
    display_summary(stats, success)
    
    if success:
        print("\n✅ Featurization completed successfully!")
        print("\n➡️     Next Step: '[2] Only Predict'.")
    else:
        print("\n❌ Featurization failed, check the report file!")
        sys.exit(1)

if __name__ == '__main__':
    main()
