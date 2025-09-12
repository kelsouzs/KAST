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

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
warnings.filterwarnings('ignore', category=UserWarning)
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=DeprecationWarning)

import tensorflow as tf
tf.get_logger().setLevel('ERROR')
logging.getLogger('deepchem').setLevel('ERROR')

import pandas as pd
import deepchem as dc
import numpy as np
from tqdm import tqdm

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import settings as cfg
from utils import ensure_dir_exists


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


def load_data_and_model() -> Optional[Tuple[dc.data.Dataset, dc.models.Model]]:
    from rdkit import RDLogger
    RDLogger.DisableLog('rdApp.*')
    
    print("\nLoading featurized dataset for prediction...")
    
    if not os.path.exists(cfg.PREDICTION_FEATURIZED_DIR):
        print(f"\nERROR: Directory '{cfg.PREDICTION_FEATURIZED_DIR}' not found.")
        print("➡️ Run '[5] -> [1] Prepare database' first.")
        return None, None
    
    try:
        prediction_dataset = dc.data.DiskDataset(cfg.PREDICTION_FEATURIZED_DIR)
        print(f"\n✅ Dataset loaded: {len(prediction_dataset)} molecules")
    except Exception as e:
        print(f"\n⚠️ ERROR loading featurized dataset: {e}")
        return None, None
    
    print(f"\nLoading trained model...")

    if not os.path.exists(cfg.MODEL_DIR):
        print(f"\n ⚠️ ERROR: Model '{cfg.MODEL_DIR}' not found.")
        print("➡️ Run '[3] Train Model' first.")
        return None, None
        
    try:
        n_features = prediction_dataset.get_shape()[0][1]
        
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
        print("\n✅ Model loaded successfully")
        
        return prediction_dataset, model
        
    except Exception as e:
        print(f"\n⚠️ ERROR loading model: {e}")
        return None, None


def run_prediction(model: dc.models.Model, dataset: dc.data.Dataset) -> pd.DataFrame:
    print(f"\nRunning predictions for {len(dataset)} molecules...")
    
    try:
        y_pred_proba_list = []
        batch_size = 512
        total_batches = len(dataset) // batch_size + (1 if len(dataset) % batch_size != 0 else 0)
        print(f"\nStarting...")
        
        for X, _, _, _ in tqdm(dataset.iterbatches(batch_size), 
                               total=total_batches, 
                               desc="Prediction Progress"):
            batch_preds = model.predict_on_batch(X)
            y_pred_proba_list.append(batch_preds)

        y_pred_proba_raw = np.concatenate(y_pred_proba_list, axis=0)

        if y_pred_proba_raw.ndim == 3:
            y_pred_proba_active = y_pred_proba_raw[:, 0, 1]
        elif y_pred_proba_raw.ndim == 2 and y_pred_proba_raw.shape[1] >= 2:
            y_pred_proba_active = y_pred_proba_raw[:, 1]
        else:
            print(f"\n⚠️ WARNING: Unexpected prediction shape: {y_pred_proba_raw.shape}")
            y_pred_proba_active = y_pred_proba_raw.flatten()

        molecule_names = generate_molecule_names(dataset.ids, prefix="KAST")
        
        results_df = pd.DataFrame({
            'molecule_name': molecule_names,
            'smiles': dataset.ids,
            'K-prediction Score': y_pred_proba_active
        })

        results_df = results_df.sort_values(
            by='K-prediction Score', 
            ascending=False
        ).reset_index(drop=True)
        
        return results_df
        
    except Exception as e:
        print(f"\n⚠️ ERROR during prediction: {e}")
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
            print("   ⚠️ Warning: No molecules met the specified criteria!")
        
    except Exception as e:
        print(f"⚠️ ERROR saving SMI file: {e}")


def display_and_save_results(results_df: pd.DataFrame, custom_filename: str, smi_export_type: str, smi_value: float = None):
    """Save results with custom filename and SMI preferences."""
    if results_df.empty:
        print("\n⚠️ ERROR: No results to save.")
        return
    
    try:
        output_csv = os.path.join(cfg.RESULTS_DIR, f"{custom_filename}.csv")
        output_report = os.path.join(cfg.RESULTS_DIR, f"{custom_filename}_report.txt")
        output_smi = os.path.join(cfg.RESULTS_DIR, f"{custom_filename}.smi")
        
        ensure_dir_exists(cfg.RESULTS_DIR)

        results_df.to_csv(output_csv, index=False)

        total_molecules = len(results_df)
        very_high = len(results_df[results_df['K-prediction Score'] > 0.9])
        high_prob = len(results_df[(results_df['K-prediction Score'] > 0.7) & 
                                  (results_df['K-prediction Score'] <= 0.9)])
        medium_prob = len(results_df[(results_df['K-prediction Score'] > 0.5) & 
                                    (results_df['K-prediction Score'] <= 0.7)])
        low_medium = len(results_df[(results_df['K-prediction Score'] > 0.3) & 
                                   (results_df['K-prediction Score'] <= 0.5)])
        low_prob = len(results_df[results_df['K-prediction Score'] <= 0.3])

        with open(output_report, "w", encoding="utf-8") as rep:
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

        save_smi_file(results_df, output_smi, smi_export_type, smi_value)
        
        print(f"\n✅ All files saved successfully:")
        print(f"   📊 CSV file: {output_csv}")
        print(f"   📝 Report: {output_report}")
        print(f"   🧬 SMI file: {output_smi}")
        
    except Exception as e:
        print(f"\nERROR saving results: {e}")


def main():
    ensure_reproducibility(seed=42)
    
    print("\n--- K-talysticFlow | Step 5.1: Running Predictions ---")
    print("\n🔮 This script ONLY runs predictions on already featurized data")
    
    loaded_assets = load_data_and_model()
    if loaded_assets[0] is None:
        print("\n❌ Failed to load data or model.")
        sys.exit(1)
        
    prediction_dataset, model = loaded_assets

    results_df = run_prediction(model, prediction_dataset)
    if results_df.empty:
        print("\n❌ Prediction execution failed.")
        sys.exit(1)
    stats = display_prediction_statistics(results_df)
    custom_filename = get_custom_filename()
    smi_export_type, smi_value = get_smi_export_preference(stats)
    display_and_save_results(results_df, custom_filename, smi_export_type, smi_value)
    
    print("\n✅ Prediction completed successfully!")


if __name__ == '__main__':
    main()