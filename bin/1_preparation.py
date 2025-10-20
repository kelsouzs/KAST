"""
K-talysticFlow - Step 1: Data Preparation

This script is the first stage of the workflow and is responsible for:
1. Interactively selecting the SMILES files for actives and inactives.
2. Loading the molecules from the selected files.
3. Assigning labels (1 for actives, 0 for inactives).
4. Combining and shuffling the data.
5. Splitting the dataset into training and testing sets using the Scaffold Split method.
6. Saving the resulting sets as CSV files in the results folder.

Input:
    - SMILES files for actives and inactives selected by the user.
Output:
    - Training and testing CSV files saved in the results folder.
"""

import os
import sys
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

import pandas as pd
import numpy as np
import deepchem as dc

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import settings as cfg
from utils import ensure_dir_exists, load_smiles_from_file, validate_smiles
from main import display_splash_screen

def warn_and_confirm_data_files(data_dir):
    """
    Displays a warning about the need for .smi files in the /data folder
    and asks for user confirmation before proceeding.
    """
    print("⚠️  WARNING: To create the model, you need the following files:")
    print("    - Actives (.smi) and Inactives/Decoys (.smi)")
    print("    - They must be in the '/data' folder in SMILES (.smi) format")
    print("------------------------------------------------------------")
    print("Example of expected files:")
    print(f"{data_dir}/actives.smi")
    print(f"{data_dir}/inactives.smi   or")
    print(f"{data_dir}/decoys.smi")
    print("------------------------------------------------------------")

def select_smiles_file(prompt, logger):
    data_dir = os.path.join(project_root, 'data')  
    files = [f for f in os.listdir(data_dir) if f.endswith('.smi')]
    if not files:
        error_msg = f"No .smi file found in '{data_dir}'"
        print(f"\nERROR: {error_msg}")
        logger.error(error_msg)
        sys.exit(1)
    print(f"\n{prompt}")
    for i, f in enumerate(files, 1):
        print(f"{i}. {f}")
    while True:
        try:
            choice = int(input("Select the file number: "))
            if 1 <= choice <= len(files):
                return os.path.join(data_dir, files[choice - 1])
            else:
                print("Invalid number. Try again.")
        except ValueError:
            print("\nInvalid input. Please enter a number.")

# --- Helper Functions ---

def load_and_label_data(active_path, inactive_path):
    """Loads, labels, and combines data from active and inactive molecules."""
    print(f"\n-------------------------------------------------------------")
    print(f"Loading actives from: {active_path}")
    active_smiles_raw = load_smiles_from_file(active_path)
    
    print(f"\nLoading inactives from: {inactive_path}")
    inactive_smiles_raw = load_smiles_from_file(inactive_path)
    print(f"-------------------------------------------------------------")

    if not active_smiles_raw or not inactive_smiles_raw:
        print("\nERROR: Failed to load SMILES files. Check the selected files.")
        return None

    print(f"\nTotal actives loaded: {len(active_smiles_raw)}")
    print(f"Total inactives loaded: {len(inactive_smiles_raw)}")
    
    # Validate and canonicalize SMILES
    print("\nValidating and canonicalizing SMILES...")
    
    # Show progress for large datasets (> 1000 molecules each)
    show_progress_actives = len(active_smiles_raw) >= 1000
    show_progress_inactives = len(inactive_smiles_raw) >= 1000
    
    if show_progress_actives or show_progress_inactives:
        print("  (Progress bars shown for large datasets)")
    
    print("  • Processing actives...")
    active_smiles = validate_smiles(active_smiles_raw, show_progress=show_progress_actives)
    
    print("  • Processing inactives...")
    inactive_smiles = validate_smiles(inactive_smiles_raw, show_progress=show_progress_inactives)
    
    if not active_smiles or not inactive_smiles:
        print("\nERROR: No valid SMILES found after validation.")
        return None
    
    print(f"\nValid molecules after validation:")
    print(f"  -> Actives: {len(active_smiles)} (removed: {len(active_smiles_raw) - len(active_smiles)})")
    print(f"  -> Inactives: {len(inactive_smiles)} (removed: {len(inactive_smiles_raw) - len(inactive_smiles)})")

    actives_df = pd.DataFrame({cfg.SMILES_COL: active_smiles, cfg.LABEL_COL: 1})
    inactives_df = pd.DataFrame({cfg.SMILES_COL: inactive_smiles, cfg.LABEL_COL: 0})
    

    all_data_df = pd.concat([actives_df, inactives_df], ignore_index=True)
    all_data_df = all_data_df.sample(frac=1, random_state=cfg.RANDOM_STATE).reset_index(drop=True)

    print(f"\nTotal combined molecules: {len(all_data_df)}")
    print("\nClass distribution in the complete dataset:")
    class_counts = all_data_df[cfg.LABEL_COL].value_counts()
    print(f"  - Class 0 (Inactives): {class_counts.get(0, 0)} samples")
    print(f"  - Class 1 (Actives):   {class_counts.get(1, 0)} samples")
    
    return all_data_df


def get_train_test_split():
    """
    Asks user for train/test split preference.
    Returns test_fraction (e.g., 0.2 for 80/20)
    """
    print("\n" + "="*70)
    print("� TRAIN/TEST SPLIT CONFIGURATION")
    print("="*70)
    print(f"\nCurrent default (settings.py): {(1-cfg.TEST_SET_FRACTION)*100:.0f}% train / {cfg.TEST_SET_FRACTION*100:.0f}% test")
    print("\nCommon splits:")
    print("  1. 80% train / 20% test  (recommended for small datasets)")
    print("  2. 70% train / 30% test  (more data for testing)")
    print("  3. 90% train / 10% test  (maximize training data)")
    print("  4. Use settings.py value (no change)")
    print("  5. Custom split")
    
    while True:
        try:
            choice = input("\nSelect option (1-5): ").strip()
            
            if choice == '1':
                return 0.2  # 80/20
            elif choice == '2':
                return 0.3  # 70/30
            elif choice == '3':
                return 0.1  # 90/10
            elif choice == '4':
                return cfg.TEST_SET_FRACTION  # From settings.py
            elif choice == '5':
                while True:
                    try:
                        test_pct = float(input("Enter test percentage (e.g., 20 for 20%): "))
                        if 5 <= test_pct <= 50:
                            return test_pct / 100.0
                        else:
                            print("⚠️  Test percentage should be between 5% and 50%")
                    except ValueError:
                        print("⚠️  Invalid input. Enter a number (e.g., 20)")
            else:
                print("⚠️  Invalid option. Choose 1-5")
        except KeyboardInterrupt:
            print("\n\n⚠️  Using default from settings.py")
            return cfg.TEST_SET_FRACTION


def split_data_scaffold(df, test_fraction=None):
    """
    Splits data using scaffold split method.
    
    Args:
        df: DataFrame with SMILES and labels
        test_fraction: Fraction for test set (if None, asks user)
    """
    # Ask user if test_fraction not provided
    if test_fraction is None:
        test_fraction = get_train_test_split()
    
    try:
        dataset = dc.data.NumpyDataset(
            X=np.zeros((len(df), 1)),
            y=df[cfg.LABEL_COL].values,
            ids=df[cfg.SMILES_COL].values
        )
        
        splitter = dc.splits.ScaffoldSplitter()
        train_frac = 1.0 - test_fraction
        
        print(f"\n✅ Using split: {train_frac*100:.1f}% train / {test_fraction*100:.1f}% test")
        print(f"\nSplitting data...")
        
        train_indices, test_indices, _ = splitter.split(
            dataset, frac_train=train_frac, frac_valid=test_fraction, frac_test=0.0
        )
        
        train_set = df.iloc[train_indices]
        test_set = df.iloc[test_indices]
        
        print("\nSplit completed:")
        print(f"  Training set size: {len(train_set)}")
        print(f"  Test set size: {len(test_set)}")
        

        print("\nClass distribution:")
        train_counts = train_set[cfg.LABEL_COL].value_counts()
        test_counts = test_set[cfg.LABEL_COL].value_counts()
        
        print(f"  Train - Inactives: {train_counts.get(0, 0)}, Actives: {train_counts.get(1, 0)}")
        print(f"  Test  - Inactives: {test_counts.get(0, 0)}, Actives: {test_counts.get(1, 0)}")
        
        return train_set, test_set
        
    except Exception as e:
        print(f"\nERROR in data splitting: {e}")
        print("Oops, something went wrong. Trying random split...")
        
        shuffled_df = df.sample(frac=1, random_state=cfg.RANDOM_STATE).reset_index(drop=True)
        split_idx = int(len(shuffled_df) * (1.0 - test_fraction))
        
        train_set = shuffled_df[:split_idx]
        test_set = shuffled_df[split_idx:]
        
        print(f"✅ Random split successful (using {train_frac*100:.0f}/{test_fraction*100:.0f})")
        
        # Fallback distribution statistics
        print("\nClass distribution:")
        train_counts = train_set[cfg.LABEL_COL].value_counts()
        test_counts = test_set[cfg.LABEL_COL].value_counts()
        print(f"\n  Train - Inactives: {train_counts.get(0, 0)}, Actives: {train_counts.get(1, 0)}")
        print(f"  Test  - Inactives: {test_counts.get(0, 0)}, Actives: {test_counts.get(1, 0)}")
        
        return train_set, test_set

def save_datasets(train_df, test_df):
    """Saves the training and testing dataframes as CSV files."""
    ensure_dir_exists(cfg.RESULTS_DIR)
    
    train_csv_path = os.path.join(cfg.RESULTS_DIR, cfg.OUTPUT_TRAIN_CSV)
    test_csv_path = os.path.join(cfg.RESULTS_DIR, cfg.OUTPUT_TEST_CSV)
    
    print(f"\nSaving training set to: {train_csv_path}")
    train_df.to_csv(train_csv_path, index=False)
    
    print(f"Saving test set to: {test_csv_path}")
    test_df.to_csv(test_csv_path, index=False)



def main():
    from utils import print_script_banner, setup_script_logging, log_error
    logger = setup_script_logging("1_preparation")
    
    display_splash_screen()
    print_script_banner("K-talysticFlow | Step 1: Preparing and Splitting Data")
    logger.info("Starting data preparation and splitting")

    data_dir = os.path.join(project_root, 'data')
    warn_and_confirm_data_files(data_dir)

    active_file = select_smiles_file("Select the file with ACTIVE molecules (.smi):", logger)
    inactive_file = select_smiles_file("Select the file with INACTIVE molecules (.smi):", logger)

    all_data = load_and_label_data(active_file, inactive_file)
    if all_data is None:
        log_error(logger, "Failed to load and label data")
        sys.exit(1)  


    print("\nChecking dataset size requirements...")
    active_count = len(all_data[all_data[cfg.LABEL_COL] == 1])
    inactive_count = len(all_data[all_data[cfg.LABEL_COL] == 0])
    
    if active_count < cfg.MIN_MOLECULES_PER_CLASS:
        print(f"\n⚠️ WARNING: Few active molecules ({active_count}).")
        print(f"   Minimum recommended: {cfg.MIN_MOLECULES_PER_CLASS}")
    
    if inactive_count < cfg.MIN_MOLECULES_PER_CLASS:
        print(f"\n⚠️ WARNING: Few inactive molecules ({inactive_count}).")
        print(f"   Minimum recommended: {cfg.MIN_MOLECULES_PER_CLASS}")
    
    total_valid = active_count + inactive_count
    print(f"\n✅ Dataset ready: {total_valid} valid molecules ({active_count} actives, {inactive_count} inactives)")
    
    train_dataset, test_dataset = split_data_scaffold(all_data)

    save_datasets(train_dataset, test_dataset)

    print("\n✅ Data preparation and splitting completed successfully!")
    print("\n➡️ Next step: '[2] Generate Fingerprints'.")
    
    from utils import setup_script_logging
    logger = setup_script_logging("1_preparation")
    logger.info("Data preparation completed successfully")


if __name__ == '__main__':
    main()
