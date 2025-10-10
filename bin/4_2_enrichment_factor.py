# bin/4_2_enrichment_factor.py

"""
K-talysticFlow - Step 4.2: Enrichment Factor (EF) Calculation

This script evaluates the model's performance in a virtual screening task.
The Enrichment Factor measures how concentrated the active molecules are
at the top of a list ranked by the model's predictions, compared to a
random selection.

An EF of 10 at 1% (EF@1%) means that the model is 10 times better at
finding actives in the top 1% of the list than a random search.
"""

import os
import warnings
import logging
from typing import Tuple, Optional, Dict

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
warnings.filterwarnings('ignore', category=DeprecationWarning)
warnings.filterwarnings('ignore', category=UserWarning)
warnings.filterwarnings('ignore', category=FutureWarning)

import tensorflow as tf
tf.get_logger().setLevel('ERROR')
logging.getLogger('deepchem').setLevel('ERROR')

import sys
import pandas as pd
import numpy as np

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import settings as cfg
from utils import ensure_dir_exists

# --- Helper Functions ---

def load_predictions(file_path: str) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    """Loads predictions from a CSV file, returning the vectors or None in case of error."""
    from rdkit import RDLogger
    RDLogger.DisableLog('rdApp.*')
    
    print(f"\nLoading predictions from: {file_path}")
    
    if not os.path.exists(file_path):
        print(f"\nERROR: Predictions file '{os.path.basename(file_path)}' not found.")
        print("➡️ Run option '[4] Evaluate the Model' first.")
        return None
    
    try:
        df = pd.read_csv(file_path)
        required_cols = ['true_label', 'predicted_probability']
        
        if not all(col in df.columns for col in required_cols):
            print(f"\nERROR: The predictions CSV file must contain the columns: {required_cols}")
            print(f"Columns found: {list(df.columns)}")
            return None
        
        y_true = df['true_label'].values
        y_score = df['predicted_probability'].values
        
        print(f"\n✅ Predictions loaded: {len(y_true)} samples")
        print(f"  -> Active molecules: {np.sum(y_true)}")
        print(f"  -> Inactive molecules: {len(y_true) - np.sum(y_true)}")
        
        return y_true, y_score
        
    except Exception as e:
        print(f"\n ⚠️ Unexpected ERROR reading predictions file: {e}")
        return None

def calculate_ef(y_true: np.ndarray, y_score: np.ndarray, fraction_percent: float) -> float:
    """Calculates the Enrichment Factor for a given population fraction."""
    try:
        fraction = fraction_percent / 100.0
        n_total = len(y_true)
        ns_total = np.sum(y_true)
        
        if ns_total == 0:
            print(f"⚠️ WARNING: No actives in the set. EF for {fraction_percent}% is 0.")
            return 0.0
        
        n_top = int(np.ceil(n_total * fraction))
        if n_top == 0: 
            return 0.0
            
        # Sort by descending score and take the top fraction
        indices = np.argsort(y_score)[::-1]
        ns_top = np.sum(y_true[indices][:n_top])
        
        # Calculate EF
        ef = (ns_top / n_top) / (ns_total / n_total)
        
        return ef
        
    except Exception as e:
        print(f"⚠️ ERROR calculating EF for {fraction_percent}%: {e}")
        return 0.0

def generate_ef_report(y_true: np.ndarray, y_score: np.ndarray) -> Tuple[str, Dict[str, float]]:
    """Calculates EFs, generates a text report, and returns results for summary."""
    print("\nCalculating Enrichment Factor (EF) for different fractions...")
    
    report_lines = [
        "=== ENRICHMENT FACTOR (EF) RESULTS ===",
        f"Date/Time: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"Total samples: {len(y_true)}",
        f"Active molecules: {np.sum(y_true)}",
        f"Inactive molecules: {len(y_true) - np.sum(y_true)}",
        "",
        "=== RESULTS BY FRACTION ===",
    ]
    
    ef_results = {}
    
    for frac in cfg.EF_FRACTIONS_PERCENT:
        ef_value = calculate_ef(y_true, y_score, frac)
        result_key = f"EF @ {frac}%"
        result_line = f"  {result_key:<15}: {ef_value:.4f}"
        
        report_lines.append(result_line)
        ef_results[result_key] = ef_value
        
        print(f"  -> {result_key}: {ef_value:.4f}")
        
    return "\n".join(report_lines), ef_results

def display_summary(ef_results: dict):
    """Displays a formatted summary of Enrichment Factor results."""
    print("\n" + "-"*57)
    print("ENRICHMENT FACTOR (EF) SUMMARY".center(57))
    print("-"*57)

    for name, value in ef_results.items():
        print(f"  {name:<20}: {value:.4f}")

    print("-"*57)
    
    valid_efs = [v for v in ef_results.values() if v > 0]
    if valid_efs:
        mean_ef = np.mean(valid_efs)
        print(f"  Mean EF            : {mean_ef:.4f}")

    print("-"*57)

def main():
    from utils import print_script_banner, setup_script_logging
    logger = setup_script_logging("4_2_enrichment_factor")
    
    print_script_banner("K-talysticFlow | Step 4.2: Enrichment Factor (EF) Calculation")
    logger.info("Starting enrichment factor calculation")

    predictions_path = os.path.join(cfg.RESULTS_DIR, '4_0_test_predictions.csv')
    loaded_data = load_predictions(predictions_path)
    
    if loaded_data is None:
        logger.error("Failed to load predictions")
        sys.exit(1)
        
    y_true, y_score = loaded_data


    report_content, ef_results = generate_ef_report(y_true, y_score)
    

    try:
        results_file_path = os.path.join(cfg.RESULTS_DIR, '4_2_enrichment_factor_results.txt')
        ensure_dir_exists(cfg.RESULTS_DIR)
        
        with open(results_file_path, 'w') as f:
            f.write(report_content)
            
        print(f"\n✅ Full report saved at: {results_file_path}")
        
    except Exception as e:
        print(f"\n ⚠️ ERROR saving report: {e}")

    # Step 4: Display summary
    display_summary(ef_results)

    print("\n✅ Enrichment Factor calculation completed successfully!")
    print("\n➡️ Next step: run the remaining detailed evaluation scripts.")
    logger.info("Enrichment factor calculation completed successfully")

if __name__ == '__main__':
    main()
