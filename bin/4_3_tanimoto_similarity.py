# bin/4_3_tanimoto_similarity.py

"""
K-talysticFlow - Step 4.3: Tanimoto Similarity Analysis

This script evaluates the degree of similarity between the active molecules
in the test set and all molecules in the training set. The goal is to check
whether the model is learning to generalize to new structures (low similarity)
or if its good performance is just due to "memorizing" very similar examples
(high similarity).
"""

import os
import warnings
import logging
from typing import List, Tuple, Optional, Dict

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
warnings.filterwarnings('ignore', category=UserWarning)
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=DeprecationWarning)

import tensorflow as tf
tf.get_logger().setLevel('ERROR')
logging.getLogger('deepchem').setLevel('ERROR')

import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

from rdkit import Chem, RDLogger
from rdkit.Chem import DataStructs
from rdkit.DataStructs import ExplicitBitVect

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import settings as cfg
from utils import ensure_dir_exists, get_morgan_fp

def load_datasets(train_path: str, test_path: str) -> Optional[Tuple[pd.DataFrame, pd.DataFrame]]:
    """Loads the training and test datasets from CSV files."""
    RDLogger.DisableLog('rdApp.*')
    
    print("Loading training and test datasets...")
    
    if not os.path.exists(train_path):
        print(f"\nERROR: Training file '{train_path}' not found.")
        print("➡️    Run option '[1] Prepare and Split Data' first.")
        return None
        
    if not os.path.exists(test_path):
        print(f"\nERROR: Test file '{test_path}' not found.")
        print("➡️   Run option '[1] Prepare and Split Data' first.")
        return None
    
    try:
        train_df = pd.read_csv(train_path)
        test_df = pd.read_csv(test_path)
        
        required_cols = [cfg.SMILES_COL, cfg.LABEL_COL]
        
        for col in required_cols:
            if col not in train_df.columns:
                print(f"\nERROR: Column '{col}' not found in the training file.")
                return None
            if col not in test_df.columns:
                print(f"\nERROR: Column '{col}' not found in the test file.")
                return None
        
        print(f"\n✅ Training dataset: {len(train_df)} samples")
        print(f"✅ Test dataset: {len(test_df)} samples")
        
        return train_df, test_df
        
    except Exception as e:
        print(f"\n⚠️ ERROR reading CSV files: {e}")
        return None

def calculate_fingerprints(smiles_list: List[str]) -> List[ExplicitBitVect]:
    """Calculates Morgan fingerprints for a list of SMILES."""
    print(f"\nCalculating fingerprints for {len(smiles_list)} training molecules...")
    print()
    fps = []
    failed_count = 0
    
    try:
        for smi in tqdm(smiles_list, desc="Calculating..."):
            fp = get_morgan_fp(smi, cfg.FP_RADIUS, cfg.FP_SIZE)
            if fp is not None:
                fps.append(fp)
            else:
                failed_count += 1
        
        print(f"\n✅ Fingerprints calculated: {len(fps)}")
        if failed_count > 0:
            print(f"⚠️ Failed molecules: {failed_count}")
            
        return fps
        
    except Exception as e:
        print(f"\n⚠️ ERROR calculating fingerprints: {e}")
        return []

def calculate_max_similarities(test_actives_smiles: List[str], train_fps: List[ExplicitBitVect]) -> List[float]:
    """Calculates the maximum similarity of each test active with the training set."""
    print(f"\nCalculating maximum similarity for {len(test_actives_smiles)} test actives...")
    print()
    if not train_fps:
        print("\n⚠️ ERROR: No training fingerprints available.")
        return []
    
    max_similarities = []
    failed_count = 0
    
    try:
        for smi in tqdm(test_actives_smiles, desc="Analyzing Test Actives"):
            fp_test = get_morgan_fp(smi, cfg.FP_RADIUS, cfg.FP_SIZE)
            
            if fp_test is None:
                failed_count += 1
                continue
            
            sims = DataStructs.BulkTanimotoSimilarity(fp_test, train_fps)
            if sims:
                max_similarities.append(np.max(sims))
        
        print(f"\n✅ Similarities calculated: {len(max_similarities)}")
        if failed_count > 0:
            print(f"⚠️ Failed molecules: {failed_count}")
            
        return max_similarities
        
    except Exception as e:
        print(f"\n⚠️ ERROR calculating similarities: {e}")
        return []

def plot_and_save_histogram(similarities: List[float], mean_sim: float):
    """Generates and saves the similarity histogram."""
    try:
        output_path = os.path.join(cfg.RESULTS_DIR, '4_3_tanimoto_similarity_histogram.png')
        ensure_dir_exists(cfg.RESULTS_DIR)
        
        plt.figure(figsize=(10, 6))
        plt.hist(similarities, bins=25, color="#6A5ACD", edgecolor='black', alpha=0.8)
        
        plt.title("Maximum Similarity Distribution (Test Actives vs. Training)", fontsize=16)
        plt.xlabel("Maximum Tanimoto Similarity Coefficient", fontsize=12)
        plt.ylabel("Count of Test Active Molecules", fontsize=12)
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        plt.xlim(0, 1)
        
        plt.axvline(mean_sim, color='red', linestyle='dashed', linewidth=2)
        plt.text(mean_sim * 1.05, plt.ylim()[1] * 0.9, f'Mean: {mean_sim:.3f}', 
                color='red', fontweight='bold')
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"✅ Histogram saved: {output_path}")
        
    except Exception as e:
        print(f"\n⚠️ ERROR saving histogram: {e}")

def generate_similarity_report(similarities: List[float]) -> str:
    """Generates a detailed text report of the similarities."""
    mean_sim = np.mean(similarities)
    std_sim = np.std(similarities)
    min_sim = np.min(similarities)
    max_sim = np.max(similarities)
    median_sim = np.median(similarities)
    
    if mean_sim < 0.3:
        interpretation = "Low similarity - Excellent generalization"
    elif mean_sim < 0.5:
        interpretation = "Moderate similarity - Good generalization"
    elif mean_sim < 0.7:
        interpretation = "High similarity - Limited generalization"
    else:
        interpretation = "Very high similarity - Possible memorization"
    
    low_sim = len([s for s in similarities if s < 0.3])
    mod_sim = len([s for s in similarities if 0.3 <= s < 0.7])
    high_sim = len([s for s in similarities if s >= 0.7])
    
    report = f"""=== TANIMOTO SIMILARITY ANALYSIS REPORT ===
Date/Time: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}
Total active molecules analyzed: {len(similarities)}

=== DESCRIPTIVE STATISTICS ===
Mean similarity: {mean_sim:.4f}
Standard deviation: {std_sim:.4f}
Minimum similarity: {min_sim:.4f}
Maximum similarity: {max_sim:.4f}
Median: {median_sim:.4f}

=== INTERPRETATION ===
{interpretation}

=== RANGE DISTRIBUTION ===
Low similarity (< 0.3): {low_sim} molecules ({low_sim/len(similarities)*100:.1f}%)
Moderate similarity (0.3-0.7): {mod_sim} molecules ({mod_sim/len(similarities)*100:.1f}%)
High similarity (≥ 0.7): {high_sim} molecules ({high_sim/len(similarities)*100:.1f}%)

=== CONCLUSIONS ===
- If most molecules have low similarity, the model is generalizing well
- If many molecules have high similarity, the model might be "memorizing"
- Ideal results show a distribution peaking at low to moderate similarities
"""
    return report

def display_summary(similarities: List[float]) -> float:
    """Calculates and displays a statistical summary of similarities."""
    mean_sim = np.mean(similarities)
    std_sim = np.std(similarities)
    min_sim = np.min(similarities)
    max_sim = np.max(similarities)
    
    print("\n=========================================================")
    print("==      TANIMOTO SIMILARITY ANALYSIS SUMMARY           ==")
    print("=========================================================")
    print(f"  Mean similarity         : {mean_sim:.4f}")
    print(f"  Standard deviation      : {std_sim:.4f}")
    print(f"  Minimum similarity      : {min_sim:.4f}")
    print(f"  Maximum similarity      : {max_sim:.4f}")
    print("=========================================================")

    return mean_sim

def main():
    print("\n--- K-talysticFlow | Step 4.3: Tanimoto Similarity Analysis ---")

    train_path = os.path.join(cfg.RESULTS_DIR, '01_train_set.csv')
    test_path = os.path.join(cfg.RESULTS_DIR, '01_test_set.csv')
    
    loaded_data = load_datasets(train_path, test_path)
    if loaded_data is None:
        sys.exit(1)
        
    train_df, test_df = loaded_data

    train_fps = calculate_fingerprints(train_df[cfg.SMILES_COL].tolist())
    if not train_fps:
        print("\n⚠️ ERROR: No training fingerprints could be calculated.")
        sys.exit(1)

    test_actives_smiles = test_df[test_df[cfg.LABEL_COL] == 1][cfg.SMILES_COL].tolist()
    if not test_actives_smiles:
        print("\n⚠️ WARNING: No active molecules found in the test set.")
        print("➡️ Similarity analysis cannot be performed.")
        sys.exit(1)
    
    print(f"\nActive molecules in test set: {len(test_actives_smiles)}")

    max_similarities = calculate_max_similarities(test_actives_smiles, train_fps)
    if not max_similarities:
        print("\n⚠️ ERROR: No similarities could be calculated.")
        sys.exit(1)

    try:
        report_content = generate_similarity_report(max_similarities)
        report_path = os.path.join(cfg.RESULTS_DIR, '4_3_tanimoto_similarity_results.txt')
        ensure_dir_exists(cfg.RESULTS_DIR)

        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(report_content)
        print(f"✅ Report saved: {report_path}")
        
    except Exception as e:
        print(f"⚠️ ERROR: Could not save report. {e}")

    mean_similarity = display_summary(max_similarities)
    plot_and_save_histogram(max_similarities, mean_similarity)

    print("\n✅ Tanimoto Similarity Analysis completed successfully!")
    print("\n➡️     Next step: run the final evaluation script (4.4).")

if __name__ == '__main__':
    main()
