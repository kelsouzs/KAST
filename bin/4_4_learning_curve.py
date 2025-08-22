# bin/4_4_learning_curve.py

"""
K-talysticFlow - Step 4.4: Learning Curve Generation

This script generates a Learning Curve to diagnose the model's behavior.
The curve plots performance (AUC Score) on the training set and a fixed
validation set as a function of the number of training samples.

Interpretation:
- If training and validation converge: well-fitted model
- If large gap between training and validation: overfitting
- If both are low: underfitting or insufficient data
"""

import os
import warnings
import logging
import pandas as pd
from typing import Tuple, List, Optional

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
warnings.filterwarnings('ignore', category=UserWarning)
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=DeprecationWarning)

import tensorflow as tf
tf.get_logger().setLevel('ERROR')
logging.getLogger('deepchem').setLevel('ERROR')

# --- Main Imports ---
import sys
import numpy as np
import deepchem as dc
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from tqdm import tqdm

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import settings as cfg
from utils import ensure_dir_exists, load_smiles_from_file

# --- Helper Functions ---

def load_and_featurize_data() -> Optional[Tuple[np.ndarray, np.ndarray]]:
    """Loads all data, combines, labels, and featurizes."""
    from rdkit import RDLogger
    RDLogger.DisableLog('rdApp.*')
    
    print("\nLoading data...")
    
    try:
        actives = load_smiles_from_file(cfg.ACTIVE_SMILES_FILE)
        inactives = load_smiles_from_file(cfg.INACTIVE_SMILES_FILE)
        
        if not actives or not inactives: 
            print("\n⚠️ ERROR: Active or inactive files not found.")
            print("➡️ Verify that the SMILES files exist.")
            return None

        all_smiles = actives + inactives
        all_labels = np.array([1] * len(actives) + [0] * len(inactives))
        
        print(f"\nActive molecules: {len(actives)}")
        print(f"Inactive molecules: {len(inactives)}")
        print("\nCalculating fingerprints...")
        
        featurizer = dc.feat.CircularFingerprint(size=cfg.FP_SIZE, radius=cfg.FP_RADIUS)
        features = featurizer.featurize(all_smiles)
        
        valid_indices = [i for i, x in enumerate(features) if x is not None and x.size > 0]
        features_valid = features[valid_indices]
        labels_valid = all_labels[valid_indices]
        
        print(f"\n✅ {len(features_valid)} molecules successfully featurized")
        
        return features_valid, labels_valid
        
    except Exception as e:
        print(f"\n⚠️ ERROR in featurization: {e}")
        return None

def generate_learning_curve_points(features: np.ndarray, labels: np.ndarray) -> Tuple[List[int], List[float], List[float]]:
    """Generates training and validation scores for different dataset sizes."""
    print("\nSplitting validation set...")
    
    try:
        features_train_pool, val_features, labels_train_pool, val_labels = train_test_split(
            features, labels, test_size=0.2, random_state=cfg.RANDOM_STATE, stratify=labels
        )
        val_dataset = dc.data.NumpyDataset(X=val_features, y=val_labels)
        
        print(f"Training pool: {len(features_train_pool)} samples")
        print(f"Fixed validation: {len(val_features)} samples")

        # Define training sizes (10% to 100% in 10 points)
        train_sizes_fractions = np.linspace(0.1, 1.0, 10, endpoint=True)
        train_scores, val_scores, actual_train_sizes = [], [], []

        print(f"\nGenerating {len(train_sizes_fractions)} learning curve points...")
        
        for i, frac in enumerate(tqdm(train_sizes_fractions, desc="Curve Progress")):
            try:
                # Select subset of training pool
                if frac < 1.0:
                    current_train_features, _, current_train_labels, _ = train_test_split(
                        features_train_pool, labels_train_pool, 
                        train_size=frac, 
                        random_state=cfg.RANDOM_STATE, 
                        stratify=labels_train_pool
                    )
                else:
                    current_train_features, current_train_labels = features_train_pool, labels_train_pool
                
                # Check minimum size
                if len(current_train_features) < 10: 
                    print(f"    -> Point {i+1}: too few samples ({len(current_train_features)}), skipping...")
                    continue
                
                train_subset = dc.data.NumpyDataset(X=current_train_features, y=current_train_labels)
                
                model_lc = dc.models.MultitaskClassifier(
                    n_tasks=cfg.MODEL_PARAMS['n_tasks'],
                    n_features=cfg.FP_SIZE,
                    layer_sizes=cfg.MODEL_PARAMS['layer_sizes'],
                    dropouts=cfg.MODEL_PARAMS['dropouts'],
                    mode=cfg.MODEL_PARAMS['mode'],
                    learning_rate=cfg.MODEL_PARAMS['learning_rate']
                )

                model_lc.fit(train_subset, nb_epoch=cfg.MODEL_PARAMS['nb_epoch'])
                
                # Predictions on training
                y_train_true = train_subset.y.flatten()
                y_train_pred_raw = model_lc.predict(train_subset)
                y_train_pred = y_train_pred_raw[:, 0, 1] if y_train_pred_raw.ndim == 3 else y_train_pred_raw[:, 1]
                
                # Predictions on validation
                y_val_true = val_dataset.y.flatten()
                y_val_pred_raw = model_lc.predict(val_dataset)
                y_val_pred = y_val_pred_raw[:, 0, 1] if y_val_pred_raw.ndim == 3 else y_val_pred_raw[:, 1]
                
                # Calculate AUC scores
                train_auc = roc_auc_score(y_train_true, y_train_pred)
                val_auc = roc_auc_score(y_val_true, y_val_pred)
                
                train_scores.append(train_auc)
                val_scores.append(val_auc)
                actual_train_sizes.append(len(current_train_features))
                
                print(f"\nPoint {i+1}: {len(current_train_features)} samples | "
                      f"Training: {train_auc:.3f} | Validation: {val_auc:.3f}")
                
            except Exception as e:
                print(f"\n⚠️ ERROR at point {i+1}: {e}")
                continue

        return actual_train_sizes, train_scores, val_scores
        
    except Exception as e:
        print(f"\n⚠️ ERROR generating learning curve: {e}")
        return [], [], []

def generate_learning_curve_report(train_sizes: List[int], train_scores: List[float], 
                                 val_scores: List[float]) -> str:
    """Generates a detailed learning curve report."""
    if not train_sizes:
        return "⚠️ ERROR: No curve points generated."
    
    final_train_score = train_scores[-1]
    final_val_score = val_scores[-1]
    gap = final_train_score - final_val_score
    
    if gap > 0.1:
        diagnosis = "Possible overfitting - High gap between training and validation"
        recommendation = "Consider regularization, higher dropout, or more data"
    elif final_val_score < 0.7:
        diagnosis = "Possible underfitting - Low scores in both sets"
        recommendation = "Consider a more complex model or more features"
    elif gap < 0.05 and final_val_score > 0.8:
        diagnosis = "Well-fitted model - Good convergence"
        recommendation = "Model is performing adequately"
    else:
        diagnosis = "Moderate behavior"
        recommendation = "Monitor performance on independent data"
    
    report = f"""=== LEARNING CURVE REPORT ===
Date/Time: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}
Number of points generated: {len(train_sizes)}

=== FINAL SCORES ===
Final Training AUC: {final_train_score:.4f}
Final Validation AUC: {final_val_score:.4f}
Gap (Training - Validation): {gap:.4f}

=== DIAGNOSIS ===
{diagnosis}

=== RECOMMENDATION ===
{recommendation}

=== DETAILS PER POINT ===
"""
    
    for i, (size, train_auc, val_auc) in enumerate(zip(train_sizes, train_scores, val_scores)):
        report += f"Point {i+1:2d}: {size:4d} samples | Training: {train_auc:.4f} | Validation: {val_auc:.4f}\n"
    
    return report

def plot_and_save_curve(train_sizes: List[int], train_scores: List[float], val_scores: List[float]):
    """Plots and saves the learning curve graph."""
    try:
        output_path = os.path.join(cfg.RESULTS_DIR, '4_4_learning_curve.png')
        ensure_dir_exists(cfg.RESULTS_DIR)
        
        plt.figure(figsize=(10, 6))
        plt.plot(train_sizes, train_scores, 'o-', color="#8A2BE2", linewidth=2, 
                markersize=6, label="Training Score")
        plt.plot(train_sizes, val_scores, 'o-', color="#32CD32", linewidth=2, 
                markersize=6, label="Validation Score")
        
        plt.title("Model Learning Curve", fontsize=16, fontweight='bold')
        plt.xlabel("Number of Training Samples", fontsize=12)
        plt.ylabel("AUC Score", fontsize=12)
        plt.legend(loc="best", fontsize=11)
        plt.grid(True, linestyle='--', alpha=0.6)
        plt.ylim(0.4, 1.05)
        
        if train_scores and val_scores:
            final_gap = train_scores[-1] - val_scores[-1]
            plt.text(0.02, 0.98, f'Final Gap: {final_gap:.3f}', 
                    transform=plt.gca().transAxes, fontsize=10,
                    bbox=dict(boxstyle="round,pad=0.3", facecolor="yellow", alpha=0.7),
                    verticalalignment='top')
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"✅ Graph saved: {output_path}")
        
    except Exception as e:
        print(f"\n⚠️ ERROR saving graph: {e}")

def display_summary(train_scores: List[float], val_scores: List[float]):
    """Displays a summary with final scores of the learning curve."""
    if not train_scores or not val_scores:
        print("\n❌ No results to display.")
        return
        
    final_train_score = train_scores[-1]
    final_val_score = val_scores[-1]
    gap = final_train_score - final_val_score
    
    print("\n=========================================================")
    print("==        LEARNING CURVE SUMMARY                        ==")
    print("=========================================================")
    print(f"  Final Training AUC        : {final_train_score:.4f}")
    print(f"  Final Validation AUC      : {final_val_score:.4f}")
    print(f"  Gap (Training - Validation): {gap:.4f}")
    print("---------------------------------------------------------")
    
    # Interpretation
    if gap > 0.1:
        interpretation = "Possible overfitting"
    elif final_val_score < 0.7:
        interpretation = "Possible underfitting"
    elif gap < 0.05 and final_val_score > 0.8:
        interpretation = "Well-fitted model"
    else:
        interpretation = "Moderate behavior"
        
    print(f"  Interpretation             : {interpretation}")
    print("=========================================================")

# --- Main Function ---
def main():
    """Orchestrates the learning curve generation."""
    print("\n--- K-talysticFlow | Step 4.4: Learning Curve Generation ---")

    # Step 1: Load and featurize all data
    loaded_data = load_and_featurize_data()
    if loaded_data is None:
        sys.exit(1)
        
    features, labels = loaded_data

    # Step 2: Generate curve points
    train_sizes, train_scores, val_scores = generate_learning_curve_points(features, labels)

    if not train_sizes:
        print("\n❌ No learning curve points could be generated.")
        sys.exit(1)

    # Step 3: Save detailed report
    try:
        report_content = generate_learning_curve_report(train_sizes, train_scores, val_scores)
        report_path = os.path.join(cfg.RESULTS_DIR, '4_4_learning_curve_results.txt')
        
        with open(report_path, 'w') as f:
            f.write(report_content)
            
        print(f"✅ Detailed report saved: {report_path}")
        
    except Exception as e:
        print(f"\n⚠️ ERROR saving report: {e}")

    # Step 4: Display summary and save graph
    display_summary(train_scores, val_scores)
    plot_and_save_curve(train_sizes, train_scores, val_scores)

    print("\n✅ Learning Curve Generation completed successfully!")

if __name__ == '__main__':
    main()
