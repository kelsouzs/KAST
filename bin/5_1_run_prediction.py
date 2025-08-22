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
"""

import sys
import os
import logging
import warnings
from typing import Tuple, Optional

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


def load_data_and_model() -> Optional[Tuple[dc.data.Dataset, dc.models.Model]]:
    """Loads the featurized dataset and the trained model."""
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
        print(f"  -> Number of detected features: {n_features}")
        
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
    """Runs predictions on the featurized dataset."""
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

        results_df = pd.DataFrame({
            'smiles': dataset.ids,
            'K-prediction Score': y_pred_proba_active
        })

        results_df = results_df.sort_values(
            by='K-prediction Score', 
            ascending=False
        ).reset_index(drop=True)
        
        print(f"✅ Predictions completed successfully")
        
        return results_df
        
    except Exception as e:
        print(f"\n⚠️ ERROR during prediction: {e}")
        return pd.DataFrame()


def display_and_save_results(results_df: pd.DataFrame):
    """Displays a summary of predictions and saves them to CSV."""
    if results_df.empty:
        print("\n⚠️ ERROR: No results to save.")
        return
    
    try:
        output_path = os.path.join(cfg.RESULTS_DIR, cfg.OUTPUT_PREDICTIONS_CSV)
        ensure_dir_exists(cfg.RESULTS_DIR)
        results_df.to_csv(output_path, index=False)
        
        total_molecules = len(results_df)
        high_prob = len(results_df[results_df['K-prediction Score'] > 0.7])
        medium_prob = len(results_df[(results_df['K-prediction Score'] > 0.3) & 
                                     (results_df['K-prediction Score'] <= 0.7)])
        low_prob = len(results_df[results_df['K-prediction Score'] <= 0.3])
        
        print("\n=========================================================")
        print("==              PREDICTION RESULTS                     ==")
        print("=========================================================")
        print(f"  Total molecules analyzed   : {total_molecules}")
        print(f"  High probability (> 0.7)   : {high_prob} ({high_prob/total_molecules*100:.1f}%)")
        print(f"  Medium probability (0.3-0.7): {medium_prob} ({medium_prob/total_molecules*100:.1f}%)")
        print(f"  Low probability (≤ 0.3)    : {low_prob} ({low_prob/total_molecules*100:.1f}%)")
        print("---------------------------------------------------------")
        
        print("\n---------------------------------------------------------")
        print("  TOP 5 MOST PROMISING CANDIDATES:")
        print("---------------------------------------------------------")
        
        for i, (_, row) in enumerate(results_df.head().iterrows(), 1):
            prob = row['K-prediction Score']
            smi = row['smiles']
            display_smi = smi if len(smi) <= 60 else smi[:57] + "..."
            print(f"  {i}. {prob:.4f} | {display_smi}")
        
        print("=========================================================")
        
    except Exception as e:
        print(f"\nERROR saving results: {e}")


def main():
    print("\n--- K-talysticFlow | Step 5.1: Running Predictions ---")
    print("🔮 This script ONLY runs predictions on already featurized data")
    
    loaded_assets = load_data_and_model()
    if loaded_assets[0] is None:
        print("\n❌ Failed to load data or model.")
        sys.exit(1)
        
    prediction_dataset, model = loaded_assets

    results_df = run_prediction(model, prediction_dataset)
    if results_df.empty:
        print("\n❌ Prediction execution failed.")
        sys.exit(1)

    display_and_save_results(results_df)
    
    print("\n✅ Prediction completed successfully!")


if __name__ == '__main__':
    main()
