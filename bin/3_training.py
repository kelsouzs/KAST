# bin/3_training.py

"""
K-talysticFlow - Step 3: Model Training

This script trains a Deep Learning model for molecular activity classification
using the featurized data generated in the previous step.
"""

import sys
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

import deepchem as dc
import json
from datetime import datetime

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import settings as cfg
from utils import ensure_dir_exists

def main():
    """Train the classification model."""
    print("\n--- K-talysticFlow | Step 3: Training the Model ---")


    from rdkit import RDLogger
    RDLogger.DisableLog('rdApp.*')
    
    start_time = datetime.now()

    train_data_dir = os.path.join(cfg.FEATURIZED_DATA_DIR, 'train')  
    log_file_path = os.path.join(cfg.RESULTS_DIR, '03_training_log.txt')
    
    ensure_dir_exists(cfg.MODEL_DIR)


    log_content = f"--- Training Log ---\nStarted at: {start_time.strftime('%Y-%m-%d %H:%M:%S')}\n"
    log_content += f"Model Parameters:\n{json.dumps(cfg.MODEL_PARAMS, indent=2)}\n\n"

    if not os.path.exists(train_data_dir):
        error_msg = f"\nERROR: Training directory '{train_data_dir}' not found."
        print(f"\n{error_msg}")
        print("\n⚠️ Please run '[2] Generate Fingerprints' first.")
        log_content += f"{error_msg}\n"
        with open(log_file_path, 'w') as f: 
            f.write(log_content)
        sys.exit(1)

    try:
        print(f"\nLoading training dataset from: {train_data_dir}")
        train_dataset = dc.data.DiskDataset(train_data_dir)
        n_features = train_dataset.get_shape()[0][1]
        
        print(f"  -> Dataset loaded: {len(train_dataset)} samples")
        print(f"  -> Number of features: {n_features}")
        
        log_content += f"\nTraining dataset: {len(train_dataset)} samples, {n_features} features.\n"
        
        print(f"\nCreating model...")
        
        model = dc.models.MultitaskClassifier(
            n_tasks=cfg.MODEL_PARAMS['n_tasks'],
            n_features=n_features,
            layer_sizes=cfg.MODEL_PARAMS['layer_sizes'],
            dropouts=cfg.MODEL_PARAMS['dropouts'],
            mode=cfg.MODEL_PARAMS['mode'],
            learning_rate=cfg.MODEL_PARAMS['learning_rate'],
            model_dir=cfg.MODEL_DIR
        )

        print(f"\nStarting training for {cfg.MODEL_PARAMS['nb_epoch']} epochs...")
        print("\n⚠️ This process may take several minutes...")
        
        model.fit(train_dataset, nb_epoch=cfg.MODEL_PARAMS['nb_epoch'])
        
        print("\n✅ Training completed successfully!")
        print(f"Model saved in: {cfg.MODEL_DIR}")

        duration = datetime.now() - start_time
        log_content += "Training completed successfully.\n"
        log_content += f"Model saved in: {cfg.MODEL_DIR}\n"
        log_content += f"Total duration: {str(duration).split('.')[0]}s\n"
        log_content += f"--- End of Log ---\n"
        
        with open(log_file_path, 'w') as f:
            f.write(log_content)
        
        print(f"\nLog saved in: {log_file_path}")
        print("\n✅ Model trained and saved successfully!")
        print("➡️ Next: '[4] Evaluate the Model'.")
        
    except Exception as e:
        error_msg = f"\n⚠️ ERROR during training: {str(e)}"
        print(f"\n{error_msg}")
        log_content += f"{error_msg}\n"
        
        with open(log_file_path, 'w') as f:
            f.write(log_content)
        
        sys.exit(1)

if __name__ == '__main__':
    main()
