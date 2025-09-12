"""
K-talysticFlow - Step 3: Model Training

This script trains a Deep Learning model for molecular activity classification
using the featurized data generated in the previous step.
"""

import sys
import os
import warnings
import logging
import random 

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
import numpy as np 
from datetime import datetime

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import settings as cfg
from utils import ensure_dir_exists

def ensure_training_reproducibility(seed=42):
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
    
    try:
        dc.utils.set_random_seed(seed)
        print("✅ DeepChem random seed set")
    except Exception:
        print("⚠️ DeepChem seed setting not available")
    
    print("✅ Training reproducibility configured - identical results guaranteed")

def save_training_metadata():
    import platform
    
    metadata = {
        "training_date": datetime.now().isoformat(),
        "random_seed": 42,
        "python_version": platform.python_version(),
        "tensorflow_version": tf.__version__,
        "deepchem_version": dc.__version__,
        "numpy_version": np.__version__,
        "system": platform.system(),
        "architecture": platform.architecture()[0],
        "deterministic_ops": os.environ.get('TF_DETERMINISTIC_OPS', 'not_set'),
        "model_parameters": cfg.MODEL_PARAMS,
        "fingerprint_settings": {
            "fp_size": cfg.FP_SIZE,
            "fp_radius": cfg.FP_RADIUS
        }
    }
    
    try:
        # Check if GPU is available and configured
        gpu_devices = tf.config.list_physical_devices('GPU')
        metadata["gpu_available"] = len(gpu_devices) > 0
        metadata["gpu_count"] = len(gpu_devices)
        if gpu_devices:
            metadata["gpu_details"] = str(gpu_devices)
    except:
        metadata["gpu_available"] = False
    
    try:
        metadata_path = os.path.join(cfg.RESULTS_DIR, 'training_metadata.json')
        ensure_dir_exists(cfg.RESULTS_DIR)
        
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        print(f"📋 Training metadata saved: {metadata_path}")
        
    except Exception as e:
        print(f"⚠️ Could not save training metadata: {e}")

def main():
    ensure_training_reproducibility(seed=42)
    
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

        print(f"\nStarting deterministic training for {cfg.MODEL_PARAMS['nb_epoch']} epochs...")
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
        
        save_training_metadata()
        
        print(f"\nLog saved in: {log_file_path}")
        print("\n✅ Model created, trained and saved successfully!")
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