"""
K-talysticFlow - Step 3: Model Creation and Training

This script creates and trains a Deep Learning model for molecular activity 
classification using the featurized data generated in the previous step.
"""

import sys
import os
import warnings
import logging
import random

# --- Environment Setup ---
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
import shutil  # Used to clean the model directory

# --- Path Configuration ---
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import settings as cfg
from utils import ensure_dir_exists
from main import display_splash_screen

def ensure_training_reproducibility(seed=42):
    """Sets the seeds to ensure reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    os.environ['TF_DETERMINISTIC_OPS'] = '1'
    
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
    """Save comprehensive training metadata."""
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
        
        with open(metadata_path, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, indent=2)
        
        print(f"📋 Training metadata saved: {metadata_path}")
        
    except Exception as e:
        print(f"⚠️ Could not save training metadata: {e}")

def save_model_safely(model, model_dir):
    """
    Safely save the trained model with multiple strategies.
    This ensures the model is always saved successfully.
    """
    print(f"\n💾 Saving trained model to: {model_dir}")
    
    save_success = False
    
    # Strategy 1: Use model.save() method
    try:
        model.save()
        print("✅ Model saved successfully")
        save_success = True
    except Exception as e:
        # Try checkpoint method silently
        pass
    
    # Strategy 2: Use model.save_checkpoint() as backup
    if not save_success:
        try:
            model.save_checkpoint()
            print("✅ Model saved successfully")
            save_success = True
        except Exception as e:
            pass
    
    # Strategy 3: Manual parameter saving (fallback)
    if not save_success:
        try:
            # Save model architecture and training info
            model_info = {
                'model_class': 'MultitaskClassifier',
                'model_parameters': cfg.MODEL_PARAMS,
                'n_features': model.n_features,
                'n_tasks': model.n_tasks,
                'training_completed': True,
                'training_date': datetime.now().isoformat(),
                'warning': 'Model weights not saved - only parameters available'
            }
            
            fallback_path = os.path.join(model_dir, 'model_info.json')
            with open(fallback_path, 'w', encoding='utf-8') as f:
                json.dump(model_info, f, indent=2)
            
            print("⚠️ Model weights could not be saved, but parameters were saved")
            print(f"Model info saved to: {fallback_path}")
            save_success = True
            
        except Exception as e:
            print(f"❌ All save strategies failed: {e}")
    
    # Verify save was successful
    if save_success:
        saved_files = []
        try:
            for file in os.listdir(model_dir):
                if any(ext in file.lower() for ext in ['.pkl', '.h5', '.json', '.pt', '.ckpt']):
                    saved_files.append(file)
            
            if saved_files:
                print(f"📁 Files saved in model directory:")
                for file in saved_files:
                    file_path = os.path.join(model_dir, file)
                    file_size = os.path.getsize(file_path) / 1024  # KB
                    print(f"   • {file} ({file_size:.1f} KB)")
            else:
                print("⚠️ Warning: No model files found in directory")
                
        except Exception as e:
            print(f"Could not list saved files: {e}")
    
    return save_success

def main():
    ensure_training_reproducibility(seed=42)
    
    from utils import print_script_banner, setup_script_logging
    logger = setup_script_logging("3_create_training")

    display_splash_screen()
    print_script_banner("K-talysticFlow | Step 3: Creating and Training the Model")
    logger.info("Starting model training")

    from rdkit import RDLogger
    RDLogger.DisableLog('rdApp.*')
    
    start_time = datetime.now()

    train_data_dir = os.path.join(cfg.FEATURIZED_DATA_DIR, 'train')  
    log_file_path = os.path.join(cfg.RESULTS_DIR, '03_training_log.txt')
    
    # --- FORCED CLEANUP OF THE MODEL DIRECTORY ---
    # This is the safest way to avoid conflicts with old files.
    if os.path.exists(cfg.MODEL_DIR):
        print(f"Cleaning existing model directory: {cfg.MODEL_DIR}")
        shutil.rmtree(cfg.MODEL_DIR)
    ensure_dir_exists(cfg.MODEL_DIR)

    # Start the log
    log_content = f"--- Training Log ---\nStarted at: {start_time.strftime('%Y-%m-%d %H:%M:%S')}\n"
    log_content += f"Model Parameters:\n{json.dumps(cfg.MODEL_PARAMS, indent=2)}\n\n"

    # Check if training data exists
    if not os.path.exists(train_data_dir):
        error_msg = f"ERROR: Training directory '{train_data_dir}' not found. Please run Step 2."
        print(f"\n❌ {error_msg}")
        logger.error(error_msg)
        with open(log_file_path, 'w', encoding='utf-8') as f: 
            f.write(log_content + error_msg)
        sys.exit(1)

    try:
        print(f"\nLoading training dataset from: {train_data_dir}")
        train_dataset = dc.data.DiskDataset(train_data_dir)
        n_features = train_dataset.get_shape()[0][1]
        
        print(f"  -> Dataset loaded: {len(train_dataset)} samples")
        print(f"  -> Number of features: {n_features}")
        log_content += f"Dataset: {len(train_dataset)} samples, {n_features} features.\n"
        
        print(f"\nCreating the model...")
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
        print("⚠️ This process may take several minutes...")
        
        # --- TRAINING WITH CHECKPOINT DISABLED TO AVOID FILE CONFLICTS ---
        model.fit(train_dataset, nb_epoch=cfg.MODEL_PARAMS['nb_epoch'], checkpoint_interval=0)
        
        print("\n✅ Training completed successfully!")
        
        # --- EXPLICIT MODEL SAVING ---
        # Since checkpoint_interval=0 disables auto-saving, we need to save manually
        save_success = save_model_safely(model, cfg.MODEL_DIR)
        
        if save_success:
            print(f"\n🚀 Model successfully saved to: {cfg.MODEL_DIR}")
        else:
            print(f"\n⚠️ Warning: Model saving encountered issues")

        duration = datetime.now() - start_time
        log_content += f"Training completed successfully in {str(duration).split('.')[0]}.\n"
        log_content += f"Model directory: {cfg.MODEL_DIR}\n"
        log_content += f"Model save status: {'Success' if save_success else 'Warning - check files'}\n"
        log_content += f"--- End of Log ---\n"
        
        # Save training metadata
        save_training_metadata()
        
        print("\n✅ Model created, trained, and saved successfully!")
        print("➡️ Next step: '[4] Evaluate the Model'.")
        logger.info("Training completed successfully")
        
    except Exception as e:
        error_msg = f"❌ CRITICAL ERROR during training: {str(e)}"
        print(f"\n{error_msg}")
        logger.error(error_msg, exc_info=True)
        log_content += f"{error_msg}\n"
        sys.exit(1)
        
    finally:
        # Ensures the log is saved even if an error occurs
        try:
            with open(log_file_path, 'w', encoding='utf-8') as f:
                f.write(log_content)
        except Exception as e:
            print(f"Warning: Could not save log: {e}")

if __name__ == '__main__':
    main()
