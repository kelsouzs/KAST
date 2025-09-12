# bin/4_evaluation_main.py 

"""
K-talysticFlow - Step 4: Main Model Evaluation

This script performs the main evaluation of the trained model on the test set.
It performs the following actions:
1. Checks if there are old evaluation files and asks for confirmation to overwrite them.
2. Loads the test dataset and the trained model.
3. Generates probability and class predictions, with a progress bar.
4. Calculates a comprehensive set of classification metrics.
5. Generates and saves a ROC Curve plot, a text report, and a CSV with predictions.
6. Displays a formatted summary of the results in the terminal.
"""

import os
import warnings
import logging

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
warnings.filterwarnings('ignore', category=UserWarning)
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=DeprecationWarning)

import tensorflow as tf
tf.get_logger().setLevel('ERROR')
logging.getLogger('deepchem').setLevel('ERROR')


import sys
import json
from datetime import datetime
import deepchem as dc
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import (
    roc_curve, auc, confusion_matrix, accuracy_score,
    precision_score, recall_score, f1_score
)
from tqdm import tqdm 

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import settings as cfg
from utils import ensure_dir_exists


def clean_previous_evaluation_files(results_dir):
    """Checks if there are old evaluation files and asks for confirmation to delete them."""
    files_to_check = ['4_evaluation_report.txt', '4_roc_curve.png', '4_test_predictions.csv']
    existing_files = [f for f in files_to_check if os.path.exists(os.path.join(results_dir, f))]
    if not existing_files:
        return
    print(f"\n\033[\n 93mWARNING: Files from a previous evaluation were found.\033[0m")
    while True:
        choice = input("Do you want to overwrite them for a new evaluation? (y/n): ").lower()
        if choice in ['y', 'yes']:
            print("Cleaning previous evaluation files...")
            for f in existing_files:
                os.remove(os.path.join(results_dir, f))
            break
        elif choice in ['n', 'no']:
            print("\nOperation canceled.")
            sys.exit(0)

def load_test_data_and_model(test_data_dir, model_dir):
    from rdkit import RDLogger     
    RDLogger.DisableLog('rdApp.*')
    print(f"\nLoading test dataset from: {test_data_dir}")
    if not os.path.exists(test_data_dir):
        print(f"\n ⚠️ ERROR: Test directory '{test_data_dir}' not found.")
        return None, None
    test_dataset = dc.data.DiskDataset(test_data_dir)
    n_features = test_dataset.get_shape()[0][1]

    print(f"Loading the previously created model...")

    model = dc.models.MultitaskClassifier(
    n_tasks=cfg.MODEL_PARAMS['n_tasks'],        
    n_features=n_features,
    layer_sizes=cfg.MODEL_PARAMS['layer_sizes'], 
    dropouts=cfg.MODEL_PARAMS['dropouts'],        
    mode=cfg.MODEL_PARAMS['mode'],               
    learning_rate=cfg.MODEL_PARAMS['learning_rate'], 
    model_dir=model_dir
)
    model.restore()
    return test_dataset, model

def generate_predictions(model, dataset):
    print("\nGenerating predictions...")
    
    y_true = dataset.y.flatten()
    y_pred_proba_raw = model.predict(dataset)
    
    if y_pred_proba_raw.ndim == 3 and y_pred_proba_raw.shape[1] == cfg.MODEL_PARAMS['n_tasks'] and y_pred_proba_raw.shape[2] == 2:
        y_pred_proba_active = y_pred_proba_raw[:, 0, 1]  
    elif y_pred_proba_raw.ndim == 2 and y_pred_proba_raw.shape[1] == 2:
        y_pred_proba_active = y_pred_proba_raw[:, 1]    
    else:
        print(f"\n❌ ERROR: Unexpected prediction format: {y_pred_proba_raw.shape}")
        return None, None, None
    
    y_pred_class = (y_pred_proba_active >= cfg.CLASSIFICATION_THRESHOLD).astype(int)
    
    print("\n✅ Predictions successfully generated!")
    return y_true, y_pred_proba_active, y_pred_class

def calculate_metrics(y_true, y_pred_proba, y_pred_class):

    metrics = {}
    fpr, tpr, _ = roc_curve(y_true, y_pred_proba)
    metrics['roc_auc'] = auc(fpr, tpr)
    metrics['accuracy'] = accuracy_score(y_true, y_pred_class)
    metrics['precision'] = precision_score(y_true, y_pred_class, zero_division=0)
    metrics['recall_sensitivity'] = recall_score(y_true, y_pred_class, zero_division=0)
    metrics['f1_score'] = f1_score(y_true, y_pred_class, zero_division=0)
    try:
        tn, fp, _, _ = confusion_matrix(y_true, y_pred_class).ravel()
        metrics['specificity'] = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    except ValueError:
        metrics['specificity'] = 0.0
    return metrics, fpr, tpr

def create_roc_plot(fpr, tpr, roc_auc, save_path):

    plt.rcParams.update({
        'font.size': 12,
        'font.family': 'serif',
        'font.serif': ['Times New Roman'],
        'axes.linewidth': 1.5,
        'axes.spines.top': False,
        'axes.spines.right': False,
        'xtick.major.size': 8,
        'xtick.major.width': 1.5,
        'ytick.major.size': 8,
        'ytick.major.width': 1.5,
        'legend.frameon': True,
        'legend.fancybox': True,
        'legend.shadow': True
    })
    
    fig, ax = plt.subplots(figsize=(8, 8), dpi=300)
    fig.patch.set_facecolor('white')
    
    roc_color = '#1f77b4'      
    baseline_color = '#d62728'  
    grid_color = '#f0f0f0'     

    ax.plot(fpr, tpr, 
           color=roc_color, 
           linewidth=3, 
           label=f'ROC Curve (AUC = {roc_auc:.3f})',
           antialiased=True,
           zorder=3)
    

    ax.plot([0, 1], [0, 1], 
           color=baseline_color, 
           linewidth=2, 
           linestyle='--', 
           alpha=0.8,
           label='Random Classifier',
           zorder=2)
    

    ax.fill_between(fpr, tpr, alpha=0.2, color=roc_color, zorder=1)
    

    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.0])
    ax.set_xlabel('False Positive Rate', fontsize=14, fontweight='bold', color='#2c3e50')
    ax.set_ylabel('True Positive Rate', fontsize=14, fontweight='bold', color='#2c3e50')
    ax.set_title('ROC Curve Analysis', fontsize=16, fontweight='bold', color='#2c3e50', pad=20)
    

    ax.grid(True, color=grid_color, linewidth=1, alpha=0.8, zorder=0)
    ax.set_axisbelow(True)
 
    ax.tick_params(axis='both', which='major', labelsize=12, colors='#34495e')
    ax.set_xticks(np.arange(0, 1.1, 0.2))
    ax.set_yticks(np.arange(0, 1.1, 0.2))
    
    legend = ax.legend(loc='lower right', 
                      fontsize=12, 
                      frameon=True, 
                      fancybox=True, 
                      shadow=True,
                      framealpha=0.95,
                      edgecolor='#bdc3c7',
                      facecolor='white')
    legend.get_frame().set_linewidth(1.5)
    
    for spine in ax.spines.values():
        spine.set_linewidth(2)
        spine.set_color('#34495e')
    
    plt.tight_layout()
    plt.savefig(save_path, 
               dpi=300, 
               bbox_inches='tight', 
               facecolor='white', 
               edgecolor='none',
               format='png')
    plt.close()
    
    plt.rcParams.update(plt.rcParamsDefault)

def display_summary(metrics):

    print("\n" + "="*65)
    print("------------------EVALUATION SUMMARY--------------------------")
    print("="*65)
    print(f"  AUC-ROC Score            : {metrics.get('roc_auc', 0):.4f}")
    print("-"*65)
    print(f"  Accuracy                 : {metrics.get('accuracy', 0):.4f}")
    print(f"  Precision                : {metrics.get('precision', 0):.4f}")
    print(f"  Recall                   : {metrics.get('recall_sensitivity', 0):.4f}")
    print(f"  Specificity              : {metrics.get('specificity', 0):.4f}")
    print(f"  F1-Score                 : {metrics.get('f1_score', 0):.4f}")
    print("="*65)
    print("----------------------------------------------------------------")
   
    auc_score = metrics.get('roc_auc', 0)
    if auc_score >= 0.9:
        print("🌟 EXCELLENT! Model with fantastic performance! 🌟")
    elif auc_score >= 0.8:
        print("✨ VERY GOOD! Solid model performance! ✨")
    elif auc_score >= 0.7:
        print("👍 GOOD! Acceptable performance! 👍")
    else:
        print("⚠️  IMPROVE: Performance below expectations. ⚠️")
    print()

def save_artifacts(metrics, fpr, tpr, roc_auc, y_true, y_pred_proba, y_pred_class, results_dir):
    """Saves the evaluation results to text, CSV, and PNG files."""
    ensure_dir_exists(results_dir)
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    roc_curve_path = os.path.join(results_dir, "4_0_roc_curve.png")
    create_roc_plot(fpr, tpr, roc_auc, roc_curve_path)

    report_path = os.path.join(results_dir, "4_0_evaluation_report.txt")
    with open(report_path, "w") as f:
        f.write(f"--- Model Evaluation Report ---\n")
        f.write(f"Generated on: {timestamp}\n\n")
        for key, value in metrics.items():
            f.write(f"{key}: {value:.4f}\n")
        f.write("\nROC Curve saved at: " + roc_curve_path + "\n")

    # 3. Save CSV with predictions
    predictions_path = os.path.join(results_dir, "4_0_test_predictions.csv")
    df_preds = pd.DataFrame({
        "true_label": y_true,
        "predicted_class": y_pred_class,
        "predicted_probability": y_pred_proba
    })
    df_preds.to_csv(predictions_path, index=False)

    print(f"\n✅ Evaluation artifacts saved in: {results_dir}")


def main():
    """Main function to run model evaluation."""
    print("\n--- K-talysticFlow | Step 4: Main Model Evaluation ---")

    start_time = datetime.now()

    test_data_dir = os.path.join(cfg.FEATURIZED_DATA_DIR, "test")
    model_dir = cfg.MODEL_DIR
    results_dir = cfg.RESULTS_DIR


    clean_previous_evaluation_files(results_dir)


    test_dataset, model = load_test_data_and_model(test_data_dir, model_dir)
    if test_dataset is None or model is None:
        sys.exit(1)

    y_true, y_pred_proba, y_pred_class = generate_predictions(model, test_dataset)
    if y_true is None:
        sys.exit(1)

    metrics, fpr, tpr = calculate_metrics(y_true, y_pred_proba, y_pred_class)

    display_summary(metrics)

  
    save_artifacts(metrics, fpr, tpr, metrics["roc_auc"], y_true, y_pred_proba, y_pred_class, results_dir)


    duration = datetime.now() - start_time
    print(f"\n⏱ Total duration: {str(duration).split('.')[0]} seconds")
    print("\n➡️    Next step: [5] Model Fine-Tuning.")


if __name__ == "__main__":
    main()
