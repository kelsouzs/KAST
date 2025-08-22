# -*- coding: utf-8 -*-
"""
==============================================================================
===                      K-talysticFlow (KAST)                             ===
==============================================================================
Description:
    This control panel manages a Deep Learning pipeline for molecular activity
    prediction, from data preparation to model evaluation and prediction on
    new molecules.
==============================================================================
"""
import os
import warnings

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
warnings.filterwarnings('ignore', category=UserWarning)
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=DeprecationWarning)

try:
    import tensorflow as tf
    tf.get_logger().setLevel('ERROR')
    tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
except ImportError:
    pass  

try:
    import logging
    logging.getLogger('deepchem').setLevel('ERROR')
except ImportError:
    pass

import sys
import subprocess
import argparse
import logging
from datetime import datetime
import settings as cfg

# =============================================================================
# Project Metadata
# =============================================================================

__author__ = "Késsia Souza Santos"
__version__ = "1.0.0"
__status__ = "Under Development"
__email__ = "kelsouzs.uefs@gmail.com"
__github__ = "https://github.com/kelsouzs"
__linkedin__ = "https://www.linkedin.com/in/kelsouzs"
__project_name__ = "K-talysticFlow"
__project_acronym__ = "KAST"

SCRIPTS_DIR = "bin"

EVALUATION_SCRIPTS = [
    "4_0_evaluation_main.py",
    "4_1_cross_validation.py",
    "4_2_enrichment_factor.py",
    "4_3_tanimoto_similarity.py",
    "4_4_learning_curve.py"
]

EVALUATION_SCRIPTS_NAMES = [
    "Main Report (AUC, Accuracy, etc.)",
    "Cross-Validation",
    "Enrichment Factor",
    "Tanimoto Similarity Analysis",
    "Learning Curve Generation"
]

# =============================================================================
# Dependency Check
# =============================================================================
try:
    from bin.check_env import check_dependencies
except (ModuleNotFoundError, ImportError):
    print("\033[91mCRITICAL ERROR: Could not find 'bin/check_env.py'.\033[0m")
    sys.exit(1)

# =============================================================================
# Control Panel Functions
# =============================================================================

def display_splash_screen():
    art = r"""
         __  __    _     ____  _____ 
            | |/ /   / \   / ___||_   _|
            | ' /   / _ \  \___ \  | |  
            | . \  / ___ \  ___) | | |  
            |_|\_\/_/   \_\|____/  |_|  
"""
    print(art)
    print(f"\033[90mVersion {__version__} | Developed by: {__author__}\033[0m")

def setup_logging():
    """Configures logging to track executions"""
    log_dir = os.path.join('results', 'logs')
    os.makedirs(log_dir, exist_ok=True)
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(os.path.join(log_dir, f'kast_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log')),
            logging.StreamHandler()
        ]
    )

def check_data_files():
    required_files = [cfg.ACTIVE_SMILES_FILE, cfg.INACTIVE_SMILES_FILE]
    missing_files = [f for f in required_files if not os.path.exists(f)]
    
    if missing_files:
        print(f"\n⚠️ WARNING: Required files not found:")
        for f in missing_files:
            print(f"   - {f}")
        return False
    return True

def show_credits():
    os.system('cls' if os.name == 'nt' else 'clear')
    print("=========================================================")
    print("==                 ABOUT & HOW TO CITE                 ==")
    print("=========================================================")
    print(f"\n  Project: \t{__project_name__} ({__project_acronym__})")
    print(f"  Version: \t{__version__} ({__status__})")
    print("\n  Developed by:")
    print(f"  \t{__author__}")
    print(f"  \tEmail: {__email__}")
    print(f"  \tGitHub: {__github__}")
    print(f"  \tLinkedIn: {__linkedin__}")
    print("\n  Under the supervision of:")
    print("\tProf. Dr. Manoelito Coelho dos Santos Junior")
    print("\n  How to cite this project:")
    print("\tSANTOS, K. S.; SANTOS JUNIOR, M. C. K-talysticFlow: A Deep Learning Workflow for Molecular Activity Prediction. 2025.")
    print("\n  Institutional Support:")
    print("\tMolecular Modeling Laboratory (LMM - UEFS)")
    print("\tNational Council for Scientific and Technological Development (CNPq)")
    print("\n  License: MIT")
    print("\n=========================================================")

def run_script(script_name):
    script_path = os.path.join(SCRIPTS_DIR, script_name)
    try:
        print(f"\nRUNNING ({script_name}) \n")
        subprocess.run([sys.executable, script_path], check=True, text=True)
        print(f"\n'{script_name}' EXECUTED SUCCESSFULLY!\n")
        return True
    except FileNotFoundError:
        print(f"\n\033[91mERROR: Script '{script_path}' not found.\033[0m")
        return False
    except subprocess.CalledProcessError:
        print(f"\n\033[91mERROR: An issue occurred while executing '{script_name}'. Check output above.\033[0m")
        return False
    except KeyboardInterrupt:
        print("\n⚠️ Execution interrupted by user.")
        return False

def pause_and_clear():
    input("\nPress Enter to continue to the next step...")
    os.system('cls' if os.name == 'nt' else 'clear')

def step_1_preparation():
    if not check_data_files():
        input("\nFix missing files and try again. Press Enter...")
        return False
    logging.info(" Starting data preparation...")
    result = run_script("1_preparation.py")
    if result:
        logging.info(" Step completed successfully")
    else:
        logging.error(" Data preparation failed")
    return result

def step_2_featurization(): 
    logging.info(" Starting featurization")
    result = run_script("2_featurization.py")
    if result:
        logging.info(" Step completed successfully")
    else:
        logging.error("\n Featurization failed")
    return result

def step_3_training(): 
    logging.info(" Starting training")
    result = run_script("3_training.py")
    if result:
        logging.info(" Step completed successfully")
    else:
        logging.error(" Training failed")
    return result

def step_4_evaluation_menu():
    """Displays evaluation submenu allowing single or all evaluations"""
    while True:
        os.system('cls' if os.name == 'nt' else 'clear')
        print("=========================================================")
        print("==           Submenu [4] - Model Evaluation               ==")
        print("=========================================================")
        print("\n  [1] Run ALL evaluations in sequence")
        print("  [2] Run a SPECIFIC evaluation")
        print("\n  [0] Return to Main Menu")
        print("=========================================================")
        sub_choice = input("Enter your choice: ")

        if sub_choice == '1':
            print("\n====================================================")
            print("==           RUNNING ALL EVALUATIONS IN SEQUENCE    ==")
            print("====================================================")
            for script in EVALUATION_SCRIPTS:
                if not run_script(script):
                    print(f"\n\033[91mEVALUATION INTERRUPTED DUE TO ERROR IN SCRIPT: '{script}'\033[0m")
                    break
            input("\nPress Enter to return to submenu...")
            continue
        elif sub_choice == '2':
            while True:
                os.system('cls' if os.name == 'nt' else 'clear')
                print("--- Choose the evaluation to run ---")
                for i, friendly_name in enumerate(EVALUATION_SCRIPTS_NAMES, 1):
                    print(f"  [{i}] {friendly_name}")
                print("\n  [0] Return to Previous Submenu")
                print("=========================================================")
                script_choice_str = input("Enter the test number: ")
                if script_choice_str.isdigit():
                    script_choice_int = int(script_choice_str)
                    if 1 <= script_choice_int <= len(EVALUATION_SCRIPTS):
                        script_to_run = EVALUATION_SCRIPTS[script_choice_int - 1]
                        run_script(script_to_run)
                        input("\nPress Enter to continue...")
                        continue
                    elif script_choice_int == 0:
                        break
                input("Invalid input. Press Enter...")
        elif sub_choice == '0':
            return
        else:
            input(f"Invalid option '{sub_choice}'. Press Enter...")

def step_5_prediction_menu():
    """Submenu for predictions on new molecules with modular options"""
    while True:
        os.system('cls' if os.name == 'nt' else 'clear')
        print("=========================================================")
        print("==        Submenu [5] - Predict New Molecules          ==")
        print("=========================================================")
        print("\n  [1] Only Featurize (Prepare dataset)")
        print("  [2] Only Predict (Use existing dataset)")
        print("  [3] Full Flow (Featurize + Predict)")
        print("\n  [0] Return to Main Menu")
        print("=========================================================")
        
        prediction_choice = input("Enter your choice: ")
        
        if prediction_choice == '1':
            print("\n🔄 Running featurization...")
            logging.info("Starting featurization for prediction")
            result = run_script("5_0_featurize_for_prediction.py")
            if result:
                print("✅ Step completed!")
                logging.info("Featurization for prediction completed successfully")
            else:
                print("❌ Featurization error.")
                logging.error("Featurization for prediction failed")
            input("\nPress Enter to continue...")
                
        elif prediction_choice == '2':
            print("\n🔄 Running prediction...")
            logging.info("Starting prediction on molecules")
            result = run_script("5_1_run_prediction.py")
            if result:
                print("✅ Step completed!")
                logging.info("Prediction completed successfully")
            else:
                print("❌ Prediction error.")
                logging.error("Prediction failed")
            input("\nPress Enter to continue...")
                
        elif prediction_choice == '3':
            print("\n🔄 Running full flow...")
            logging.info("Starting full prediction flow")
            result1 = run_script("5_0_featurize_for_prediction.py")
            if not result1:
                print("❌ Featurization error. Aborting flow.")
                logging.error("Featurization failed - flow aborted")
            else:
                print("✅ Step completed!")
                result2 = run_script("5_1_run_prediction.py")
                if result2:
                    print("✅ Full flow completed successfully!")
                    logging.info("Full prediction flow completed successfully")
                else:
                    print("❌ Prediction error.")
                    logging.error("Prediction failed - partially completed flow")
            input("\nPress Enter to continue...")
                    
        elif prediction_choice == '0':
            break
        else:
            print("Invalid option.")
            input("Press Enter to try again...")

def run_full_pipeline():
    """Runs the full pipeline with pauses and clearing between steps"""
    print("\n------------------ RUNNING FULL PIPELINE -------------------")
    if not step_1_preparation(): return
    pause_and_clear()
    if not step_2_featurization(): return
    pause_and_clear()
    if not step_3_training(): return
    pause_and_clear()
    print("\n--- Starting full evaluation as part of the flow ---")
    for script_name in EVALUATION_SCRIPTS:
        if not run_script(script_name):
            print(f"\nFLOW INTERRUPTED DUE TO ERROR IN EVALUATION SCRIPT: '{script_name}'")
            return
        if script_name != EVALUATION_SCRIPTS[-1]:
            pause_and_clear()
    print("\n\n✅ FULL PIPELINE COMPLETED SUCCESSFULLY!")

def display_menu():
    os.system('cls' if os.name == 'nt' else 'clear')
    display_splash_screen()
    print("\n=========================================================")
    print(f"==           Control Panel - {__project_acronym__}          ==")
    print("=========================================================")
    print("\n  [1] Prepare and Split Data")
    print("  [2] Generate Fingerprints")
    print("  [3] Train the Model")
    print("  [4] Evaluate the Model")
    print("  [5] Predict Activity of New Molecules")
    print("\n------------------- FULL FLOW ----------------------")
    print("\n  [6] Run Full Pipeline (Options 1 to 4)")
    print("\n---------------------------------------------------------")
    print("\n  [9] About & How to Cite")
    print("  [0] Exit Program")
    print("\n=========================================================")

def run_interactive_menu():
    """Manages the menu and user selection"""
    while True:
        display_menu()
        choice = input("Enter your choice number: ")
        actions = {
            '1': step_1_preparation,
            '2': step_2_featurization,
            '3': step_3_training,
            '4': step_4_evaluation_menu,
            '5': step_5_prediction_menu,
            '6': run_full_pipeline,
            '9': show_credits,
        }
        if choice in actions:
            actions[choice]()
            if choice not in ['4', '5']:
                input("\nPress Enter to return to main menu...")
        elif choice == '0':
            print(f"\nThank you for using {__project_name__}! Developed by {__author__}.\n")
            break
        else:
            input(f"\nInvalid option '{choice}'. Press Enter to try again...")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=f"Main script for the project {__project_name__}.")
    parser.add_argument('--check-env', action='store_true', help="Only checks if dependencies are installed and exits.")
    args = parser.parse_args()

    setup_logging()
    logging.info("K-talysticFlow started")

    if args.check_env:
        sys.exit(0 if check_dependencies() else 1)

    if not check_dependencies():
        print("\n\033[91mExecution aborted. Please install missing dependencies.\033[0m")
        sys.exit(1)

    run_interactive_menu()
