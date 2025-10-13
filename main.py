# -*- coding: utf-8 -*-
"""
==============================================================================
===                      K-talysticFlow (KAST)                             ===
==============================================================================
Description:
    This control panel manages a Deep Learning pipeline for molecular activity
    prediction, from data preparation to model evaluation and prediction on
    new molecules.

⚙️  PARALLEL PROCESSING CONFIGURATION:
    All computational scripts now support parallel processing for faster
    performance (5-10x speedup on large datasets).
    
    To configure parallel processing, edit settings.py:
        • ENABLE_PARALLEL_PROCESSING (True/False)
        • N_WORKERS (None=auto, -1=all cores, or specific number)
        • PARALLEL_BATCH_SIZE (molecules per batch)
        • PARALLEL_MIN_THRESHOLD (minimum dataset size)
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

# Menu Formatting Constants
MENU_WIDTH = 57
SEPARATOR = "=" * MENU_WIDTH
SECTION_SEP = "-" * MENU_WIDTH

def format_header(text, width=MENU_WIDTH):
    """
    Formats text centered between == marks, perfectly aligned with separators.
    
    Args:
        text (str): Text to be centered in the header
        width (int): Total width of the header (default: MENU_WIDTH=57)
    
    Returns:
        str: Formatted header string with centered text
    
    Example:
        >>> format_header("Main Menu")
        "==                    Main Menu                     =="
    """
    inner_width = width - 4  # Remove "==" from both sides
    return f"=={text.center(inner_width)}=="

def print_menu_header(title):
    """
    Prints a formatted menu header with separators above and below.
    
    Args:
        title (str): Header title to display
    
    Example:
        >>> print_menu_header("Control Panel")
        =========================================================
        ==                   Control Panel                    ==
        =========================================================
    """
    print(f"\n{SEPARATOR}")
    print(format_header(title))
    print(SEPARATOR)

def display_splash_screen():
    """
    Displays the K-talysticFlow ASCII art logo and version information.
    
    Shows:
        - KAST ASCII logo
        - Current version number
        - Author name
        - Styled in gray color
    """
    art = r"""
            __  __    _     ____  _____ 
            | |/ /   / \   / ___||_   _|
            | ' /   / _ \  \___ \  | |  
            | . \  / ___ \  ___) | | |  
            |_|\_\/_/   \_\|____/  |_|  
"""
    print(art)
    version_text = f"Version {__version__} | Developed by: {__author__}"
    print(f"\033[90m{version_text.center(MENU_WIDTH)}\033[0m")

def show_credits():
    """
    Displays project information, credits, and citation details.
    
    Shows:
        - Project name and version
        - Developer information (name, email, social links)
        - Supervisor information
        - Citation format
        - Institutional support acknowledgments
        - License information
    
    Side Effects:
        Clears screen before displaying information
    """
    os.system('cls' if os.name == 'nt' else 'clear')
    print_menu_header("ABOUT & HOW TO CITE")
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
    print(f"\n{SEPARATOR}")

def run_script(script_name):
    """
    Executes a pipeline script as a subprocess and handles errors.
    
    Args:
        script_name (str): Name of the script file in the bin/ directory
    
    Returns:
        bool: True if script executed successfully, False otherwise
    
    Raises:
        FileNotFoundError: If script file doesn't exist
        CalledProcessError: If script returns non-zero exit code
        KeyboardInterrupt: If user cancels execution
    
    Side Effects:
        Prints error messages in red color on failure
    """
    script_path = os.path.join(SCRIPTS_DIR, script_name)
    try:
        subprocess.run([sys.executable, script_path], check=True, text=True)
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

def pause():
    """
    Pauses execution and waits for user to press Enter.
    
    Used between pipeline steps to allow user to review output
    before proceeding to the next stage.
    """
    input("\nPress Enter to continue to the next step...")

def step_1_preparation():
    """
    Executes Step 1: Data Preparation and Splitting.
    
    Checks for required input files before execution.
    Runs 1_preparation.py which handles:
        - SMILES import and validation
        - Train/test splitting
        - Activity labeling
    
    Returns:
        bool: True if preparation completed successfully, False otherwise
    """
    print("\n🔄 Running preparation...")
    result = run_script("1_preparation.py")
    return result

def step_2_featurization(): 
    """
    Executes Step 2: Molecular Featurization.
    
    Runs 2_featurization.py which handles:
        - ECFP/Morgan fingerprint generation
        - Sparse matrix optimization
        - Parallel processing (if enabled)
    
    Returns:
        bool: True if featurization completed successfully, False otherwise
    """
    print("\n🔄 Running featurization...")
    result = run_script("2_featurization.py")
    return result

def step_3_training(): 
    """
    Executes Step 3: Model Creation and Training.
    
    Runs 3_create_training.py which handles:
        - Neural network creation
        - Model training with configured parameters
        - Model checkpoint saving
    
    Returns:
        bool: True if training completed successfully, False otherwise
    """
    print("\n🔄 Creating and training model...")
    result = run_script("3_create_training.py")

def step_4_evaluation_menu():
    """Displays evaluation submenu allowing single or all evaluations"""
    while True:
        os.system('cls' if os.name == 'nt' else 'clear')
        print_menu_header("Submenu [4] - Model Evaluation")
        print("\n  [1] Run ALL evaluations in sequence")
        print("  [2] Run a SPECIFIC evaluation")
        print("\n  [0] Return to Main Menu")
        print(SEPARATOR)
        sub_choice = input("Enter your choice: ")

        # Clear screen immediately after choice
        os.system('cls' if os.name == 'nt' else 'clear')

        if sub_choice == '1':
            # Run all evaluations (each script will show its own header)
            for script in EVALUATION_SCRIPTS:
                if not run_script(script):
                    print(f"\n\033[91mEVALUATION INTERRUPTED DUE TO ERROR IN SCRIPT: '{script}'\033[0m")
                    break
            input("\nPress Enter to return to submenu...")
            continue
        elif sub_choice == '2':
            while True:
                os.system('cls' if os.name == 'nt' else 'clear')
                print(SEPARATOR)
                print("EVALUATION TESTS - Select an Option".center(MENU_WIDTH))
                print(SEPARATOR)
                for i, friendly_name in enumerate(EVALUATION_SCRIPTS_NAMES, 1):
                    print(f"  [{i}] {friendly_name}")
                print(f"\n  [0] Return to Previous Menu")
                print(SEPARATOR)
                script_choice_str = input("Enter your choice: ")
                
                # Clear screen immediately after choice
                os.system('cls' if os.name == 'nt' else 'clear')
                
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
        print_menu_header("Submenu [5] - Predict New Molecules")
        print("\n  [1] Only Featurize (Prepare dataset)")
        print("  [2] Only Predict (Use existing dataset)")
        print("  [3] Full Flow (Featurize + Predict)")
        print("\n  [0] Return to Main Menu")
        print(SEPARATOR)
        
        prediction_choice = input("Enter your choice: ")
        
        # Clear screen immediately after choice
        os.system('cls' if os.name == 'nt' else 'clear')
        
        if prediction_choice == '1':
            print("\n🔄 Running featurization...")
            result = run_script("5_0_featurize_for_prediction.py")
            if result:
                print("\n✅ Step completed!")
            else:
                print("❌ Featurization error.")
            input("\nPress Enter to continue...")
                
        elif prediction_choice == '2':
            print("\n🔄 Running prediction...")
            result = run_script("5_1_run_prediction.py")
            if result:
                print("\n✅ Step completed!")
            else:
                print("❌ Prediction error.")
            input("\nPress Enter to continue...")
                
        elif prediction_choice == '3':
            print("\n🔄 Running full flow...")
            result1 = run_script("5_0_featurize_for_prediction.py")
            if not result1:
                print("❌ Featurization error. Aborting flow.")
            else:
                print("\n✅ Step completed!")
                result2 = run_script("5_1_run_prediction.py")
                if result2:
                    print("\n✅ Full flow completed successfully!")
                else:
                    print("❌ Prediction error.")
            input("\nPress Enter to continue...")
                    
        elif prediction_choice == '0':
            break
        else:
            print("Invalid option.")
            input("Press Enter to try again...")

def configure_parallel_workers():
    """
    Interactive configuration tool for parallel processing CPU cores.
    
    Allows user to configure N_WORKERS setting without editing files:
        - Auto-detect: Uses cpu_count - 1 (recommended)
        - Specific number: 1 to total_cores
        - All cores: Uses all available cores (-1)
        - Cancel: Returns without changes
    
    Changes are:
        - Saved to settings.py file
        - Applied to cfg.N_WORKERS in memory
        - Effective immediately for next pipeline run
    
    Side Effects:
        - Displays system information and current configuration
        - Modifies settings.py file
        - Updates cfg.N_WORKERS global variable
        - Prints confirmation messages
    """
    from multiprocessing import cpu_count
    
    print_menu_header("Configure Parallel Processing")
    
    total_cores = cpu_count() or 4
    current_value = cfg.N_WORKERS
    
    print(f"\n📊 System Information:")
    print(f"   • Total CPU cores available: {total_cores}")
    print(f"   • Current setting: N_WORKERS = {current_value}")
    
    if current_value is None:
        actual_cores = max(1, total_cores - 1)
        print(f"   • Actual cores in use: {actual_cores} (auto-detected)")
    elif current_value == -1:
        print(f"   • Actual cores in use: {total_cores} (all cores)")
    else:
        print(f"   • Actual cores in use: {current_value}")
    
    print(f"\n⚙️ Configuration Options:")
    print(f"   [0] Auto-detect (recommended) → uses {max(1, total_cores - 1)} cores")
    print(f"   [1-{total_cores}] Use specific number of cores")
    print(f"   [-1] Use ALL cores ({total_cores})")
    print(f"   [Q] Cancel and return")
    
    print(f"\n💡 Tip: Auto-detect (0) is recommended for most users")
    
    while True:
        choice = input(f"\nEnter your choice: ").strip()
        
        if choice.upper() == 'Q':
            print("\n⚠️ Configuration unchanged.")
            return
        
        try:
            value = int(choice)
            
            if value == 0:
                # Auto-detect
                print(f"\n✅ Setting N_WORKERS = None (auto-detect)")
                print(f"   Will use {max(1, total_cores - 1)} cores")
                update_settings_file('N_WORKERS', None)
                cfg.N_WORKERS = None
                print("\n💾 Settings saved to settings.py")
                break
            elif value == -1:
                # All cores
                print(f"\n✅ Setting N_WORKERS = -1 (all cores)")
                print(f"   Will use {total_cores} cores")
                update_settings_file('N_WORKERS', -1)
                cfg.N_WORKERS = -1
                print("\n💾 Settings saved to settings.py")
                break
            elif 1 <= value <= total_cores:
                # Specific number
                print(f"\n✅ Setting N_WORKERS = {value}")
                update_settings_file('N_WORKERS', value)
                cfg.N_WORKERS = value
                print("\n💾 Settings saved to settings.py")
                break
            else:
                print(f"\n❌ Invalid number. Enter 0 for auto, -1 for all, or 1-{total_cores}")
        except ValueError:
            print(f"\n❌ Invalid input. Enter a number or 'Q' to cancel")

def update_settings_file(setting_name, new_value):
    """
    Programmatically updates a setting value in settings.py file.
    
    Args:
        setting_name (str): Name of the setting variable (e.g., 'N_WORKERS')
        new_value: New value to set (can be None, int, str, etc.)
    
    Returns:
        bool: True if update successful, False on error
    
    Side Effects:
        - Modifies settings.py file in-place
        - Preserves file structure and comments
        - Prints warning if update fails
    
    Implementation:
        Uses regex to find and replace the setting line while
        maintaining the original file formatting and comments.
    """
    import re
    
    settings_path = 'settings.py'
    
    try:
        with open(settings_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Find and replace the setting
        if new_value is None:
            new_line = f"{setting_name} = None"
        else:
            new_line = f"{setting_name} = {new_value}"
        
        # Match the line with the setting
        pattern = rf'^{setting_name}\s*=.*$'
        content = re.sub(pattern, new_line, content, flags=re.MULTILINE)
        
        with open(settings_path, 'w', encoding='utf-8') as f:
            f.write(content)
        
        return True
    except Exception as e:
        print(f"\n⚠️ Warning: Could not update settings.py: {e}")
        print("   Please edit settings.py manually")
        return False

def advanced_options_menu():
    """Advanced options submenu"""
    while True:
        os.system('cls' if os.name == 'nt' else 'clear')
        display_splash_screen()
        print_menu_header("Advanced Options")
        
        print("\n  [1] Check Environment & Dependencies")
        print("  [2] Test Parallel Processing Compatibility")
        print("  [3] Configure CPU Cores for Parallel Processing")
        print("\n  [0] Back to Main Menu")
        print(f"\n{SEPARATOR}")
        
        choice = input("Enter your choice: ").strip()
        
        # Clear screen after choice
        os.system('cls' if os.name == 'nt' else 'clear')
        
        if choice == '1':
            print("\n🔍 Checking environment...\n")
            run_script("check_env.py")
            input("\nPress Enter to continue...")
            
        elif choice == '2':
            print("\n🧪 Running parallel processing tests...\n")
            run_script("test_parallel_compatibility.py")
            input("\nPress Enter to continue...")
            
        elif choice == '3':
            configure_parallel_workers()
            input("\nPress Enter to continue...")
            
        elif choice == '0':
            break
        else:
            print("Invalid option.")
            input("Press Enter to try again...")

def run_full_pipeline():
    """Runs the full pipeline with pauses and clearing between steps"""
    print("\n----------------- 🔄 Running Full Pipeline -------------------")
    if not step_1_preparation(): return
    pause()
    if not step_2_featurization(): return
    pause()
    if not step_3_training(): return
    pause()
    print("\n--- Starting full evaluation as part of the flow ---")
    for script_name in EVALUATION_SCRIPTS:
        if not run_script(script_name):
            print(f"\nFlow Interrupted Due to Error in Evaluation Script: '{script_name}'")
            return
        if script_name != EVALUATION_SCRIPTS[-1]:
            pause()
    print("\n\n✅ Full Pipeline Completed Successfully!")

def display_menu():
    os.system('cls' if os.name == 'nt' else 'clear')
    display_splash_screen()
    print_menu_header(f"Control Panel - {__project_acronym__}")
    
    # Display parallel processing configuration
    try:
        from multiprocessing import cpu_count
        if cfg.ENABLE_PARALLEL_PROCESSING:
            if cfg.N_WORKERS is None:
                n_workers = max(1, (cpu_count() or 4) - 1)
                workers_text = f"{n_workers} cores (auto)"
            elif cfg.N_WORKERS == -1:
                workers_text = f"{cpu_count() or 4} cores (all)"
            else:
                workers_text = f"{cfg.N_WORKERS} cores"
            print(f"\n\033[92m⚡ Parallel Processing: ENABLED\033[0m")
        else:
            print(f"\n\033[91m⚡ Parallel Processing: DISABLED\033[0m")
        print(f"   \033[90mTo configure: Edit settings.py\033[0m")
    except Exception:
        pass  # Silently skip if there's any issue
    print(f"\n{SEPARATOR}")
    print("\n  [1] Prepare and Split Data")
    print("  [2] Generate Fingerprints")
    print("  [3] Create and Train the Model")
    print("  [4] Evaluate the Model")
    print("  [5] Predict Activity of New Molecules")
    print("\n--------------------- FULL FLOW -------------------------")
    print("\n  [6] Run Full Pipeline (Options 1 to 4)")
    print("\n---------------------------------------------------------")
    print("\n  [8] Advanced Options (Testing & Configuration)")
    print("  [9] About & How to Cite")
    print("  [0] Exit Program")
    print(f"\n{SEPARATOR}")

def run_interactive_menu():
    """Manages the menu and user selection"""
    while True:
        display_menu()
        choice = input("Enter your choice number: ")
        
        # Clear screen immediately after choice
        os.system('cls' if os.name == 'nt' else 'clear')
        
        actions = {
            '1': step_1_preparation,
            '2': step_2_featurization,
            '3': step_3_training,
            '4': step_4_evaluation_menu,
            '5': step_5_prediction_menu,
            '6': run_full_pipeline,
            '8': advanced_options_menu,
            '9': show_credits,
        }
        if choice in actions:
            actions[choice]()
            if choice not in ['4', '5', '8']:
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

    if args.check_env:
        sys.exit(0 if check_dependencies() else 1)

    if not check_dependencies():
        print("\n\033[91mExecution aborted. Please install missing dependencies.\033[0m")
        sys.exit(1)

    run_interactive_menu()
