import sys
import os
import importlib.util

# Package mapping (pip) to import names (python)
# Add others here if needed. Ex: 'beautifulsoup4': 'bs4'
PACKAGE_IMPORT_MAP = {
    'scikit-learn': 'sklearn',
    'rdkit-pypi': 'rdkit', # If you installed rdkit via pip
    'rdkit': 'rdkit'      # Common name
}

# Terminal color codes
GREEN = '\033[92m'
RED = '\033[91m'
YELLOW = '\033[93m'
RESET = '\033[0m'

def parse_requirements(file_path):
    """Reads a requirements.txt file and returns a list of package names."""
    packages = []
    if not os.path.exists(file_path):
        print(f"{RED}[ERROR]{RESET} File '{file_path}' not found.")
        return None
        
    with open(file_path, 'r') as f:
        for line in f:
            line = line.strip()
            # Ignore comments, empty lines and edit flags (-e)
            if line and not line.startswith('#') and not line.startswith('-e'):
                # Remove version specifiers (==, >=, <=, [) to get package name
                package_name = line.split('==')[0].split('>=')[0].split('<=')[0].split('[')[0]
                packages.append(package_name.strip())
    return packages

def check_dependencies():
    """Checks if requirements.txt dependencies are installed."""
    print("--- Checking environment dependencies ---")
    
    # Script is in /bin, requirements.txt is in parent directory (root)
    req_path = os.path.join(os.path.dirname(__file__), '..', 'requirements.txt')
    packages = parse_requirements(req_path)

    if packages is None:
        return False

    missing_packages = []
    
    for pkg in packages:
        import_name = PACKAGE_IMPORT_MAP.get(pkg.lower(), pkg)
        print(f"Checking '{pkg}' (importing as '{import_name}')... ", end='')
        
        spec = importlib.util.find_spec(import_name)
        if spec is None:
            missing_packages.append(pkg)
            print(f"{RED}[MISSING]{RESET}")
        else:
            print(f"{GREEN}[OK]{RESET}")

    print("-" * 40)
    if not missing_packages:
        print(f"{GREEN}Success! All dependencies are installed.{RESET}")
        return True
    else:
        print(f"{RED}ERROR: The following dependencies were not found:{RESET}")
        for pkg in missing_packages:
            print(f"  - {pkg}")
        
        print(f"\n{YELLOW}--- Recommended Actions ---{RESET}")
        print("1. Make sure your conda environment ('modelo_preditivo') is activated.")
        if 'rdkit' in [p.lower() for p in missing_packages]:
            print(f"2. For {YELLOW}RDKit{RESET}, use the specific command:")
            print(f"   {YELLOW}conda install -c conda-forge rdkit{RESET}")
        print("3. For other libraries, run:")
        print(f"   {YELLOW}pip install -r requirements.txt{RESET}")
        return False

if __name__ == "__main__":
    all_ok = check_dependencies()
    sys.exit(0 if all_ok else 1) # Returns 0 for success, 1 for error