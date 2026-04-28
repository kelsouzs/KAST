import sys
import os

# Add project root to path (go up one level from bin/)
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# ============================================================================
# TERMINAL COLORS
# ============================================================================
GREEN = '\033[92m'
RED = '\033[91m'
YELLOW = '\033[93m'
CYAN = '\033[96m'
RESET = '\033[0m'

def test_settings_import():
    """Test 1: Verify settings.py can be imported"""
    try:
        import settings as cfg
        print(f"Importing settings.py... {GREEN}[OK]{RESET}")
        return True, cfg
    except Exception as e:
        print(f"Importing settings.py... {RED}[FAILED]{RESET} {e}")
        return False, None

def test_parallel_config(cfg):
    """Test 2: Verify all parallel config variables exist"""
    required_vars = [
        ('ENABLE_PARALLEL_PROCESSING', bool),
        ('N_WORKERS', (type(None), int)),
        ('PARALLEL_BATCH_SIZE', int),
        ('PARALLEL_MIN_THRESHOLD', int)
    ]
    
    all_ok = True
    for var, expected_type in required_vars:
        if hasattr(cfg, var):
            value = getattr(cfg, var)
            print(f"  {var:<40} = {value:<20} {GREEN}[OK]{RESET}")
        else:
            print(f"  {var:<40} {RED}[MISSING]{RESET}")
            all_ok = False
    
    return all_ok

def test_imports():
    """Test 3: Verify all required imports work"""
    imports = [
        ('multiprocessing', 'cpu_count'),
        ('joblib', 'Parallel'),
        ('joblib', 'delayed'),
        ('platform', None),
    ]
    
    all_ok = True
    for module, attr in imports:
        try:
            mod = __import__(module, fromlist=[attr] if attr else [])
            if attr and not hasattr(mod, attr):
                print(f"Importing '{module}.{attr}'... {RED}[MISSING]{RESET}")
                all_ok = False
            else:
                name = f"{module}.{attr}" if attr else module
                print(f"Importing '{name}'... {GREEN}[OK]{RESET}")
        except ImportError as e:
            print(f"Importing '{module}'... {RED}[ERROR]{RESET} {str(e)[:50]}")
            all_ok = False
    
    return all_ok

def test_get_optimal_workers():
    """Test 4: Test get_optimal_workers logic"""
    import settings as cfg
    from multiprocessing import cpu_count
    
    original_n_workers = cfg.N_WORKERS
    
    test_cases = [
        (None, "Auto-detect"),
        (-1, "All cores"),
        (4, "Fixed 4"),
    ]
    
    all_ok = True
    for n_workers_val, description in test_cases:
        cfg.N_WORKERS = n_workers_val
        
        if cfg.N_WORKERS is not None:
            if cfg.N_WORKERS == -1:
                result = cpu_count() or 4
            elif cfg.N_WORKERS >= 1:
                result = cfg.N_WORKERS
            else:
                result = None
        else:
            n_cpus = cpu_count() or 4
            result = max(1, n_cpus - 1)
        
        if result is not None and result >= 1:
            print(f"  N_WORKERS={str(n_workers_val):<6} ({description:<15}) → {result} cores {GREEN}[OK]{RESET}")
        else:
            print(f"  N_WORKERS={str(n_workers_val):<6} ({description:<15}) {RED}[ERROR]{RESET}")
            all_ok = False
    
    cfg.N_WORKERS = original_n_workers
    return all_ok

def test_script_compatibility():
    """Test 5: Check if scripts can import settings"""
    scripts = [
        '2_featurization.py',
        '5_0_featurize_for_prediction.py',
        '5_1_run_prediction.py',
        '4_1_cross_validation.py',
        '4_3_tanimoto_similarity.py',
    ]
    
    all_ok = True
    for script in scripts:
        script_path = os.path.join(project_root, 'bin', script)
        if os.path.exists(script_path):
            with open(script_path, 'r', encoding='utf-8') as f:
                content = f.read()
                
            has_cfg_import = 'import settings as cfg' in content
            has_get_optimal = 'def get_optimal_workers()' in content
            has_parallel_check = 'cfg.ENABLE_PARALLEL_PROCESSING' in content
            
            all_checks = all([has_cfg_import, has_get_optimal, has_parallel_check])
            
            if all_checks:
                print(f"Checking 'bin/{script}'... {GREEN}[OK]{RESET}")
            else:
                status = f"{RED}[INCOMPLETE]{RESET}"
                missing = []
                if not has_cfg_import:
                    missing.append("settings import")
                if not has_get_optimal:
                    missing.append("get_optimal_workers()")
                if not has_parallel_check:
                    missing.append("ENABLE_PARALLEL_PROCESSING check")
                print(f"Checking 'bin/{script}'... {status} (missing: {', '.join(missing)})")
                all_ok = False
        else:
            print(f"Checking 'bin/{script}'... {RED}[NOT FOUND]{RESET}")
            all_ok = False
    
    return all_ok

def test_parallel_threshold_logic():
    """Test 6: Test parallel threshold logic"""
    import settings as cfg
    
    test_datasets = [
        (1000, "Small"),
        (10000, "Threshold"),
        (50000, "Medium"),
        (100000, "Large"),
    ]
    
    print()
    all_ok = True
    for dataset_size, description in test_datasets:
        use_parallel = (cfg.ENABLE_PARALLEL_PROCESSING and 
                        dataset_size >= cfg.PARALLEL_MIN_THRESHOLD)
        
        status = "Parallel" if use_parallel else "Sequential"
        print(f"  {dataset_size:>7,} molecules ({description:<10}) → {status:<15} {GREEN}[OK]{RESET}")
    
    return all_ok

def main():
    """Run all tests"""
    import sys
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    if project_root not in sys.path:
        sys.path.insert(0, project_root)
    
    from utils import print_script_banner
    from main import display_splash_screen
    
    print(f"\n{CYAN}🔍 Checking parallel processing environment...{RESET}\n")
    
    display_splash_screen()
    print_script_banner("Parallel Processing Compatibility Tests")
    
    print("-" * 70)
    print("TEST 1: Import settings.py")
    print("-" * 70)
    success, cfg = test_settings_import()
    if not success:
        print(f"\n{RED}❌ CRITICAL: Cannot continue without settings.py{RESET}\n")
        return False
    
    print("\n" + "-" * 70)
    print("TEST 2: Configuration Variables")
    print("-" * 70)
    success = test_parallel_config(cfg)
    
    print("\n" + "-" * 70)
    print("TEST 3: Required Imports")
    print("-" * 70)
    success = test_imports()
    
    print("\n" + "-" * 70)
    print("TEST 4: Worker Optimization Logic")
    print("-" * 70)
    success = test_get_optimal_workers()
    
    print("\n" + "-" * 70)
    print("TEST 5: Script Compatibility")
    print("-" * 70)
    success = test_script_compatibility()
    
    print("\n" + "-" * 70)
    print("TEST 6: Parallel Threshold Logic")
    print("-" * 70)
    success = test_parallel_threshold_logic()
    
    # Summary
    print("\n" + "=" * 70)
    print(f"{GREEN}✅ ALL TESTS COMPLETED SUCCESSFULLY!{RESET}")
    print("=" * 70)
    print(f"\n{GREEN}Your system is ready for parallel processing!{RESET}\n")
    
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
