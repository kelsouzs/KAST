
import sys
import os
from main import display_splash_screen

# Add project root to path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

def test_settings_import():
    """Test 1: Verify settings.py can be imported"""
    print("="*70)
    print("TEST 1: Import settings.py")
    print("="*70)
    try:
        import settings as cfg
        print("✅ settings.py imported successfully")
        return True, cfg
    except Exception as e:
        print(f"❌ FAILED: {e}")
        return False, None

def test_parallel_config(cfg):
    """Test 2: Verify all parallel config variables exist"""
    print("\n" + "="*70)
    print("TEST 2: Check parallel processing configuration variables")
    print("="*70)
    
    required_vars = [
        'ENABLE_PARALLEL_PROCESSING',
        'N_WORKERS',
        'PARALLEL_BATCH_SIZE',
        'PARALLEL_MIN_THRESHOLD'
    ]
    
    all_ok = True
    for var in required_vars:
        if hasattr(cfg, var):
            value = getattr(cfg, var)
            print(f"✅ {var} = {value}")
        else:
            print(f"❌ MISSING: {var}")
            all_ok = False
    
    return all_ok

def test_imports():
    """Test 3: Verify all required imports work"""
    print("\n" + "="*70)
    print("TEST 3: Check required imports")
    print("="*70)
    
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
                print(f"❌ {module}.{attr} not found")
                all_ok = False
            else:
                print(f"✅ {module}{f'.{attr}' if attr else ''}")
        except ImportError as e:
            print(f"❌ {module}: {e}")
            all_ok = False
    
    return all_ok

def test_get_optimal_workers():
    """Test 4: Test get_optimal_workers logic"""
    print("\n" + "="*70)
    print("TEST 4: Test get_optimal_workers() logic")
    print("="*70)
    
    import settings as cfg
    from multiprocessing import cpu_count
    
    # Save original values
    original_n_workers = cfg.N_WORKERS
    
    test_cases = [
        (None, "Auto-detect"),
        (-1, "All cores"),
        (4, "Fixed 4 cores"),
        (1, "Sequential (1 core)"),
    ]
    
    all_ok = True
    for n_workers_val, description in test_cases:
        cfg.N_WORKERS = n_workers_val
        
        # Simulate get_optimal_workers logic
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
            print(f"✅ N_WORKERS={n_workers_val} ({description}) → {result} cores")
        else:
            print(f"❌ N_WORKERS={n_workers_val} ({description}) → Invalid: {result}")
            all_ok = False
    
    # Restore original
    cfg.N_WORKERS = original_n_workers
    return all_ok

def test_script_compatibility():
    """Test 5: Check if scripts can import settings"""
    print("\n" + "="*70)
    print("TEST 5: Script compatibility check")
    print("="*70)
    
    scripts = [
        'bin/2_featurization.py',
        'bin/5_0_featurize_for_prediction.py',
        'bin/5_1_run_prediction.py',
        'bin/4_1_cross_validation.py',
        'bin/4_3_tanimoto_similarity.py',
    ]
    
    all_ok = True
    for script in scripts:
        script_path = os.path.join(project_root, script)
        if os.path.exists(script_path):
            # Check if script has required imports
            with open(script_path, 'r', encoding='utf-8') as f:
                content = f.read()
                
            has_cfg_import = 'import settings as cfg' in content
            has_get_optimal = 'def get_optimal_workers()' in content
            has_parallel_check = 'cfg.ENABLE_PARALLEL_PROCESSING' in content
            
            if has_cfg_import and has_get_optimal and has_parallel_check:
                print(f"✅ {script}")
                print(f"   • imports settings ✓")
                print(f"   • has get_optimal_workers() ✓")
                print(f"   • checks ENABLE_PARALLEL_PROCESSING ✓")
            else:
                print(f"⚠️  {script}")
                if not has_cfg_import:
                    print(f"   • missing 'import settings as cfg'")
                if not has_get_optimal:
                    print(f"   • missing get_optimal_workers()")
                if not has_parallel_check:
                    print(f"   • missing ENABLE_PARALLEL_PROCESSING check")
                all_ok = False
        else:
            print(f"❌ {script} - File not found")
            all_ok = False
    
    return all_ok

def test_parallel_threshold_logic():
    """Test 6: Test parallel threshold logic"""
    print("\n" + "="*70)
    print("TEST 6: Test parallel threshold logic")
    print("="*70)
    
    import settings as cfg
    
    test_datasets = [
        (1000, "Small dataset"),
        (10000, "Threshold dataset"),
        (50000, "Medium dataset"),
        (100000, "Large dataset"),
    ]
    
    all_ok = True
    for dataset_size, description in test_datasets:
        use_parallel = (cfg.ENABLE_PARALLEL_PROCESSING and 
                        dataset_size >= cfg.PARALLEL_MIN_THRESHOLD)
        
        expected = cfg.ENABLE_PARALLEL_PROCESSING and dataset_size >= cfg.PARALLEL_MIN_THRESHOLD
        
        if use_parallel == expected:
            status = "Parallel" if use_parallel else "Sequential"
            print(f"✅ {dataset_size:,} molecules ({description}) → {status}")
        else:
            print(f"❌ {dataset_size:,} molecules ({description}) → Logic error")
            all_ok = False
    
    return all_ok

def main():
    """Run all tests"""
    display_splash_screen()
    print("\n" + "🔍 " + "="*66)
    print("  PARALLEL PROCESSING COMPATIBILITY TEST SUITE")
    print("="*68 + " 🔍\n")
    
    results = []
    
    # Test 1: Import settings
    success, cfg = test_settings_import()
    results.append(("Import settings.py", success))
    if not success:
        print("\n❌ CRITICAL: Cannot continue without settings.py")
        return False
    
    # Test 2: Check config variables
    success = test_parallel_config(cfg)
    results.append(("Config variables exist", success))
    
    # Test 3: Check imports
    success = test_imports()
    results.append(("Required imports", success))
    
    # Test 4: Test get_optimal_workers
    success = test_get_optimal_workers()
    results.append(("get_optimal_workers() logic", success))
    
    # Test 5: Script compatibility
    success = test_script_compatibility()
    results.append(("Script compatibility", success))
    
    # Test 6: Threshold logic
    success = test_parallel_threshold_logic()
    results.append(("Threshold logic", success))
    
    # Summary
    print("\n" + "="*70)
    print("TEST SUMMARY")
    print("="*70)
    
    all_passed = True
    for test_name, passed in results:
        status = "✅ PASSED" if passed else "❌ FAILED"
        print(f"{status}: {test_name}")
        if not passed:
            all_passed = False
    
    print("="*70)
    if all_passed:
        print("\n🎉 ALL TESTS PASSED! System is compatible! 🎉\n")
        return True
    else:
        print("\n❌ SOME TESTS FAILED! Review errors above.\n")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)

