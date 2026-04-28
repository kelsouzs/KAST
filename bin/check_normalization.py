"""
K-talysticFlow - Molecular Normalization Checker

This script analyzes the impact of molecular normalization on your SMILES files.
Shows what changes after applying salt removal, charge neutralization, etc.

USAGE:
    python bin/check_normalization.py [file.smi]
    
    If no file provided, checks all .smi files in /data folder

OUTPUT:
    - Statistics on normalized vs unchanged molecules
    - Examples of salts removed
    - Examples of charges neutralized
    - Before/after comparison
"""

import sys
import os
from typing import List, Tuple, Dict

# Add project root to path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from rdkit import Chem
from rdkit import RDLogger
RDLogger.DisableLog('rdApp.*')

from utils import load_smiles_from_file, normalize_molecule
import settings as cfg


def analyze_normalization(filepath: str) -> Dict:
    """
    Analyzes normalization impact on a single SMILES file.
    """
    print(f"\n{'='*70}")
    print(f"📂 File: {os.path.basename(filepath)}")
    print(f"{'='*70}")
    
    # Load raw SMILES
    smiles_raw = load_smiles_from_file(filepath)
    
    if not smiles_raw:
        print("❌ ERROR: Could not load file or file is empty")
        return {}
    
    print(f"\n📊 Total SMILES in file: {len(smiles_raw):,}")
    
    # Analyze normalization effects
    unchanged = []
    salt_removed = []
    charge_neutralized = []
    both_changed = []
    invalid = []
    
    print("\n🔄 Analyzing normalization impact...")
    
    for smi_raw in smiles_raw:
        mol_raw = Chem.MolFromSmiles(smi_raw)
        
        if mol_raw is None:
            invalid.append(smi_raw)
            continue
        
        # Apply normalization
        mol_normalized = normalize_molecule(mol_raw)
        
        if mol_normalized is None:
            invalid.append(smi_raw)
            continue
        
        # Convert back to SMILES
        smi_normalized = Chem.MolToSmiles(mol_normalized)
        smi_canonical = Chem.MolToSmiles(mol_raw)  # Without normalization
        
        # Detect what changed
        has_salt = '.' in smi_raw
        had_charge = any(c in smi_raw for c in ['+', '-'])
        
        if smi_raw == smi_normalized:
            unchanged.append(smi_raw)
        elif has_salt and had_charge:
            both_changed.append((smi_raw, smi_normalized))
        elif has_salt:
            salt_removed.append((smi_raw, smi_normalized))
        elif smi_canonical != smi_normalized:
            charge_neutralized.append((smi_raw, smi_normalized))
        else:
            unchanged.append(smi_raw)
    
    # Print Results
    total_valid = len(unchanged) + len(salt_removed) + len(charge_neutralized) + len(both_changed)
    
    print(f"\n{'─'*70}")
    print("📈 RESULTS:")
    print(f"{'─'*70}")
    
    print(f"\n✅ Valid molecules:        {total_valid:6,} ({total_valid/len(smiles_raw)*100:5.1f}%)")
    print(f"❌ Invalid molecules:      {len(invalid):6,} ({len(invalid)/len(smiles_raw)*100:5.1f}%)")
    
    print(f"\n🔄 Normalization Impact:")
    print(f"   Unchanged:              {len(unchanged):6,} ({len(unchanged)/total_valid*100:5.1f}%)")
    print(f"   Salt removed:           {len(salt_removed):6,} ({len(salt_removed)/total_valid*100:5.1f}%)")
    print(f"   Charge neutralized:     {len(charge_neutralized):6,} ({len(charge_neutralized)/total_valid*100:5.1f}%)")
    print(f"   Both (salt + charge):   {len(both_changed):6,} ({len(both_changed)/total_valid*100:5.1f}%)")
    
    total_changed = len(salt_removed) + len(charge_neutralized) + len(both_changed)
    print(f"\n📊 Total normalized:       {total_changed:6,} ({total_changed/total_valid*100:5.1f}%)")
    
    # Show examples
    if salt_removed:
        print(f"\n{'─'*70}")
        print("🧂 EXAMPLES: Salt/Fragment Removal (first 5)")
        print(f"{'─'*70}")
        for i, (raw, norm) in enumerate(salt_removed[:5], 1):
            print(f"\n{i}. Before: {raw}")
            print(f"   After:  {norm}")
    
    if charge_neutralized:
        print(f"\n{'─'*70}")
        print("⚡ EXAMPLES: Charge Neutralization (first 5)")
        print(f"{'─'*70}")
        for i, (raw, norm) in enumerate(charge_neutralized[:5], 1):
            print(f"\n{i}. Before: {raw}")
            print(f"   After:  {norm}")
    
    if both_changed:
        print(f"\n{'─'*70}")
        print("🔧 EXAMPLES: Salt + Charge Removal (first 5)")
        print(f"{'─'*70}")
        for i, (raw, norm) in enumerate(both_changed[:5], 1):
            print(f"\n{i}. Before: {raw}")
            print(f"   After:  {norm}")
    
    # Impact Assessment
    print(f"\n{'='*70}")
    print("🎯 IMPACT ASSESSMENT:")
    print(f"{'='*70}")
    
    impact_pct = (total_changed / total_valid * 100) if total_valid > 0 else 0
    
    if impact_pct == 0:
        print("\n✅ EXCELLENT: No normalization needed!")
        print("   → All molecules are already clean and standardized")
    
    elif impact_pct < 1:
        print("\n✅ VERY LOW IMPACT: < 1% normalized")
        print("   → Dataset is very clean")
    
    elif impact_pct < 5:
        print("\n⚠️  LOW IMPACT: 1-5% normalized")
        print("   → Minor cleanup, good data quality")
    
    elif impact_pct < 10:
        print("\n⚠️  MODERATE IMPACT: 5-10% normalized")
        print("   → Some cleaning needed, typical for vendor libraries")
    
    elif impact_pct < 25:
        print("\n🚨 HIGH IMPACT: 10-25% normalized")
        print("   → Significant cleanup, data from mixed sources")
    
    else:
        print("\n🚨 VERY HIGH IMPACT: > 25% normalized")
        print("   → Heavy cleanup needed, data quality concerns")
        print("   → May include many salts/counterions")
    
    print(f"\n💡 Recommendation:")
    if impact_pct > 5:
        print("   ✅ Keep normalization ENABLED (default)")
        print("   → Significant improvements to data consistency")
    elif impact_pct > 0:
        print("   ✅ Keep normalization ENABLED (default)")
        print("   → Minor improvements, no downside")
    else:
        print("   → Normalization has no effect (dataset already perfect)")
    
    print(f"\n{'='*70}\n")
    
    return {
        'total': len(smiles_raw),
        'valid': total_valid,
        'invalid': len(invalid),
        'unchanged': len(unchanged),
        'salt_removed': len(salt_removed),
        'charge_neutralized': len(charge_neutralized),
        'both_changed': len(both_changed),
        'total_changed': total_changed
    }


def check_all_data_files():
    """Check all .smi files in /data folder."""
    data_dir = os.path.join(project_root, 'data')
    
    if not os.path.exists(data_dir):
        print(f"❌ ERROR: Data directory not found: {data_dir}")
        return
    
    smi_files = [f for f in os.listdir(data_dir) if f.endswith('.smi')]
    
    if not smi_files:
        print(f"❌ No .smi files found in {data_dir}")
        return
    
    print(f"\n{'='*70}")
    print(f"🔍 CHECKING NORMALIZATION FOR ALL SMILES FILES")
    print(f"{'='*70}")
    print(f"\nFound {len(smi_files)} .smi file(s):")
    for f in smi_files:
        print(f"  • {f}")
    
    all_stats = {}
    
    for smi_file in smi_files:
        filepath = os.path.join(data_dir, smi_file)
        stats = analyze_normalization(filepath)
        if stats:
            all_stats[smi_file] = stats
    
    # Summary table
    if all_stats:
        print(f"\n{'='*70}")
        print("📊 SUMMARY TABLE:")
        print(f"{'='*70}\n")
        
        header = f"{'File':<25} {'Total':>10} {'Changed':>10} {'Impact':>10}"
        print(header)
        print("-" * len(header))
        
        for filename, stats in all_stats.items():
            impact_pct = (stats['total_changed'] / stats['valid'] * 100) if stats['valid'] > 0 else 0
            impact_str = f"{impact_pct:.1f}%"
            
            print(f"{filename:<25} {stats['total']:>10,} {stats['total_changed']:>10,} {impact_str:>10}")
        
        print(f"\n{'='*70}\n")


def main():
    """Main function."""
    from main import display_splash_screen
    display_splash_screen()
    print("\n" + "="*70)
    print(" 🧪 K-TALYSTICFLOW - MOLECULAR NORMALIZATION CHECKER")
    print("="*70)
    
    if len(sys.argv) > 1:
        # Check specific file
        filepath = sys.argv[1]
        
        if not os.path.exists(filepath):
            print(f"\n❌ ERROR: File not found: {filepath}")
            print("\nUsage:")
            print(f"  python {sys.argv[0]} [file.smi]")
            print(f"  python {sys.argv[0]}              (checks all files in /data)")
            sys.exit(1)
        
        analyze_normalization(filepath)
    
    else:
        # Check all files in /data
        check_all_data_files()
    
    print("\n💡 NOTES:")
    print("-" * 70)
    print("• Normalization is now ENABLED by default in all scripts")
    print("• Changes include: salt removal, charge neutralization")
    print("• Tautomer canonicalization is DISABLED (can be enabled in utils.py)")
    print("• To disable normalization: pass normalize=False to validate_smiles()")
    print("=" * 70 + "\n")


if __name__ == '__main__':
    main()
