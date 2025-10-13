import os
import sys
import pytest
from unittest.mock import patch, mock_open

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from rdkit import Chem
from utils import (
    normalize_molecule,
    validate_smiles,
    print_script_banner,
    load_smiles_from_file
)

# Test cases for normalize_molecule
# Parameters: (input_smiles, expected_smiles)
normalize_molecule_test_cases = [
    ("CCO.Cl", "CCO"),  # Test salt removal
    ("CC(=O)[O-]", "CC(=O)O"),  # Test charge neutralization
    ("C[N+]", "C[N+]"),  # Test un-neutralizable charges
    ("invalid", None),  # Test invalid SMILES
    ("CC(O)=CC", "CC=C(C)O"),  # Test tautomer canonicalization
]

@pytest.mark.parametrize("input_smiles, expected_smiles", normalize_molecule_test_cases)
def test_normalize_molecule(input_smiles, expected_smiles):
    """
    Tests the normalize_molecule function for correct handling of salts,
    charges, and tautomers.
    """
    mol = Chem.MolFromSmiles(input_smiles)
    normalized_mol = normalize_molecule(mol)
    if expected_smiles is None:
        assert normalized_mol is None
    else:
        assert Chem.MolToSmiles(normalized_mol) == expected_smiles

# Test cases for validate_smiles
# Parameters: (input_list, normalize, expected_list)
validate_smiles_test_cases = [
    (["CCO", "CCC"], True, ["CCO", "CCC"]),
    (["CCO", "invalid"], True, ["CCO"]),
    ([], True, []),
]

@pytest.mark.parametrize("input_list, normalize, expected_list", validate_smiles_test_cases)
def test_validate_smiles(input_list, normalize, expected_list):
    """
    Tests the validate_smiles function for correct filtering and normalization.
    """
    assert validate_smiles(input_list, normalize=normalize, verbose=False) == expected_list

@patch('builtins.print')
def test_print_script_banner(mock_print):
    """
    Tests the print_script_banner function to ensure it prints a correctly
    formatted banner.
    """
    title = "Test Title"
    description = "Test Description"
    width = 70
    separator = "=" * width

    print_script_banner(title, description)

    expected_calls = [
        f"\n{separator}",
        title.center(width),
        description.center(width),
        f"{separator}\n"
    ]

    # Get actual calls from mock_print
    actual_calls = [call[0][0] for call in mock_print.call_args_list]

    assert actual_calls == expected_calls

@patch('os.path.exists', return_value=True)
def test_load_smiles_from_file_success(mock_exists):
    """
    Tests load_smiles_from_file with a mock file to ensure it reads data correctly.
    """
    m = mock_open(read_data="CCO\nCCC\n")
    with patch('builtins.open', m):
        smiles = load_smiles_from_file("dummy_path.smi", verbose=False)
        assert smiles == ["CCO", "CCC"]

@patch('os.path.exists', return_value=False)
def test_load_smiles_from_file_not_found(mock_exists):
    """
    Tests load_smiles_from_file for a non-existent file to ensure it
    returns an empty list.
    """
    smiles = load_smiles_from_file("non_existent_path.smi", verbose=False)
    assert smiles == []