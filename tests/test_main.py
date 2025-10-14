import os
import sys
import pytest

# Add project root to path to allow module imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from main import format_header

# Test cases for format_header
# Parameters: (text, width, expected_output)
format_header_test_cases = [
    ("Main Menu", 57, "==                      Main Menu                      =="),
    ("Short", 20, "==     Short      =="),
    ("A Long Title That Is Quite Wide", 40, "==  A Long Title That Is Quite Wide   =="),
    ("Odd Width", 21, "==    Odd Width    =="),
    ("Exact Fit", 10, "==Exact Fit=="),
]

@pytest.mark.parametrize("text, width, expected", format_header_test_cases)
def test_format_header(text, width, expected):
    """
    Tests the format_header function with various inputs to ensure correct
    alignment and formatting.
    """
    assert format_header(text, width) == expected

def test_format_header_default_width():
    """
    Tests the format_header function with the default width (57).
    """
    assert format_header("Default Width Test") == "==                  Default Width Test                 =="