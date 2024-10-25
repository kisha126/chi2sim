# tests/test_chi2sim.py
import numpy as np
import pytest
from chi2sim import chi2_cont_sim

def test_basic_2x2():
    """Test with a basic 2x2 contingency table"""
    table = np.array([
        [10, 20],
        [30, 40]
    ], dtype=int)
    result = chi2_cont_sim(table)
    assert 0 <= result['p_value'] <= 1
    assert result['simulations'] == 10000

def test_3x3_table():
    """Test with a 3x3 contingency table"""
    table = np.array([
        [10, 20, 15],
        [5, 15, 10],
        [25, 30, 20]
    ], dtype=int)
    result = chi2_cont_sim(table, simulations=5000)
    assert 0 <= result['p_value'] <= 1
    assert result['simulations'] == 5000

def test_input_validation():
    """Test input validation"""
    # Test non-2D array
    with pytest.raises(ValueError):
        chi2_cont_sim(np.array([1, 2, 3]))
    
    # Test non-integer array
    with pytest.raises(TypeError):
        chi2_cont_sim(np.array([[1.5, 2.5], [3.5, 4.5]]))
    
    # Test negative values
    with pytest.raises(ValueError):
        chi2_cont_sim(np.array([[-1, 2], [3, 4]]))

def test_simulation_count():
    """Test different simulation counts"""
    table = np.array([[5, 10], [15, 20]], dtype=int)
    
    # Test default
    result1 = chi2_cont_sim(table)
    assert result1['simulations'] == 10000
    
    # Test custom simulation count
    result2 = chi2_cont_sim(table, simulations=100)
    assert result2['simulations'] == 100

def test_reproducibility():
    """Test reproducibility with same input"""
    table = np.array([[5, 10], [15, 20]], dtype=int)
    np.random.seed(42)  # Set seed for reproducibility
    result1 = chi2_cont_sim(table, simulations=1000)
    np.random.seed(42)  # Reset seed
    result2 = chi2_cont_sim(table, simulations=1000)
    assert abs(result1['p_value'] - result2['p_value']) < 1e-10
