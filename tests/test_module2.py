"""
Module 2 Test File — SCSA Quantum Noise Filter
====================================================
Run with pytest:
    cd c:\\Users\\nuri_\\OneDrive\\Masaüstü\\DNA
    pytest tests/test_module2.py -v
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import scipy.sparse as sp
import pytest

# Add project root to Python path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from scsa_filter import SCSAFilter

# ---------------------------------------------------------------------------
# Test Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def sample_signal() -> np.ndarray:
    """A simple 1D synthesized signal for Module 2 tests."""
    np.random.seed(42)
    # A signal structure resembling Z-score normalized data (mean around 0)
    x = np.linspace(0, 10, 200)
    signal = np.sin(x) + np.random.normal(0, 0.2, size=len(x))
    signal -= np.mean(signal)  # Set mean to 0 to simulate Z-score
    return signal

# ---------------------------------------------------------------------------
# Test Classes
# ---------------------------------------------------------------------------

class TestSCSAFilter:
    
    def test_hamiltonian_shape(self, sample_signal: np.ndarray) -> None:
        """The size of the Tridiagonal Hamiltonian matrix should be (N, N)."""
        n = len(sample_signal)
        scsa = SCSAFilter(h=1.0)
        H = scsa._build_hamiltonian(sample_signal)
        
        assert H.shape == (n, n), f"Expected: ({n}, {n}), Got: {H.shape}"
        assert sp.issparse(H), "The H matrix should be sparse"

    def test_hamiltonian_symmetry(self, sample_signal: np.ndarray) -> None:
        """H should be Hermitian (symmetric for real numbers): H == H.T"""
        scsa = SCSAFilter(h=0.5)
        H = scsa._build_hamiltonian(sample_signal)
        
        H_dense = H.toarray()
        H_transposed = H_dense.T
        
        np.testing.assert_allclose(
            H_dense, H_transposed, 
            err_msg="The Hamiltonian matrix is not symmetric!"
        )

    def test_negative_eigenvalues_filtered(self, sample_signal: np.ndarray) -> None:
        """
        SCSA should strictly reconstruct over the negative eigenvalues.
        If all are positive, it should not find negative roots.
        """
        scsa = SCSAFilter(h=0.5, n_components=10)
        
        # 1. Normal run: Should contain negative values
        scsa.fit_transform(sample_signal)
        assert scsa.eigenvalues_ is not None
        assert np.all(scsa.eigenvalues_ < 0), "Only negative eigenvalues should be filtered"
        
        # 2. Entirely massive positive signal: Should yield no negative roots
        positive_signal = sample_signal + 1000.0
        scsa.fit_transform(positive_signal)
        assert len(scsa.eigenvalues_) == 0, "No negative eigenvalues should emerge at purely positive limits"

    def test_output_shape(self, sample_signal: np.ndarray) -> None:
        """The output signal shape should be identical to the input."""
        scsa = SCSAFilter(h=1.0)
        filtered_signal = scsa.fit_transform(sample_signal)
        
        assert filtered_signal.shape == sample_signal.shape
        assert filtered_signal.ndim == 1

    def test_default_heuristic_h(self, sample_signal: np.ndarray) -> None:
        """If h is not provided, it should be assigned via np.sqrt(np.var(signal))."""
        scsa = SCSAFilter(h=None)
        scsa.fit_transform(sample_signal)
        
        expected_h = np.sqrt(np.var(sample_signal))
        np.testing.assert_approx_equal(scsa.h, expected_h, significant=5)

    def test_assertions_prevent_wrong_input(self) -> None:
        """Assertion error should be thrown when passed the wrong matrix dimensions (e.g., 2D)."""
        scsa = SCSAFilter(h=1.0)
        wrong_input = np.ones((10, 10))
        
        with pytest.raises(AssertionError, match="1D"):
            scsa.fit_transform(wrong_input)

