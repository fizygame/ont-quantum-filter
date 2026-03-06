"""
Module 3 Test File — Richardson-Lucy Deconvolution
====================================================
Run with pytest:
    cd c:\\Users\\nuri_\\OneDrive\\Masaüstü\\DNA
    pytest tests/test_module3.py -v
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from scipy.signal import fftconvolve
import pytest

# Add project root to Python path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from rl_deconvolution import (
    gaussian_psf_1d,
    shift_to_positive,
    inverse_shift,
    richardson_lucy_1d
)

# ---------------------------------------------------------------------------
# Test Classes
# ---------------------------------------------------------------------------

class TestGaussianPSF:
    
    def test_psf_sums_to_one(self) -> None:
        """The total sum of the PSF should be 1 for Mass/Energy conservation."""
        psf = gaussian_psf_1d(size=11, sigma=2.0)
        assert np.isclose(psf.sum(), 1.0), "PSF is not normalized!"
        
    def test_psf_is_symmetric(self) -> None:
        """The Gaussian PSF should be symmetric."""
        psf = gaussian_psf_1d(size=15, sigma=3.0)
        np.testing.assert_array_almost_equal(psf, psf[::-1], err_msg="PSF is not symmetric")
        
    def test_psf_shape(self) -> None:
        """PSF must be generated in the requested shape."""
        psf = gaussian_psf_1d(size=5, sigma=1.0)
        assert len(psf) == 5

    def test_raises_on_invalid_params(self) -> None:
        """Should prevent invalid sizes and sigma bounds."""
        with pytest.raises(AssertionError):
            gaussian_psf_1d(size=0, sigma=1.0)
        with pytest.raises(AssertionError):
            gaussian_psf_1d(size=5, sigma=-1.0)

class TestShiftFunctions:
    
    def test_min_shift_makes_strictly_positive(self) -> None:
        """The signal should be strictly shifted into the positive space, even if negative."""
        signal = np.array([-10.0, -5.0, 0.0, 5.0, 10.0])
        pos_signal, params = shift_to_positive(signal, offset=1e-6, method="min_shift")
        
        assert np.all(pos_signal > 0)
        np.testing.assert_approx_equal(pos_signal.min(), 1e-6)
        
    def test_minmax_makes_strictly_positive(self) -> None:
        """The MinMax scaling should force the signal to the [offset, 1+offset] boundaries."""
        signal = np.array([-50.0, 50.0])
        pos_signal, params = shift_to_positive(signal, offset=1e-3, method="minmax")
        
        assert np.all(pos_signal > 0)
        np.testing.assert_approx_equal(pos_signal.min(), 1e-3)
        np.testing.assert_approx_equal(pos_signal.max(), 1.0 + 1e-3)

    def test_inverse_shift_restores_original(self) -> None:
        """The transformation mechanism must be reversibly lossless."""
        np.random.seed(42)
        original = np.random.normal(-20, 10, 100)
        
        # Test Min-Shift
        pos1, params1 = shift_to_positive(original, method="min_shift")
        restored1 = inverse_shift(pos1, params1)
        np.testing.assert_allclose(original, restored1)
        
        # Test Min-Max
        pos2, params2 = shift_to_positive(original, method="minmax")
        restored2 = inverse_shift(pos2, params2)
        np.testing.assert_allclose(original, restored2)
        
    def test_invalid_method_raises_error(self) -> None:
        """An incorrect method name should raise an exception."""
        with pytest.raises(ValueError):
            shift_to_positive(np.array([1, 2]), method="magic")

class TestRichardsonLucy:
    
    @pytest.fixture
    def setup_data(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        np.random.seed(42)
        x = np.zeros(100)
        x[40:60] = 10.0  # Sharp Square Step (True Signal)
        
        psf = gaussian_psf_1d(9, 2.0)
        # Blur the signal
        blurred = fftconvolve(x, psf, mode='same')
        
        # Make strictly positive for RL restoration
        blurred += 0.001 
        
        return x, psf, blurred
        
    def test_output_shape_preserved(self, setup_data) -> None:
        """Deconvolution output should equal the input shape."""
        true_sig, psf, blurred = setup_data
        restored = richardson_lucy_1d(blurred, psf, iterations=5)
        
        assert restored.shape == blurred.shape

    def test_output_non_negative(self, setup_data) -> None:
        """The RL algorithm should never inherently produce negative components."""
        true_sig, psf, blurred = setup_data
        restored = richardson_lucy_1d(blurred, psf, iterations=10)
        
        assert np.all(restored >= 0)

    def test_convergence_reduces_blur(self, setup_data) -> None:
        """As deconvolution iterates, the signal should better resemble the original sharp wave (Error Drops)."""
        true_sig, psf, blurred = setup_data
        
        # Initial error (Blurred vs True)
        initial_mse = np.mean((blurred - true_sig)**2)
        
        # Error after 10 iterations
        restored_10 = richardson_lucy_1d(blurred, psf, iterations=10)
        restored_10_mse = np.mean((restored_10 - true_sig)**2)
        
        # Error after 50 iterations
        restored_50 = richardson_lucy_1d(blurred, psf, iterations=50)
        restored_50_mse = np.mean((restored_50 - true_sig)**2)
        
        # Prove steady architectural improvement
        assert restored_10_mse < initial_mse, "10 iterations failed to drop the error"
        assert restored_50_mse < restored_10_mse, "50 iterations failed to lower the error further"

    def test_raises_on_negative_input(self) -> None:
        """RL cannot accept negative limits; it must throw an exception."""
        psf = gaussian_psf_1d(3, 1.0)
        neg_signal = np.array([-1.0, 5.0, 2.0])
        
        with pytest.raises(ValueError, match="negative"):
            richardson_lucy_1d(neg_signal, psf)
