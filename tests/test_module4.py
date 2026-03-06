"""
Module 4 Test File — PnP-ADMM Optimization
=============================================
Run with pytest:
    cd c:\\Users\\nuri_\\OneDrive\\Masaüstü\\DNA
    pytest tests/test_module4.py -v
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

# Add project root to Python path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from pnp_admm import PnPADMM

# ---------------------------------------------------------------------------
# Helper Classes and Fixtures
# ---------------------------------------------------------------------------

def dummy_denoiser(x: np.ndarray) -> np.ndarray:
    """A simplistic halving denoiser for the tests (designed purely for mathematical isolation)."""
    return x * 0.5

@pytest.fixture
def sample_signal() -> np.ndarray:
    """1D synthetic test sequence."""
    np.random.seed(42)
    return np.random.normal(5.0, 2.0, size=100)

# ---------------------------------------------------------------------------
# Test Classes
# ---------------------------------------------------------------------------

class TestPnPADMM:

    def test_output_shape(self, sample_signal: np.ndarray) -> None:
        """The output signal shape must equal the input length securely."""
        admm = PnPADMM(denoiser=dummy_denoiser, max_iter=2)
        out = admm.run(sample_signal)
        assert out.shape == sample_signal.shape
        assert out.ndim == 1

    def test_data_fidelity_step(self) -> None:
        """Validates the exact logic formulation of the proximal operator x."""
        admm = PnPADMM(denoiser=dummy_denoiser)
        y = np.array([10.0, 10.0])
        z = np.array([5.0, 5.0])
        u = np.array([1.0, 1.0])
        rho = 1.0
        
        # Expected formula constraints: (y + rho*(z - u)) / (1 + rho)
        # = (10 + 1*(5 - 1)) / (2) = 14 / 2 = 7.0
        x_next = admm._data_fidelity_step(y, z, u, rho)
        np.testing.assert_allclose(x_next, np.array([7.0, 7.0]))

    def test_dual_update(self) -> None:
        """Ensures absolute validity of the Lagrangian matrix multiplier (u) update routing."""
        admm = PnPADMM(denoiser=dummy_denoiser)
        u = np.array([0.5, 0.5])
        x = np.array([3.0, 3.0])
        z = np.array([2.0, 2.0])
        
        # Expected: u + x - z = 0.5 + 3 - 2 = 1.5
        u_next = admm._dual_update(u, x, z)
        np.testing.assert_allclose(u_next, np.array([1.5, 1.5]))

    def test_rho_update_adaptive(self) -> None:
        """Verifies differing scenarios triggering automated balancing parameter (Rho) inflation/deflation."""
        admm = PnPADMM(denoiser=dummy_denoiser)
        
        mu = 10.0
        tau = 2.0
        base_rho = 1.0
        
        # If primal residual is heavily overpowering -> increase severity/penalty
        increased_rho = admm._rho_update(base_rho, primal_res=100.0, dual_res=1.0, mu=mu, tau=tau)
        assert increased_rho == 2.0
        
        # If dual residual overwhelms operations -> drop limits strictly
        decreased_rho = admm._rho_update(base_rho, primal_res=1.0, dual_res=100.0, mu=mu, tau=tau)
        assert decreased_rho == 0.5
        
        # If stabilized within tolerance barriers -> constraint locked completely
        stable_rho = admm._rho_update(base_rho, primal_res=5.0, dual_res=5.0, mu=mu, tau=tau)
        assert stable_rho == 1.0

    def test_admm_converges(self, sample_signal: np.ndarray) -> None:
        """System operations iteratively MUST exhibit residual dropping (Convergence Ruleset)."""
        # A static simple denoiser must forcefully compel ADMM into a fixed steady limit matrix layout eventually.
        admm = PnPADMM(denoiser=lambda x: x * 0.9, max_iter=30, tol=1e-5, adaptive_rho=True)
        admm.run(sample_signal)
        
        history = admm.history_
        
        assert len(history["primal_res"]) > 0
        
        initial_primal = history["primal_res"][0]
        final_primal = history["primal_res"][-1]
        
        # Verify mathematically that error margins diminish as it cascades downstream functionally
        assert final_primal < initial_primal, "ADMM is diverging, iteration residuals fail to compress downward structurally."

    def test_assertion_on_bad_signal(self) -> None:
        admm = PnPADMM(denoiser=dummy_denoiser)
        with pytest.raises(AssertionError, match="1D"):
            admm.run(np.ones((10, 10)))
