"""
Modül 4 Test Dosyası — PnP-ADMM Optimizasyonu
=============================================
pytest ile çalıştırın:
    cd c:\\Users\\nuri_\\OneDrive\\Masaüstü\\DNA
    pytest tests/test_module4.py -v
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

# Proje kökünü Python yoluna ekle
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from pnp_admm import PnPADMM

# ---------------------------------------------------------------------------
# Yardımcı Sınıflar ve Fixtures
# ---------------------------------------------------------------------------

def dummy_denoiser(x: np.ndarray) -> np.ndarray:
    """Testler için basit bir yarıya indirme denoiser'ı (matematiksel yalıtım için)."""
    return x * 0.5

@pytest.fixture
def sample_signal() -> np.ndarray:
    """1D sentetik test dizisi."""
    np.random.seed(42)
    return np.random.normal(5.0, 2.0, size=100)

# ---------------------------------------------------------------------------
# Test Sınıfları
# ---------------------------------------------------------------------------

class TestPnPADMM:

    def test_output_shape(self, sample_signal: np.ndarray) -> None:
        """Çıktı sinyal boyutu girişle aynı olmalıdır."""
        admm = PnPADMM(denoiser=dummy_denoiser, max_iter=2)
        out = admm.run(sample_signal)
        assert out.shape == sample_signal.shape
        assert out.ndim == 1

    def test_data_fidelity_step(self) -> None:
        """Proximal operator x formülünün doğruluğunu test eder."""
        admm = PnPADMM(denoiser=dummy_denoiser)
        y = np.array([10.0, 10.0])
        z = np.array([5.0, 5.0])
        u = np.array([1.0, 1.0])
        rho = 1.0
        
        # Beklenen: (y + rho*(z - u)) / (1 + rho)
        # = (10 + 1*(5 - 1)) / (2) = 14 / 2 = 7.0
        x_next = admm._data_fidelity_step(y, z, u, rho)
        np.testing.assert_allclose(x_next, np.array([7.0, 7.0]))

    def test_dual_update(self) -> None:
        """Lagrangian çarpanı olan u matrisinin güncellenme doğruluğu."""
        admm = PnPADMM(denoiser=dummy_denoiser)
        u = np.array([0.5, 0.5])
        x = np.array([3.0, 3.0])
        z = np.array([2.0, 2.0])
        
        # Beklenen: u + x - z = 0.5 + 3 - 2 = 1.5
        u_next = admm._dual_update(u, x, z)
        np.testing.assert_allclose(u_next, np.array([1.5, 1.5]))

    def test_rho_update_adaptive(self) -> None:
        """Rho dengeleyici ceza oranının değişme senaryoları."""
        admm = PnPADMM(denoiser=dummy_denoiser)
        
        mu = 10.0
        tau = 2.0
        base_rho = 1.0
        
        # Primal residual ezici şekilde büyükse -> ceza artmalı
        increased_rho = admm._rho_update(base_rho, primal_res=100.0, dual_res=1.0, mu=mu, tau=tau)
        assert increased_rho == 2.0
        
        # Dual residual ezici şekilde büyükse -> ceza azalmalı
        decreased_rho = admm._rho_update(base_rho, primal_res=1.0, dual_res=100.0, mu=mu, tau=tau)
        assert decreased_rho == 0.5
        
        # Denk (veya tolerans içinde) -> ceza aynı kalmalı
        stable_rho = admm._rho_update(base_rho, primal_res=5.0, dual_res=5.0, mu=mu, tau=tau)
        assert stable_rho == 1.0

    def test_admm_converges(self, sample_signal: np.ndarray) -> None:
        """Fonksiyon residual'ları düşürerek yakınsamalıdır."""
        # Basit bir denoiser ile PnP ADMM mutlaka belli bir değere yakınsar
        admm = PnPADMM(denoiser=lambda x: x * 0.9, max_iter=30, tol=1e-5, adaptive_rho=True)
        admm.run(sample_signal)
        
        history = admm.history_
        
        assert len(history["primal_res"]) > 0
        
        initial_primal = history["primal_res"][0]
        final_primal = history["primal_res"][-1]
        
        # İlerleyen iterasyonlarda hata marjının azaldığını test et
        assert final_primal < initial_primal, "ADMM sapıyor (diverging), residuallar düşmüyor."

    def test_assertion_on_bad_signal(self) -> None:
        admm = PnPADMM(denoiser=dummy_denoiser)
        with pytest.raises(AssertionError, match="1D"):
            admm.run(np.ones((10, 10)))
