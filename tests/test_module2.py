"""
Modül 2 Test Dosyası — SCSA Kuantum Gürültü Filtresi
====================================================
pytest ile çalıştırın:
    cd c:\\Users\\nuri_\\OneDrive\\Masaüstü\\DNA
    pytest tests/test_module2.py -v
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import scipy.sparse as sp
import pytest

# Proje kökünü Python yoluna ekle
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from scsa_filter import SCSAFilter

# ---------------------------------------------------------------------------
# Test Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def sample_signal() -> np.ndarray:
    """Modül 2 testleri için 1D boyutunda basit bir sentezlenmiş sinyal."""
    np.random.seed(42)
    # Z-score normalize edilmiş gibi (- ortalamalı) bir sinyal yapısı
    x = np.linspace(0, 10, 200)
    signal = np.sin(x) + np.random.normal(0, 0.2, size=len(x))
    signal -= np.mean(signal)  # Z-score simülasyonu için ortalamayı 0 yap
    return signal

# ---------------------------------------------------------------------------
# Test Sınıfları
# ---------------------------------------------------------------------------

class TestSCSAFilter:
    
    def test_hamiltonian_shape(self, sample_signal: np.ndarray) -> None:
        """Tridiagonal Hamiltonian matrisinin boyutu (N, N) olmalı."""
        n = len(sample_signal)
        scsa = SCSAFilter(h=1.0)
        H = scsa._build_hamiltonian(sample_signal)
        
        assert H.shape == (n, n), f"Beklenen: ({n}, {n}), Alınan: {H.shape}"
        assert sp.issparse(H), "H matrisi sparse (seyrek) yapıda olmalı"

    def test_hamiltonian_symmetry(self, sample_signal: np.ndarray) -> None:
        """H Hermitian (gerçel sayılar için simetrik) olmalı: H == H.T"""
        scsa = SCSAFilter(h=0.5)
        H = scsa._build_hamiltonian(sample_signal)
        
        H_dense = H.toarray()
        H_transposed = H_dense.T
        
        np.testing.assert_allclose(
            H_dense, H_transposed, 
            err_msg="Hamiltonian matrisi simetrik değil!"
        )

    def test_negative_eigenvalues_filtered(self, sample_signal: np.ndarray) -> None:
        """
        SCSA yalnızca negatif özdeğerler üzerinden yeniden yapılandırma yapmalı.
        Hepsini pozitif yaptığımızda negatif root bulmamalı.
        """
        scsa = SCSAFilter(h=0.5, n_components=10)
        
        # 1. Normal koşu: Negatif değerler içermeli
        scsa.fit_transform(sample_signal)
        assert scsa.eigenvalues_ is not None
        assert np.all(scsa.eigenvalues_ < 0), "Sadece negatif özdeğerler filtrelenmiş olmalı"
        
        # 2. Sadece devasa pozitif sinyal: Negatif kök beklenmez (h yeterince küçük değilse)
        positive_signal = sample_signal + 1000.0
        scsa.fit_transform(positive_signal)
        assert len(scsa.eigenvalues_) == 0, "Tamamen pozitif sınırda negatif özdeğer çıkmamalı"

    def test_output_shape(self, sample_signal: np.ndarray) -> None:
        """Çıktı sinyal boyutu girişle birebir aynı olmalı."""
        scsa = SCSAFilter(h=1.0)
        filtered_signal = scsa.fit_transform(sample_signal)
        
        assert filtered_signal.shape == sample_signal.shape
        assert filtered_signal.ndim == 1

    def test_default_heuristic_h(self, sample_signal: np.ndarray) -> None:
        """h değeri verilmediğinde np.sqrt(np.var(signal)) üzerinden atanmalı."""
        scsa = SCSAFilter(h=None)
        scsa.fit_transform(sample_signal)
        
        expected_h = np.sqrt(np.var(sample_signal))
        np.testing.assert_approx_equal(scsa.h, expected_h, significant=5)

    def test_assertions_prevent_wrong_input(self) -> None:
        """Yanlış boyut matris verildiğinde (2D) assertion hatası alınmalı."""
        scsa = SCSAFilter(h=1.0)
        wrong_input = np.ones((10, 10))
        
        with pytest.raises(AssertionError, match="1D"):
            scsa.fit_transform(wrong_input)
