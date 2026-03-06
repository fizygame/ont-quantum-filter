"""
Modül 6 Test Dosyası — Kıyaslama ve Çıktı Üretimi
=================================================
pytest ile çalıştırın:
    cd c:\\Users\\nuri_\\OneDrive\\Masaüstü\\DNA
    pytest tests/test_module6.py -v
"""

from __future__ import annotations

import sys
import os
from pathlib import Path

import numpy as np
import pytest

# Proje kökünü Python yoluna ekle
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from benchmarking import (
    compute_snr,
    savitzky_golay_baseline,
    plot_comparison,
    export_npy,
    export_fallback
)

# ---------------------------------------------------------------------------
# Test Sınıfları
# ---------------------------------------------------------------------------

class TestBenchmarkingMetrics:
    
    def test_snr_positive(self) -> None:
        """
        Daha az gürültülü (temiz) bir sinyalin SNR değeri 
        çok gürültülü bir sinyale kıyasla pozitif yönde yüksek olmalıdır.
        """
        np.random.seed(42)
        true_sig = np.ones(50)
        
        # Çok gürültülü (Noisy base)
        noisy = true_sig + np.random.normal(0, 2.0, 50)
        
        # Olay1: Biraz temizlenmiş sinyal
        semi_clean = true_sig + np.random.normal(0, 1.0, 50)
        
        # Olay2: Tam temizlenmiş sinyal
        very_clean = true_sig + np.random.normal(0, 0.1, 50)
        
        snr_semi = compute_snr(semi_clean, noisy)
        snr_very = compute_snr(very_clean, noisy)
        
        assert snr_very > snr_semi, "Daha temiz sinyal daha yüksek SNR vermelidir."
        
    def test_snr_shape_mismatch(self) -> None:
        """Farklı boyuttaki diziler SNR hesaplarken AssertionError vermelidir."""
        a = np.ones(10)
        b = np.ones(11)
        with pytest.raises(AssertionError):
            compute_snr(a, b)

class TestBaselines:
    
    def test_savitzky_golay_shape(self) -> None:
        """Savitzky-Golay baseline filtresi shape değiştirmemeli."""
        sig = np.random.randn(20)
        filtered = savitzky_golay_baseline(sig, window_length=5, polyorder=2)
        assert filtered.shape == sig.shape

class TestExportAndPlot:

    @pytest.fixture
    def temp_out_dir(self, tmpdir) -> Path:
        """Geçici test klasörü."""
        return Path(tmpdir)
        
    def test_npy_export(self, temp_out_dir: Path) -> None:
        """NP array .npy olarak dışa aktarılabilmeli ve yüklenebilmelidir."""
        sig = np.array([1., 2., 3., 4.])
        out_path = temp_out_dir / "test.npy"
        
        export_npy(sig, out_path)
        
        assert out_path.exists()
        loaded = np.load(out_path)
        np.testing.assert_allclose(sig, loaded)
        
    def test_fallback_export(self, temp_out_dir: Path) -> None:
        """Text fallback (.txt) dışa aktarılabilmeli ve içerik barındırmalıdır."""
        sig = np.array([1.123, 2.456, 3.789])
        out_path = temp_out_dir / "test.txt"
        
        export_fallback(sig, out_path)
        
        assert out_path.exists()
        # Header'ı geçip datayı yükle
        loaded = np.loadtxt(out_path, delimiter=',', skiprows=1)
        np.testing.assert_allclose(sig, loaded, rtol=1e-3)
        
    def test_plot_runs_without_error(self, temp_out_dir: Path) -> None:
        """plot_comparison fonksiyonu matplotlib çökmelerine yol açmadan png üretmelidir."""
        raw = np.zeros(10)
        classic = np.ones(10)
        quantum = np.ones(10) * 2
        out_path = temp_out_dir / "fig.png"
        
        plot_comparison(raw, classic, quantum, save_path=out_path, show=False)
        
        assert out_path.exists()
        assert out_path.stat().st_size > 1024  # En az 1KB veri olmalı
