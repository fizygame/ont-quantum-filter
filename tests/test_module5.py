"""
Modül 5 Test Dosyası — DQGA Hiperparametre Optimizasyonu
========================================================
pytest ile çalıştırın:
    cd c:\\Users\\nuri_\\OneDrive\\Masaüstü\\DNA
    pytest tests/test_module5.py -v
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

# Proje kökünü Python yoluna ekle
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from dqga_optimizer import QuantumChromosome, QuantumGate, DQGA, bits_to_float

# ---------------------------------------------------------------------------
# Yardımcı Sınıflar ve Fonksiyonlar
# ---------------------------------------------------------------------------

def dummy_fitness(bits: np.ndarray) -> float:
    """Hedefin tamamı '1' olan bitlerden oluştuğu basit test fonksuyonu."""
    return float(np.sum(bits))

# ---------------------------------------------------------------------------
# Test Sınıfları
# ---------------------------------------------------------------------------

class TestQuantumChromosome:
    
    def test_initial_superposition(self) -> None:
        """Başlangıçta tüm genler |a|^2 = 0.5 ve |b|^2 = 0.5 olmalı (Süperpozisyon)."""
        chrom = QuantumChromosome(n_genes=10)
        
        np.testing.assert_allclose(chrom.alpha ** 2, 0.5)
        np.testing.assert_allclose(chrom.beta ** 2, 0.5)
        
    def test_normalization(self) -> None:
        """Manuel değişikliklerden sonra _normalize |a|^2 + |b|^2 = 1 kuralını sağlar."""
        chrom = QuantumChromosome(n_genes=2)
        chrom.alpha = np.array([10.0, 0.0])
        chrom.beta  = np.array([0.0, 5.0])
        
        chrom._normalize()
        
        sum_sq = chrom.alpha**2 + chrom.beta**2
        np.testing.assert_allclose(sum_sq, 1.0)
        
    def test_measurement_returns_binary(self) -> None:
        """Ölçüm işlemi 0 ve 1'lerden oluşan belirtilen uzunlukta dizi dönmelidir."""
        chrom = QuantumChromosome(n_genes=8)
        bits = chrom.measure()
        
        assert len(bits) == 8
        assert bits.dtype == np.int8
        assert np.all(np.isin(bits, [0, 1]))

class TestQuantumGate:

    def test_rotation_is_unitary(self) -> None:
        """Rotasyon kapısı normu korumalıdır (Unitary Transform)."""
        chrom = QuantumChromosome(n_genes=5)
        best = QuantumChromosome(n_genes=5)
        
        chrom.measure()
        best.measure()
        
        # Tüm bitleri kasıtlı farklı yapalım ki rotasyon gerçekleşsin
        chrom.binary_string = np.zeros(5, dtype=np.int8)
        best.binary_string = np.ones(5, dtype=np.int8)
        
        QuantumGate.rotate(chrom, best, theta=0.1)
        
        # Kural: |a|^2 + |b|^2 hala 1 mi?
        sum_sq = chrom.alpha**2 + chrom.beta**2
        np.testing.assert_allclose(sum_sq, 1.0)
        
    def test_raises_without_measurement(self) -> None:
        """Kromozom ölçülmeden döndürülmeye çalışılırsa hata vermeli."""
        chrom = QuantumChromosome(5)
        best = QuantumChromosome(5)
        
        with pytest.raises(ValueError):
            QuantumGate.rotate(chrom, best)

class TestDQGA:
    
    def test_fitness_improves(self) -> None:
        """
        Zamanla fitness değerinin artması optimizasyon kabiliyetini ispatlar.
        """
        np.random.seed(42) # Tekrarlanabilirlik için
        
        dqga = DQGA(pop_size=10, n_genes=16, fitness_fn=dummy_fitness, n_generations=20)
        best_bits, best_score = dqga.run()
        
        history = dqga.history_best_fitness
        
        # İlk gen ile son gen arasında düzelme var mı?
        assert history[-1] >= history[0], "Fitness zamanla bozuluyor, düzelmiyor!"
        # Çoğu durumda ideal skor n_genes olur.
        assert best_score > 0

class TestParameterMapping:
    
    def test_bits_to_float_at_boundaries(self) -> None:
        """Bit değerlerinin Min ve Max float aralıklarına dönüşümünü doğrula."""
        # Hepsi 0 --> Val Min (10.0)
        bits_min = np.array([0, 0, 0, 0])
        assert bits_to_float(bits_min, 10.0, 50.0) == 10.0
        
        # Hepsi 1 --> Val Max (50.0)
        bits_max = np.array([1, 1, 1, 1])
        assert bits_to_float(bits_max, 10.0, 50.0) == 50.0
        
        # Yarısı
        bits_mid = np.array([1, 0, 0, 0]) # 8 (for 4 bits, max is 15 -> 8/15)
        expected = 10.0 + (8 / 15.0) * (50.0 - 10.0)
        np.testing.assert_approx_equal(bits_to_float(bits_mid, 10.0, 50.0), expected)
