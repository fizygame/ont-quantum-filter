"""
Modül 1 Test Dosyası — Veri Çekme ve Ön İşleme
===============================================
pytest ile çalıştırın:
    cd c:\\Users\\nuri_\\OneDrive\\Masaüstü\\DNA
    pytest tests/test_module1.py -v

Tüm testler gerçek ağ bağlantısı gerektirmez;
sentetik sinyal üreteci ile çevrimdışı çalışır.
"""

from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path
from unittest.mock import patch, MagicMock

import numpy as np
import pytest

# Proje kökünü Python yoluna ekle
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from data_ingestion import (
    zscore_normalize,
    generate_synthetic_nanopore_signal,
    download_ont_data,
    plot_signal,
    RANDOM_SEED,
)


# ---------------------------------------------------------------------------
# Sabitler
# ---------------------------------------------------------------------------
SAMPLE_SIZE = 10_000


# ---------------------------------------------------------------------------
# Test Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def synthetic_signal() -> np.ndarray:
    """Modül genelinde yeniden kullanılacak sentetik ONT sinyali."""
    return generate_synthetic_nanopore_signal(n_samples=SAMPLE_SIZE, seed=RANDOM_SEED)


@pytest.fixture(scope="module")
def normalized_signal(synthetic_signal: np.ndarray) -> np.ndarray:
    """Z-score normalize edilmiş sinyal."""
    return zscore_normalize(synthetic_signal)


# ---------------------------------------------------------------------------
# 1. Sentetik Sinyal Üretimi Testleri
# ---------------------------------------------------------------------------

class TestSyntheticSignal:
    """generate_synthetic_nanopore_signal fonksiyonunu doğrular."""

    def test_output_is_1d_numpy_array(self, synthetic_signal: np.ndarray) -> None:
        """Çıktı 1D numpy dizisi olmalı."""
        assert isinstance(synthetic_signal, np.ndarray), "numpy.ndarray bekleniyor"
        assert synthetic_signal.ndim == 1, f"1D bekleniyor, {synthetic_signal.ndim}D alındı"

    def test_output_shape(self, synthetic_signal: np.ndarray) -> None:
        """İstenen örnek sayısı üretilmeli."""
        assert synthetic_signal.shape == (SAMPLE_SIZE,), (
            f"Beklenen shape: ({SAMPLE_SIZE},), alınan: {synthetic_signal.shape}"
        )

    def test_output_dtype_is_float64(self, synthetic_signal: np.ndarray) -> None:
        """Dtype float64 olmalı."""
        assert synthetic_signal.dtype == np.float64, (
            f"dtype=float64 bekleniyor, alınan={synthetic_signal.dtype}"
        )

    def test_signal_in_realistic_pa_range(self, synthetic_signal: np.ndarray) -> None:
        """Sinyal nanopore için gerçekçi pikoamper aralığında olmalı (~40–200 pA)."""
        assert synthetic_signal.min() > 20.0, "Alt sınır 20 pA'nın üzerinde olmalı"
        assert synthetic_signal.max() < 250.0, "Üst sınır 250 pA'nın altında olmalı"

    def test_reproducibility_with_seed(self) -> None:
        """Aynı seed ile üretilen sinyaller özdeş olmalı."""
        s1 = generate_synthetic_nanopore_signal(n_samples=1000, seed=42)
        s2 = generate_synthetic_nanopore_signal(n_samples=1000, seed=42)
        np.testing.assert_array_equal(s1, s2, err_msg="Seed ile tekrarlanabilirlik başarısız")

    def test_different_seeds_produce_different_signals(self) -> None:
        """Farklı seed'ler farklı sinyal üretmeli."""
        s1 = generate_synthetic_nanopore_signal(n_samples=1000, seed=0)
        s2 = generate_synthetic_nanopore_signal(n_samples=1000, seed=99)
        assert not np.array_equal(s1, s2), "Farklı seed'ler aynı sinyali üretmemeli"


# ---------------------------------------------------------------------------
# 2. Z-Score Normalizasyon Testleri
# ---------------------------------------------------------------------------

class TestZscoreNormalize:
    """zscore_normalize fonksiyonunu matematiksel olarak doğrular."""

    def test_mean_is_zero(self, normalized_signal: np.ndarray) -> None:
        """Normalize sinyalin ortalaması 0'a yakın olmalı (< 1e-5)."""
        mean_val = abs(normalized_signal.mean())
        assert mean_val < 1e-5, f"Ortalama 0'a yakın olmalı, alınan: {mean_val:.2e}"

    def test_std_is_one(self, normalized_signal: np.ndarray) -> None:
        """Normalize sinyalin standart sapması 1'e yakın olmalı (< 1e-5)."""
        std_val = abs(normalized_signal.std() - 1.0)
        assert std_val < 1e-5, f"Standart sapma ≈ 1 olmalı, sapma: {std_val:.2e}"

    def test_shape_preserved(
        self, synthetic_signal: np.ndarray, normalized_signal: np.ndarray
    ) -> None:
        """Normalizasyon sonrası shape değişmemeli."""
        assert synthetic_signal.shape == normalized_signal.shape, (
            f"Shape değişti: {synthetic_signal.shape} → {normalized_signal.shape}"
        )

    def test_dtype_is_float64(self, normalized_signal: np.ndarray) -> None:
        """Çıktı dtype float64 olmalı."""
        assert normalized_signal.dtype == np.float64

    def test_raises_on_non_1d_input(self) -> None:
        """2D dizi verilince ValueError fırlatmalı."""
        with pytest.raises(ValueError, match="1D"):
            zscore_normalize(np.random.randn(10, 10))

    def test_constant_signal_does_not_divide_by_zero(self) -> None:
        """Sabit sinyalde (std=0) sıfıra bölme hatası olmamalı."""
        constant = np.ones(100) * 75.0
        result = zscore_normalize(constant)
        assert np.all(np.isfinite(result)), "Sabit sinyal için sonuç sonlu olmalı"

    def test_output_has_no_nan_or_inf(self, normalized_signal: np.ndarray) -> None:
        """Çıktıda NaN veya Inf olmamalı."""
        assert np.all(np.isfinite(normalized_signal)), "NaN veya Inf değerler var"

    def test_normalization_is_invertible(self, synthetic_signal: np.ndarray) -> None:
        """Normalizasyon tersine çevrilebilir olmalı (mu ve sigma korunarak)."""
        mu = synthetic_signal.mean()
        sigma = synthetic_signal.std()
        normed = zscore_normalize(synthetic_signal)
        # Ters dönüşüm: x = z * sigma + mu
        reconstructed = normed * sigma + mu
        np.testing.assert_allclose(
            reconstructed, synthetic_signal, rtol=1e-5, atol=1e-5,
            err_msg="Normalizasyon tersine çevrilemiyor"
        )


# ---------------------------------------------------------------------------
# 3. İndirme Mekanizması Testleri (Mock ile)
# ---------------------------------------------------------------------------

class TestDownloadOntData:
    """download_ont_data retry mekanizmasını mock ile doğrular."""

    def test_skips_download_if_file_exists(self, tmp_path: Path) -> None:
        """Dosya zaten varsa indirme atlanmalı (requests çağrılmamalı)."""
        existing_file = tmp_path / "test.pod5"
        existing_file.write_bytes(b"mock pod5 data" * 100)  # Gerçekçi boyut

        with patch("data_ingestion.requests.get") as mock_get:
            result = download_ont_data("http://fake.url", existing_file)
            mock_get.assert_not_called()

        assert result == existing_file

    def test_raises_runtime_error_after_all_retries_fail(self, tmp_path: Path) -> None:
        """Tüm retry denemeleri başarısız olursa RuntimeError fırlatmalı."""
        import requests as req_lib

        dest = tmp_path / "fail.pod5"

        with patch("data_ingestion.requests.get") as mock_get:
            mock_get.side_effect = req_lib.RequestException("Bağlantı hatası")
            with pytest.raises(RuntimeError):
                download_ont_data(
                    "http://invalid.url",
                    dest,
                    retries=2,
                    retry_delay=0.01,
                )

    def test_retry_count_is_respected(self, tmp_path: Path) -> None:
        """requests.get tam olarak retries kadar çağrılmalı."""
        import requests as req_lib

        dest = tmp_path / "retry_test.pod5"

        with patch("data_ingestion.requests.get") as mock_get:
            mock_get.side_effect = req_lib.RequestException("Sunucu hatası")
            with pytest.raises(RuntimeError):
                download_ont_data(
                    "http://retry.test",
                    dest,
                    retries=3,
                    retry_delay=0.01,
                    fallback_url=None,
                )
            # Ana URL için 3 deneme
            assert mock_get.call_count == 3, (
                f"Beklenen 3 deneme, gerçekleşen: {mock_get.call_count}"
            )

    def test_creates_parent_directory(self, tmp_path: Path) -> None:
        """Hedef dizin yoksa otomatik oluşturulmalı."""
        import requests as req_lib
        nested_dest = tmp_path / "a" / "b" / "c" / "test.pod5"

        with patch("data_ingestion.requests.get") as mock_get:
            mock_get.side_effect = req_lib.RequestException("Hata")
            with pytest.raises(RuntimeError):
                download_ont_data("http://x.url", nested_dest, retries=1, retry_delay=0.01)

        assert nested_dest.parent.exists(), "Üst dizin oluşturulmalıydı"


# ---------------------------------------------------------------------------
# 4. Görselleştirme Testleri
# ---------------------------------------------------------------------------

class TestPlotSignal:
    """plot_signal fonksiyonunun hatasız çalıştığını doğrular."""

    def test_plot_runs_without_error(self, synthetic_signal: np.ndarray, tmp_path: Path) -> None:
        """plot_signal hata fırlatmamalı ve dosyayı kaydetmeli."""
        save_path = tmp_path / "test_plot.png"
        plot_signal(synthetic_signal, title="Test", save_path=save_path, show=False)
        assert save_path.exists(), "Grafik dosyası oluşturulmalıydı"
        assert save_path.stat().st_size > 1000, "Grafik dosyası çok küçük"

    def test_plot_works_without_save_path(self, synthetic_signal: np.ndarray) -> None:
        """save_path verilmese de hata fırlatmamalı."""
        plot_signal(synthetic_signal, title="No Save Test", show=False)

    def test_plot_with_sampling_rate(self, synthetic_signal: np.ndarray, tmp_path: Path) -> None:
        """sampling_rate_hz parametresiyle zaman eksenli grafik çalışmalı."""
        save_path = tmp_path / "test_time_axis.png"
        plot_signal(
            synthetic_signal,
            title="Zaman Eksenli",
            sampling_rate_hz=4000.0,
            save_path=save_path,
            show=False,
        )
        assert save_path.exists()


# ---------------------------------------------------------------------------
# 5. Entegrasyon Testi
# ---------------------------------------------------------------------------

class TestIntegration:
    """Tüm Modül 1 adımlarının uçtan uca çalıştığını doğrular."""

    def test_full_pipeline_synthetic(self, tmp_path: Path) -> None:
        """
        Sentetik modda tam pipeline çalışmalı:
        üret → normalize → kaydet → yükle → doğrula
        """
        # 1. Üret
        raw = generate_synthetic_nanopore_signal(n_samples=5000, seed=RANDOM_SEED)
        assert raw.shape == (5000,)

        # 2. Normalize
        normed = zscore_normalize(raw)
        assert abs(normed.mean()) < 1e-5
        assert abs(normed.std() - 1.0) < 1e-5

        # 3. Görselleştir
        plot_path = tmp_path / "integration_plot.png"
        plot_signal(normed, title="Integration Test", save_path=plot_path, show=False)
        assert plot_path.exists()

        # 4. Kaydet / Yükle
        npy_path = tmp_path / "test_signal.npy"
        np.save(npy_path, normed)
        loaded = np.load(npy_path)
        np.testing.assert_array_equal(normed, loaded, err_msg=".npy kayıt/yükleme başarısız")
