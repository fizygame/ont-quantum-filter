"""
Module 1 Test File — Data Ingestion & Preprocessing
===============================================
Run with pytest:
    cd c:\\Users\\nuri_\\OneDrive\\Masaüstü\\DNA
    pytest tests/test_module1.py -v

All tests work offline using the synthetic signal generator without requiring a real network connection.
"""

from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path
from unittest.mock import patch, MagicMock

import numpy as np
import pytest

# Add project root to Python path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from data_ingestion import (
    zscore_normalize,
    generate_synthetic_nanopore_signal,
    download_ont_data,
    plot_signal,
    RANDOM_SEED,
)


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
SAMPLE_SIZE = 10_000


# ---------------------------------------------------------------------------
# Test Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def synthetic_signal() -> np.ndarray:
    """Synthetic ONT signal to be reused across the module."""
    return generate_synthetic_nanopore_signal(n_samples=SAMPLE_SIZE, seed=RANDOM_SEED)


@pytest.fixture(scope="module")
def normalized_signal(synthetic_signal: np.ndarray) -> np.ndarray:
    """Z-score normalized signal."""
    return zscore_normalize(synthetic_signal)


# ---------------------------------------------------------------------------
# 1. Synthetic Signal Generator Tests
# ---------------------------------------------------------------------------

class TestSyntheticSignal:
    """Validates the generate_synthetic_nanopore_signal function."""

    def test_output_is_1d_numpy_array(self, synthetic_signal: np.ndarray) -> None:
        """The output must be a 1D numpy array."""
        assert isinstance(synthetic_signal, np.ndarray), "Expected numpy.ndarray"
        assert synthetic_signal.ndim == 1, f"Expected 1D, got {synthetic_signal.ndim}D"

    def test_output_shape(self, synthetic_signal: np.ndarray) -> None:
        """The requested number of samples must be generated."""
        assert synthetic_signal.shape == (SAMPLE_SIZE,), (
            f"Expected shape: ({SAMPLE_SIZE},), got: {synthetic_signal.shape}"
        )

    def test_output_dtype_is_float64(self, synthetic_signal: np.ndarray) -> None:
        """Dtype must be float64."""
        assert synthetic_signal.dtype == np.float64, (
            f"Expected dtype=float64, got={synthetic_signal.dtype}"
        )

    def test_signal_in_realistic_pa_range(self, synthetic_signal: np.ndarray) -> None:
        """The signal must be within a realistic picoampere range for nanopores (~40–200 pA)."""
        assert synthetic_signal.min() > 20.0, "Lower bound should be above 20 pA"
        assert synthetic_signal.max() < 250.0, "Upper bound should be below 250 pA"

    def test_reproducibility_with_seed(self) -> None:
        """Signals generated with the same seed must be identical."""
        s1 = generate_synthetic_nanopore_signal(n_samples=1000, seed=42)
        s2 = generate_synthetic_nanopore_signal(n_samples=1000, seed=42)
        np.testing.assert_array_equal(s1, s2, err_msg="Seed reproducibility failed")

    def test_different_seeds_produce_different_signals(self) -> None:
        """Different seeds must generate different signals."""
        s1 = generate_synthetic_nanopore_signal(n_samples=1000, seed=0)
        s2 = generate_synthetic_nanopore_signal(n_samples=1000, seed=99)
        assert not np.array_equal(s1, s2), "Different seeds should not produce identical signals"


# ---------------------------------------------------------------------------
# 2. Z-Score Normalization Tests
# ---------------------------------------------------------------------------

class TestZscoreNormalize:
    """Mathematically validates the zscore_normalize function."""

    def test_mean_is_zero(self, normalized_signal: np.ndarray) -> None:
        """The mean of the normalized signal must be close to 0 (< 1e-5)."""
        mean_val = abs(normalized_signal.mean())
        assert mean_val < 1e-5, f"Mean should be near 0, got: {mean_val:.2e}"

    def test_std_is_one(self, normalized_signal: np.ndarray) -> None:
        """The standard deviation of the normalized signal must be close to 1 (< 1e-5)."""
        std_val = abs(normalized_signal.std() - 1.0)
        assert std_val < 1e-5, f"Standard deviation should be ≈ 1, deviation: {std_val:.2e}"

    def test_shape_preserved(
        self, synthetic_signal: np.ndarray, normalized_signal: np.ndarray
    ) -> None:
        """The shape must not change after normalization."""
        assert synthetic_signal.shape == normalized_signal.shape, (
            f"Shape changed: {synthetic_signal.shape} → {normalized_signal.shape}"
        )

    def test_dtype_is_float64(self, normalized_signal: np.ndarray) -> None:
        """Output dtype must be float64."""
        assert normalized_signal.dtype == np.float64

    def test_raises_on_non_1d_input(self) -> None:
        """It should raise a ValueError when a 2D array is provided."""
        with pytest.raises(ValueError, match="1D"):
            zscore_normalize(np.random.randn(10, 10))

    def test_constant_signal_does_not_divide_by_zero(self) -> None:
        """No divide-by-zero error should occur for a constant signal (std=0)."""
        constant = np.ones(100) * 75.0
        result = zscore_normalize(constant)
        assert np.all(np.isfinite(result)), "Result must be finite for a constant signal"

    def test_output_has_no_nan_or_inf(self, normalized_signal: np.ndarray) -> None:
        """There should be no NaN or Inf in the output."""
        assert np.all(np.isfinite(normalized_signal)), "Values contain NaN or Inf"

    def test_normalization_is_invertible(self, synthetic_signal: np.ndarray) -> None:
        """Normalization must be invertible (preserving mu and sigma)."""
        mu = synthetic_signal.mean()
        sigma = synthetic_signal.std()
        normed = zscore_normalize(synthetic_signal)
        # Inverse transform: x = z * sigma + mu
        reconstructed = normed * sigma + mu
        np.testing.assert_allclose(
            reconstructed, synthetic_signal, rtol=1e-5, atol=1e-5,
            err_msg="Normalization is not invertible"
        )


# ---------------------------------------------------------------------------
# 3. Download Mechanism Tests (with Mock)
# ---------------------------------------------------------------------------

class TestDownloadOntData:
    """Validates the download_ont_data retry mechanism using mock."""

    def test_skips_download_if_file_exists(self, tmp_path: Path) -> None:
        """Download should be skipped if the file exists (requests shouldn't be called)."""
        existing_file = tmp_path / "test.pod5"
        existing_file.write_bytes(b"mock pod5 data" * 100)  # Realistic size

        with patch("data_ingestion.requests.get") as mock_get:
            result = download_ont_data("http://fake.url", existing_file)
            mock_get.assert_not_called()

        assert result == existing_file

    def test_raises_runtime_error_after_all_retries_fail(self, tmp_path: Path) -> None:
        """It should raise RuntimeError if all retry attempts fail."""
        import requests as req_lib

        dest = tmp_path / "fail.pod5"

        with patch("data_ingestion.requests.get") as mock_get:
            mock_get.side_effect = req_lib.RequestException("Connection error")
            with pytest.raises(RuntimeError):
                download_ont_data(
                    "http://invalid.url",
                    dest,
                    retries=2,
                    retry_delay=0.01,
                )

    def test_retry_count_is_respected(self, tmp_path: Path) -> None:
        """requests.get should be called exactly 'retries' times."""
        import requests as req_lib

        dest = tmp_path / "retry_test.pod5"

        with patch("data_ingestion.requests.get") as mock_get:
            mock_get.side_effect = req_lib.RequestException("Server error")
            with pytest.raises(RuntimeError):
                download_ont_data(
                    "http://retry.test",
                    dest,
                    retries=3,
                    retry_delay=0.01,
                    fallback_url=None,
                )
            # 3 attempts for the primary URL
            assert mock_get.call_count == 3, (
                f"Expected 3 attempts, got: {mock_get.call_count}"
            )

    def test_creates_parent_directory(self, tmp_path: Path) -> None:
        """Parent directory should be automatically created if missing."""
        import requests as req_lib
        nested_dest = tmp_path / "a" / "b" / "c" / "test.pod5"

        with patch("data_ingestion.requests.get") as mock_get:
            mock_get.side_effect = req_lib.RequestException("Error")
            with pytest.raises(RuntimeError):
                download_ont_data("http://x.url", nested_dest, retries=1, retry_delay=0.01)

        assert nested_dest.parent.exists(), "Parent directory should have been created"


# ---------------------------------------------------------------------------
# 4. Visualization Tests
# ---------------------------------------------------------------------------

class TestPlotSignal:
    """Validates that the plot_signal function works without errors."""

    def test_plot_runs_without_error(self, synthetic_signal: np.ndarray, tmp_path: Path) -> None:
        """plot_signal should not raise an error and must save the file."""
        save_path = tmp_path / "test_plot.png"
        plot_signal(synthetic_signal, title="Test", save_path=save_path, show=False)
        assert save_path.exists(), "Plot file should have been generated"
        assert save_path.stat().st_size > 1000, "Plot file is too small"

    def test_plot_works_without_save_path(self, synthetic_signal: np.ndarray) -> None:
        """It should not raise an error even if save_path is omitted."""
        plot_signal(synthetic_signal, title="No Save Test", show=False)

    def test_plot_with_sampling_rate(self, synthetic_signal: np.ndarray, tmp_path: Path) -> None:
        """A time-axis plot should work with the sampling_rate_hz parameter."""
        save_path = tmp_path / "test_time_axis.png"
        plot_signal(
            synthetic_signal,
            title="Time Axis",
            sampling_rate_hz=4000.0,
            save_path=save_path,
            show=False,
        )
        assert save_path.exists()


# ---------------------------------------------------------------------------
# 5. Integration Test
# ---------------------------------------------------------------------------

class TestIntegration:
    """Validates that all Module 1 steps work end-to-end."""

    def test_full_pipeline_synthetic(self, tmp_path: Path) -> None:
        """
        The full pipeline should work in synthetic mode:
        generate → normalize → save → load → verify
        """
        # 1. Generate
        raw = generate_synthetic_nanopore_signal(n_samples=5000, seed=RANDOM_SEED)
        assert raw.shape == (5000,)

        # 2. Normalize
        normed = zscore_normalize(raw)
        assert abs(normed.mean()) < 1e-5
        assert abs(normed.std() - 1.0) < 1e-5

        # 3. Visualize
        plot_path = tmp_path / "integration_plot.png"
        plot_signal(normed, title="Integration Test", save_path=plot_path, show=False)
        assert plot_path.exists()

        # 4. Save / Load
        npy_path = tmp_path / "test_signal.npy"
        np.save(npy_path, normed)
        loaded = np.load(npy_path)
        np.testing.assert_array_equal(normed, loaded, err_msg=".npy save/load failed")
