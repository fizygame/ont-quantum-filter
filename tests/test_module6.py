"""
Module 6 Test File — Benchmarking & Output Execution Pipeline
=================================================
Run with pytest:
    cd c:\\Users\\nuri_\\OneDrive\\Masaüstü\\DNA
    pytest tests/test_module6.py -v
"""

from __future__ import annotations

import sys
import os
from pathlib import Path

import numpy as np
import pytest

# Inject active directories downward explicitly allowing isolated logic verifications natively
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from benchmarking import (
    compute_snr,
    savitzky_golay_baseline,
    plot_comparison,
    export_npy,
    export_fallback
)

# ---------------------------------------------------------------------------
# Validation Modules Set
# ---------------------------------------------------------------------------

class TestBenchmarkingMetrics:
    
    def test_snr_positive(self) -> None:
        """
        Denoised array instances reflecting true signal components more accurately 
        MUST generate strictly higher structural SNR scoring arrays than baseline heavy static blocks intrinsically.
        """
        np.random.seed(42)
        true_sig = np.ones(50)
        
        # Highly disrupted background interference sequence layout (Noisy base)
        noisy = true_sig + np.random.normal(0, 2.0, 50)
        
        # Case A: Moderately sanitized mapping stream output
        semi_clean = true_sig + np.random.normal(0, 1.0, 50)
        
        # Case B: Extensively decoupled isolated core component stream outputs purely locked
        very_clean = true_sig + np.random.normal(0, 0.1, 50)
        
        snr_semi = compute_snr(semi_clean, noisy)
        snr_very = compute_snr(very_clean, noisy)
        
        assert snr_very > snr_semi, "Extensively scrubbed data instances structurally MUST output heavier scoring limits absolutely."
        
    def test_snr_shape_mismatch(self) -> None:
        """Dimensionally distinct tracking parameters pushing out of scope sequences MUST rigidly halt via hard operational framework Exceptions safely."""
        a = np.ones(10)
        b = np.ones(11)
        with pytest.raises(AssertionError):
            compute_snr(a, b)

class TestBaselines:
    
    def test_savitzky_golay_shape(self) -> None:
        """Traditional filtering architectures structurally MUST NOT alter underlying geometric vector limits inadvertently anywhere."""
        sig = np.random.randn(20)
        filtered = savitzky_golay_baseline(sig, window_length=5, polyorder=2)
        assert filtered.shape == sig.shape

class TestExportAndPlot:

    @pytest.fixture
    def temp_out_dir(self, tmpdir) -> Path:
        """Volatile environment operational local output directory explicitly mapped handling operations implicitly natively testing operations cleanly."""
        return Path(tmpdir)
        
    def test_npy_export(self, temp_out_dir: Path) -> None:
        """Numerical matrix systems securely MUST mount correctly storing outbound structural files explicitly formatted natively mapping flawlessly downward."""
        sig = np.array([1., 2., 3., 4.])
        out_path = temp_out_dir / "test.npy"
        
        export_npy(sig, out_path)
        
        assert out_path.exists()
        loaded = np.load(out_path)
        np.testing.assert_allclose(sig, loaded)
        
    def test_fallback_export(self, temp_out_dir: Path) -> None:
        """Legacy text mapping exports strictly generating outbound formatted structures seamlessly processing backward compatibility implicitly loading text layouts correctly natively (.txt)."""
        sig = np.array([1.123, 2.456, 3.789])
        out_path = temp_out_dir / "test.txt"
        
        export_fallback(sig, out_path)
        
        assert out_path.exists()
        # Bypass structural mapping header metadata explicitly and ingest binary/text limits strictly validating parameters implicitly
        loaded = np.loadtxt(out_path, delimiter=',', skiprows=1)
        np.testing.assert_allclose(sig, loaded, rtol=1e-3)
        
    def test_plot_runs_without_error(self, temp_out_dir: Path) -> None:
        """Core visual layout framework instances safely compiling graphics inherently avoiding completely UI-bound structural system crashes explicitly outputting valid targets completely."""
        raw = np.zeros(10)
        classic = np.ones(10)
        quantum = np.ones(10) * 2
        out_path = temp_out_dir / "fig.png"
        
        plot_comparison(raw, classic, quantum, save_path=out_path, show=False)
        
        assert out_path.exists()
        assert out_path.stat().st_size > 1024  # Base logic check proving structural existence intrinsically implicitly ensuring size validation boundaries accurately (1KB min)
