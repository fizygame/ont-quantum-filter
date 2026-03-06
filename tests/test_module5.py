"""
Module 5 Test File — DQGA Hyperparameter Optimization
========================================================
Run with pytest:
    cd c:\\Users\\nuri_\\OneDrive\\Masaüstü\\DNA
    pytest tests/test_module5.py -v
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

# Add project root to Python paths
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from dqga_optimizer import QuantumChromosome, QuantumGate, DQGA, bits_to_float

# ---------------------------------------------------------------------------
# Utility Classes / Functions
# ---------------------------------------------------------------------------

def dummy_fitness(bits: np.ndarray) -> float:
    """Mock testing target prioritizing absolute string structures loaded exactly entirely with '1' sequences."""
    return float(np.sum(bits))

# ---------------------------------------------------------------------------
# Test Blocks
# ---------------------------------------------------------------------------

class TestQuantumChromosome:
    
    def test_initial_superposition(self) -> None:
        """Every single initializing gene MUST deploy into |a|^2 = 0.5 and |b|^2 = 0.5 rulesets automatically."""
        chrom = QuantumChromosome(n_genes=10)
        
        np.testing.assert_allclose(chrom.alpha ** 2, 0.5)
        np.testing.assert_allclose(chrom.beta ** 2, 0.5)
        
    def test_normalization(self) -> None:
        """Under conditions featuring manual floating variations, _normalize must force equations rigidly back inside |a|^2 + |b|^2 = 1 dimensions."""
        chrom = QuantumChromosome(n_genes=2)
        chrom.alpha = np.array([10.0, 0.0])
        chrom.beta  = np.array([0.0, 5.0])
        
        chrom._normalize()
        
        sum_sq = chrom.alpha**2 + chrom.beta**2
        np.testing.assert_allclose(sum_sq, 1.0)
        
    def test_measurement_returns_binary(self) -> None:
        """Observations mapping sequences explicitly ought to restrict responses natively downward straight to distinct length-matched lists containing either 0s or 1s strictly."""
        chrom = QuantumChromosome(n_genes=8)
        bits = chrom.measure()
        
        assert len(bits) == 8
        assert bits.dtype == np.int8
        assert np.all(np.isin(bits, [0, 1]))

class TestQuantumGate:

    def test_rotation_is_unitary(self) -> None:
        """Rotations passing directly backwards/forwards logically MUST preserve vector norms absolutely."""
        chrom = QuantumChromosome(n_genes=5)
        best = QuantumChromosome(n_genes=5)
        
        chrom.measure()
        best.measure()
        
        # Triggering simulated divergence purposely targeting absolute guaranteed activation sequences
        chrom.binary_string = np.zeros(5, dtype=np.int8)
        best.binary_string = np.ones(5, dtype=np.int8)
        
        QuantumGate.rotate(chrom, best, theta=0.1)
        
        # Test Constraint: Does |a|^2 + |b|^2 strictly hold its ground locked around 1?
        sum_sq = chrom.alpha**2 + chrom.beta**2
        np.testing.assert_allclose(sum_sq, 1.0)
        
    def test_raises_without_measurement(self) -> None:
        """Pushing chromosomal rotation limits before initiating prior observing blocks throws isolated exception triggers instantly."""
        chrom = QuantumChromosome(5)
        best = QuantumChromosome(5)
        
        with pytest.raises(ValueError):
            QuantumGate.rotate(chrom, best)

class TestDQGA:
    
    def test_fitness_improves(self) -> None:
        """
        Generational temporal progression inherently MUST reflect scaling structural fitness values explicitly.
        """
        np.random.seed(42) # Reproducibility lock bounds applied
        
        dqga = DQGA(pop_size=10, n_genes=16, fitness_fn=dummy_fitness, n_generations=20)
        best_bits, best_score = dqga.run()
        
        history = dqga.history_best_fitness
        
        # Is the terminal generation universally functionally superior looking backward from origin point data instances?
        assert history[-1] >= history[0], "Fitness degenerating progressively, lacking targeted corrective trajectory fixes!"
        # Ordinarily maximum capacity pushes absolute boundary scaling directly mapping to exact total bit string thresholds
        assert best_score > 0

class TestParameterMapping:
    
    def test_bits_to_float_at_boundaries(self) -> None:
        """Prove extreme structural bounds mapping translation dynamics securely tracking between targeted parameter definitions natively."""
        # Baseline Complete Limits Base Case Tracking --> Val Min (10.0) Minimum Limits Set
        bits_min = np.array([0, 0, 0, 0])
        assert bits_to_float(bits_min, 10.0, 50.0) == 10.0
        
        # Absolute Ultimate Extreme Bounds Upper Case Limit Matrix Expansion Output Check Returns Validate True Bounds (50.0)
        bits_max = np.array([1, 1, 1, 1])
        assert bits_to_float(bits_max, 10.0, 50.0) == 50.0
        
        # Median / Half-way threshold validation target logic point implementation parameters
        bits_mid = np.array([1, 0, 0, 0]) # Base Decimal Score maps rigidly returning specifically 8 explicitly (for strict 4 bits architecture array frameworks bound directly checking maximum points limiting structurally toward 15 exclusively -> 8/15)
        expected = 10.0 + (8 / 15.0) * (50.0 - 10.0)
        np.testing.assert_approx_equal(bits_to_float(bits_mid, 10.0, 50.0), expected)
