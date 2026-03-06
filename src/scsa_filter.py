"""
Module 2: Quantum-Inspired Noise Filter (SCSA)
=================================================
Filters microscopic noise (Poisson/Anderson localization) in 
1-dimensional nanopore signals using the principles of 
Semi-Classical Signal Analysis (SCSA).

By treating the signal as a potential well, this module applies 
the discrete Schrödinger equation from quantum mechanics.

Author: FizyGame
Date: 2026
"""

from __future__ import annotations

import logging
from typing import Optional, Tuple

import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import eigsh

logger = logging.getLogger("scsa_filter")

class SCSAFilter:
    """
    SCSA (Semi-Classical Signal Analysis) discrete spectral noise filter.
    
    Attributes:
        h (float): Semi-classical heuristic parameter (equivalent to Planck's constant).
                   This balances the kinetic energy in the Hamiltonian.
        n_components (int): Maximum number of eigenvalues to compute.
                            Limits processing time for very large signals.
    """

    def __init__(
        self,
        h: Optional[float] = None,
        n_components: Optional[int] = 50,
    ) -> None:
        self.h = h
        self.n_components = n_components
        self.eigenvalues_: Optional[np.ndarray] = None
        self.eigenfunctions_: Optional[np.ndarray] = None

    def _build_hamiltonian(self, signal: np.ndarray) -> sp.spmatrix:
        """
        Builds the N x N tridiagonal Hamiltonian (H) matrix.
        
        Formula:
            H(i, i)   = x(i) + 2/h^2
            H(i, i±1) = -1/h^2
            
        Args:
            signal (np.ndarray): 1D signal array (potential well V).
            
        Returns:
            scipy.sparse.spmatrix: H matrix in CSC format.
        """
        n = len(signal)
        h = self.h
        
        assert h is not None and h > 0, "The h parameter must be greater than zero."

        # Diagonal: x(i) + 2/h^2
        main_diag = signal + (2.0 / (h ** 2))
        
        # Off-diagonal: -1/h^2
        off_diag = np.full(n - 1, -1.0 / (h ** 2), dtype=np.float64)
        
        # Sparse matrix assembly
        H = sp.diags([off_diag, main_diag, off_diag], [-1, 0, 1], format='csc')
        
        assert H.shape == (n, n), f"Expected shape: ({n}, {n}), got: {H.shape}"
        return H

    def fit_transform(self, signal: np.ndarray) -> np.ndarray:
        """
        Applies the SCSA filter to the signal.
        
        Contains np.ndarray type and shape assertions to prevent
        dimension mismatches.
        
        Args:
            signal (np.ndarray): 1D input signal. A portion (or all) of the signal
                                 must be negative (e.g., via Z-score).
                                 
        Returns:
            np.ndarray: The filtered (reconstructed) signal.
        """
        assert isinstance(signal, np.ndarray), "Input signal must be a numpy array."
        assert signal.ndim == 1, f"Expected a 1D signal, got {signal.ndim}D."
        
        n = len(signal)
        if n < 3:
            return signal.copy()
            
        # If h is not assigned, use heuristic initialization (sqrt of signal variance)
        # If the signal is standardized (Module 1), the variance is 1.
        if self.h is None:
            var = np.var(signal)
            self.h = np.sqrt(var) if var > 1e-6 else 1.0
            logger.debug("Calculated heuristic h: %.4f", self.h)
            
        H = self._build_hamiltonian(signal)
        
        # The 'k' parameter for scipy eigsh must be less than the matrix dimension (n-1).
        k_eig = min(self.n_components if self.n_components else 50, n - 2)
        k_eig = max(1, k_eig)
        
        try:
            # Find only the smallest algebraic eigenvalues (which='SA')
            evals, evecs = eigsh(H, k=k_eig, which='SA', tol=1e-4)
        except Exception as e:
            logger.warning("eigsh solution failed: %s. Returning original signal.", e)
            return signal.copy()
            
        # Filter negative eigenvalues (bound quantum states)
        neg_mask = evals < 0
        neg_evals = evals[neg_mask]
        neg_evecs = evecs[:, neg_mask]
        
        self.eigenvalues_ = neg_evals
        self.eigenfunctions_ = neg_evecs
        
        assert neg_evecs.shape[0] == n, (
            f"Eigenfunction spatial dimension {neg_evecs.shape[0]} does not match signal dimension {n}."
        )
        
        if len(neg_evals) == 0:
            logger.debug("SCSA: No negative eigenvalues (bound states) found.")
            return signal.copy()
            
        # Signal Reconstruction
        # SCSA formula: y_rec ~ sum( kappa_k * psi_k(x)^2 )
        # kappa_k = sqrt(-E_k)
        kappa = np.sqrt(-neg_evals)
        
        reconstructed = np.zeros(n, dtype=np.float64)
        for i in range(len(neg_evals)):
            # Add the square of psi_k as probability density
            reconstructed += kappa[i] * (neg_evecs[:, i] ** 2)
            
        # The amplitude of the reconstructed signal depends on h and N.
        # Therefore, we map the output signal back to the amplitude range 
        # (Min-Max) of the input signal while retaining its structural features.
        ptp_rec = reconstructed.max() - reconstructed.min()
        if ptp_rec > 1e-12:
            reconstructed = (reconstructed - reconstructed.min()) / ptp_rec
            
            # Necessary Min-Max back-transformation (matched to original signal's min/max)
            ptp_sig = signal.max() - signal.min()
            reconstructed = (reconstructed * ptp_sig) + signal.min()
            
        return reconstructed

# CLI Test
if __name__ == "__main__":
    import matplotlib.pyplot as plt
    
    # A simple test signal (Noisy Sine)
    np.random.seed(42)
    x = np.linspace(0, 4*np.pi, 500)
    clean = np.sin(x) - 0.5 # Add negative regions
    noisy = clean + np.random.normal(0, 0.3, size=len(x))
    
    scsa = SCSAFilter(h=0.5, n_components=30)
    filtered = scsa.fit_transform(noisy)
    
    print(f"Number of negative eigenvalues (E_neg) found: {len(scsa.eigenvalues_)}")
    
    plt.plot(x, noisy, label="Noisy", alpha=0.5)
    plt.plot(x, clean, label="Clean (True)", linestyle='--')
    plt.plot(x, filtered, label="SCSA Filtered")
    plt.legend()
    plt.title("SCSA Filter Demo")
    plt.show()
