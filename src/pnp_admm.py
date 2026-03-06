"""
Module 4: PnP-ADMM Optimization
===============================
The Plug-and-Play Alternating Direction Method of Multipliers (PnP-ADMM)
algorithm provides a global optimization framework by combining a 
data fidelity step with any independent denoiser (e.g., SCSA).

Author: FizyGame
Date: 2026
"""

from __future__ import annotations

import logging
from typing import Callable, Optional, Tuple, List

import numpy as np

logger = logging.getLogger("pnp_admm")

class PnPADMM:
    """
    Plug-and-Play ADMM Framework.
    
    Attributes:
        denoiser (Callable): Executable function that takes a signal (np.ndarray)
                             and returns a filtered signal (e.g., scsa.fit_transform).
        rho (float): ADMM penalty parameter (Lagrangian multiplier).
        max_iter (int): Maximum number of iterations.
        tol (float): Convergence tolerance (limits for primal and dual residuals).
        adaptive_rho (bool): Whether to perform dynamic rho updates between iterations.
    """

    def __init__(
        self,
        denoiser: Callable[[np.ndarray], np.ndarray],
        rho: float = 1.0,
        max_iter: int = 50,
        tol: float = 1e-4,
        adaptive_rho: bool = True
    ) -> None:
        self.denoiser = denoiser
        self.rho = rho
        self.max_iter = max_iter
        self.tol = tol
        self.adaptive_rho = adaptive_rho
        
        # Tracking history for logging
        self.history_: dict[str, list[float]] = {
            "primal_res": [],
            "dual_res": [],
            "rho": []
        }

    def _data_fidelity_step(
        self, 
        y: np.ndarray, 
        z: np.ndarray, 
        u: np.ndarray, 
        rho: float
    ) -> np.ndarray:
        """
        Data Fidelity Step (x-update).
        Assumption: H = I (Identity Matrix), meaning the degradation is purely 
        Additive White Gaussian Noise (AWGN).
        
        Formula (Proximal Operator for H=I):
            x^{k+1} = (y + rho * (z^k - u^k)) / (1 + rho)
            
        Args:
            y (np.ndarray): Original noisy observation signal.
            z (np.ndarray): ADMM z variable (denoised signal).
            u (np.ndarray): ADMM dual variable (Lagrangian multiplier).
            rho (float): Current penalty parameter.
            
        Returns:
            np.ndarray: Evaluated signal x^{k+1}
        """
        return (y + rho * (z - u)) / (1.0 + rho)

    def _denoising_step(self, x: np.ndarray, u: np.ndarray) -> np.ndarray:
        """
        Plug-in Denoising Step (z-update).
        
        Signal target: x^{k+1} + u^k
        
        Args:
            x (np.ndarray): Updated x variable.
            u (np.ndarray): Dual variable.
            
        Returns:
            np.ndarray: Passed through the denoiser: z^{k+1}
        """
        noisy_target = x + u
        # Plug-and-Play magic: External algorithm call instead of a purely mathematical formulation
        return self.denoiser(noisy_target)

    def _dual_update(
        self, 
        u: np.ndarray, 
        x: np.ndarray, 
        z: np.ndarray
    ) -> np.ndarray:
        """
        Lagrangian Multiplier (Dual Variable) Update.
        
        Formula:
            u^{k+1} = u^k + (x^{k+1} - z^{k+1})
        """
        return u + (x - z)

    def _rho_update(
        self, 
        rho: float, 
        primal_res: float, 
        dual_res: float, 
        mu: float = 10.0, 
        tau: float = 2.0
    ) -> float:
        """
        Adaptive Rho Update (Residual Balancing).
        Prevents ADMM from converging too slowly or diverging.
        """
        if primal_res > mu * dual_res:
            return rho * tau
        elif dual_res > mu * primal_res:
            return rho / tau
        return rho

    def run(self, y: np.ndarray) -> np.ndarray:
        """
        Executes the full PnP-ADMM loop.
        
        Args:
            y (np.ndarray): 1D noisy signal to be optimized.
            
        Returns:
            np.ndarray: Optimized/cleaned z signal.
        """
        assert isinstance(y, np.ndarray) and y.ndim == 1, "Signal must be a 1D numpy array."
        
        n = len(y)
        x = y.copy()
        z = y.copy()
        u = np.zeros(n, dtype=np.float64)
        rho = self.rho
        
        # Reset history
        self.history_ = {"primal_res": [], "dual_res": [], "rho": []}
        
        logger.info("Initializing PnP-ADMM Optimization (Max Iter: %d)", self.max_iter)
        
        for k in range(self.max_iter):
            z_old = z.copy()
            
            # 1. x Update (Data Fidelity)
            x = self._data_fidelity_step(y, z, u, rho)
            
            # 2. z Update (Denoising by Plug-in)
            z = self._denoising_step(x, u)
            
            # 3. u Update (Dual)
            u = self._dual_update(u, x, z)
            
            # Calculate Errors (Residuals)
            primal_residual = np.linalg.norm(x - z)
            dual_residual = np.linalg.norm(rho * (z - z_old))
            
            # Save history
            self.history_["primal_res"].append(primal_residual)
            self.history_["dual_res"].append(dual_residual)
            self.history_["rho"].append(rho)
            
            # Early Convergence Check
            if primal_residual < self.tol and dual_residual < self.tol:
                logger.info("PnP-ADMM converged at iteration %d.", k+1)
                break
                
            # Adaptive rho update (Optional)
            if self.adaptive_rho:
                rho = self._rho_update(rho, primal_residual, dual_residual)
                
        else:
            logger.info("PnP-ADMM reached max iterations (%d) (Convergence Incomplete).", self.max_iter)

        return z

# CLI Demo
if __name__ == "__main__":
    import matplotlib.pyplot as plt
    
    # Create a simple denoiser (Moving average filter)
    def simple_moving_average_denoiser(sig, window_size=5):
        return np.convolve(sig, np.ones(window_size)/window_size, mode='same')
        
    np.random.seed(42)
    x = np.linspace(0, 10, 300)
    ideal = np.sin(x)
    noisy_signal = ideal + np.random.normal(0, 0.4, size=len(x))
    
    # Initialize PnP-ADMM
    admm = PnPADMM(
        denoiser=simple_moving_average_denoiser,
        rho=1.5,
        max_iter=100,
        tol=1e-3
    )
    
    optimized_signal = admm.run(noisy_signal)
    
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 2, 1)
    plt.plot(ideal, 'k--', label="Original")
    plt.plot(noisy_signal, 'r-', alpha=0.5, label="Noisy")
    plt.plot(optimized_signal, 'g-', linewidth=2, label="PnP-ADMM")
    plt.legend()
    plt.title("PnP-ADMM Real-time Behavior")
    
    plt.subplot(1, 2, 2)
    plt.plot(admm.history_["primal_res"], label="Primal Residual")
    plt.plot(admm.history_["dual_res"], label="Dual Residual")
    plt.yscale('log')
    plt.legend()
    plt.title("ADMM Convergence")
    
    plt.tight_layout()
    plt.show()
