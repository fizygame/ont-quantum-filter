"""
Module 6: Benchmarking & Output Generation
===========================================================
Performs quantitative comparisons between classical baseline filters 
(e.g., Savitzky-Golay) and the Quantum-Inspired (SCSA + PnP-ADMM) engines 
developed herein. Finally, exports the clean signal array to disk (.npy / .txt).

Author: FizyGame
Date: 2026
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Union

import numpy as np
from scipy.signal import savgol_filter
import matplotlib
import matplotlib.pyplot as plt

# GUI Isolation (For Headless CI/CD tests)
matplotlib.use("Agg")

logger = logging.getLogger("benchmarking")

def compute_snr(signal: np.ndarray, ground_truth: np.ndarray) -> float:
    """
    Calculates the Signal-to-Noise Ratio (SNR) in Decibels (dB).
    
    SNR = 10 * log10( P_signal / P_noise )
    A higher SNR means the signal conforms more closely to the physical ground truth.
    
    Args:
        signal (np.ndarray): Evaluated / Filtered signal.
        ground_truth (np.ndarray): Error-free reference (True) signal.
        
    Returns:
        float: SNR score in dB.
    """
    assert signal.shape == ground_truth.shape, "Signal dimensions must match."
    
    # Power of Signal
    p_signal = np.mean(ground_truth ** 2)
    
    # Noise Power (Mean Squared Error against ground truth)
    noise = signal - ground_truth
    p_noise = np.mean(noise ** 2)
    
    if p_noise == 0:
        return float('inf')
        
    snr_db = 10 * np.log10(p_signal / p_noise)
    return float(snr_db)

def savitzky_golay_baseline(signal: np.ndarray, window_length: int = 11, polyorder: int = 3) -> np.ndarray:
    """
    Utilizes a Savitzky-Golay filter as a classical baseline reference.
    This acts as a standard bioinformatics smoothing algorithm.
    
    Args:
        signal (np.ndarray): 1D Noisy signal array.
        window_length (int): Filter window size (must be an odd integer, less than signal len).
        polyorder (int): The polynomial degree.
        
    Returns:
        np.ndarray: Classically filtered output signal.
    """
    if len(signal) < window_length:
        window_length = len(signal) if len(signal) % 2 == 1 else len(signal) - 1
        
    if window_length <= polyorder:
        polyorder = window_length - 1
        
    return savgol_filter(signal, window_length=window_length, polyorder=polyorder)

def plot_comparison(
    raw: np.ndarray, 
    classic: np.ndarray, 
    quantum: np.ndarray, 
    save_path: Union[str, Path, None] = None,
    show: bool = False
) -> None:
    """
    Generates a 3-axis matplotlib subplot comparing the Raw (Noisy), 
    Classical (Savitzky-Golay), and Quantum (SCSA/ADMM) responses side by side.
    
    Args:
        raw (np.ndarray): Original noisy sequence.
        classic (np.ndarray): Classical algorithm (Savitzky-Golay) output.
        quantum (np.ndarray): Quantum-inspired pipeline output.
        save_path (Path|str|None): Target figure filepath.
        show (bool): Should display on screen? (Requires GUI backends)
    """
    fig, axes = plt.subplots(3, 1, figsize=(12, 8), sharex=True, sharey=True)
    
    colors = ['#d62728', '#1f77b4', '#2ca02c'] # Red, Blue, Green
    titles = ["Raw (Noisy) Signal", "Classical Filter (Savitzky-Golay)", "Quantum-Inspired Pipeline (SCSA + PnP-ADMM)"]
    signals = [raw, classic, quantum]
    
    for i, (sig, ax, c, t) in enumerate(zip(signals, axes, colors, titles)):
        ax.plot(sig, color=c, alpha=0.8)
        ax.set_title(t, fontweight='bold')
        ax.grid(True, alpha=0.3)
        if i == 2:
            ax.set_xlabel("Time / Sample", fontweight='bold')
    
    plt.tight_layout()
    
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Comparison graph stored at: {save_path}")
        
    if show:
        try:
            plt.show()
        except Exception as e:
            logger.warning(f"plot_comparison -> show() failed (Headless environment?): {e}")
    else:
        plt.close(fig)

def export_npy(signal: np.ndarray, path: Union[str, Path]) -> None:
    """
    Discharges the filtered numpy sequence as a binary (.npy) store structure.
    
    Args:
        signal (np.ndarray): The processed 1D signal.
        path (Path|str): Target system destination path (.npy appended).
    """
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Enforces size optimization mapping against float32 architectures 
    np.save(str(out_path), signal.astype(np.float32))
    logger.info(f"Saved binary .npy: {out_path} ({signal.nbytes / 1024:.2f} KB)")

def export_fallback(signal: np.ndarray, path: Union[str, Path]) -> None:
    """
    Fall-back raw ASCII Text (.txt) or CSV single-column dump designed to load easily 
    in legacy ecosystems incapable of mounting Python's specific .npy protocol.
    
    Args:
        signal (np.ndarray): Output target sequence array.
        path (Path|str): Fall-back plain-text layout (.txt or .csv extending route).
    """
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    
    np.savetxt(str(out_path), signal, fmt='%.6f', delimiter=',', header='Signal_pA')
    logger.info(f"Fallback plain-text file registered: {out_path}")

# Rapid CLI Component Inspection Boot
if __name__ == "__main__":
    np.random.seed(42)
    t = np.linspace(0, 10, 500)
    true_sig = np.sin(t)
    noisy = true_sig + np.random.normal(0, 0.3, len(t))
    
    # Distorted fake 'quantum' test response block
    fake_quant = true_sig + np.random.normal(0, 0.05, len(t))
    
    sg_baseline = savitzky_golay_baseline(noisy)
    
    snr_noisy = compute_snr(noisy, true_sig)
    snr_sg = compute_snr(sg_baseline, true_sig)
    snr_quant = compute_snr(fake_quant, true_sig)
    
    print(f"Raw Signal SNR: {snr_noisy:.2f} dB")
    print(f"Classical SG SNR : {snr_sg:.2f} dB")
    print(f"Quantum SNR   : {snr_quant:.2f} dB")
    
    output_dir = Path("outputs")
    plot_comparison(noisy, sg_baseline, fake_quant, save_path=output_dir / "test_comparison.png")
    export_npy(fake_quant, output_dir / "test_clean.npy")
    export_fallback(fake_quant, output_dir / "test_clean.txt")
    print("Mock inspection passed.")
