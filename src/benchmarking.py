"""
Modül 6: Kıyaslama ve Çıktı Üretimi (Benchmarking & Output)
===========================================================
Geliştirilen Quantum-Inspired (SCSA + PnP-ADMM) filtrelerin 
klasik filtrelerle (örn. Savitzky-Golay) kantitatif kıyaslamasını
yapar. Sinyali (clean) diske dışa aktarır (.npy / .txt).

Yazar: DeepTech Pipeline
Tarih: 2026
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Union

import numpy as np
from scipy.signal import savgol_filter
import matplotlib
import matplotlib.pyplot as plt

# GUI yalıtımı (Headless CI/CD testleri için)
matplotlib.use("Agg")

logger = logging.getLogger("benchmarking")

def compute_snr(signal: np.ndarray, ground_truth: np.ndarray) -> float:
    """
    Sinyal-Gürültü Oranını (SNR) Desibel (dB) cinsinden hesaplar.
    
    SNR = 10 * log10( P_signal / P_noise )
    Daha yüksek SNR, sinyalin ground truth'a daha yakın olduğunu gösterir.
    
    Args:
        signal (np.ndarray): Değerlendirilecek / Filtrelenmiş sinyal.
        ground_truth (np.ndarray): Hatasız referans (True) sinyal.
        
    Returns:
        float: dB cinsinden SNR skoru.
    """
    assert signal.shape == ground_truth.shape, "Sinyal boyutları eşit olmalıdır."
    
    # Sinyal Gücü (Power of Signal)
    p_signal = np.mean(ground_truth ** 2)
    
    # Gürültü Gücü (Mean Squared Error)
    noise = signal - ground_truth
    p_noise = np.mean(noise ** 2)
    
    if p_noise == 0:
        return float('inf')
        
    snr_db = 10 * np.log10(p_signal / p_noise)
    return float(snr_db)

def savitzky_golay_baseline(signal: np.ndarray, window_length: int = 11, polyorder: int = 3) -> np.ndarray:
    """
    Savitzky-Golay filtresini klasik bir referans (Baseline) olarak uygular.
    Standart biyoinformatik ardışık veri düzeltme/yumuşatma algoritmasıdır.
    
    Args:
        signal (np.ndarray): Gürültülü 1D sinyal.
        window_length (int): Filtre pencere boyutu (tek sayı, sinyalden küçük olmalı).
        polyorder (int): Polinom derecesi.
        
    Returns:
        np.ndarray: Klasik filtreleme işlemi görmüş sinyal.
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
    Ham (Noisy), Klasik (Savitzky-Golay) ve Kuantum (SCSA/ADMM) sinyallerinin
    yan yana kıyaslanabileceği 3'lü matplotlib figürü oluşturur.
    
    Args:
        raw (np.ndarray): Orijinal gürültülü dizilim.
        classic (np.ndarray): Klasik algoritma (Savitzky-Golay vs.) çıktısı.
        quantum (np.ndarray): Quantum-inspired pipeline çıktısı.
        save_path (Path|str|None): Figürün kaydedileceği yol.
        show (bool): Ekranda gösterilsin mi? (GUI gerektirir)
    """
    fig, axes = plt.subplots(3, 1, figsize=(12, 8), sharex=True, sharey=True)
    
    colors = ['#d62728', '#1f77b4', '#2ca02c'] # Kırmızı, Mavi, Yeşil
    titles = ["Ham (Noisy) Sinyal", "Klasik Filtre (Savitzky-Golay)", "Kuantum Esintili Pipeline (SCSA + PnP-ADMM)"]
    signals = [raw, classic, quantum]
    
    for i, (sig, ax, c, t) in enumerate(zip(signals, axes, colors, titles)):
        ax.plot(sig, color=c, alpha=0.8)
        ax.set_title(t, fontweight='bold')
        ax.grid(True, alpha=0.3)
        if i == 2:
            ax.set_xlabel("Zaman / Örnek (Sample)", fontweight='bold')
    
    plt.tight_layout()
    
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Kıyaslama grafiği kaydedildi: {save_path}")
        
    if show:
        try:
            plt.show()
        except Exception as e:
            logger.warning(f"plot_comparison -> show() başarısız (Headless ortam?): {e}")
    else:
        plt.close(fig)

def export_npy(signal: np.ndarray, path: Union[str, Path]) -> None:
    """
    Filtrelenmiş numpy dizisini ikili (.npy) formatında diske kaydeder.
    
    Args:
        signal (np.ndarray): Kaydedilecek 1D dizi.
        path (Path|str): Hedef dosya yolu (.npy uzantılı).
    """
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Güvenlik için astype(float32) ile boyut optimize edilir
    np.save(str(out_path), signal.astype(np.float32))
    logger.info(f".npy olarak kaydedildi: {out_path} ({signal.nbytes / 1024:.2f} KB)")

def export_fallback(signal: np.ndarray, path: Union[str, Path]) -> None:
    """
    .npy okuyamayan eski sistemler için raw ASCII Text (.txt)
    veya CSV tabanlı tek sütunluk geri dönüş (fallback) aktarımı.
    
    Args:
        signal (np.ndarray): Kaydedilecek 1D dizi.
        path (Path|str): Hedef dosya yolu (.txt veya .csv uzantılı).
    """
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    
    np.savetxt(str(out_path), signal, fmt='%.6f', delimiter=',', header='Signal_pA')
    logger.info(f"Fallback metin dosyası kaydedildi: {out_path}")

# Hızlı CLI Modülü Testi
if __name__ == "__main__":
    np.random.seed(42)
    t = np.linspace(0, 10, 500)
    true_sig = np.sin(t)
    noisy = true_sig + np.random.normal(0, 0.3, len(t))
    
    # Sahte bir "quantum" filtresi çıktısı
    fake_quant = true_sig + np.random.normal(0, 0.05, len(t))
    
    sg_baseline = savitzky_golay_baseline(noisy)
    
    snr_noisy = compute_snr(noisy, true_sig)
    snr_sg = compute_snr(sg_baseline, true_sig)
    snr_quant = compute_snr(fake_quant, true_sig)
    
    print(f"Ham Sinyal SNR: {snr_noisy:.2f} dB")
    print(f"Klasik SG SNR : {snr_sg:.2f} dB")
    print(f"Kuantum SNR   : {snr_quant:.2f} dB")
    
    output_dir = Path("outputs")
    plot_comparison(noisy, sg_baseline, fake_quant, save_path=output_dir / "test_comparison.png")
    export_npy(fake_quant, output_dir / "test_clean.npy")
    export_fallback(fake_quant, output_dir / "test_clean.txt")
    print("Test tamamlandı.")
