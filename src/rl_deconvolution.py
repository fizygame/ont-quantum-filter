"""
Modül 3: 1D Fizik Tabanlı Restorasyon (Richardson-Lucy & BID)
=============================================================
Homopolimer bölgelerinde bant genişliği limitleri nedeniyle
düzleşmiş veya birleşmiş ONT nanopore sinyallerinin çözünürlüğünü 
artırmak için 1 Boyutlu Richardson-Lucy dekonvolüsyon algoritması.

Yazar: DeepTech Pipeline
Tarih: 2026
"""

from __future__ import annotations

import logging

import numpy as np
from scipy.signal import fftconvolve

logger = logging.getLogger("rl_deconvolution")

def gaussian_psf_1d(size: int, sigma: float) -> np.ndarray:
    """
    1 boyutlu Gauss Nokta Yayılım Fonksiyonu (Point Spread Function - PSF) oluşturur.
    
    Args:
        size (int): Filtre boyutu (tek sayı olması convolution merkezlemesi için idealdir).
        sigma (float): Gauss penceresinin standart sapması (yayılım genişliği).
        
    Returns:
        np.ndarray: Toplamı 1'e normalize edilmiş 1D Gauss filtresi.
        
    Example:
        >>> psf = gaussian_psf_1d(11, 2.0)
        >>> assert np.isclose(psf.sum(), 1.0)
    """
    assert size > 0, "PSF boyutu 0'dan büyük olmalıdır."
    assert sigma > 0, "Sigma 0'dan büyük olmalıdır."
    
    # 0 merkezli x ekseni oluştur 
    # örn. size=5 -> [-2, -1, 0, 1, 2]
    x = np.arange(-size // 2 + 1 if size % 2 == 0 else -(size // 2), size // 2 + 1)
    
    psf = np.exp(-(x ** 2) / (2 * sigma ** 2))
    
    # L1 normalizasyonu (Görüntü/Sinyal enerjisinin korunması için)
    psf /= psf.sum()
    
    return psf

def shift_to_positive(
    signal: np.ndarray, 
    offset: float = 1e-6, 
    method: str = "min_shift"
) -> tuple[np.ndarray, dict]:
    """
    Sinyali Richardson-Lucy dekonvolüsyonunun çalışabileceği pozitif uzaya dönüştürür.
    RL algoritması matematiksel olarak negatif değerlerle çalışamaz.
    
    Args:
        signal (np.ndarray): Orijinal sinyal (negatif değerler içerebilir).
        offset (float): Sıfıra çarpmanın/bölünmenin engellenmesi için eklenen tampon değer.
        method (str): Dönüşüm metodu ("min_shift" veya "minmax").
        
    Returns:
        tuple[np.ndarray, dict]: 
            - Pozitif uzaya kaydırılmış sinyal
            - Geri dönüşüm (inverse) işlemi için gerekli parametreleri içeren sözlük
            
    Raises:
        ValueError: Bilinmeyen bir method girilirse.
    """
    signal = np.asarray(signal, dtype=np.float64)
    params = {"method": method, "offset": offset}
    
    if method == "min_shift":
        sig_min = signal.min()
        shift_val = abs(sig_min) + offset if sig_min <= 0 else 0.0
        pos_signal = signal + shift_val
        params["shift_val"] = shift_val
        
    elif method == "minmax":
        sig_min = signal.min()
        sig_max = signal.max()
        ptp = sig_max - sig_min
        if ptp < 1e-12:
            ptp = 1.0
        pos_signal = (signal - sig_min) / ptp + offset
        params["min"] = sig_min
        params["ptp"] = ptp
        
    else:
        raise ValueError(f"Geçersiz metod: {method}. 'min_shift' veya 'minmax' kullanın.")
        
    return pos_signal, params

def inverse_shift(signal: np.ndarray, params: dict) -> np.ndarray:
    """
    `shift_to_positive` fonksiyonu ile pozitif uzaya geçirilen sinyali
    orijinal uzayına (genlik aralığına) geri döndürür.
    
    Args:
        signal (np.ndarray): İşlenmiş, pozitif sinyal.
        params (dict): `shift_to_positive` fonksiyonundan dönen parametre sözlüğü.
        
    Returns:
        np.ndarray: Orijinal fiziksel uzaydaki sinyal.
    """
    method = params.get("method")
    offset = params.get("offset", 0.0)
    
    if method == "min_shift":
        shift_val = params["shift_val"]
        return signal - shift_val
        
    elif method == "minmax":
        sig_min = params["min"]
        ptp = params["ptp"]
        return (signal - offset) * ptp + sig_min
        
    else:
        raise ValueError(f"Geçersiz dönüşüm yöntem parametreleri: {method}")

def richardson_lucy_1d(
    signal: np.ndarray,
    psf: np.ndarray,
    iterations: int = 50,
    epsilon: float = 1e-12
) -> np.ndarray:
    """
    1D sinyaller için Vektörel Richardson-Lucy (RL) Dekonvolüsyonu.
    
    İteratif formül:
        I_{k+1} = I_k * conv(PSF_flipped, I_o / (conv(I_k, PSF) + eps))
        
    (PSF simetrik Gaussian olduğu varsayımıyla PSF_flipped = PSF alınır).
    
    Dikkat: Giriş sinyali strictly positive (kesinlikle pozitif) olmalıdır. 
    Aksi halde `shift_to_positive` kullanılmalıdır.
    
    Args:
        signal (np.ndarray): Pozitif 1D gözlem sinyali (I_o).
        psf (np.ndarray): 1D Nokta yayılım fonksiyonu (Point Spread Function).
        iterations (int): Maksimum RL döngüsü (iterasyon sayısı).
        epsilon (float): Sıfıra bölünme yalıtım parametresi.
        
    Returns:
        np.ndarray: Çözünürlüğü artırılmış (deconvolved) 1D sinyal (I_k).
        
    Raises:
        ValueError: Sinyalde negatif veya sıfır değer varsa.
    """
    signal = np.asarray(signal, dtype=np.float64)
    psf = np.asarray(psf, dtype=np.float64)
    
    if np.any(signal < 0):
        raise ValueError(
            "Richardson-Lucy algoritması tanımsız matematiksek hataları "
            "önlemek için negatif girişlerle çalışamaz. "
            "Önce shift_to_positive() fonksiyonunu kullanın."
        )
        
    # İterasyon için başlangıç tahmini (genellikle gözlemin kendisi)
    estimate = signal.copy()
    
    # Simetrik bir PSF kullanıldığını varsayıyoruz.
    # Dacă asimetrik olsaydı: psf_flipped = psf[::-1]
    psf_flipped = psf[::-1]
    
    # Vektörel RL İterasyon Döngüsü
    for _ in range(iterations):
        # 1. Tahmini sinyalin sistem yanıtı
        # mode='same' çıkarımın girişle aynı boyutta olmasını sağlar
        conv_est = fftconvolve(estimate, psf, mode='same')
        
        # 2. Oran: Gerçek gözlem / Tahmini gözlem
        # Epsilon, division-by-zero hatalarını engeller
        ratio = signal / (conv_est + epsilon)
        
        # 3. Hata geri yayılımı (Error back-projection)
        error_proj = fftconvolve(ratio, psf_flipped, mode='same')
        
        # 4. Tahmini güncelle (Multiplicative Update Rule)
        estimate *= error_proj
        
        # Fiziksel zorlama: Işıma/Akım pozitif olmalıdır
        # Çok küçük sayılara (underflow) karşı koruma
        estimate = np.clip(estimate, epsilon, None)
        
    return estimate

# CLI Demo
if __name__ == "__main__":
    import matplotlib.pyplot as plt
    
    # 1. Hedef Sinyal Üret (Keskin tepe noktaları - ideal homopolimer geçişleri)
    np.random.seed(42)
    x = np.arange(200)
    true_signal = np.zeros_like(x, dtype=float)
    true_signal[40:50] = 50.0  # Keskin basamak
    true_signal[120:125] = 40.0 # Daha kısa keskin basamak
    
    # 2. Bulanıklaştırıcı Etken (ONT Sensor/Pore bant genişliği kısıtlaması simülasyonu)
    psf = gaussian_psf_1d(size=15, sigma=3.0)
    
    # 3. Bulanık Sinyal Oluştur
    blurred_signal = fftconvolve(true_signal, psf, mode='same')
    noisy_blurred = blurred_signal + np.random.normal(0, 1.0, len(x))
    
    # 4. Richardson-Lucy Onarımı
    # Öncelikle pozitif uzaya taşıyalım (Gürültü negatife inmiş olabilir)
    pos_noisy, params = shift_to_positive(noisy_blurred, method="min_shift")
    
    # RL iterasyonu
    restored_pos = richardson_lucy_1d(pos_noisy, psf, iterations=40)
    
    # Orijinal uzaya geri dön
    restored = inverse_shift(restored_pos, params)
    
    plt.figure(figsize=(10, 5))
    plt.plot(x, true_signal, 'k--', label="Orijinal Keskin (Bilinmeyen)", alpha=0.6)
    plt.plot(x, noisy_blurred, 'r-', label="ONT Bulanık + Gürültü", alpha=0.7)
    plt.plot(x, restored, 'g-', label="RL Dekonvolüsyon (Onarılmış)", linewidth=2)
    plt.title("Richardson-Lucy 1D Dekonvolüsyon Demosu")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()
