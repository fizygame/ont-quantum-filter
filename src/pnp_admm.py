"""
Modül 4: PnP-ADMM Optimizasyonu
===============================
Plug-and-Play Alternating Direction Method of Multipliers (PnP-ADMM)
algoritması, veriye uygunluk (data fidelity) adımı ile 
herhangi bir bağımsız gürültü azaltıcıyı (denoiser, örn: SCSA)
birleştirerek global bir optimizasyon sunar.

Yazar: DeepTech Pipeline
Tarih: 2026
"""

from __future__ import annotations

import logging
from typing import Callable, Optional, Tuple, List

import numpy as np

logger = logging.getLogger("pnp_admm")

class PnPADMM:
    """
    Plug-and-Play ADMM Çerçevesi.
    
    Attributes:
        denoiser (Callable): Sinyal (np.ndarray) alıp filtrelenmiş sinyal döndüren
                             çalıştırılabilir fonksiyon (Örn: scsa.fit_transform).
        rho (float): ADMM ceza parametresi (Lagrangian multiplier).
        max_iter (int): Maksimum iterasyon sayısı.
        tol (float): Yakınsama toleransı (primal ve dual residual için limit).
        adaptive_rho (bool): İterasyonlar arası dinamik rho güncellemeleri yapılsın mı?
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
        
        # Loglama için takip geçmişi
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
        Data Fidelity (Veri Uygunluğu) Adımı (x-guncellemesi).
        Varsayım: H = I (Birim Matris), yani bozulma sadece gürültü/Additive Noise.
        
        Formül (H=I için Proximal Operator):
            x^{k+1} = (y + rho * (z^k - u^k)) / (1 + rho)
            
        Args:
            y (np.ndarray): Orijinal gürültülü gözlem sinyali.
            z (np.ndarray): ADMM z değişkeni (denoised sinyal).
            u (np.ndarray): ADMM dual değişkeni (Lagrangian çarpanı).
            rho (float): Güncel ceza parametresi.
            
        Returns:
            np.ndarray: x^{k+1} güncel sinyal
        """
        return (y + rho * (z - u)) / (1.0 + rho)

    def _denoising_step(self, x: np.ndarray, u: np.ndarray) -> np.ndarray:
        """
        Plug-in Denoising Adımı (z-güncellemesi).
        
        Sinyal hedefi: x^{k+1} + u^k
        
        Args:
            x (np.ndarray): Güncellenmiş x değişkeni.
            u (np.ndarray): Dual değişken.
            
        Returns:
            np.ndarray: Denoiser'dan geçen z^{k+1}
        """
        noisy_target = x + u
        # Plug-and-Play büyüsü: Matematiksel formülasyon yerine dış algoritma çağrısı
        return self.denoiser(noisy_target)

    def _dual_update(
        self, 
        u: np.ndarray, 
        x: np.ndarray, 
        z: np.ndarray
    ) -> np.ndarray:
        """
        Lagrangian Multiplier (Dual Değişken) Güncellemesi.
        
        Formül:
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
        Adaptif Rho Güncellemesi (Residual Dengeleme).
        ADMM'in çok yavaş veya çok hızlı yakınsamasını önler.
        """
        if primal_res > mu * dual_res:
            return rho * tau
        elif dual_res > mu * primal_res:
            return rho / tau
        return rho

    def run(self, y: np.ndarray) -> np.ndarray:
        """
        Tam PnP-ADMM döngüsünü çalıştırır.
        
        Args:
            y (np.ndarray): Optimize edilecek 1D gürültülü sinyal.
            
        Returns:
            np.ndarray: Optimize edilmiş/temizlenmiş z sinyali.
        """
        assert isinstance(y, np.ndarray) and y.ndim == 1, "Sinyal 1D numpy array olmalıdır."
        
        n = len(y)
        x = y.copy()
        z = y.copy()
        u = np.zeros(n, dtype=np.float64)
        rho = self.rho
        
        # Geçmişi sıfırla
        self.history_ = {"primal_res": [], "dual_res": [], "rho": []}
        
        logger.info("PnP-ADMM Optimizasyonu Başlatılıyor (Max Iter: %d)", self.max_iter)
        
        for k in range(self.max_iter):
            z_old = z.copy()
            
            # 1. x Güncellemesi (Data Fidelity)
            x = self._data_fidelity_step(y, z, u, rho)
            
            # 2. z Güncellemesi (Denoising by Plug-in)
            z = self._denoising_step(x, u)
            
            # 3. u Güncellemesi (Dual)
            u = self._dual_update(u, x, z)
            
            # Hatayı hesapla (Residuals)
            primal_residual = np.linalg.norm(x - z)
            dual_residual = np.linalg.norm(rho * (z - z_old))
            
            # Geçmişi kaydet
            self.history_["primal_res"].append(primal_residual)
            self.history_["dual_res"].append(dual_residual)
            self.history_["rho"].append(rho)
            
            # Erken durma (Early Convergence)
            if primal_residual < self.tol and dual_residual < self.tol:
                logger.info("PnP-ADMM %d. iterasyonda yakınsadı.", k+1)
                break
                
            # Adaptif rho güncellemesi (İsteğe bağlı)
            if self.adaptive_rho:
                rho = self._rho_update(rho, primal_residual, dual_residual)
                
        else:
            logger.info("PnP-ADMM max iterasyona (%d) ulaştı (Yakınsama Tamamlanmadı).", self.max_iter)

        return z

# CLI Demo
if __name__ == "__main__":
    import matplotlib.pyplot as plt
    
    # Basit bir denoiser oluştur (Hareketli ortalama filtresi)
    def simple_moving_average_denoiser(sig, window_size=5):
        return np.convolve(sig, np.ones(window_size)/window_size, mode='same')
        
    np.random.seed(42)
    x = np.linspace(0, 10, 300)
    ideal = np.sin(x)
    noisy_signal = ideal + np.random.normal(0, 0.4, size=len(x))
    
    # PnP-ADMM başlat
    admm = PnPADMM(
        denoiser=simple_moving_average_denoiser,
        rho=1.5,
        max_iter=100,
        tol=1e-3
    )
    
    optimized_signal = admm.run(noisy_signal)
    
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 2, 1)
    plt.plot(ideal, 'k--', label="Orijinal")
    plt.plot(noisy_signal, 'r-', alpha=0.5, label="Gürültülü")
    plt.plot(optimized_signal, 'g-', linewidth=2, label="PnP-ADMM")
    plt.legend()
    plt.title("PnP-ADMM Gerçek Zamanlı Davranış")
    
    plt.subplot(1, 2, 2)
    plt.plot(admm.history_["primal_res"], label="Primal Residual")
    plt.plot(admm.history_["dual_res"], label="Dual Residual")
    plt.yscale('log')
    plt.legend()
    plt.title("ADMM Convergence")
    
    plt.tight_layout()
    plt.show()
