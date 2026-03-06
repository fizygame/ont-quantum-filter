"""
Modül 2: Kuantum Esintili Gürültü Filtresi (SCSA)
=================================================
Semi-Classical Signal Analysis (SCSA) prensiplerini
kullanarak 1 boyutlu nanopore sinyalindeki mikroskobik
gürültüyü (Poisson/Anderson localization) filtreler.

Kuantum mekaniğindeki ayrık Schrödinger denklemi 
kullanılarak sinyal bir potansiyel kuyusu olarak modellenir.

Yazar: DeepTech Pipeline
Tarih: 2026
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
    SCSA (Semi-Classical Signal Analysis) ayrık spektral gürültü filtresi.
    
    Attributes:
        h (float): Yarı-klasik Planck sabiti benzeri heuristik parametre.
                   Bu değer kinetik enerjiyi dengeler.
        n_components (int): Hesaplanacak maksimum özdeğer (eigenvalue) sayısı.
                            Büyük sinyallerde performansı artırmak için sınırlandırılır.
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
        N x N boyutunda tridiagonal Hamiltonian (H) matrisini oluşturur.
        
        Formül:
            H(i, i)   = x(i) + 2/h^2
            H(i, i±1) = -1/h^2
            
        Args:
            signal (np.ndarray): 1D sinyal dizisi (V potansiyel kuyusu).
            
        Returns:
            scipy.sparse.spmatrix: CSC formatında H matrisi.
        """
        n = len(signal)
        h = self.h
        
        assert h is not None and h > 0, "h parametresi sıfırdan büyük olmalıdır."

        # Diagonal: x(i) + 2/h^2
        main_diag = signal + (2.0 / (h ** 2))
        
        # Off-diagonal: -1/h^2
        off_diag = np.full(n - 1, -1.0 / (h ** 2), dtype=np.float64)
        
        # Sparse matris oluşturma
        H = sp.diags([off_diag, main_diag, off_diag], [-1, 0, 1], format='csc')
        
        assert H.shape == (n, n), f"Beklenen shape: ({n}, {n}), alınan: {H.shape}"
        return H

    def fit_transform(self, signal: np.ndarray) -> np.ndarray:
        """
        Sinyale SCSA filtresi uygular.
        
        Boyut uyuşmazlığını önlemek için np.ndarray tip ve boyut
        doğrulamaları (assert) içerir.
        
        Args:
            signal (np.ndarray): 1D giriş sinyali. Sinyalin bir kısmı (veya 
                                 tamamı) negatif olmalıdır (örn. Z-score).
                                 
        Returns:
            np.ndarray: Filtrelenmiş sinyal (yeniden yapılandırılmış).
        """
        assert isinstance(signal, np.ndarray), "Giriş sinyali numpy dizisi olmalıdır."
        assert signal.ndim == 1, f"1D sinyal bekleniyor, {signal.ndim}D alındı."
        
        n = len(signal)
        if n < 3:
            return signal.copy()
            
        # h değeri atanmamışsa heuristik başlatma (sinyal varyansının karekökü)
        # Sinyal zaten standardize edilmişse (Modül 1) varyans 1'dir.
        if self.h is None:
            var = np.var(signal)
            self.h = np.sqrt(var) if var > 1e-6 else 1.0
            logger.debug("Heuristik h hesaplandı: %.4f", self.h)
            
        H = self._build_hamiltonian(signal)
        
        # Scipy eigsh 'k' parametresi matris boyutundan (n-1) küçük olmalıdır.
        k_eig = min(self.n_components if self.n_components else 50, n - 2)
        k_eig = max(1, k_eig)
        
        try:
            # Sadece en küçük özdeğerleri (which='SA') bul
            # SA = Smallest Algebraic (negatif olanlar dahil en küçük)
            evals, evecs = eigsh(H, k=k_eig, which='SA', tol=1e-4)
        except Exception as e:
            logger.warning("eigsh çözümü başarısız: %s. Orijinal sinyal döndürülüyor.", e)
            return signal.copy()
            
        # Negatif özdeğerleri filtrele (bağlı kuantum durumları)
        neg_mask = evals < 0
        neg_evals = evals[neg_mask]
        neg_evecs = evecs[:, neg_mask]
        
        self.eigenvalues_ = neg_evals
        self.eigenfunctions_ = neg_evecs
        
        assert neg_evecs.shape[0] == n, (
            f"Özfonksiyon boyutu {neg_evecs.shape[0]} sinyal boyutuyla {n} eşleşmiyor."
        )
        
        if len(neg_evals) == 0:
            logger.debug("SCSA: Negatif özdeğer (bağlı durum) bulunamadı.")
            return signal.copy()
            
        # Sinyalin Yeniden Yapılandırılması (Reconstruction)
        # SCSA formülü: y_rec ~ sum( kappa_k * psi_k(x)^2 )
        # kappa_k = sqrt(-E_k)
        kappa = np.sqrt(-neg_evals)
        
        reconstructed = np.zeros(n, dtype=np.float64)
        for i in range(len(neg_evals)):
            # psi_k'nın karesini yoğunluk (probability density) olarak ekle
            reconstructed += kappa[i] * (neg_evecs[:, i] ** 2)
            
        # Reconstructed sinyalin genliği h ve N'e bağımlıdır.
        # Bu nedenle, özelliklerin (shape) korunduğu çıktı sinyalini
        # girdi sinyalinin genlik yelpazesine (Min-Max) geri oturtuyoruz.
        ptp_rec = reconstructed.max() - reconstructed.min()
        if ptp_rec > 1e-12:
            reconstructed = (reconstructed - reconstructed.min()) / ptp_rec
            
            # Gerekli Min-Max geri dönüşümü (orijinal sinyalin min/max uyumu)
            ptp_sig = signal.max() - signal.min()
            reconstructed = (reconstructed * ptp_sig) + signal.min()
            
        return reconstructed

# CLI Test
if __name__ == "__main__":
    import matplotlib.pyplot as plt
    
    # Basit bir test sinyali (Gürültülü Sinüs)
    np.random.seed(42)
    x = np.linspace(0, 4*np.pi, 500)
    clean = np.sin(x) - 0.5 # Negatif bölgeleri olsun
    noisy = clean + np.random.normal(0, 0.3, size=len(x))
    
    scsa = SCSAFilter(h=0.5, n_components=30)
    filtered = scsa.fit_transform(noisy)
    
    print(f"Bulan E_neg sayısı: {len(scsa.eigenvalues_)}")
    
    plt.plot(x, noisy, label="Noisy", alpha=0.5)
    plt.plot(x, clean, label="Clean (True)", linestyle='--')
    plt.plot(x, filtered, label="SCSA Filtered")
    plt.legend()
    plt.title("SCSA Filter Demo")
    plt.show()
