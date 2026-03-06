"""
Modül 5: Ayrık Kuantum Genetik Algoritma (DQGA) Optimizatörü
============================================================
Kuantum hesaplama prensiplerinden esinlenerek (qubit genlikleri ve 
döndürme kapıları) hiperparametre optimizasyonu yapar. 
Bu modül PnP-ADMM ve SCSA filtrelerindeki kritik hiperparametrelerin 
(h, rho vb.) en uygun değerlerini otomatik bulmak için tasarlanmıştır.

Yazar: DeepTech Pipeline
Tarih: 2026
"""

from __future__ import annotations

import logging
from typing import Callable, Tuple, List, Optional

import numpy as np

logger = logging.getLogger("dqga_optimizer")

class QuantumChromosome:
    """
    Qubit dizisini (Quantum Chromosome) temsil eden veri yapısı.
    Her gen (qubit) α ve β olasılık genliklerine (probability amplitudes) sahiptir.
    Matematiksel kural: |α|^2 + |β|^2 = 1.
    """
    def __init__(self, n_genes: int) -> None:
        self.n_genes = n_genes
        # Başlangıçta hepsi süperpozisyonda: alpha = 1/sqrt(2), beta = 1/sqrt(2)
        base_val = 1.0 / np.sqrt(2.0)
        self.alpha = np.full(n_genes, base_val, dtype=np.float64)
        self.beta = np.full(n_genes, base_val, dtype=np.float64)
        
        # Olasılık ölçümü (çökme - collapse) sonrasındaki klasik bit dizisi
        self.binary_string: Optional[np.ndarray] = None
        self.fitness: float = -np.inf

    def measure(self) -> np.ndarray:
        """
        Qubitleri klasik bitlere (0 veya 1) çöktürür (Wave-function collapse).
        1 gelme olasılığı: |β|^2
        0 gelme olasılığı: |α|^2
        """
        probabilities_of_1 = self.beta ** 2
        # np.random.rand kullanarak olasılık simülasyonu
        random_vals = np.random.rand(self.n_genes)
        self.binary_string = (random_vals <= probabilities_of_1).astype(np.int8)
        return self.binary_string

    def _normalize(self) -> None:
        """Kayan nokta (float) hataları sonrası |α|² + |β|² = 1 koşulunu garanti eder."""
        norm = np.sqrt(self.alpha**2 + self.beta**2)
        # Sıfıra bölünme yalıtımı
        norm[norm == 0] = 1.0
        self.alpha /= norm
        self.beta /= norm

class QuantumGate:
    """Kuantum kapısı operasyonlarını barındırır."""
    
    @staticmethod
    def rotate(chromosome: QuantumChromosome, best_chromosome: QuantumChromosome, theta: float = 0.05 * np.pi) -> None:
        """
        Kromozomu, popülasyondaki en iyi bireye doğru evrimleştiren 
        Kuantum Döndürme Kapısı (Quantum Rotation Gate).
        
        Args:
            chromosome: Güncellenecek hedef kromozom
            best_chromosome: Popülasyonun en iyi fitnesli kromozomu
            theta: Rotasyon açısı büyüklüğü (radyan)
        """
        # Hangi genlerin dönüşüme uğrayacağı, bireylerin klasik bit karşılaştırmalarına bağlıdır
        if chromosome.binary_string is None or best_chromosome.binary_string is None:
            raise ValueError("Kromozom ölçülmemiş (measure metodu çağrılmamış)!")
            
        x = chromosome.binary_string
        b = best_chromosome.binary_string
        
        delta_theta = np.zeros(chromosome.n_genes, dtype=np.float64)
        
        # Algoritma kuralı: Mevcut bit '0' ve en iyi bit '1' ise pozitife döndür
        #                  Mevcut bit '1' ve en iyi bit '0' ise negatife döndür
        #                  Aksi halde açıyı değiştirme (süperpozisyonu koru)
        mask_01 = (x == 0) & (b == 1)
        mask_10 = (x == 1) & (b == 0)
        
        delta_theta[mask_01] = theta
        delta_theta[mask_10] = -theta
        
        # 2x2 Unitary Rotasyon Matrisi Çarpımı Vektörize
        # [ alpha' ] = [ cos(th) -sin(th) ] [ alpha ]
        # [ beta'  ]   [ sin(th)  cos(th) ] [ beta  ]
        cos_t = np.cos(delta_theta)
        sin_t = np.sin(delta_theta)
        
        new_alpha = cos_t * chromosome.alpha - sin_t * chromosome.beta
        new_beta  = sin_t * chromosome.alpha + cos_t * chromosome.beta
        
        chromosome.alpha = new_alpha
        chromosome.beta = new_beta
        chromosome._normalize()

class DQGA:
    """
    Kuantum Genetik Optimizatör.
    
    Attributes:
        pop_size (int): Popülasyondaki birey sayısı.
        n_genes (int): Her kromozomdaki qubit sayısı.
        fitness_fn (Callable): 1D ikili dizi (np.ndarray) alıp 
                               uygunluk skoru (float) döndüren fonksiyon.
        n_generations (int): Evrim döngüsü sayısı.
    """
    
    def __init__(
        self,
        pop_size: int,
        n_genes: int,
        fitness_fn: Callable[[np.ndarray], float],
        n_generations: int = 50,
        theta: float = 0.05 * np.pi
    ) -> None:
        assert pop_size > 0
        assert n_genes > 0
        
        self.pop_size = pop_size
        self.n_genes = n_genes
        self.fitness_fn = fitness_fn
        self.n_generations = n_generations
        self.theta = theta
        
        self.population: List[QuantumChromosome] = [QuantumChromosome(n_genes) for _ in range(pop_size)]
        self.global_best_chromosome: Optional[QuantumChromosome] = None
        
        # Geçmiş kayıtları
        self.history_best_fitness: List[float] = []

    def _evaluate_population(self) -> None:
        """Tüm popülasyonu ölçer ve fitness fonksiyonundan geçirir."""
        for chrom in self.population:
            # 1. Klasik bitleri elde et (Ölçüm)
            bits = chrom.measure()
            
            # 2. Fitness Skorunu al
            chrom.fitness = self.fitness_fn(bits)
            
            # Global En iyiyi güncelle
            if self.global_best_chromosome is None or chrom.fitness > self.global_best_chromosome.fitness:
                # Kopyalayarak saklıyoruz ki ileride üstüne yazılmasın
                best_copy = QuantumChromosome(self.n_genes)
                best_copy.alpha = chrom.alpha.copy()
                best_copy.beta = chrom.beta.copy()
                best_copy.binary_string = chrom.binary_string.copy()
                best_copy.fitness = chrom.fitness
                self.global_best_chromosome = best_copy

    def run(self) -> Tuple[np.ndarray, float]:
        """
        Evrimsel algoritmayı çalıştırır.
        
        Returns:
            Tuple[np.ndarray, float]: En iyi klasik bit dizisi ve en yüksek skor.
        """
        logger.info("DQGA Başlatılıyor: pop=%d, genes=%d, gen=%d", self.pop_size, self.n_genes, self.n_generations)
        
        self.history_best_fitness = []
        
        for gen in range(self.n_generations):
            # 1. Ölçüm ve Değerlendirme
            self._evaluate_population()
            
            best_fit = self.global_best_chromosome.fitness
            self.history_best_fitness.append(best_fit)
            
            # 2. Kuantum Rotasyonu (Evrim Adımı)
            for chrom in self.population:
                QuantumGate.rotate(chrom, self.global_best_chromosome, self.theta)
                
            if (gen + 1) % 10 == 0:
                logger.debug("DQGA Jenerasyon %d | Best Fitness: %.4f", gen+1, best_fit)
                
        logger.info("DQGA Tamamlandı. Max Fitness: %.4f", self.global_best_chromosome.fitness)
        return self.global_best_chromosome.binary_string, self.global_best_chromosome.fitness

# Değer Haritalama Fonksiyonları (DQGA bitlerini float aralıklara dönüştürmek için)
def bits_to_float(bits: np.ndarray, val_min: float, val_max: float) -> float:
    """
    Binary bit dizisini, verilen [val_min, val_max] aralığına doğrusal dönüştürür.
    """
    n_bits = len(bits)
    # İkili tabandan (binary) ondalık tabana (decimal) çevrim (vektörel)
    powers = 2 ** np.arange(n_bits - 1, -1, -1)
    decimal_val = np.sum(bits * powers)
    
    max_decimal = (2 ** n_bits) - 1
    if max_decimal <= 0:
        return val_min
        
    normalized = decimal_val / max_decimal
    return val_min + normalized * (val_max - val_min)
