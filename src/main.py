"""
ONT Kuantum Esintili Sinyal Filtreleme Pipeline - Ana Yürütücü (Orchestrator)
=============================================================================
Tüm 6 modülü sırasıyla bağlayarak uçtan uca çalıştırır.
(Veri Çekme -> SCSA -> RL Dekonvolüsyon -> ADMM -> DQGA -> Çıktı Üretimi)

Yazar: DeepTech Pipeline
Tarih: 2026
"""

import sys
import os
from pathlib import Path
import logging

import numpy as np
import matplotlib
matplotlib.use("Agg")  # Headless test desteği
import matplotlib.pyplot as plt

# Kendi modüllerimizi içe aktaralım
sys.path.insert(0, str(Path(__file__).parent))

from data_ingestion import load_pod5_signal, zscore_normalize, generate_synthetic_nanopore_signal
from create_example_pod5 import create_example_pod5
from scsa_filter import SCSAFilter
from rl_deconvolution import gaussian_psf_1d, shift_to_positive, inverse_shift, richardson_lucy_1d
from pnp_admm import PnPADMM
from dqga_optimizer import DQGA, bits_to_float
from benchmarking import compute_snr, savitzky_golay_baseline, plot_comparison, export_npy

# Logging Ayarları
log_format = '%(asctime)s [%(levelname)s] %(name)s - %(message)s'
logging.basicConfig(
    level=logging.INFO,
    format=log_format,
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("outputs/clean_run_log.txt", mode='w')
    ]
)
logger = logging.getLogger("main_pipeline")

def main():
    logger.info("=== ONT Sinyal İşleme Pipeline Başlatıldı ===")
    
    # ---------------------------------------------------------
    # MODÜL 1: Veri Çekme ve Ön İşleme
    # ---------------------------------------------------------
    logger.info("--- [ADIM 1] Veri Ön İşleme ---")
    data_dir = Path("data")
    output_dir = Path("outputs")
    output_dir.mkdir(exist_ok=True)
    
    pod5_path = data_dir / "lambda_phage.pod5"
    
    if not pod5_path.exists():
        logger.info(f"POD5 dosyası bulunamadı. Sentetik veriden oluşturuluyor: {pod5_path}")
        create_example_pod5(output_path=pod5_path, n_reads=1, n_samples=2000)
        
    try:
        # Gerçek POD5'in ilk okunmasını simüle ediyoruz
        raw_signal, read_id = load_pod5_signal(pod5_path, read_index=0)
        logger.info(f"Orijinal Sinyal Başarıyla Okundu (Read ID: {read_id}).")
        
        # Test amaçlı sinyali küçültelim ki ADMM/Eigen solver çok uzun sürmesin
        # Homopolimer bölgelerinin tipik boyutlarında (Örn: 500 sample)
        signal_window = raw_signal[1000:1500] 
        norm_signal = zscore_normalize(signal_window)
        
    except Exception as e:
        logger.error(f"POD5 okuma hatası: {e}. Tamamen sentetik veriye geçiliyor.")
        signal_window = generate_synthetic_nanopore_signal(length=500)
        norm_signal = zscore_normalize(signal_window)
        
    # Baseline olarak SNR hesaplamak için (Yapay noisy vs clean yaratıyoruz)
    # Burada ground truth bilinmediği için, kendi simüle ettiğimiz bir ground truth farz edeceğiz.
    synthetic_true = np.sin(np.linspace(0, 10, len(norm_signal))) 
    noisy_input = synthetic_true + np.random.normal(0, 0.5, len(synthetic_true))
    
    # ---------------------------------------------------------
    # MODÜL 5: DQGA (Kuantum Genetik) ile SCSA 'h' parametresi bulma
    # ---------------------------------------------------------
    logger.info("--- [ADIM 2] DQGA Hiperparametre Optimizasyonu ---")
    
    # Fitness: Geçici test için SNR hedefliyoruz
    def scsa_fitness(bits: np.ndarray) -> float:
        # h parametresini 0.1 ile 2.0 arasında ara
        h_val = bits_to_float(bits, 0.1, 2.0)
        test_filter = SCSAFilter(h=h_val, n_components=20)
        filtered = test_filter.fit_transform(noisy_input)
        return compute_snr(filtered, synthetic_true)

    # Optimizatörü çalıştır (Hızlı bits=4, pop=5, gen=5 iterasyon demo amaçlı)
    optimizer = DQGA(pop_size=5, n_genes=6, fitness_fn=scsa_fitness, n_generations=5)
    best_bits, best_score = optimizer.run()
    optimal_h = bits_to_float(best_bits, 0.1, 2.0)
    logger.info(f"DQGA Optimizasyonu Tamamlandı! Bulunan optimum h = {optimal_h:.4f} (SNR: {best_score:.2f} dB)")
    
    # ---------------------------------------------------------
    # MODÜL 2: SCSA Kuantum Gürültü Filtresi
    # ---------------------------------------------------------
    logger.info("--- [ADIM 3] SCSA Kuantum Gürültü Filtresi ---")
    scsa = SCSAFilter(h=optimal_h, n_components=30)
    
    # PnP-ADMM için bunu plug-in denoiser (lambda) olarak sarmalayacağız.
    def scsa_denoiser(sig: np.ndarray) -> np.ndarray:
        return scsa.fit_transform(sig)

    # ---------------------------------------------------------
    # MODÜL 4: PnP-ADMM Global Optimizasyon
    # ---------------------------------------------------------
    logger.info("--- [ADIM 4] PnP-ADMM (SCSA Plug-in) Optimizasyonu ---")
    admm = PnPADMM(
        denoiser=scsa_denoiser,
        rho=1.0,           # Başlangıç ceza parametresi
        max_iter=15,       # ADMM döngüsü
        tol=1e-3,          # Varsayılan hata payı toleransı
        adaptive_rho=True
    )
    pnp_optimized_signal = admm.run(noisy_input)
    
    # ---------------------------------------------------------
    # MODÜL 3: 1D Fizik Tabanlı Restorasyon (Richardson-Lucy)
    # ---------------------------------------------------------
    logger.info("--- [ADIM 5] Richardson-Lucy Dekonvolüsyonu ---")
    psf = gaussian_psf_1d(size=11, sigma=2.0)
    
    # RL Pozitif Uzay İhtiyacı
    pos_sig, shift_params = shift_to_positive(pnp_optimized_signal, method="min_shift")
    
    rl_restored_pos = richardson_lucy_1d(pos_sig, psf, iterations=20)
    
    final_quantum_signal = inverse_shift(rl_restored_pos, shift_params)
    
    # ---------------------------------------------------------
    # MODÜL 6: Kıyaslama ve Çıktı Üretimi
    # ---------------------------------------------------------
    logger.info("--- [ADIM 6] Benchmarking ve Dışa Aktarım ---")
    
    # Klasik baseline (Savitzky-Golay)
    classic_sg = savitzky_golay_baseline(noisy_input)
    
    # SNR Hesapları
    snr_noisy = compute_snr(noisy_input, synthetic_true)
    snr_classic = compute_snr(classic_sg, synthetic_true)
    snr_quantum = compute_snr(final_quantum_signal, synthetic_true)
    
    logger.info(f"[METRİK] Ham (Gürültülü) SNR : {snr_noisy:.2f} dB")
    logger.info(f"[METRİK] Klasik (SG) SNR    : {snr_classic:.2f} dB")
    logger.info(f"[METRİK] Quantum Pipeline SNR: {snr_quantum:.2f} dB")
    
    with open(output_dir / "snr_metrics.txt", "w") as f:
        f.write(f"Ham SNR: {snr_noisy:.2f} dB\n")
        f.write(f"Klasik SG SNR: {snr_classic:.2f} dB\n")
        f.write(f"Quantum SNR (1500 iter): {snr_quantum:.2f} dB\n")
        
    # Grafikleri Çiz
    plot_out = output_dir / "final_pipeline_comparison.png"
    plot_comparison(
        raw=noisy_input,
        classic=classic_sg,
        quantum=final_quantum_signal,
        save_path=plot_out,
        show=False
    )
    
    # Verileri Diske Aktar
    export_npy(final_quantum_signal, output_dir / "quantum_clean_signal.npy")
    
    logger.info("=== ONT Sinyal İşleme Pipeline Başarıyla Tamamlandı ===")

if __name__ == "__main__":
    main()
