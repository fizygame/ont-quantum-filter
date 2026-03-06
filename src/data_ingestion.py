"""
Modül 1: Veri Çekme ve Ön İşleme (Data Ingestion & Preprocessing)
==================================================================
Oxford Nanopore (ONT) Lambda Phage açık veri setinden ham iyonik akım
sinyallerini indirir, POD5 formatında okur, Z-score normalize eder ve
görselleştirir.

Veri Kaynağı:
    ONT Open Datasets — Lambda phage control run (POD5 format)
    https://github.com/nanoporetech/ont-open-datasets
    Direct S3: https://s3.amazonaws.com/nanopore-human-wgs/rna/referenceSamples/

Kütüphaneler: numpy, scipy, pod5, matplotlib, requests
Yazar: DeepTech Pipeline
Tarih: 2026
"""

from __future__ import annotations

import os
import time
import logging
import hashlib
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import matplotlib
matplotlib.use("Agg")  # Headless ortamlar için — GUI backend gerektirmez
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import requests
import pod5
import tempfile
import shutil
import uuid

# ---------------------------------------------------------------------------
# Sabitler
# ---------------------------------------------------------------------------
RANDOM_SEED: int = 42
np.random.seed(RANDOM_SEED)

# ONT / nanoporetech açık POD5 test verisi — doğrulanmış gerçek POD5 dosyaları
# Kaynak: https://github.com/nanoporetech/pod5-file-format (Apache-2.0, Public)
ONT_LAMBDA_POD5_URL: str = (
    "https://raw.githubusercontent.com/nanoporetech/pod5-file-format/"
    "master/python/pod5/test_data/multi_fast5_zip.pod5"
)

# Yedek 1: Diğer resmi ONT test dosyası
ONT_LAMBDA_FALLBACK_URL: str = (
    "https://raw.githubusercontent.com/nanoporetech/pod5-file-format/"
    "master/python/pod5/test_data/multi_reads.pod5"
)

# POD5 dosya imzası (magic bytes) — ilk 8 byte
# Arrow IPC/Feather formatı: b'ARROW1\x00\x00' veya b'\x41\x52\x52\x4f\x57\x31\x00\x00'
_POD5_MIN_FILE_SIZE_BYTES: int = 10_000  # Geçerli POD5 en az ~10 KB olmalı

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
)
logger = logging.getLogger("data_ingestion")


# ---------------------------------------------------------------------------
# 1. Veri İndirme
# ---------------------------------------------------------------------------

def download_ont_data(
    url: str,
    dest_path: str | Path,
    retries: int = 3,
    retry_delay: float = 2.0,
    chunk_size: int = 1024 * 1024,  # 1 MB
    fallback_url: Optional[str] = None,
) -> Path:
    """
    ONT açık veri setinden POD5 dosyasını indirir.

    Retry (yeniden deneme) mekanizması ile ağ kopmalarına karşı dayanıklıdır.
    Dosya zaten mevcutsa yeniden indirmez.

    Args:
        url (str): İndirilecek dosyanın URL'si.
        dest_path (str | Path): Kaydedilecek yerel dosya yolu.
        retries (int): Maksimum yeniden deneme sayısı. Varsayılan: 3.
        retry_delay (float): Denemeler arası bekleme süresi (saniye).
        chunk_size (int): Akış indirme için parça boyutu (bayt).
        fallback_url (Optional[str]): Ana URL başarısız olursa kullanılacak yedek URL.

    Returns:
        Path: İndirilen veya mevcut dosyanın yerel yolu.

    Raises:
        RuntimeError: Tüm denemeler ve yedek başarısız olursa.

    Example:
        >>> path = download_ont_data(ONT_LAMBDA_POD5_URL, "data/lambda.pod5")
    """
    dest_path = Path(dest_path)
    dest_path.parent.mkdir(parents=True, exist_ok=True)

    if dest_path.exists() and dest_path.stat().st_size > 0:
        logger.info("Dosya zaten mevcut, indirme atlandı: %s", dest_path)
        return dest_path

    urls_to_try = [url]
    if fallback_url:
        urls_to_try.append(fallback_url)

    for attempt_url in urls_to_try:
        logger.info("İndirme deneniyor: %s", attempt_url)
        for attempt in range(1, retries + 1):
            try:
                response = requests.get(
                    attempt_url,
                    stream=True,
                    timeout=60,
                    headers={"User-Agent": "ONT-Pipeline/1.0"},
                )
                response.raise_for_status()

                # Content-Type kontrolü — HTML/XML hata sayfası değilse indir
                content_type = response.headers.get("Content-Type", "")
                if "html" in content_type or "xml" in content_type:
                    raise IOError(
                        f"Sunucu HTML/XML döndürdü (POD5 değil): Content-Type={content_type}"
                    )

                total_size = int(response.headers.get("content-length", 0))
                downloaded = 0

                with open(dest_path, "wb") as f:
                    for chunk in response.iter_content(chunk_size=chunk_size):
                        if chunk:
                            f.write(chunk)
                            downloaded += len(chunk)

                if total_size > 0 and downloaded < total_size:
                    raise IOError(
                        f"Eksik indirme: {downloaded}/{total_size} bayt"
                    )

                # Minimum boyut kontrolü — HTML hata sayfası mı?
                if downloaded < _POD5_MIN_FILE_SIZE_BYTES:
                    dest_path.unlink(missing_ok=True)
                    raise IOError(
                        f"İndirilen dosya çok küçük ({downloaded} bayt < "
                        f"{_POD5_MIN_FILE_SIZE_BYTES} bayt). "
                        f"Muhtemelen geçersiz URL veya hata sayfası."
                    )

                # POD5/Arrow imza kontrolü (ilk 4 byte: 'ARRO')
                with open(dest_path, "rb") as f_check:
                    magic = f_check.read(4)
                if magic != b"ARRO" and magic != b"\x50\x4f\x44\x35":
                    dest_path.unlink(missing_ok=True)
                    raise IOError(
                        f"Geçersiz POD5 imzası: {magic!r}. "
                        f"Dosya POD5/Arrow formatında değil."
                    )

                logger.info(
                    "İndirme tamamlandı (%.2f MB): %s",
                    downloaded / 1024 / 1024,
                    dest_path,
                )
                return dest_path

            except (requests.RequestException, IOError) as exc:
                logger.warning(
                    "Deneme %d/%d başarısız [%s]: %s",
                    attempt, retries, attempt_url, exc,
                )
                if attempt < retries:
                    time.sleep(retry_delay * attempt)
                else:
                    logger.error("URL başarısız: %s", attempt_url)
                    if dest_path.exists():
                        dest_path.unlink(missing_ok=True)

    raise RuntimeError(
        f"Tüm URL denemeleri başarısız oldu. "
        f"Lütfen interneti ve veri kaynaklarını kontrol edin."
    )


# ---------------------------------------------------------------------------
# 2. POD5 Okuma
# ---------------------------------------------------------------------------

def load_pod5_signal(
    filepath: str | Path,
    read_id: Optional[str] = None,
    read_index: int = 0,
) -> Tuple[np.ndarray, str]:
    """
    POD5 dosyasından ham iyonik akım sinyalini (pA) okur.

    Args:
        filepath (str | Path): POD5 dosyasının yolu.
        read_id (Optional[str]): Belirli bir read UUID'si. None ise read_index kullanılır.
        read_index (int): read_id verilmezse kullanılacak okuma indeksi. Varsayılan: 0.

    Returns:
        Tuple[np.ndarray, str]:
            - signal (np.ndarray): pA cinsinden ham iyonik akım zaman serisi, shape=(N,).
            - actual_read_id (str): Okunan read'in UUID'si.

    Raises:
        FileNotFoundError: Dosya bulunamazsa.
        ValueError: Belirtilen read_id dosyada yoksa.
        IndexError: read_index dosyadaki read sayısını aşıyorsa.

    Example:
        >>> signal, rid = load_pod5_signal("data/lambda.pod5", read_index=0)
        >>> print(signal.shape, signal.dtype)
    """
    filepath = Path(filepath)
    if not filepath.exists():
        raise FileNotFoundError(f"POD5 dosyası bulunamadı: {filepath}")

    logger.info("POD5 dosyası açılıyor: %s", filepath)

    # Windows Unicode path bypass for pod5 C++ bindings
    # If the path contains non-ascii characters (like 'Masaüstü'),
    # we must copy it to an ASCII-only temp directory first.
    temp_path = None
    is_ascii = all(ord(c) < 128 for c in str(filepath.resolve()))
    if not is_ascii or os.name == 'nt':
        temp_dir = Path(tempfile.gettempdir())
        temp_path = temp_dir / f"read_{uuid.uuid4().hex[:8]}.pod5"
        shutil.copy2(filepath, temp_path)
        actual_read_path = temp_path
    else:
        actual_read_path = filepath

    try:
        with pod5.Reader(actual_read_path) as reader:
            if read_id is not None:
                # Belirli read_id ile arama
                target_id = read_id
                for read in reader.reads():
                    if str(read.read_id) == target_id:
                        signal = read.signal_pa.astype(np.float64)
                        assert signal.ndim == 1, (
                            f"Sinyal 1D olmalı, fakat {signal.ndim}D alındı."
                        )
                        logger.info(
                            "Read yüklendi: %s, uzunluk=%d, dtype=%s",
                            target_id, len(signal), signal.dtype,
                        )
                        return signal, target_id
                raise ValueError(f"read_id bulunamadı: {read_id}")
            else:
                # İndeks ile erişim
                reads = list(reader.reads())
                if read_index >= len(reads):
                    raise IndexError(
                        f"read_index={read_index} geçersiz, "
                        f"dosyada {len(reads)} read var."
                    )
                read = reads[read_index]
                signal = read.signal_pa.astype(np.float64)
                actual_id = str(read.read_id)

                assert signal.ndim == 1, (
                    f"Sinyal 1D olmalı, fakat {signal.ndim}D alındı."
                )
                logger.info(
                    "Read #%d yüklendi: %s, uzunluk=%d samples",
                    read_index, actual_id, len(signal),
                )
                return signal, actual_id
    finally:
        if temp_path and temp_path.exists():
            try:
                temp_path.unlink()
            except BaseException:
                pass



# ---------------------------------------------------------------------------
# 3. Z-Score Normalizasyonu
# ---------------------------------------------------------------------------

def zscore_normalize(
    signal: np.ndarray,
    epsilon: float = 1e-10,
) -> np.ndarray:
    """
    1D sinyale Z-score (standart skor) normalizasyonu uygular.

    Formül: z = (x - μ) / (σ + ε)

    Args:
        signal (np.ndarray): Ham iyonik akım sinyali, shape=(N,).
        epsilon (float): Sıfıra bölünmeyi önlemek için küçük sabit.

    Returns:
        np.ndarray: Normalize edilmiş sinyal, shape=(N,), ortalama≈0, std≈1.

    Raises:
        ValueError: signal 1D değilse.

    Example:
        >>> normed = zscore_normalize(signal)
        >>> assert abs(normed.mean()) < 1e-6
    """
    signal = np.asarray(signal, dtype=np.float64)

    if signal.ndim != 1:
        raise ValueError(
            f"zscore_normalize yalnızca 1D dizilerle çalışır, "
            f"fakat shape={signal.shape} alındı."
        )

    mu: float = signal.mean()
    sigma: float = signal.std()

    normalized = (signal - mu) / (sigma + epsilon)

    assert abs(normalized.mean()) < 1e-5, "Normalizasyon sonrası ortalama 0'a yakın olmalı."
    logger.debug(
        "Z-score normalizasyon: μ=%.4f, σ=%.4f → yeni μ=%.6f, σ=%.6f",
        mu, sigma, normalized.mean(), normalized.std(),
    )
    return normalized


# ---------------------------------------------------------------------------
# 4. Görselleştirme
# ---------------------------------------------------------------------------

def plot_signal(
    signal: np.ndarray,
    title: str = "ONT Ham İyonik Akım Sinyali",
    sampling_rate_hz: Optional[float] = None,
    save_path: Optional[str | Path] = None,
    show: bool = True,
) -> None:
    """
    1D ONT sinyalini matplotlib ile görselleştirir.

    Args:
        signal (np.ndarray): Görüntülenecek sinyal, shape=(N,).
        title (str): Grafik başlığı.
        sampling_rate_hz (Optional[float]): Örnekleme frekansı (Hz). Verilirse
            x ekseni zaman (ms) cinsinden gösterilir, aksi hâlde örnek indeksi.
        save_path (Optional[str | Path]): Kaydedilecek dosya yolu (.png).
        show (bool): plt.show() çağrılsın mı. Test ortamlarında False yapılabilir.

    Returns:
        None

    Example:
        >>> plot_signal(normed_signal, "Lambda Phage — Normalize Sinyal", show=False)
    """
    signal = np.asarray(signal, dtype=np.float64)

    fig = plt.figure(figsize=(14, 5), facecolor="#0d1117")
    ax = fig.add_subplot(111)

    # Zaman ekseni
    if sampling_rate_hz is not None:
        x = np.arange(len(signal)) / sampling_rate_hz * 1000  # ms
        xlabel = "Zaman (ms)"
    else:
        x = np.arange(len(signal))
        xlabel = "Örnek İndeksi"

    # Sinyal çizimi
    ax.plot(x, signal, linewidth=0.6, color="#39d353", alpha=0.85)
    ax.fill_between(x, signal, alpha=0.12, color="#39d353")

    # Stil
    ax.set_facecolor("#0d1117")
    ax.set_title(title, color="white", fontsize=14, pad=12)
    ax.set_xlabel(xlabel, color="#8b949e", fontsize=11)
    ax.set_ylabel("Sinyal Genliği (pA veya Z-score)", color="#8b949e", fontsize=11)
    ax.tick_params(colors="#8b949e")
    for spine in ax.spines.values():
        spine.set_edgecolor("#30363d")
    ax.grid(True, alpha=0.15, color="#30363d", linestyle="--")

    # Annotasyon
    info_text = (
        f"N={len(signal):,} örnek | "
        f"μ={signal.mean():.3f} | "
        f"σ={signal.std():.3f} | "
        f"min={signal.min():.3f} | "
        f"max={signal.max():.3f}"
    )
    ax.annotate(
        info_text, xy=(0.01, 0.97), xycoords="axes fraction",
        color="#8b949e", fontsize=8.5, va="top",
    )

    plt.tight_layout()

    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight", facecolor="#0d1117")
        logger.info("Grafik kaydedildi: %s", save_path)

    if show:
        plt.show()
    else:
        plt.close(fig)


# ---------------------------------------------------------------------------
# 5. Sentetik Test Sinyali Üreteci (POD5 yoksa)
# ---------------------------------------------------------------------------

def generate_synthetic_nanopore_signal(
    n_samples: int = 50_000,
    seed: int = RANDOM_SEED,
) -> np.ndarray:
    """
    Gerçekçi bir ONT (nanopore) sinyalini simüle eden sentetik veri üretir.

    Kademeleri (level) ve Poisson gürültüsünü gerçek nanopore verisiyle
    benzer istatistiksel özelliklere sahip olacak şekilde modeller.
    POD5 dosyasına erişilemeyen ortamlarda test/geliştirme için kullanılır.

    Args:
        n_samples (int): Üretilecek örnek sayısı. Varsayılan: 50_000.
        seed (int): Rastgelelik tohumu. Varsayılan: RANDOM_SEED (42).

    Returns:
        np.ndarray: Pikoamper (pA) cinsinden simüle edilmiş sinyal, shape=(n_samples,).

    Example:
        >>> signal = generate_synthetic_nanopore_signal(n_samples=10_000)
        >>> assert signal.shape == (10_000,)
    """
    rng = np.random.default_rng(seed)

    # Nanopore kılavuz kanalı bantları (pA cinsinden tipik düzeyler)
    levels = np.array([60.0, 75.0, 90.0, 105.0, 120.0, 85.0, 70.0, 95.0])
    dwell_times = rng.integers(200, 2000, size=len(levels))

    # Kademeli sinyal oluştur
    signal_parts = []
    for level, dwell in zip(levels, dwell_times):
        n = min(dwell, n_samples - sum(len(s) for s in signal_parts))
        if n <= 0:
            break
        # Gaussian + Poisson gürültüsü
        segment = (
            level
            + rng.normal(0, 3.0, size=n)       # Termal gürültü
            + rng.poisson(lam=1.5, size=n)     # Sayısal/shot gürültüsü
        )
        signal_parts.append(segment)

    signal = np.concatenate(signal_parts)

    # Hedef uzunluğa ulaşana kadar tekrar et
    while len(signal) < n_samples:
        repeats = int(np.ceil(n_samples / len(signal)))
        base = np.tile(signal, repeats)
        signal = base[:n_samples] + rng.normal(0, 1.5, n_samples)

    signal = signal[:n_samples].astype(np.float64)
    logger.info(
        "Sentetik sinyal üretildi: %d örnek, μ=%.2f pA, σ=%.2f pA",
        n_samples, signal.mean(), signal.std(),
    )
    return signal


# ---------------------------------------------------------------------------
# Yardımcı: Dosya MD5 doğrulama
# ---------------------------------------------------------------------------

def verify_file_md5(filepath: str | Path, expected_md5: Optional[str] = None) -> str:
    """
    Dosyanın MD5 özetini hesaplar ve isteğe bağlı olarak doğrular.

    Args:
        filepath (str | Path): Kontrol edilecek dosya.
        expected_md5 (Optional[str]): Beklenen MD5 değeri. Verilirse eşleşme kontrol edilir.

    Returns:
        str: Dosyanın MD5 hex özeti.

    Raises:
        ValueError: MD5 eşleşmezse.
    """
    h = hashlib.md5()
    with open(filepath, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    digest = h.hexdigest()
    if expected_md5 and digest != expected_md5:
        raise ValueError(
            f"MD5 uyuşmazlığı: beklenen={expected_md5}, hesaplanan={digest}"
        )
    return digest


# ---------------------------------------------------------------------------
# CLI Çalıştırma
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="ONT Lambda Phage veri indirme ve ön işleme"
    )
    parser.add_argument(
        "--dest", default="data/lambda_phage.pod5",
        help="İndirilecek dosyanın yerel yolu"
    )
    parser.add_argument(
        "--read-index", type=int, default=0,
        help="Okunacak read'in indeksi"
    )
    parser.add_argument(
        "--synthetic", action="store_true",
        help="POD5 yerine sentetik sinyal kullan (test modu)"
    )
    parser.add_argument(
        "--plot-save", default="outputs/raw_signal.png",
        help="Grafik kayıt yolu"
    )
    args = parser.parse_args()

    if args.synthetic:
        logger.info("Sentetik mod aktif.")
        raw_signal = generate_synthetic_nanopore_signal()
        read_uuid = "synthetic-lambda-phage"
    else:
        pod5_path = download_ont_data(
            url=ONT_LAMBDA_POD5_URL,
            dest_path=args.dest,
            fallback_url=ONT_LAMBDA_FALLBACK_URL,
        )
        raw_signal, read_uuid = load_pod5_signal(pod5_path, read_index=args.read_index)

    print(f"\n[READ ID ] {read_uuid}")
    print(f"[SHAPE   ] {raw_signal.shape}")
    print(f"[DTYPE   ] {raw_signal.dtype}")
    print(f"[MEAN pA ] {raw_signal.mean():.4f}")
    print(f"[STD  pA ] {raw_signal.std():.4f}")

    # Ham sinyal grafiği
    plot_signal(
        raw_signal,
        title=f"Ham ONT Sinyali - {read_uuid[:12]}...",
        save_path=args.plot_save,
        show=False,
    )

    # Z-score normalizasyon
    norm_signal = zscore_normalize(raw_signal)
    print(f"\n[NORM mean] {norm_signal.mean():.6f}")
    print(f"[NORM std ] {norm_signal.std():.6f}")

    plot_signal(
        norm_signal,
        title=f"Z-Score Normalize Sinyal - {read_uuid[:12]}...",
        save_path=args.plot_save.replace("raw", "normalized"),
        show=False,
    )

    # Normalize sinyali kaydet
    np.save("outputs/normalized_signal.npy", norm_signal)
    logger.info("Normalize sinyal kaydedildi: outputs/normalized_signal.npy")
