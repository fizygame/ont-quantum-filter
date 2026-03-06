"""
Module 1: Data Ingestion & Preprocessing
==================================================================
Downloads raw ionic current signals from the Oxford Nanopore (ONT) 
Lambda Phage open dataset, reads them in POD5 format, applies Z-score 
normalization, and visualizes them.

Data Source:
    ONT Open Datasets — Lambda phage control run (POD5 format)
    https://github.com/nanoporetech/ont-open-datasets
    Direct S3: https://s3.amazonaws.com/nanopore-human-wgs/rna/referenceSamples/

Libraries: numpy, scipy, pod5, matplotlib, requests
Author: FizyGame
Date: 2026
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
matplotlib.use("Agg")  # For headless environments — no GUI backend required
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import requests
import pod5
import tempfile
import shutil
import uuid

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
RANDOM_SEED: int = 42
np.random.seed(RANDOM_SEED)

# ONT / nanoporetech open POD5 test data — verified real POD5 files
# Source: https://github.com/nanoporetech/pod5-file-format (Apache-2.0, Public)
ONT_LAMBDA_POD5_URL: str = (
    "https://raw.githubusercontent.com/nanoporetech/pod5-file-format/"
    "master/python/pod5/test_data/multi_fast5_zip.pod5"
)

# Fallback 1: Another official ONT test file
ONT_LAMBDA_FALLBACK_URL: str = (
    "https://raw.githubusercontent.com/nanoporetech/pod5-file-format/"
    "master/python/pod5/test_data/multi_reads.pod5"
)

# POD5 file signatures (magic bytes) — first 8 bytes
# Arrow IPC/Feather format: b'ARROW1\x00\x00' or b'\x41\x52\x52\x4f\x57\x31\x00\x00'
_POD5_MIN_FILE_SIZE_BYTES: int = 10_000  # Valid POD5 should be at least ~10 KB

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
)
logger = logging.getLogger("data_ingestion")


# ---------------------------------------------------------------------------
# 1. Data Downloading
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
    Downloads a POD5 file from the ONT open dataset.

    Features a retry mechanism to handle network interruptions.
    If the file already exists, the download is skipped.

    Args:
        url (str): The URL of the file to download.
        dest_path (str | Path): Local path to save the file.
        retries (int): Maximum number of retries. Default: 3.
        retry_delay (float): Wait time between attempts (seconds).
        chunk_size (int): Chunk size for streaming download (bytes).
        fallback_url (Optional[str]): Backup URL to use if the primary fails.

    Returns:
        Path: The local path of the downloaded or existing file.

    Raises:
        RuntimeError: If all attempts and the fallback URL fail.

    Example:
        >>> path = download_ont_data(ONT_LAMBDA_POD5_URL, "data/lambda.pod5")
    """
    dest_path = Path(dest_path)
    dest_path.parent.mkdir(parents=True, exist_ok=True)

    if dest_path.exists() and dest_path.stat().st_size > 0:
        logger.info("File already exists, skipping download: %s", dest_path)
        return dest_path

    urls_to_try = [url]
    if fallback_url:
        urls_to_try.append(fallback_url)

    for attempt_url in urls_to_try:
        logger.info("Attempting download: %s", attempt_url)
        for attempt in range(1, retries + 1):
            try:
                response = requests.get(
                    attempt_url,
                    stream=True,
                    timeout=60,
                    headers={"User-Agent": "ONT-Pipeline/1.0"},
                )
                response.raise_for_status()

                # Content-Type check — Only download if not an HTML/XML error page
                content_type = response.headers.get("Content-Type", "")
                if "html" in content_type or "xml" in content_type:
                    raise IOError(
                        f"Server returned HTML/XML (not POD5): Content-Type={content_type}"
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
                        f"Incomplete download: {downloaded}/{total_size} bytes"
                    )

                # Minimum size check — Is it an HTML error page?
                if downloaded < _POD5_MIN_FILE_SIZE_BYTES:
                    dest_path.unlink(missing_ok=True)
                    raise IOError(
                        f"Downloaded file is too small ({downloaded} bytes < "
                        f"{_POD5_MIN_FILE_SIZE_BYTES} bytes). "
                        f"Possibly an invalid URL or error page."
                    )

                # POD5/Arrow signature check (first 4 bytes: 'ARRO')
                with open(dest_path, "rb") as f_check:
                    magic = f_check.read(4)
                if magic != b"ARRO" and magic != b"\x50\x4f\x44\x35":
                    dest_path.unlink(missing_ok=True)
                    raise IOError(
                        f"Invalid POD5 signature: {magic!r}. "
                        f"File is not in POD5/Arrow format."
                    )

                logger.info(
                    "Download complete (%.2f MB): %s",
                    downloaded / 1024 / 1024,
                    dest_path,
                )
                return dest_path

            except (requests.RequestException, IOError) as exc:
                logger.warning(
                    "Attempt %d/%d failed [%s]: %s",
                    attempt, retries, attempt_url, exc,
                )
                if attempt < retries:
                    time.sleep(retry_delay * attempt)
                else:
                    logger.error("URL failed: %s", attempt_url)
                    if dest_path.exists():
                        dest_path.unlink(missing_ok=True)

    raise RuntimeError(
        f"All URL attempts failed. "
        f"Please check your internet connection and data sources."
    )


# ---------------------------------------------------------------------------
# 2. POD5 Reading
# ---------------------------------------------------------------------------

def load_pod5_signal(
    filepath: str | Path,
    read_id: Optional[str] = None,
    read_index: int = 0,
) -> Tuple[np.ndarray, str]:
    """
    Reads the raw ionic current signal (pA) from a POD5 file.

    Args:
        filepath (str | Path): Path to the POD5 file.
        read_id (Optional[str]): A specific read UUID. If None, read_index is used.
        read_index (int): The read index to use if read_id is not provided. Default: 0.

    Returns:
        Tuple[np.ndarray, str]:
            - signal (np.ndarray): Raw ionic current time series in pA, shape=(N,).
            - actual_read_id (str): The UUID of the read that was loaded.

    Raises:
        FileNotFoundError: If the file does not exist.
        ValueError: If the specified read_id is not found in the file.
        IndexError: If the read_index exceeds the number of reads in the file.

    Example:
        >>> signal, rid = load_pod5_signal("data/lambda.pod5", read_index=0)
        >>> print(signal.shape, signal.dtype)
    """
    filepath = Path(filepath)
    if not filepath.exists():
        raise FileNotFoundError(f"POD5 file not found: {filepath}")

    logger.info("Opening POD5 file: %s", filepath)

    # Windows Unicode path bypass for pod5 C++ bindings
    # If the path contains non-ascii characters,
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
                # Search by specific read_id
                target_id = read_id
                for read in reader.reads():
                    if str(read.read_id) == target_id:
                        signal = read.signal_pa.astype(np.float64)
                        assert signal.ndim == 1, (
                            f"Signal should be 1D, but received {signal.ndim}D."
                        )
                        logger.info(
                            "Read loaded: %s, length=%d, dtype=%s",
                            target_id, len(signal), signal.dtype,
                        )
                        return signal, target_id
                raise ValueError(f"read_id not found: {read_id}")
            else:
                # Access by index
                reads = list(reader.reads())
                if read_index >= len(reads):
                    raise IndexError(
                        f"read_index={read_index} is invalid, "
                        f"the file has {len(reads)} reads."
                    )
                read = reads[read_index]
                signal = read.signal_pa.astype(np.float64)
                actual_id = str(read.read_id)

                assert signal.ndim == 1, (
                    f"Signal should be 1D, but received {signal.ndim}D."
                )
                logger.info(
                    "Read #%d loaded: %s, length=%d samples",
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
# 3. Z-Score Normalization
# ---------------------------------------------------------------------------

def zscore_normalize(
    signal: np.ndarray,
    epsilon: float = 1e-10,
) -> np.ndarray:
    """
    Applies Z-score (standard score) normalization to a 1D signal.

    Formula: z = (x - mean) / (std + epsilon)

    Args:
        signal (np.ndarray): Raw ionic current signal, shape=(N,).
        epsilon (float): Small constant to avoid division by zero.

    Returns:
        np.ndarray: Normalized signal, shape=(N,), mean≈0, std≈1.

    Raises:
        ValueError: If the signal is not 1D.

    Example:
        >>> normed = zscore_normalize(signal)
        >>> assert abs(normed.mean()) < 1e-6
    """
    signal = np.asarray(signal, dtype=np.float64)

    if signal.ndim != 1:
        raise ValueError(
            f"zscore_normalize only works with 1D arrays, "
            f"but received shape={signal.shape}."
        )

    mu: float = signal.mean()
    sigma: float = signal.std()

    normalized = (signal - mu) / (sigma + epsilon)

    assert abs(normalized.mean()) < 1e-5, "Mean should be close to 0 after normalization."
    logger.debug(
        "Z-score normalization: mean=%.4f, std=%.4f → new mean=%.6f, new std=%.6f",
        mu, sigma, normalized.mean(), normalized.std(),
    )
    return normalized


# ---------------------------------------------------------------------------
# 4. Visualization
# ---------------------------------------------------------------------------

def plot_signal(
    signal: np.ndarray,
    title: str = "ONT Raw Ionic Current Signal",
    sampling_rate_hz: Optional[float] = None,
    save_path: Optional[str | Path] = None,
    show: bool = True,
) -> None:
    """
    Visualizes the 1D ONT signal using matplotlib.

    Args:
        signal (np.ndarray): Signal to visualize, shape=(N,).
        title (str): Plot title.
        sampling_rate_hz (Optional[float]): Sampling frequency (Hz). If provided,
            the x-axis is shown in time (ms), otherwise by sample index.
        save_path (Optional[str | Path]): Path to save the file (.png).
        show (bool): Whether to call plt.show(). Set to False in testing environments.

    Returns:
        None

    Example:
        >>> plot_signal(normed_signal, "Lambda Phage — Normalized Signal", show=False)
    """
    signal = np.asarray(signal, dtype=np.float64)

    fig = plt.figure(figsize=(14, 5), facecolor="#0d1117")
    ax = fig.add_subplot(111)

    # Time axis
    if sampling_rate_hz is not None:
        x = np.arange(len(signal)) / sampling_rate_hz * 1000  # ms
        xlabel = "Time (ms)"
    else:
        x = np.arange(len(signal))
        xlabel = "Sample Index"

    # Signal plotting
    ax.plot(x, signal, linewidth=0.6, color="#39d353", alpha=0.85)
    ax.fill_between(x, signal, alpha=0.12, color="#39d353")

    # Styling
    ax.set_facecolor("#0d1117")
    ax.set_title(title, color="white", fontsize=14, pad=12)
    ax.set_xlabel(xlabel, color="#8b949e", fontsize=11)
    ax.set_ylabel("Signal Amplitude (pA or Z-score)", color="#8b949e", fontsize=11)
    ax.tick_params(colors="#8b949e")
    for spine in ax.spines.values():
        spine.set_edgecolor("#30363d")
    ax.grid(True, alpha=0.15, color="#30363d", linestyle="--")

    # Annotation
    info_text = (
        f"N={len(signal):,} samples | "
        f"mean={signal.mean():.3f} | "
        f"std={signal.std():.3f} | "
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
        logger.info("Plot saved: %s", save_path)

    if show:
        plt.show()
    else:
        plt.close(fig)


# ---------------------------------------------------------------------------
# 5. Synthetic Test Signal Generator (If POD5 is missing)
# ---------------------------------------------------------------------------

def generate_synthetic_nanopore_signal(
    n_samples: int = 50_000,
    seed: int = RANDOM_SEED,
) -> np.ndarray:
    """
    Generates synthetic data that simulates a realistic ONT (nanopore) signal.

    Models states (levels) and Poisson noise to mirror the statistical properties
    of real nanopore data. Useful for testing/development when POD5 files are
    inaccessible.

    Args:
        n_samples (int): Number of samples to generate. Default: 50_000.
        seed (int): Random seed. Default: RANDOM_SEED (42).

    Returns:
        np.ndarray: Simulated signal in picoamperes (pA), shape=(n_samples,).

    Example:
        >>> signal = generate_synthetic_nanopore_signal(n_samples=10_000)
        >>> assert signal.shape == (10_000,)
    """
    rng = np.random.default_rng(seed)

    # Nanopore guide channel bands (typical levels in pA)
    levels = np.array([60.0, 75.0, 90.0, 105.0, 120.0, 85.0, 70.0, 95.0])
    dwell_times = rng.integers(200, 2000, size=len(levels))

    # Construct the step signal
    signal_parts = []
    for level, dwell in zip(levels, dwell_times):
        n = min(dwell, n_samples - sum(len(s) for s in signal_parts))
        if n <= 0:
            break
        # Gaussian + Poisson noise
        segment = (
            level
            + rng.normal(0, 3.0, size=n)       # Thermal noise
            + rng.poisson(lam=1.5, size=n)     # Shot noise
        )
        signal_parts.append(segment)

    signal = np.concatenate(signal_parts)

    # Repeat until the target length is reached
    while len(signal) < n_samples:
        repeats = int(np.ceil(n_samples / len(signal)))
        base = np.tile(signal, repeats)
        signal = base[:n_samples] + rng.normal(0, 1.5, n_samples)

    signal = signal[:n_samples].astype(np.float64)
    logger.info(
        "Synthetic signal generated: %d samples, mean=%.2f pA, std=%.2f pA",
        n_samples, signal.mean(), signal.std(),
    )
    return signal


# ---------------------------------------------------------------------------
# Helper: File MD5 Verification
# ---------------------------------------------------------------------------

def verify_file_md5(filepath: str | Path, expected_md5: Optional[str] = None) -> str:
    """
    Calculates the MD5 digest of a file, verifying it if expected_md5 is provided.

    Args:
        filepath (str | Path): The file to check.
        expected_md5 (Optional[str]): Expected MD5 value. If provided, checks for a match.

    Returns:
        str: MD5 hex digest of the file.

    Raises:
        ValueError: If the MD5 does not match.
    """
    h = hashlib.md5()
    with open(filepath, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    digest = h.hexdigest()
    if expected_md5 and digest != expected_md5:
        raise ValueError(
            f"MD5 mismatch: expected={expected_md5}, calculated={digest}"
        )
    return digest


# ---------------------------------------------------------------------------
# CLI Execution
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="ONT Lambda Phage data ingestion and preprocessing"
    )
    parser.add_argument(
        "--dest", default="data/lambda_phage.pod5",
        help="Local path to download the file to"
    )
    parser.add_argument(
        "--read-index", type=int, default=0,
        help="Index of the read to load"
    )
    parser.add_argument(
        "--synthetic", action="store_true",
        help="Use synthetic signal instead of POD5 (test mode)"
    )
    parser.add_argument(
        "--plot-save", default="outputs/raw_signal.png",
        help="Path to save the plot"
    )
    args = parser.parse_args()

    if args.synthetic:
        logger.info("Synthetic mode active.")
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

    # Plot raw signal
    plot_signal(
        raw_signal,
        title=f"Raw ONT Signal - {read_uuid[:12]}...",
        save_path=args.plot_save,
        show=False,
    )

    # Z-score normalization
    norm_signal = zscore_normalize(raw_signal)
    print(f"\n[NORM mean] {norm_signal.mean():.6f}")
    print(f"[NORM std ] {norm_signal.std():.6f}")

    plot_signal(
        norm_signal,
        title=f"Z-Score Normalized Signal - {read_uuid[:12]}...",
        save_path=args.plot_save.replace("raw", "normalized"),
        show=False,
    )

    # Save normalized signal
    np.save("outputs/normalized_signal.npy", norm_signal)
    logger.info("Normalized signal saved to: outputs/normalized_signal.npy")
