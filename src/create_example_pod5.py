"""
ONT Lambda Phage Simüle POD5 Dosyası Oluşturucu
Modül 1 için gerçek POD5 verisine yerel alternatif.
Çalıştırma: python src/create_example_pod5.py
"""
import pathlib
import uuid
import datetime
import tempfile
import shutil

import numpy as np
import pod5
from pod5.writer import Read, RunInfo
import pod5.pod5_types as pt

RANDOM_SEED = 42
N_READS = 5
N_SAMPLES = 50_000
OUTPUT_PATH = pathlib.Path("data/lambda_phage.pod5")

def create_example_pod5(
    output_path: pathlib.Path = OUTPUT_PATH,
    n_reads: int = N_READS,
    n_samples: int = N_SAMPLES,
    seed: int = RANDOM_SEED,
) -> pathlib.Path:
    """
    Gerçek ONT kalibrasyon parametreleriyle simüle edilmiş
    bir POD5 dosyası oluşturur.

    Args:
        output_path: Çıktı POD5 yolu.
        n_reads: Oluşturulacak read sayısı.
        n_samples: Her read için örnek sayısı.
        seed: Rastgelelik tohumu.

    Returns:
        Oluşturulan POD5 dosyasının yolu.
    """
    rng = np.random.default_rng(seed)
    output_path = pathlib.Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.exists():
        output_path.unlink()

    # ---- RunInfo (tek bir flowcell koşusu) ----
    run_info = RunInfo(
        acquisition_id=str(uuid.UUID(int=seed)),
        acquisition_start_time=datetime.datetime(2024, 1, 1, 0, 0, 0),
        adc_max=32767,
        adc_min=-32768,
        context_tags={},
        experiment_name="lambda_phage_sim",
        flow_cell_id="FAK00000",
        flow_cell_product_code="FLO-MIN114",
        protocol_name="seq_PRO114_DNA_e8.2",
        protocol_run_id=str(uuid.UUID(int=seed + 1)),
        protocol_start_time=datetime.datetime(2024, 1, 1, 0, 0, 0),
        sample_id="lambda_phage_R941_2024",
        sample_rate=4000,           # 4 kHz — standart ONT örnekleme hızı
        sequencing_kit="SQK-LSK114",
        sequencer_position="MN00000",
        sequencer_position_type="MN",
        software="ONT-Pipeline/1.0-sim",
        system_name="SimPC",
        system_type="Mk1b",
        tracking_id={},
    )

    # Kalibrasyon parametreleri (gerçekçi ONT değerleri)
    OFFSET  = -200.0   # pA offset
    SCALE   = 0.01     # pA/ADC birimi

    reads = []
    for i in range(n_reads):
        # Kademeli nanopore sinyali simülasyonu
        levels = rng.choice([60., 70., 80., 90., 100., 110.], size=n_samples)
        noise  = rng.normal(0., 5.0, size=n_samples)
        signal_pa = (levels + noise).astype(np.float32)

        # pA → ADC dönüşümü
        signal_adc = ((signal_pa - OFFSET) / SCALE).astype(np.int16)

        reads.append(Read(
            read_id=uuid.UUID(int=seed * 1000 + i),
            pore=pt.Pore(channel=i + 1, well=1, pore_type="not_set"),
            calibration=pt.Calibration(offset=OFFSET, scale=SCALE),
            read_number=i,
            start_sample=i * n_samples,
            median_before=70.0,
            end_reason=pt.EndReason(
                reason=pt.EndReasonEnum.SIGNAL_POSITIVE,
                forced=False,
            ),
            run_info=run_info,
            signal=signal_adc,
        ))

    # pod5 C++ kütüphanesi Windows'ta Türkçe karakterli ('ü') yollarda hata veriyor.
    # Bu yüzden ASCII garantili geçici bir dizinde dosyayı oluşturup sonra hedefe taşıyoruz.
    temp_dir = pathlib.Path(tempfile.gettempdir())
    temp_pod5 = temp_dir / f"temp_{uuid.uuid4().hex[:8]}.pod5"

    try:
        with pod5.Writer(str(temp_pod5)) as writer:
            writer.add_reads(reads)
        
        # Geçici dosyayı asıl hedefe taşı (shutil Unicode yolları sorunsuz destekler)
        shutil.move(str(temp_pod5), str(output_path))
    finally:
        if temp_pod5.exists():
            temp_pod5.unlink(missing_ok=True)

    print(f"POD5 oluşturuldu: {output_path} | {output_path.stat().st_size:,} bytes | {n_reads} reads")
    return output_path


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Lambda Phage simüle POD5 oluşturucu")
    parser.add_argument("--output", default=str(OUTPUT_PATH))
    parser.add_argument("--n-reads", type=int, default=N_READS)
    parser.add_argument("--n-samples", type=int, default=N_SAMPLES)
    parser.add_argument("--seed", type=int, default=RANDOM_SEED)
    args = parser.parse_args()

    path = create_example_pod5(
        output_path=pathlib.Path(args.output),
        n_reads=args.n_reads,
        n_samples=args.n_samples,
        seed=args.seed,
    )
    print(f"Tamamlandı: {path}")
