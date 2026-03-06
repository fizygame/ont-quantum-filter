"""
ONT Lambda Phage Simulated POD5 File Generator
A local alternative to real POD5 data for Module 1.
Run: python src/create_example_pod5.py
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
    Creates a simulated POD5 file using realistic ONT calibration parameters.

    Args:
        output_path: Output POD5 path.
        n_reads: Number of reads to generate.
        n_samples: Number of samples per read.
        seed: Random seed.

    Returns:
        The path to the generated POD5 file.
    """
    rng = np.random.default_rng(seed)
    output_path = pathlib.Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.exists():
        output_path.unlink()

    # ---- RunInfo (A single flowcell run) ----
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
        sample_rate=4000,           # 4 kHz — standard ONT sampling rate
        sequencing_kit="SQK-LSK114",
        sequencer_position="MN00000",
        sequencer_position_type="MN",
        software="ONT-Pipeline/1.0-sim",
        system_name="SimPC",
        system_type="Mk1b",
        tracking_id={},
    )

    # Calibration parameters (realistic ONT values)
    OFFSET  = -200.0   # pA offset
    SCALE   = 0.01     # pA/ADC unit

    reads = []
    for i in range(n_reads):
        # Stepped nanopore signal simulation
        levels = rng.choice([60., 70., 80., 90., 100., 110.], size=n_samples)
        noise  = rng.normal(0., 5.0, size=n_samples)
        signal_pa = (levels + noise).astype(np.float32)

        # pA → ADC conversion
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

    # The pod5 C++ library on Windows throws errors for paths with Unicode characters.
    # Therefore, we generate the file in an ASCII-safe temporary directory and move it.
    temp_dir = pathlib.Path(tempfile.gettempdir())
    temp_pod5 = temp_dir / f"temp_{uuid.uuid4().hex[:8]}.pod5"

    try:
        with pod5.Writer(str(temp_pod5)) as writer:
            writer.add_reads(reads)
        
        # Move the temporary file to the final destination (shutil handles Unicode paths fine)
        shutil.move(str(temp_pod5), str(output_path))
    finally:
        if temp_pod5.exists():
            temp_pod5.unlink(missing_ok=True)

    print(f"POD5 created: {output_path} | {output_path.stat().st_size:,} bytes | {n_reads} reads")
    return output_path


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Lambda Phage Simulated POD5 Generator")
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
    print(f"Completed: {path}")
