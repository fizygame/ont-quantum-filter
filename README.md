# Quantum-Inspired Ionic Signal Filtering Pipeline for Oxford Nanopore

![Python Version](https://img.shields.io/badge/python-3.10%2B-blue.svg)
![License](https://img.shields.io/badge/license-MIT-green.svg)
![Status](https://img.shields.io/badge/status-Stable-brightgreen.svg)

A DeepTech pipeline designed to clean, denoise, and enhance the resolution of raw ionic current signals from Oxford Nanopore (ONT) DNA/RNA sequencing devices. It utilizes a Quantum-Inspired Iterative Deconvolution and Noise Filtering approach, combining **SCSA** (Semi-Classical Signal Analysis), **PnP-ADMM**, and a **Discrete Quantum Genetic Algorithm (DQGA)**.

---

## 🚀 Key Features

*   **SCSA Quantum Noise Filter:** Models the signal as a potential well and uses negative eigenvalues of a Hamiltonian matrix to perform physics-based noise reduction.
*   **Plug-and-Play ADMM (PnP-ADMM):** A global optimization framework that seamlessly separates the data fidelity constraint from the denoising prior using adaptive Lagrangian multipliers.
*   **1D Richardson-Lucy Deconvolution:** Solves the blurring issue in homopolymer regions by mathematically deconvolving the signal using a symmetric Gaussian Point Spread Function (PSF).
*   **DQGA Hyperparameter Optimizer:** A Discrete Quantum Genetic Algorithm that uses qubit superpositions and unitary rotation gates to automatically find the optimal noise thresholds (like Planck's constant equivalent $h$).
*   **Robust Data Handling:** Built-in safeguards for `.pod5` file ingestion, automated synthetic calibration data generation for offline testing, and safe mapping across spaces.
*   **Benchmarking & Output:** Built-in comparisons against classical bioinformatics baselines (e.g., Savitzky-Golay filters), automated SNR (Signal-to-Noise Ratio) calculation, and exporting `.npy` vectors.

---

## 🛠️ Installation

```bash
# 1. Clone the repository
git clone https://github.com/your-username/ont-quantum-filter.git
cd ont-quantum-filter

# 2. Create a virtual environment (Recommended)
python -m venv .venv
source .venv/bin/activate  # On Windows use: .venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt
```

---

## 📖 Usage

You can run the entire pipeline end-to-end using the main orchestrator script:

```bash
python src/main.py
```

### What does `main.py` do?
1. **Data Ingestion:** Reads the raw `.pod5` sequencing file. If no data exists, it generates a mathematically calibrated synthetic biological signal.
2. **DQGA Optimization:** Spins up a quantum chromosome population to find the best $h$ parameter mathematically.
3. **PnP-ADMM & SCSA:** Iteratively filters out the physical noise using quantum bound states.
4. **Deconvolution:** Enhances the resolution using 1D RL techniques.
5. **Benchmarking:** Calculates physical SNR improvements and saves an overlay plot (`outputs/final_pipeline_comparison.png`).
6. **Export:** Saves the final clean signal array as a numpy binary `outputs/quantum_clean_signal.npy`.

---

## 🧪 Testing

The pipeline is mathematically verified with over 50 test cases covering constraints, energy conservation (unitary transforms), and proximal mapping bounds.

To run the test suite:

```bash
pip install pytest
pytest tests/ -v
```

---

## 📂 Project Structure

```text
├── data/                    # Raw ONT pod5 data directory
├── outputs/                 # Clean signals and generated benchmark plots
├── src/                     # Core Algorithm Modules
│   ├── benchmarking.py      # SNR math and output generation
│   ├── create_example_pod5.py # Mock calibration and data generator
│   ├── data_ingestion.py    # Pod5 loaders and normalization
│   ├── dqga_optimizer.py    # Quantum Genetic hyperparameter engine
│   ├── main.py              # End-to-end orchestrator 
│   ├── pnp_admm.py          # Proximal ADMM iteration framework
│   ├── rl_deconvolution.py  # Richardson-Lucy resolution enhancer
│   └── scsa_filter.py       # Schrödinger-based quantum denoiser
├── tests/                   # Pytest validation suite (Modules 1-6)
├── requirements.txt         # Python dependencies
├── .gitignore               # Ignored system files
└── README.md                # Project documentation
```

---

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🤝 Contributing

Contributions, issues, and feature requests are welcome! Feel free to check the issues page.
