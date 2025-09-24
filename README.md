# ThermoSAC

<p align="center">
  <img src=https://github.com/ivanantolo/thermosac/raw/main/TOC.png alt="TOC Figure" width="500"/>
</p>

ThermoSAC is a Python package for studying phase equilibria with the COSMO-SAC model. The project accompanies the paper [*High-Throughput Application and Evaluation of the COSMO-SAC Model for Predictions of Liquid–Liquid Equilibria*](https://doi.org/10.1039/D5DD00259A) published in *Digital Discovery*. It provides tools to analyse miscibility, solubility and phase stability of binary mixtures and reproduces the automated workflow described in the article.

## Features

- **COSMO-SAC activity model** with optional dispersion and different combinatorial terms.
- **Gibbs energy scanning** to locate binodal and spinodal points.
- **Liquid–liquid equilibrium tracing** for miscibility curves.
- **Solid–liquid equilibrium utilities** for solubility predictions.
- **High‑throughput modes** for screening many systems in parallel.
- **Extensive dataset** – predicted LLE curves for 2478 binary systems with auxiliary metadata, plus the NIST/TRC reference dataset.
- **Support for COSMO‑SAC‑2010 and COSMO‑SAC‑dsp**, matching the variants assessed in the paper.
- **Precomputed results** in `data/calculated` including initial values and full miscibility curves.

Five example scripts (`ex_01_GMixScanner.py` to `ex_05_EvaluateNISTBenchmark.py`) demonstrating typical workflows.

## Installation

1. Install the [`cCOSMO`](https://github.com/usnistgov/COSMOSAC) library from NIST (required for COSMO‑SAC calculations).
2. Install ThermoSAC via pip:
   ```
   pip install thermosac
   ```
   ⚠️ **Note:** The PyPI version includes only the core package functionality. Example data and auxiliary files (e.g., `data/profiles`) required for full reproducibility and examples are **not** included in the pip package to keep it lightweight. To access these materials, please clone the repository:
   ```
   git clone https://github.com/ivanantolo/thermosac
   cd thermosac
   pip install .
   ```

Python 3.12 or later is required.

## Quick start
The snippet below mirrors the single-system scanning routine used for the paper's Figure 2.

```python
import numpy as np
from thermosac import Component, Mixture, COSMOSAC
from thermosac.equilibrium.lle import GMixScanner
from thermosac.utils.spacing import spacing

# Set up a binary mixture
names = ["ETHYLENE_GLYCOL", "2,5-Dimethyltetrahydrofuran"]
mixture = Mixture(*[Component(n) for n in names])
model = COSMOSAC(mixture)
model._import_delaware(names, "./data/profiles/UD/sigma3")

# Scan Gibbs free energy of mixing
T = [200]
x = spacing(0, 1, 51, func_name="sigmoid", inflection=15)
scanner = GMixScanner(model, T, x)
initial_values, _ = scanner.find_all_binodal()
print(initial_values.head())
```
This snippet is adapted from the example script showing single‑system scanning (`ex_01_GMixScanner.py`).

## Examples

Additional examples are provided in the repository:

- **ex_01_GMixScanner.py** – single system Gibbs-energy scan.
- **ex_02_LLETracing.py** – trace liquid–liquid equilibria using initial values.
- **ex_03_HighThroughput-Screening.py** – high-throughput binodal detection across multiple systems.
- **ex_04_HighThroughput-Tracing.py** – parallelized LLE tracing for a list of systems.
- **ex_05_EvaluateNISTBenchmark.py** – single-system benchmark evaluation using NIST/TRC data (basis of Table 5).

Each script contains plotting commands and annotations for reproducing the figures used in the accompanying paper. For instance, the single-system tracing routine loads initial values and computes miscibility curves as shown in lines 27–33 of `ex_02_LLETracing.py`.

## Data and reproducibility

The `data` directory is included in the GitHub repository (but not in the PyPI package) and bundles everything needed to reproduce the analyses:

- `data/calculated` – initial values, critical points and the full miscibility curves generated with ThermoSAC.
- `data/experimental` – the lists of binary systems and substances used in the evaluation together with the curated reference set from the NIST Thermodynamics Research Center.
- `data/profiles` – sigma-profiles used by the COSMO-SAC model.
- `data/statistics` – summary tables and helper scripts for assessing model accuracy.

These files allow you to reproduce the statistical analyses and figures presented in the paper.

## Citation

If you find ThermoSAC useful in your research, please consider citing the associated publication. This helps acknowledge the work and supports further development of the package.  

```bibtex
@article{Antolovic2025ThermoSAC,
  title={High-throughput application and evaluation of the COSMO-SAC model for predictions of liquid--liquid equilibria},
  author={Antolović, Ivan and Stephan, Simon and Vrabec, Jadran},
  journal={Digital Discovery},
  year={2025},
  doi={10.1039/D5DD00259A}
}
```

A `CITATION.cff` file will be updated once volume, issue, and page numbers are available, so that GitHub can automatically generate citation formats (APA, BibTeX, etc.).

## Contributing

Contributions and feedback are welcome through GitHub issues and pull requests.