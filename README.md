# ThermoSAC

<p align="center">
  <img src=https://github.com/ivanantolo/thermosac/raw/main/TOC.png alt="TOC Figure" width="500"/>
</p>

ThermoSAC is a Python package for studying phase equilibria with the COSMO‑SAC model. The project accompanies the paper *High‑Throughput Application and Evaluation of the COSMO‑SAC Model for Predictions of Liquid–Liquid Equilibria* currently under review in *Digital Discovery*. It provides tools to analyse miscibility, solubility and phase stability of binary mixtures and reproduces the automated workflow described in the article.

## Features

- **COSMO-SAC activity model** with optional dispersion and different combinatorial terms.
- **Gibbs energy scanning** to locate binodal and spinodal points.
- **Liquid–liquid equilibrium tracing** for miscibility curves.
- **Solid–liquid equilibrium utilities** for solubility predictions.
- **High‑throughput modes** for screening many systems in parallel.
- **Extensive dataset** – predicted LLE curves for 2478 binary systems with auxiliary metadata, plus the NIST/TRC reference dataset.
- **Support for COSMO‑SAC‑2010 and COSMO‑SAC‑dsp**, matching the variants assessed in the paper.
- **Precomputed results** in `data/calculated` including initial values and full miscibility curves.

Four example scripts (`ex_01_GMixScanner.py` to `ex_04_HighThroughput-Tracing.py`) demonstrating typical workflows.

## Installation

1. Install the [`cCOSMO`](https://github.com/usnistgov/COSMOSAC) library from NIST (required for COSMO‑SAC calculations).
2. Install ThermoSAC via pip:
   ```
   pip install thermosac
   ```
   or clone the repository and install locally:
   ```
   git clone https://github.com/ivanantolo/thermosac
   cd thermosac
   pip install .
   ```

Python 3.12 or later is required as indicated in the project configuration【F:pyproject.toml†L1-L10】.

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

Each script contains plotting commands and annotations for reproducing the figures used in the accompanying paper. For instance, the single-system tracing routine loads initial values and computes miscibility curves as shown in lines 27–33 of `ex_02_LLETracing.py`.

## Data and reproducibility

The `data` directory bundles everything needed to reproduce the analyses:

- `data/calculated` – initial values, critical points and the full miscibility curves generated with ThermoSAC.
- `data/experimental` – the lists of binary systems and substances used in the evaluation together with the curated reference set from the NIST Thermodynamics Research Center.
- `data/profiles` – sigma-profiles used by the COSMO-SAC model.
- `data/statistics` – summary tables and helper scripts for assessing model accuracy.

These files allow you to reproduce the statistical analyses and figures presented in the paper.

## Citation

If you use ThermoSAC in your research, please cite the forthcoming paper once published. A `CITATION.cff` file will be added after publication.

## Contributing

Contributions and feedback are welcome through GitHub issues and pull requests.