# NIST/TRC Benchmark Evaluation

This directory contains the complete and reproducible evaluation pipeline for benchmarking COSMO-SAC predictions against the publicly available NIST/TRC dataset, as reported in Table 5 of the manuscript.

## Overview

The NIST/TRC benchmark consists of 96 binary systems, covering both aqueous and nonaqueous mixtures. This directory contains:

- Scripts to calculate deviations between model predictions and experimental data.
- Detailed statistics in CSV format.
- Diagnostic figures comparing COSMO-SAC to experiment for each system.

The benchmark evaluates two COSMO-SAC model variants:
- **COSMO-SAC-2010** (no dispersion correction)
- **COSMO-SAC-dsp** (with dispersion correction)

All evaluations are conducted using the average absolute logarithmic deviation (AALDS) as defined in Eq. (12) of the manuscript.

## Scripts

The evaluation is modularized into four main scripts:

### `01_approx_deviation.py`
Interpolates predicted mole fractions at experimental temperatures using precomputed LLE data. This yields fast, approximate deviation values that are already quite accurate. Output:
- `approx-SAC_2010.csv`
- `approx-SAC_dsp.csv`

### `02_calc_vs_exp.py`
Optionally refines the approximation by recalculating the exact equilibrium compositions using the ThermoSAC engine. This ensures maximum numerical accuracy. Output:
- `stats-SAC_2010.csv`
- `stats-SAC_dsp.csv`

### `03_visualize.py`
Generates diagnostic plots for all 96 systems, showing:
- Experimental phase compositions (NIST/TRC)
- Predicted LLE curves for both COSMO-SAC variants
- Deviation connectors between model and experiment

Output: PNG images in `figures/`, one for each system.

### `04_statistics.py`
Aggregates the pointwise deviations into AALDS metrics, grouped by system type (aqueous vs. nonaqueous). Final numbers reported in Table 5 are derived directly from these outputs.

## Data Files

- `stats-SAC_2010.csv`, `stats-SAC_dsp.csv`: Final statistics after exact evaluation.
- `approx-*.csv`: Interpolated approximations for quicker assessment.
- `figures/*.png`: One plot per system visualizing the experimental vs. predicted LLE.

## Missing Systems

Five binary systems included in the NIST/TRC dataset were not covered by the original DDB-based screening. These have been computed and added manually:
- Located under: `missing_systems/`
- Corresponding visualizations: `missing_systems/figures/`
- LLE data: `missing-SAC_2010.csv`, `missing-SAC_dsp.csv`

These entries ensure completeness of the benchmark dataset and full coverage of the 96 systems.

## Usage

Each script is standalone and can be executed independently:
```bash
python 01_approx_deviation.py
python 02_calc_vs_exp.py
python 03_visualize.py
python 04_statistics.py
