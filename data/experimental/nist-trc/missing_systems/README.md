# Missing Systems – NIST/TRC Benchmark Completion

This folder contains COSMO-SAC results and corresponding visualizations for five binary systems that are part of the NIST/TRC reference dataset (`LLeDbPGL6ed96b.txt`) but were not included in the main DDB-based dataset used in the manuscript. These systems were computed separately to ensure full coverage of all 96 systems listed in Table 5.

## Included Systems

All five systems involve **ANILINE** as the first component and differ in the alkane used as the second component:

| System ID | c1       | c2                     |
|-----------|----------|------------------------|
| 6154      | ANILINE  | 2,2,3-TRIMETHYLBUTANE  |
| 6155      | ANILINE  | 3-METHYLHEXANE         |
| 6156      | ANILINE  | 2,2-DIMETHYLPENTANE    |
| 6157      | ANILINE  | 3,3-DIMETHYLPENTANE    |
| 6158      | ANILINE  | 3-ETHYLPENTANE         |

## Contents

- `missing-SAC_2010.csv`  
  COSMO-SAC-2010 predictions for the five systems listed above.

- `missing-SAC_dsp.csv`  
  COSMO-SAC-dsp predictions for the same systems.

- `figures/`  
  Contains one PNG plot per system, each showing:
  - Experimental LLE points (from NIST/TRC)
  - COSMO-SAC-2010 and COSMO-SAC-dsp predicted binodals
  - Deviations between prediction and experiment visualized as dotted lines

## Purpose

These results are provided to complete the NIST/TRC benchmark evaluation and ensure that all systems included in Table 5 of the manuscript are fully represented with corresponding model predictions.
