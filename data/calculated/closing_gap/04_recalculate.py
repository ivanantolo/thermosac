"""
RECALCULATE LLE AND ADJUST SMOOTHING CURVE SCRIPT

This script recalculates the LLE for specific temperature ranges to resolve
wiggly smoothing curves.

Key functionalities:
- Allows visual inspection of wiggly curves and selection of temperature ranges.
- Supports removing visually identified outliers and adjusting outlier counts.
- Recalculates LLE points to improve smoothing curve approximations.
- Combines original, recalculated, and approximated LLE data for analysis.
- Saves results into Excel files per system, noting recalculated and approximated rows.

Usage instructions:
1. Start with `adjustment=0` and `remove_points=0`; comment out recalculation.
2. Inspect visual output to choose temperature ranges or identify outliers.
3. Adjust parameters iteratively to achieve a satisfactory smoothing curve.
4. Save and track results for each system in separate Excel files.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from equilibrium.lle.lle_generator import LLEGenerator
from tools.helper import plot_system_data, save_recalc, generate_approximation, recalculate_binodal

dispersion = True
DIR_DATA = Path('./data')
DIR_RES = Path(("with" if dispersion else "without") + "_dispersion")

file = 'lle_dsp_02.csv' if dispersion else 'lle_02.csv'
data = pd.read_csv(DIR_DATA / file, sep=';')
data = data.sort_values(by=['sys', 'T']).reset_index(drop=True)
outliers = pd.read_excel(DIR_RES / '03_adjustment.xlsx')

log_name = DIR_RES / '04_recalculate.xlsx'
if not log_name.exists():
    outliers.to_excel(log_name, index=False)
log_file = pd.read_excel(log_name)

outliers = outliers[outliers['Recalculate'].notna()]
outliers = outliers.drop(columns=["Recalculate"])
outliers = outliers.sort_values(by=['System']).reset_index(drop=True)

lleGen = LLEGenerator(dispersion)
recalc = None

# systems = outliers['System'].unique()
systems = [4608]

for i, sys in enumerate(systems):
    lle = lleGen.get_lle_handler(sys)
    dfs = data[data.sys==sys]
    outlier = outliers[outliers['System']==sys]['Outlier'].iloc[0]
# =============================================================================
    adjustment = 0 # Set to 0 at beginning
# =============================================================================
    _outlier = outlier + adjustment
    df = dfs.iloc[:-_outlier].tail(3) if _outlier > 0 else dfs.copy().tail(3)
    df = df.reset_index(drop=True)
    out = dfs.tail(_outlier + 1)
    dfs = dfs.copy().iloc[:-_outlier] if _outlier > 0 else dfs.copy()
    fig, ax = plot_system_data(sys, df, out)

    # LLE recalculation
    T_range = (195, 195.4, 9)
    temperatures = np.linspace(*T_range)
# =============================================================================
    remove_points = 1 # Set to 0 at beginning & comment out next line
    recalc = recalculate_binodal(ax, df, temperatures,lle, remove_points)
# =============================================================================

    # Approximate
    approx, wiggly_curve = generate_approximation(df, recalc, ax=ax)
    msg = '- is wiggly' if wiggly_curve else ''
    print(f"{adjustment=} - {sys=:04d} {msg}")

    # Save results
    logging = (log_file, log_name, T_range, remove_points, adjustment)

    def save(close_fig=True, log=True):
        res = pd.concat([dfs, recalc, approx], ignore_index=True)
        save_recalc(sys, fig, res, *logging, close_fig=close_fig, log=log)

    save(close_fig=False, log=False)
    break
