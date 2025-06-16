"""
FINE-TUNING OUTLIER DETECTION SCRIPT

This script generates visual comparisons to analyze differences between
outlier detection methods (DC+BC and DC+BC_slope).

Key functionalities:
- Identifies cases where the difference between DC+BC and DC+BC_slope is non-zero.
- Groups systems into bins of increasing differences for better organization.
- Highlights cases where BC_curvature may cause false positives in BC.
- Saves figures to guide user decisions on switching to BC_slope.

Instructions for the user:
1. Copy `00_screen.xlsx` to `01_fine_tune.xlsx` before running this script.
2. Review generated figures and determine which systems to switch to BC_slope.
   - A switch is most likely needed for systems with higher differences.
3. Edit `01_fine_tune.xlsx`:
   - Replace "Outlier" column values with BC_slope where necessary.
   - Delete all columns beyond the "Outlier" column and save the file.
"""

import pandas as pd
from pathlib import Path

from tools.helper import plot_lle_curve

dispersion = True
DIR_DATA = Path('./data')
DIR_RES = Path(("with" if dispersion else "without") + "_dispersion")

file = 'lle_dsp_02.csv' if dispersion else 'lle_02.csv'
data = pd.read_csv(DIR_DATA / file, sep=';')
data = data.sort_values(by=['sys', 'T']).reset_index(drop=True)
outliers = pd.read_excel(DIR_RES / '00_screen.xlsx')
# Filter out rows where 'Difference' is 0
outliers = outliers[outliers['Difference'] != 0]

i = 0
for diff, dfs in outliers.groupby('Difference'):
    for sys, row in dfs.groupby('System'):
        i+=1
        df = data[data.sys==sys]
        out1 = df.tail(row['DC+BC'].iloc[0])
        out2 = df.tail(row['DC+BC_slope'].iloc[0])
        idx = max(len(out1), len(out2)) + 3
        df = df.tail(idx)
        ax = plot_lle_curve(df, out1, out2, sys, diff, save=True)
        print(f"{i:02d} - {diff=:02d}:{sys=:04d} processed.")
