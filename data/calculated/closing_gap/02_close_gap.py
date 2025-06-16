"""
CLOSE GAP AND SMOOTH CURVE SCRIPT

This script processes datasets to remove outliers and approximate a smoothing
curve to close miscibility gap binodal branches at the UCST.

Key functionalities:
- Loads datasets and outlier information from previous steps.
- Approximates a smoothing curve connecting binodal branches at the UCST.
- Flags "wiggly" curves with curvature changes or monotonicity issues.
- Saves an Excel file (`02_close_gap.xlsx`) documenting wiggly curves.

Output:
- Visualizations comparing original, smoothed, and flagged curves.
- Excel file with systems marked as wiggly or non-wiggly for the next step.
"""

import pandas as pd
from pathlib import Path
from tools.helper import approximate, plot_system_data, save_fig

dispersion = True
DIR_DATA = Path('./data')
DIR_RES = Path(("with" if dispersion else "without") + "_dispersion")

file = 'lle_dsp_02.csv' if dispersion else 'lle_02.csv'
data = pd.read_csv(DIR_DATA / file, sep=';')
data = data.sort_values(by=['sys', 'T']).reset_index(drop=True)
outliers = pd.read_excel(DIR_RES / '01_fine_tune.xlsx')
if "Note" in outliers.columns:
    outliers = outliers.drop(columns=["Note"])
outliers = outliers.sort_values(by=['System']).reset_index(drop=True)

log_file = outliers.copy()

systems = outliers['System'].unique()
systems = [3]

for i, sys in enumerate(systems):
    dfs = data[data.sys==sys]
    outlier = outliers[outliers['System']==sys]['Outlier'].iloc[0]
    out = dfs.tail(outlier + 1)
    idx = outlier + 3
    df = dfs.iloc[:-outlier].tail(3) if outlier else dfs.copy().tail(4)
    fig, ax = plot_system_data(sys, df, out)

    wiggly_curve = False
    max_temp = dfs['T'].max()
    if max_temp < 1000:

        # Approximation
        df = dfs.iloc[:-outlier] if outlier else dfs.copy()
        t, smooth, wiggly_curve = approximate(df, npoints=7, ndata=3)
        ax.plot(t[1:-1], smooth[1:-1], 'o-', c='cyan', mec='k', label='Approx')
        ax.plot(t[:2], smooth[:2], '-', c='cyan', mec='k', zorder=0)
        ax.plot(t[-2:], smooth[-2:], '-', c='cyan', mec='k', zorder=0)

        if wiggly_curve:
            log_file.loc[log_file['System'] == sys, 'Note'] = 'Wiggly curve'

    # Customize legend
    legend = ax.legend()
    legend.get_frame().set_facecolor('gray')
    legend.get_frame().set_edgecolor('white')

    save_fig(sys, fig, wiggly_curve)

    print(f"{i:02d} - {sys=:04d} processed.")

    # break

log_file.to_excel('02_close_gap.xlsx', index=False)
