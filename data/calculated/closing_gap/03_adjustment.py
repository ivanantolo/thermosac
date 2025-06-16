"""
ADJUST OUTLIERS AND SMOOTH CURVE SCRIPT

This script adjusts the number of outliers and recalculates the smoothing
curve to resolve wiggly curves.

Key functionalities:
- Loops over a range of adjustments to add or remove outliers as needed.
- Recalculates smoothing curves after each adjustment to check for wiggliness.
- Stops adjustments for a system once wiggliness is resolved.
- Flags systems needing further processing for recalculation.

Output:
- Visualizations of adjusted smoothing curves.
- Excel file (`03_adjustment.xlsx`) with adjustment details and flags.
"""

import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from tools.helper import approximate, plot_system_data, save_adjustments

dispersion = True
DIR_DATA = Path('./data')
DIR_RES = Path(("with" if dispersion else "without") + "_dispersion")

file = 'lle_dsp_02.csv' if dispersion else 'lle_02.csv'
data = pd.read_csv(DIR_DATA / file, sep=';')
data = data.sort_values(by=['sys', 'T']).reset_index(drop=True)
outliers = pd.read_excel(DIR_RES / '02_close_gap.xlsx')
log_file = outliers.copy()

outliers = outliers[outliers['Note'].notna()]
outliers = outliers.drop(columns=["Note"])
outliers = outliers.sort_values(by=['System']).reset_index(drop=True)
log_file = log_file.drop(columns=["Note"])

systems = outliers['System'].unique()
# systems = [11]

for i, sys in enumerate(systems):
    # lle = lleGen.get_lle_handler(sys)
    dfs = data[data.sys==sys]
    outlier = outliers[outliers['System']==sys]['Outlier'].iloc[0]
    adjustments = range(1, 3 +1)
    for adjustment in adjustments:
        _outlier = outlier + adjustment
        n = (max(adjustments) + 2) - adjustment
        df = dfs.iloc[:-_outlier].tail(n) if _outlier > 0 else dfs.copy().tail(n)
        out = dfs.tail(_outlier + 1)
        idx = outlier + 3
        fig, ax = plot_system_data(sys, df, out)

        # Approximation
        _df = dfs.iloc[:-_outlier] if _outlier else dfs.copy()
        t, smooth, wiggly_curve = approximate(_df, npoints=7, ndata=3)
        ax.plot(t[1:-1], smooth[1:-1], 'o-', c='cyan', mec='k', label='Approx')
        ax.plot(t[:2], smooth[:2], '-', c='cyan', mec='k', zorder=0)
        ax.plot(t[-2:], smooth[-2:], '-', c='cyan', mec='k', zorder=0)

        if wiggly_curve:
            log_file.loc[log_file['System'] == sys, 'Note'] = 'Wiggly curve'

        # Customize legend
        legend = ax.legend()
        legend.get_frame().set_facecolor('gray')
        legend.get_frame().set_edgecolor('white')


        msg = '- is wiggly' if wiggly_curve else ''
        print(f"{adjustment=} - {sys=:04d} {msg}")
        plt.close()

        if not wiggly_curve:
            log_file.loc[log_file['System'] == sys, 'Adjustment'] = adjustment
            break

    if wiggly_curve:
        log_file.loc[log_file['System'] == sys, 'Recalculate'] = 'Recalculate'
        save_adjustments(sys, fig, 'recalculate')
    else:
        save_adjustments(sys, fig, adjustment)

    # break

log_file.to_excel('03_adjustment.xlsx', index=False)
