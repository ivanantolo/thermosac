"""
OUTLIER DETECTION AND SCREENING SCRIPT

This script processes data from various systems to detect and log outliers.

Key functionalities:
- Iterates over all systems to analyze temperature-related data.
- Allows selection of the upper or lower part of the dataset for analysis.
- Detects outliers using methods combined with an "OR" logic.
- Generates plots and logs details about detected outliers.
- Outputs an XLSX file with systems and their respective outlier counts.
"""

import pandas as pd
from pathlib import Path
import traceback

from phaseq.equilibrium import LLEOutlierDetector
from tools.helper import log_outliers, generate_and_save_plot, save_outlier_log

dispersion = True

DIR_DATA = Path('./data')
FILE = 'lle_dsp_02.csv' if dispersion else 'lle_02.csv'
data = pd.read_csv(DIR_DATA / FILE, sep=';')
data = data.sort_values(by=['sys', 'T']).reset_index(drop=True)

systems = data['sys'].unique()
systems = [5336]

dtc = LLEOutlierDetector()
outlier_log = []
for i, sys in enumerate(systems):
    try:
        dfs = data[data.sys==sys].sort_values(by='T')

        df = dtc.select_part(dfs, part='upper')
        df = dtc.normalize_data(df)
        df = dtc.filter_data(df, threshold=0.2)
        df = dtc.normalize_data(df)
        # DC: direction change, BC: big change, TA: Taylor approx.
        df = dtc.detect_outliers(df, methods=['DC','BC'])

        log_outliers(sys, df, len(dfs), outlier_log)
        generate_and_save_plot(df, sys, save=False)
        print(f"{i+1:02d}:{sys:04d} processed.")
    except Exception as e:
        print(f"Error processing system {sys:04d}: {e}")
        traceback.print_exc()
    # break

def save():
    save_outlier_log(df, outlier_log, save=True)
