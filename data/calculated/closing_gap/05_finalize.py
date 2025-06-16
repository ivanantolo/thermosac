import pandas as pd
from pathlib import Path

from tools.helper import process_data, get_approx

dispersion = True
DIR_DATA = Path('./data')
DIR_RES = Path(("with" if dispersion else "without") + "_dispersion")

file = 'lle_dsp_02.csv' if dispersion else 'lle_02.csv'
new_file = 'lle_dsp_03.csv' if dispersion else 'lle_03.csv'
data = pd.read_csv(DIR_DATA / file, sep=';')
data = data.sort_values(by=['sys', 'T']).reset_index(drop=True)
overview = pd.read_excel(DIR_RES / '04_recalculate.xlsx')
# =============================================================================
recalculation = overview[overview['Recalculate'].notna()]
recalculation = process_data(recalculation)
read = lambda sys: pd.read_excel(DIR_RES / '04_recalculate' / f'{sys:04d}.xlsx')
recalc = pd.concat([read(sys) for sys in recalculation['System'].unique()])
# =============================================================================
outliers = overview[overview['Recalculate'].isna()]
outliers = process_data(outliers)
smooth = get_approx(data, outliers, plot=False)
# =============================================================================
res = pd.concat([recalc, smooth], ignore_index=True)
res = res.sort_values(by=['sys', 'T']).reset_index(drop=True)
res.to_csv(new_file, sep=';', index=False)
