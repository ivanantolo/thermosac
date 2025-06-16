import os
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
from phaseq.equilibrium import LLE
from utils.helper import initialize_cosmo_model

# DIR_FIG = Path('figures/statistics')
DIR_STAT = Path('./')
# DIR_RES = Path("./data/calculated/lle/results")

dispersion = False
# load = lambda file: pd.read_csv(file, sep=';').sort_values(by=['sys', 'T'])
# data_lle = load(DIR_RES / 'lle_05.csv').reset_index(drop=True)
# data_dsp = load(DIR_RES / 'dsp_05.csv').reset_index(drop=True)
# data_calc = data_dsp if dispersion else data_lle

# data = pd.read_excel(DIR_STAT / 'data_cleaning.xlsx')
# df = data['LCST'].dropna().astype(int)
# systems = df.unique()
systems = [6156, 6157, 6158]

folder = 'dsp' if dispersion else 'lle'
DIR_SAVE = DIR_STAT / folder
# DIR_FIG = DIR_FIG / folder
os.makedirs(DIR_SAVE, exist_ok=True)
# os.makedirs(DIR_FIG, exist_ok=True)

for system in systems:
    actmodel = initialize_cosmo_model(system, dispersion)
    lle = LLE(actmodel)
    # calc = data_calc[data_calc['sys']==system]
    # if calc.empty:
    #     continue
    # T, *x0 = calc.iloc[0][['T','x1_L1','x1_L2']].values.T
    T, x0 = 160, [0, 1]
    # T, x0 = 198, [0.05, 0.72]
    # binodal = lle.binodal(T, x0)
    # print(system, T, *binodal, *actmodel.mixture.names)

    # if T <= 140:
    #     continue

    skip_first_iteration=False
    options = dict(
        # max_gap=0.01,
        # exponent=1,
        # max_T = 300,
        # max_change=0.02,
        # max_change_retries=20,
        # shrink_factor_change = 0.2,
        # shrink_factor_gap = 0.1,
        # max_gap_retries = 10,
        skip_first_iteration=skip_first_iteration,
        # use_dynamic_x0=True,
        # use_constant_dT=True,
        # use_initial_dT0=True,
        # print_traceback=False,
        # range_extension_factor = 5,
        # check_curve_direction = False,
        )

    # gap = abs(x0[1] - x0[0])
    # dT0 = .1 / (gap**options['exponent'])
    miscibility = lle.miscibility(T, x0, dT0=10, **options)
    if skip_first_iteration:
        miscibility = miscibility.iloc[1:].sort_values(by='T')
    else:
        miscibility = miscibility.sort_values(by='T')
    miscibility.insert(0,'sys',system)
    miscibility[['c1', 'c2']] = actmodel.mixture.names
    miscibility.to_csv(f'{DIR_SAVE}/{system:04d}.csv', sep=';', index=False)

    fig, ax = plt.subplots()
    ax.plot(miscibility['x1_L1'], miscibility['T'], '.-', label='LLE')
    ax.plot(miscibility['x1_L2'], miscibility['T'], '.-')
    # plt.savefig(f'{DIR_FIG}/{system:04d}.png')
    # plt.close(fig)

    # break
