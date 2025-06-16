''' Example: LLETracer (single system)
'''
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pathlib import Path

from thermosac import Component, Mixture, COSMOSAC
from thermosac.equilibrium.lle import LLE
from utils.helper import filter_by_components, convert_name_to_system, save_figure

DIR_PROFILES = Path("./data/profiles/UD/sigma3")
DIR_INITS = Path("data/calculated/initial_values")
DIR_FIG = Path("./figures") / Path(__file__).stem.lower()

if __name__ == "__main__":
    names = ["ETHYLENE_GLYCOL", "2,5-Dimethyltetrahydrofuran"]
    system = convert_name_to_system(names)
    dispersion = False
    model = 'SAC_dsp' if dispersion else 'SAC_2010'
    mixture = Mixture(*[Component(name) for name in names])
    actmodel = COSMOSAC(mixture, dispersion=dispersion)

    # Import Delaware COSMO-SAC profiles
    actmodel._import_delaware(names, DIR_PROFILES)

    # LLE Tracing
    lle = LLE(actmodel)
    initial_values = pd.read_csv(DIR_INITS / f"init-{model}.csv", sep=';')
    init = filter_by_components(initial_values, names)
    T0, *x0 = init.iloc[0][['T','x1_L1','x1_L2']].values.T
    miscibility = lle.miscibility(T0, x0, dT0=10)

    # Plot results
    fig, ax = plt.subplots()
    ax.plot(miscibility['x1_L1'], miscibility['T'], '.-', label='LLE')
    ax.plot(miscibility['x1_L2'], miscibility['T'], '.-')

    # Add annotations
    kwargs = dict(va='top', transform=ax.transAxes, color='k')
    bbox = dict(facecolor='yellow', alpha=0.9, edgecolor='none', pad=1)
    ax.text(0.01, 1.01, f"COSMO-{model.replace('_', '-')}", bbox=bbox, ha='right', **kwargs)
    ax.text(0.99, 1.01, f"System={system:0>4}", bbox=bbox, ha='right', **kwargs)

    # Plot settings
    ax.set_title(f'{mixture}')
    ax.set_xlabel(f'Mole fraction {mixture[0]}')
    ax.set_ylabel(r"$T$ / K")
    ax.set_xlim(0, 1)
    ax.tick_params(axis='both', direction='in') # shorten tick length

    # Save plot
    save_figure(fig, system, dispersion, None, DIR_FIG, dpi=75)
    plt.show()
