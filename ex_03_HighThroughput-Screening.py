''' Example: GMixScanner (multi system)
    Corresponds to Figure 2 of the main manuscript
'''
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

from thermosac.utils.spacing import spacing
from utils.multi_scanner import GMixMultiScan
from utils.helper import initialize_cosmo_model, convert_system_to_name, nr_to_sys
from utils.helper import plot_details, save_figure

DIR_FIG = Path("./figures") / Path(__file__).stem.lower()

if __name__ == "__main__":
    dispersion = False
    systems = nr_to_sys['sys'].unique() # Select all systems in 'systems.csv'
    systems = [3] # see 'data/experimental/systems.csv' for full list of systems

    initializer = initialize_cosmo_model
    initargs = dict(dispersion=dispersion)
    options = dict(initializer=initializer, initargs=initargs)

    temperatures = np.arange(100, 200 + 10, 10)
    temperatures = [200] # single temperature for testing
    mole_fractions = spacing(0, 1, 51, func_name="sigmoid", inflection=15)
    scanner = GMixMultiScan(systems, temperatures, mole_fractions, **options)
    initial_values, gmix_curves = scanner.find_first_binodal(mode='sequential') # mode: sequential | parallel
    details = scanner.binodal_full

    # Plot results
    for (system, T), df in gmix_curves.groupby(['sys', 'T']):

        names = convert_system_to_name(system)
        res = details[(details['sys']==system) & (details['T']==T)]

        # Spinodal
        xS = res.filter(like='x1_S', axis=1).values.flatten()
        yS = res.filter(like='g_S', axis=1).values.flatten()

        # Binodal
        xB = res.filter(like='x1_L', axis=1).values.flatten()
        yB = res.filter(like='g_L', axis=1).values.flatten()

        # Filter np.nan values caused by inner columns when 4 spinodal/binodal points are added
        xB, yB, xS, yS = [x[~np.isnan(x)] for x in [xB, yB, xS, yS]]

        fig, ax = plt.subplots()
        ax.plot(df.x1, df.gmix, 'k.-')

        ax.plot(xS, yS, 'bo', mfc='w', zorder=6, label='Spinodal')
        ax.plot(xB, yB, 'rs', mfc='w', zorder=6, label='Binodal')

        # Add tangent(s) and mark stable/unstable regions
        plot_details(ax, xS, xB, yB)

        # Add annotations
        kwargs = dict(va='top', transform=ax.transAxes, color='k')
        bbox = dict(facecolor='yellow', alpha=0.9, edgecolor='none', pad=1)
        model = 'COSMO-SAC-dsp' if dispersion else 'COSMO-SAC-2010'
        ax.text(0.01, 1.02, f"T={T}K", bbox=bbox, ha='left', **kwargs)
        ax.text(0.99, 1.02, f"{model}", bbox=bbox, ha='right', **kwargs)

        # Plot settings
        ax.set_title(f'{names[0]} + {names[1]}')
        ax.set_xlabel(f'Mole fraction {names[0]}')
        ax.set_ylabel(r"$\Delta g_{\text{mix}} / (RT)$")
        ax.set_xlim(0, 1)
        ax.tick_params(axis='both', direction='in') # shorten tick length
        ax.legend()

        # Save plot
        save_figure(fig, system, dispersion, T, DIR_FIG, dpi=75)
        plt.show()
