''' Example: GMixScanner (single system)
    Corresponds to Figure 2 of the main manuscript
'''
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

from thermosac import Component, Mixture, COSMOSAC
from thermosac.equilibrium.lle import GMixScanner
from thermosac.utils.spacing import spacing
from utils.helper import plot_details, convert_name_to_system, save_figure

DIR_PROFILES = Path("./data/profiles/UD/sigma3")
DIR_FIG = Path("./figures") / Path(__file__).stem.lower()

if __name__ == "__main__":
    names = ["ETHYLENE_GLYCOL", "2,5-Dimethyltetrahydrofuran"]
    system = convert_name_to_system(names)
    dispersion = False
    mixture = Mixture(*[Component(name) for name in names])
    actmodel = COSMOSAC(mixture, dispersion=dispersion)

    # Import Delaware COSMO-SAC profiles
    actmodel._import_delaware(names, DIR_PROFILES)

    temperatures = np.arange(100, 200 + 10, 10)
    temperatures = [200] # single temperature for testing
    mole_fractions = spacing(0, 1, 51, func_name="sigmoid", inflection=15)
    scanner = GMixScanner(actmodel, temperatures, mole_fractions)
    # Try 'find_first_binodal()' as well
    initial_values, gmix_curves = scanner.find_all_binodal(mode='sequential') # mode: sequential | parallel
    details = scanner.binodal_full

    for T, df in gmix_curves.groupby('T', sort=False):

        res = details[details['T'] == T]

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
        ax.text(0.01, 1.01, f"{model}", bbox=bbox, ha='left', **kwargs)
        ax.text(0.5, 1.01, f"T={T}K", bbox=bbox, ha='center', **kwargs)
        ax.text(0.99, 1.01, f"System={system:0>4}", bbox=bbox, ha='right', **kwargs)

        # Plot settings
        ax.set_title(f'{mixture}')
        ax.set_xlabel(f'Mole fraction {mixture[0]}')
        ax.set_ylabel(r"$\Delta g_{\text{mix}} / (RT)$")
        ax.set_xlim(0, 1)
        ax.tick_params(axis='both', direction='in') # shorten tick length
        ax.legend()

        # Save plot
        save_figure(fig, system, dispersion, T, DIR_FIG, dpi=75)
        plt.show()
