import pickle
import warnings
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d

from phaseq.equilibrium.lle import GMixScanner, LLE
from phaseq.utils.spacing import spacing
from utils.helper import initialize_cosmo_model, linear_function
from utils.plotting import plot_gmix
from utils.statistics.plotter import plot_context

plt.rcParams['figure.max_open_warning'] = 0

# =============================================================================
dispersion = True
initializer = initialize_cosmo_model

# =============================================================================
# Load list of systems
# =============================================================================
DIR_ROOT = Path("../../calculated/lle/initial_values")
DIR_FIG = Path('figures/statistics/gmix')
systems = [6158]

# =============================================================================
# Get initial values from GMIX
# =============================================================================
temperatures = np.arange(100, 210, 20)
# temperatures = [140]
mole_fractions = spacing(0, 1, 51, func_name="sigmoid", inflection=25)

if __name__ == "__main__":
    for system in systems:
        initargs = dict(system=system, dispersion=dispersion)
        options = dict(initializer=initializer, initargs=initargs)
        actmodel = initialize_cosmo_model(system, dispersion)
        # lle = LLE(actmodel)
        scanner = GMixScanner(actmodel, temperatures, mole_fractions)
        # gmix = scanner.get_all_gmix(mode='sequential', **options)
        # with open(f'gmix-sys={system:04d}-dsp={dispersion}.pkl', 'wb') as file:
        #     pickle.dump(gmix, file)
        # with open(f'gmix-sys={system:04d}-dsp={dispersion}.pkl', 'rb') as file:
        #     gmix = pickle.load(file)
        # options['gmix'] = gmix
        binodal, gmix = scanner.find_all_binodal(mode='sequential', **options)

        for T, dfs in gmix.groupby('T', sort=False):
            if T not in temperatures:
                continue

            x1, g1 = dfs.x1.values, dfs.gmix.values

            # Data
            x, y = x1, g1
            dy = np.gradient(y, x)
            ddy = np.gradient(dy, x)
            f_y = interp1d(x, y, kind='linear', bounds_error=True)
            f_dy = interp1d(x, dy, kind='linear', bounds_error=True)

            # Spinodal
            xS = scanner._approx_spinodal(x, ddy)
            yS = f_y(xS)

            # # Binodal
            xB = scanner._approx_binodal(xS, f_y, f_dy)
            yB = f_y(xB)

            # Graphs
            with plot_context(sys=system, dispersion=dispersion, legend_loc='lower center') as ax:
                plot_gmix(system, actmodel.mixture, T, x1, g1, color='k', ax=ax)
    # =============================================================================
                ax.set_ylim(None, None)
    # =============================================================================

                # Spinodal
                ax.plot(xS, yS, 'bo', mfc='w', zorder=6)
                [ax.axvline(k, color='k', ls=':') for k in xS]
                # for ax in axes:
                for i in range(0, len(xS), 2):
                    ax.fill_betweenx(ax.get_ylim(), xS[i], xS[i + 1], color='gray', alpha=1)

                # Binodal
                [ax.axvline(k, color='k', ls=':') for k in xB]
                for i in range(0, len(xB) - 1, 2):
                    xx, yy = xB[i:i+2], yB[i:i+2]
                    f = linear_function(*xx, *yy)
                    edges = [0, 1]
                    ax.plot(edges, f(edges), 'r-', mec='k', zorder=7)
                    ax.plot(xx, yy, 'rs', mec='k', zorder=7)
                    ax.fill_betweenx(ax.get_ylim(), xB[i], xB[i + 1], color='silver', alpha=1, zorder=-1)

                # # Polish results
                # inits = np.split(xB, len(xB) // 2)
                # binodal = np.concatenate([lle.binodal(T, x0) for x0 in inits if not np.isnan(x0).any()])
                # for x in binodal:
                #     ax.axvline(x, color='k', ls=':')
                # print(system, T, *binodal, *actmodel.mixture.names)

                # Suppress the specific warning
                with warnings.catch_warnings():
                    warnings.filterwarnings("ignore", message=".*The figure layout has changed to tight.*")
                    plt.tight_layout(pad=0.04)
                    plt.subplots_adjust(top=0.86, hspace=0.0, wspace=0.0)
