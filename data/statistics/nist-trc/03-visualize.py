import os
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
from contextlib import contextmanager

# =============================================================================
# BASIC
# =============================================================================
CWD = Path(__file__).resolve().parent
ROOT = CWD.parents[3]
DIR_DDB = ROOT / Path('data/experimental/ddb')
nr_to_sys = pd.read_csv(DIR_DDB / "systems.csv", sep=";")

# =============================================================================
# PLOTTING
# =============================================================================
@contextmanager
def plot_context(system=None, legend_loc='best', show_title=True, ax=None):
    plt.rcParams['figure.max_open_warning'] = 0

    # Create figure and axes only if no ax is provided
    if ax is None:
        fig, ax = plt.subplots()  # Create a new figure if no ax is provided

    kwargs = dict(va='top', transform=ax.transAxes, color='k')
    bbox = dict(facecolor='yellow', alpha=0.9, edgecolor='none', pad=1)
    names = convert_system_to_name(system)
    if system is not None:
        if show_title:
            # Set plot title
            title = f"{names[0]} + {names[1]}"
            font_size = estimate_font_size(title, max_width=297)
            ax.set_title(title, fontsize=font_size)
        # Annotate the plot
        ax.text(0.01, 1.0, f"System:{system:04d}", bbox=bbox, ha='left', **kwargs)

    ax.set_xlabel(rf'Mole fraction {names[0]}')
    ax.set_ylabel(r'$T$ / K')

    try:
        # Yield the ax handle for adding plot elements
        yield ax
    finally:

        if legend_loc is not None:
            # Check if there are any labels
            handles, labels = ax.get_legend_handles_labels()
            if handles:
                legend = ax.legend(loc=legend_loc)
                ax.add_artist(legend)

        # Post-processing
        ax.set_xlim(0, 1)
        ax.tick_params(axis='both', direction='in') # shorten tick length
        plt.show()

def plot_curve(ax, calc, *args, **kwargs):
    """Plots the curve and returns a list of Line2D objects."""
    label_added = False  # Flag to track if the label has been added
    T_col = next((c for c in calc.columns if c in ['T', 'T / K']), None)

    # If 'Curve' exists, group by it; otherwise, treat the entire DataFrame as a single group
    calcs = calc.groupby('Curve', dropna=False) if 'Curve' in calc.columns else [(None, calc)]

    for curve, dfs in calcs:
        cols_to_drop = [col for col in dfs.columns if dfs[col].isna().all() and col != T_col]
        dfs = dfs.drop(columns=cols_to_drop)  # Remove completely empty columns
        x_cols = [col for col in dfs.columns if col.startswith('x1')]  # Filter relevant columns

        for x_col in x_cols:
            df = dfs.dropna(subset=[x_col])  # Ensure contiguous data points
            if label_added:
                kwargs.pop('label', None)
            ax.plot(df[x_col], df[T_col], *args, **kwargs)  # Plot and capture the line object
            label_added = True  # Mark that the label has been used

def plot_calc_vs_exp(ax, data, zorder=None, **kwargs):
    # Common styles
    line_style = dict(lw=0.8, ls=':', zorder=zorder or 0)
    point_style = dict(marker='none', c='k')
    point_style.update(kwargs)

    # Choose x-coordinate to plot against
    if 'x1_calc' in data.columns:
        x_target = data['x1_calc']
    elif 'x1_approx' in data.columns:
        x_target = data['x1_approx']
    else:
        raise ValueError("Data must contain either 'x1_calc' or 'x1_approx' column.")

    # Extract coordinates
    x, y = data['x1'], data['T']
    line_x, line_y = [x, x_target], [y, y]

    # Plot connecting lines
    ax.plot(line_x, line_y, point_style["c"], **line_style)

    # Plot points
    ax.plot(x_target, y, **point_style, ls='none')

def save_plot(ax, sys, output_dir='.'):
    # Ensure the directory exists
    os.makedirs(output_dir, exist_ok=True)
    # Save figure
    fig = ax.get_figure()
    fig.savefig(f'{output_dir}/{sys:04}.png', dpi=75)
    plt.close(fig)

# =============================================================================
# AUXILLIARY FUNCTIONS
# =============================================================================
def convert_system_to_name(system):
    row = nr_to_sys.loc[nr_to_sys['sys'] == system, ['c1', 'c2']]
    if row.empty:
        return None  # or raise an error if preferred
    return row.values[0].tolist()

def estimate_font_size(title, max_width, default_font_size=12):
    """Estimate a suitable font size for a title to fit within max_width."""
    estimated_width = len(title) * default_font_size * 0.3  # 0.3 is an approximation factor
    if estimated_width < max_width:
        return default_font_size
    else:
        return max(5, default_font_size * max_width / estimated_width)

# =============================================================================
# MAIN
# =============================================================================
if __name__ == '__main__':
    # ─── Define root and data directories ────────────────────────────────────
    DIR_CALC = ROOT / Path('data/calculated/lle_results')
    DIR_EXP = ROOT / Path('data/experimental/nist-trc')

    # ─── Load experimental and calculated data ───────────────────────────────
    options = dict(sep=';', low_memory=False)
    data_calc_2010 = pd.read_csv(DIR_CALC / 'lle-SAC_2010.csv', **options)
    data_calc_dsp = pd.read_csv(DIR_CALC / 'lle-SAC_dsp.csv', **options)
    data_exp = pd.read_excel(DIR_EXP / 'LLeDbPGL6ed96b.xlsx')

    # ─── Load and append any ‘missing’ systems to the calculated data ────────
    add_2010 = pd.read_csv(DIR_EXP / 'missing_systems/missing-SAC_2010.csv', sep=';')
    add_dsp = pd.read_csv(DIR_EXP / 'missing_systems/missing-SAC_dsp.csv', sep=';')
    data_calc_2010 = pd.concat([data_calc_2010, add_2010], ignore_index=True)
    data_calc_dsp = pd.concat([data_calc_dsp, add_dsp], ignore_index=True)

    # ─── Load deviations (uncomment 'approx' lines to use approximated values)
    deviations_lle = pd.read_csv(CWD / 'stats-SAC_2010.csv', sep=';')
    deviations_dsp = pd.read_csv(CWD / 'stats-SAC_dsp.csv', sep=';')
    # deviations_lle = pd.read_csv(CWD / 'approx-SAC_2010.csv', sep=';')
    # deviations_dsp = pd.read_csv(CWD / 'approx-SAC_dsp.csv', sep=';')

    # ─── Extract unique system IDs from the experimental dataset ─────────────
    systems = data_exp['sys'].unique()

    # Plot results
    for i, system in enumerate(sorted(systems)):
        print(f"{system:04d}")
        calc_2010 = data_calc_2010[data_calc_2010.sys == system]
        calc_dsp = data_calc_dsp[data_calc_dsp.sys == system]
        exp = data_exp[data_exp.sys == system]

        deviation_lle = deviations_lle[deviations_lle.sys == system]
        deviation_dsp = deviations_dsp[deviations_dsp.sys == system]

        with plot_context(system, show_title=True) as ax:
            plot_curve(ax, exp.drop(columns='x1_L2'), 'o', c='C0', ms=4, label='Phase 1 (NIST/TRC)')
            plot_curve(ax, exp.drop(columns='x1_L1'), 'o', c='C1', ms=4, label='Phase 2 (NIST/TRC)')
            plot_curve(ax, calc_2010, 'k-', label='COSMO-SAC-2010')
            plot_curve(ax, calc_dsp, 'r-', label='COSMO-SAC-dsp')
            plot_calc_vs_exp(ax, deviation_lle)
            plot_calc_vs_exp(ax, deviation_dsp, c='r')

        # save_plot(ax, system, output_dir=CWD/'./figures')

        break
