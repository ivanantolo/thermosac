import os
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from contextlib import contextmanager

# =============================================================================
# BASIC
# =============================================================================
CWD = Path(__file__).resolve().parent
ROOT = CWD.parents[0]
DIR_DDB = ROOT / Path('data/experimental/ddb')
nr_to_sys = pd.read_csv(DIR_DDB / "systems.csv", sep=";")

# =============================================================================
# PLOTTING
# =============================================================================
@contextmanager
def plot_context(system=None, legend_loc='best', dispersion=None,
                 show_title=True, ax=None):
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

    if dispersion is not None:
        model = 'COSMO-SAC-dsp' if dispersion else 'COSMO-SAC-2010'
        ax.text(0.99, 1.0, f"{model}", bbox=bbox, ha='right', **kwargs)

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

def plot_details(ax, xS, xB, yB, fill_area=True):
    ylim = ax.get_ylim()
    # Spinodal
    if fill_area:
        # [ax.axvline(k, color='k', ls='-', lw=.5) for k in xS]
        for i in range(0, len(xS), 2):
            ax.fill_betweenx(ax.get_ylim(), xS[i], xS[i + 1], color='gray', alpha=1)

    # Binodal
    if fill_area:
        # [ax.axvline(k, color='k', ls='-', lw=.5) for k in xB]
        for i in range(0, len(xB), 2):
            ax.fill_betweenx(ax.get_ylim(), xB[i], xB[i + 1], color='silver', alpha=1, zorder=-1)
    for i in range(0, len(xB) - 1, 2):
        xx, yy = xB[i:i+2], yB[i:i+2]
        f = linear_function(*xx, *yy)
        edges = [0, 1]
        ax.plot(edges, f(edges), 'r-', lw=.8, zorder=5)
        ax.plot(xx, yy, 'rs', mec='k', zorder=5)

    ax.set_ylim(ylim)

def save_plot(ax, sys, output_dir='.'):
    # Ensure the directory exists
    os.makedirs(output_dir, exist_ok=True)
    # Save figure
    fig = ax.get_figure()
    fig.savefig(f'{output_dir}/{sys:04}.png', dpi=75)
    plt.close(fig)

def save_figure(fig, system, dispersion=False, T=None, dir_fig=None, dpi=75):

    if dir_fig is None:
        raise ValueError("You must provide dir_fig (target output directory).")

    # Ensure dir_fig is a Path object
    dir_fig = Path(dir_fig)
    dir_fig.mkdir(parents=True, exist_ok=True)

    # Clean model name from dispersion flag
    model = "SACdsp" if dispersion else "SAC2010"

    # Build filename
    filename = f"{model}-system={system:04d}"
    filename += f"-T={T:.0f}K.png" if T is not None else ".png"


    # Ensure directory exists
    dir_fig.mkdir(parents=True, exist_ok=True)

    # Save the figure
    fig.savefig(dir_fig / filename, dpi=dpi)

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

def linear_function(x1, x2, y1, y2):
    def func(x):
        x = np.asarray(x)
        slope = (y2 - y1) / (x2 - x1)
        intercept = y1 - slope * x1
        return slope * x + intercept
    return func
