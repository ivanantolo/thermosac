import re, os
import pandas as pd
import numpy as np
import itertools
from pathlib import Path
from contextlib import contextmanager
import matplotlib.pyplot as plt

from thermosac import Component, Mixture, COSMOSAC

# Define __all__ to control exports for wildcard imports
__all__ = ["initialize_cosmo_model"]

ROOT = Path(__file__).parent.parent
DIR_PROFILES = ROOT / Path("data/profiles/UD/sigma3")
DIR_EXP = ROOT / Path("data/experimental/ddb")

nr_to_sys = pd.read_csv(DIR_EXP / "systems.csv", sep=";")

# =============================================================================
# BASIC
# =============================================================================
def initialize_cosmo_model(system: int, dispersion: bool = False):
    names = convert_system_to_name(system)
    mixture = Mixture(*[Component(name) for name in names])
    actmodel = COSMOSAC(mixture, dispersion=dispersion)
    try:
        # Attempt to import Delaware COSMO-SAC profiles
        actmodel._import_delaware(names, DIR_PROFILES)
    except ValueError:
        return None  # Skip if profiles cannot be imported
    return actmodel

def convert_system_to_name(system):
    row = nr_to_sys.loc[nr_to_sys['sys'] == system, ['c1', 'c2']]
    if row.empty:
        return None  # or raise an error if preferred
    return row.values[0].tolist()

def linear_function(x1, x2, y1, y2):
    def func(x):
        x = np.asarray(x)
        slope = (y2 - y1) / (x2 - x1)
        intercept = y1 - slope * x1
        return slope * x + intercept
    return func

def convert_name_to_system(components):
    a, b = components
    row = nr_to_sys[
        ((nr_to_sys['c1'] == a) & (nr_to_sys['c2'] == b)) |
        ((nr_to_sys['c1'] == b) & (nr_to_sys['c2'] == a))
    ]

    if row.empty:
        return None  # or raise an error if needed

    return row['sys'].values[0]


def filter_by_components(df, target_components):
    """
    Filters rows in df where ['c1', 'c2'] match target_components (order-insensitive).

    Parameters:
        df (pd.DataFrame): The input DataFrame with 'c1' and 'c2' columns.
        target_components (list or set): The two component names to match.

    Returns:
        pd.DataFrame: Filtered DataFrame where (c1, c2) == target_components.
    """
    target_sorted = sorted(target_components)
    comp_array = np.sort(df[['c1', 'c2']].values, axis=1)
    mask = (comp_array == target_sorted).all(axis=1)
    return df[mask].dropna(axis=1)


# =============================================================================
# PLOTTING
# =============================================================================
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

def process_color_args(args, kwargs):
    """Processes color arguments to enable Matplotlib's color cycling when 'C' is used."""

    # Detect 'C' inside shorthand style (e.g., 'Co-', 'C--', 'C^')
    color_pattern = re.compile(r'^C\d?')  # Matches 'C' or 'C0'-'C9' at the start of a string
    new_args = []
    use_color_cycle = False

    for arg in args:
        arg_str = str(arg)
        if color_pattern.match(arg_str):
            use_color_cycle = True
            # Remove 'C' but keep other formatting (marker, linestyle)
            new_arg = color_pattern.sub('', arg_str)  # Strip only 'C' from start
            if new_arg:  # If something is left (like 'o-', 'x--'), keep it
                new_args.append(new_arg)
        else:
            new_args.append(arg)  # Keep args that don't contain 'C'

    # Check if 'C' is in kwargs
    use_color_cycle = use_color_cycle or kwargs.get('color') == 'C' or kwargs.get('c') == 'C'

    if use_color_cycle:
        kwargs.pop('color', None)  # Remove explicit color setting
        kwargs.pop('c', None)      # Remove shorthand color

    return tuple(new_args), kwargs, use_color_cycle

def plot_curve(ax, calc, *args, **kwargs):
    """Plots the curve and returns a list of Line2D objects."""
    label_added = False  # Flag to track if the label has been added
    kwargs = kwargs.copy()
    line_objects = []  # Store plotted lines
    T_col = next((c for c in calc.columns if c in ['T', 'T / K']), None)

    # If 'Curve' exists, group by it; otherwise, treat the entire DataFrame as a single group
    calcs = calc.groupby('Curve', dropna=False) if 'Curve' in calc.columns else [(None, calc)]

    args, kwargs, use_color_cycle = process_color_args(args, kwargs)
    marker_cycle = itertools.cycle(['o', 's', 'D', '^', 'v', '<', '>', 'p', '*', 'x', '+'])

    for curve, dfs in calcs:
        cols_to_drop = [col for col in dfs.columns if dfs[col].isna().all() and col != T_col]
        dfs = dfs.drop(columns=cols_to_drop)  # Remove completely empty columns
        x_cols = [col for col in dfs.columns if col.startswith('x1')]  # Filter relevant columns

        for x_col in x_cols:
            df = dfs.dropna(subset=[x_col])  # Ensure contiguous data points
            style = kwargs.copy()
            if style.get('marker') == 'auto':
                style['marker'] = next(marker_cycle)
            if label_added:
                style.pop('label', None)
            line, = ax.plot(df[x_col], df[T_col], *args, **style)  # Plot and capture the line object
            line_objects.append(line)  # Store the plotted line
            label_added = True  # Mark that the label has been used

    return line_objects  # Return the list of plotted lines


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
def estimate_font_size(title, max_width, default_font_size=12):
    """Estimate a suitable font size for a title to fit within max_width."""
    estimated_width = len(title) * default_font_size * 0.3  # 0.3 is an approximation factor
    if estimated_width < max_width:
        return default_font_size
    else:
        return max(5, default_font_size * max_width / estimated_width)
