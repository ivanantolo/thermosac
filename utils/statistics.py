import pandas as pd
import numpy as np
from pathlib import Path
from scipy.interpolate import interp1d
from concurrent.futures import ProcessPoolExecutor

from thermosac import Component, Mixture, COSMOSAC
from thermosac.equilibrium import LLE

# =============================================================================
# BASIC
# =============================================================================
CWD = Path(__file__).resolve().parent
ROOT = CWD.parents[0]
DIR_DDB = ROOT / Path('data/experimental/ddb')
DIR_PROFILES = ROOT / Path("data/profiles/UD/sigma3")
nr_to_sys = pd.read_csv(DIR_DDB / "systems.csv", sep=";")

# =============================================================================
# FUNCTIONS
# =============================================================================
def compute_x_with_init(args):
    system, row, dispersion = args
    phase = row['phase']
    T0, x0 = row['T'], [row['x1_L1'], row['x1_L2']]

    # Return NaN if any initial value is NaN
    if any(np.isnan(x0)):
        return np.nan

    # Initialize model instance per process (needed for parallel execution)
    actmodel = initialize_cosmo_model(system, dispersion)
    lle = LLE(actmodel)

    try:
        # Compute binodal points at given T and x
        res = lle.binodal(T0, x0)
        idx = 0 if phase == 'L1' else 1
        return res[idx]
    except Exception:
        # Return NaN on failure
        return np.nan

def get_deviation(system, approx, dispersion, mode='sequential'):
    # Filter the approximation data for the specified system
    calc_vs_exp = approx[approx.sys == system].copy()
    rows = calc_vs_exp.to_dict(orient='records')

    # Prepare arguments for computation
    args = [(system, row, dispersion) for row in rows]

    # Compute x_calc either in parallel or sequentially
    if mode == 'parallel':
        with ProcessPoolExecutor() as executor:
            x_calc = list(executor.map(compute_x_with_init, args))
    else:
        # simple list comprehension for sequential execution
        x_calc = []
        total = len(args)
        print(f"System: {system:04d}")
        print("Calculating exact deviation...")
        for i, arg in enumerate(args, 1):
            x_calc.append(compute_x_with_init(arg))

            # Calculate and display percentage
            percent = (i / total) * 100
            bar = f"[{'#' * int(percent // 2):<50}] {percent:6.2f}%"
            print(f"\r{bar}", end='', flush=True)

        print()  # move to new line after progress bar finishes

    # Store the calculated values in a new column & sort results for consistency
    calc_vs_exp['x1_calc'] = x_calc
    calc_vs_exp.sort_values(['phase', 'T'], inplace=True)

    # Organize final output columns in a logical order
    base_cols = ["sys", "T", "x1", "phase"]
    suffixes = ["", "_calc", "_approx"]
    ordered_cols = [col for base in base_cols for suffix in suffixes
                    if (col := f"{base}{suffix}") in calc_vs_exp.columns]

    return calc_vs_exp.reindex(columns=ordered_cols)

def add_metadata(data):
    file = ROOT / 'data/experimental/ddb/systems.csv'
    sys_ID = pd.read_csv(file, sep=';', index_col='sys')
    meta = {'c1': 'c1(DDB)', 'c2': 'c2(DDB)'}

    # insert each new column right after 'sys'
    pos = data.columns.get_loc('sys')
    for offset, (col, src) in enumerate(meta.items(), 1):
        name = data['sys'].map(sys_ID[src])
        data.insert(pos + offset, col, name)

def calculate_log_deviation(data):
    # Calculate logarithmic deviation (ALDS)
    if 'x1_calc' in data.columns:
        x_calc = data['x1_calc']
    elif 'x1_approx' in data.columns:
        x_calc = data['x1_approx']
    else:
        raise ValueError("Data must contain either 'x1_calc' or 'x1_approx' column.")
    data['ALDS'] = np.abs(np.log(x_calc / data.x1)) * 100

    return data


def print_aalds(stats, dispersion: bool = False):
    """
    Loads binodal deviation data, classifies systems as Aqueous/Nonaqueous,
    computes AALDS statistics, and prints a formatted summary table.

    Args:
        dispersion (bool): Whether to use dispersion-corrected model (SAC_dsp) or not (SAC_2010).
    """
    # Load data
    model = 'COSMO-SAC-dsp' if dispersion else 'COSMO-SAC-2010'

    # Classify as aqueous if either component is water
    is_water = stats['c1'].str.contains('Water', case=False, na=False) | \
               stats['c2'].str.contains('Water', case=False, na=False)
    stats['system_type'] = np.where(is_water, 'Aqueous', 'Nonaqueous')

    # Compute statistics helper
    def compute_stats(sub: pd.DataFrame):
        # Determine which column to use for predicted x1
        if 'x1_calc' in sub.columns:
            x_calc = sub['x1_calc']
        elif 'x1_approx' in sub.columns:
            x_calc = sub['x1_approx']
        else:
            raise ValueError("Data must contain either 'x1_calc' or 'x1_approx' column.")

        # Drop rows with missing values
        sub = sub.dropna(subset=['x1', x_calc.name])
        x_ref = sub['x1']
        x_calc = sub[x_calc.name]

        n_points = len(sub)
        n_systems = sub['sys'].nunique()

        if n_points == 0:
            return n_systems, 0, np.nan

        # Compute AALDS
        devs = np.abs(np.log(x_calc / x_ref))
        aalds = devs.mean() * 100

        return n_systems, n_points, aalds

    # Compute stats per group
    results = {
        'Nonaqueous': compute_stats(stats[stats['system_type'] == 'Nonaqueous']),
        'Aqueous': compute_stats(stats[stats['system_type'] == 'Aqueous']),
        'Overall': compute_stats(stats)
    }

    # Print formatted table
    print('-' * 44)
    print(f'Model variant: COSMO-{model}')
    print('-' * 44)
    print(f"{'Group':<12}{'Systems':>10}{'Points':>10}{'%AALDS':>12}")
    print('-' * 44)
    for grp, (n_sys, n_pts, aalds) in results.items():
        a_str = f"{aalds:12.2f}" if not np.isnan(aalds) else f"{'n/a':>12}"
        print(f"{grp:<12}{n_sys:10d}{n_pts:10d}{a_str:12}")


# =============================================================================
# AUXILLIARY FUNCTIONS
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



# =============================================================================
# APPROXIMATION
# =============================================================================
def pre_process_dataframe(df):
    # Automatically detect column names
    t_col = 'T / K' if 'T / K' in df.columns else 'T'
    x_col = [col for col in df.columns if col.startswith('x1') and 'inner' not in col]
    x1_L1_col, x1_L2_col = x_col

    # Phase 1
    df_L1 = df[['sys', t_col, x1_L1_col]]
    df_L1 = df_L1.rename(columns={x1_L1_col: 'x1', t_col: 'T'})
    df_L1.insert(1, 'phase', 'L1')

    # Phase 2
    df_L2 = df[['sys', t_col, x1_L2_col]]
    df_L2 = df_L2.rename(columns={x1_L2_col: 'x1', t_col: 'T'})
    df_L2.insert(1, 'phase', 'L2')

    res = pd.concat([df_L1, df_L2], ignore_index=True).dropna(subset=['x1'])
    return res

def approx_calc_vs_exp(df_calc, df_exp, polish=True):
    dev_x = [get_deviation_x(df_calc, df_exp, phase) for phase in ['L1', 'L2']]
    calc_vs_exp = pd.concat(dev_x)

    if polish:
        calc_vs_exp.sort_values(['phase', 'T'], inplace=True)
        base_cols = ["sys", "T", "x1", "phase", "section"]
        suffixes = ["", "_calc", "_approx", "_L1", "_L2"]
        ordered_cols = [col for base in base_cols for suffix in suffixes
                        if (col := f"{base}{suffix}") in calc_vs_exp.columns]
        calc_vs_exp = calc_vs_exp.reindex(columns=ordered_cols)

    return calc_vs_exp

def get_deviation_x(df_calc, df_exp, phase='L1'):
    # x-Direction
    exp = df_exp[df_exp['phase'] == phase].sort_values('T')
    calc = df_calc[df_calc['phase'] == phase].sort_values('T')
    other_phase = 'L2' if phase == 'L1' else 'L1'
    calc_other = df_calc[df_calc['phase'] == other_phase].sort_values('T')

    # Ensure required columns exist in exp
    exp = exp.copy()  # Avoid modifying the original dataframe
    exp[f'x1_{phase}'] = np.nan  # Initialize column
    exp[f'x1_{other_phase}'] = np.nan  # Initialize column
    exp['x1_approx'] = np.nan  # Initialize column

    # If calc is empty, return exp unchanged
    if calc.empty:
        return exp

    # Interpolate calculated LLE results to get approximate solution
    approx_x_from_T = interp1d(calc['T'], calc['x1'], bounds_error=False)
    exp[f'x1_{phase}'] = approx_x_from_T(exp['T'])
    exp['x1_approx'] = exp[f'x1_{phase}']

    # Approximate the other phase as well to get complete set of initial values
    approx_other_from_T = interp1d(calc_other['T'], calc_other['x1'], bounds_error=False)
    exp[f'x1_{other_phase}'] = approx_other_from_T(exp['T'])

    return exp
