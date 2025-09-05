import pandas as pd
import numpy as np
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor

from thermosac import Component, Mixture, COSMOSAC
from thermosac.equilibrium import LLE

# =============================================================================
# BASIC
# =============================================================================
CWD = Path(__file__).resolve().parent
ROOT = CWD.parents[2]
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
        x_calc = [compute_x_with_init(arg) for arg in args]

    # Store the calculated values in a new column & sort results for consistency
    calc_vs_exp['x1_calc'] = x_calc
    calc_vs_exp.sort_values(['phase', 'T'], inplace=True)

    # Organize final output columns in a logical order
    base_cols = ["sys", "T", "x1", "phase"]
    suffixes = ["", "_calc", "_approx"]
    ordered_cols = [col for base in base_cols for suffix in suffixes
                    if (col := f"{base}{suffix}") in calc_vs_exp.columns]

    return calc_vs_exp.reindex(columns=ordered_cols)

def compute_alds(data: pd.DataFrame) -> pd.Series:
    """
    Compute per-point logarithmic deviation (ALDS, %) on the dilute-component basis.
    If experimental x1 > 0.5, complements (1 - x) are used for both reference and calc.
    Returns a Series aligned with data's index.
    """
    # decide which predicted column to use
    if 'x1_calc' in data.columns:
        x_calc = data['x1_calc'].to_numpy(dtype=float)
    elif 'x1_approx' in data.columns:
        x_calc = data['x1_approx'].to_numpy(dtype=float)
    else:
        raise ValueError("Data must contain either 'x1_calc' or 'x1_approx' column.")

    x_ref = data['x1'].to_numpy(dtype=float)

    # map to dilute component basis
    use_complement = x_ref > 0.5
    x_ref_dil = np.where(use_complement, 1.0 - x_ref, x_ref)
    x_calc_dil = np.where(use_complement, 1.0 - x_calc, x_calc)

    # numerical safety
    eps = 1e-12
    x_ref_dil = np.clip(x_ref_dil, eps, 1.0 - eps)
    x_calc_dil = np.clip(x_calc_dil, eps, 1.0 - eps)

    alds = np.abs(np.log(x_calc_dil / x_ref_dil)) * 100.0
    return pd.Series(alds, index=data.index, name="ALDS")

def add_metadata(data):
    file = ROOT / 'data/experimental/ddb/systems.csv'
    sys_ID = pd.read_csv(file, sep=';', index_col='sys')
    meta = {'c1': 'c1(DDB)', 'c2': 'c2(DDB)'}

    # insert each new column right after 'sys'
    pos = data.columns.get_loc('sys')
    for offset, (col, src) in enumerate(meta.items(), 1):
        name = data['sys'].map(sys_ID[src])
        data.insert(pos + offset, col, name)

    # Calculate logarithmic deviation (ALDS)
    data['ALDS'] = compute_alds(data)

    return data

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
# MAIN
# =============================================================================
if __name__ == '__main__':
    # ─── Load approximation data based on model selection ────────────────────
    dispersion = True
    model = 'SAC_dsp' if dispersion else 'SAC_2010'
    approx = pd.read_csv(CWD / f'approx-{model}.csv', sep=';')
    options = dict(approx=approx, dispersion=dispersion)

    # ─── Select systems for which to compute deviations ──────────────────────
    systems = approx.sys.unique()
    systems = [3345]  # Example: only evaluate one system for demonstration

    # ─── Compute deviations between calculated and experimental data ─────────
    calc_vs_exp = []
    for i, system in enumerate(systems):
        print(f"{i+1}: {system:04d} ...", end="", flush=True)
        deviation = get_deviation(system, **options, mode='sequential') # 'sequential' | 'parallel'
        calc_vs_exp.append(deviation)
        print(" finished")
        # break  # Limit execution to the first system (for testing/debugging)

    # ─── Concatenate results and enrich with metadata ────────────────────────
    calc_vs_exp = pd.concat(calc_vs_exp, ignore_index=True)
    calc_vs_exp = add_metadata(calc_vs_exp)

    # ─── Save results to disk (optional) ─────────────────────────────────────
    calc_vs_exp.to_csv(CWD / f"stats-{model}.csv", sep=";", index=False)
