import pandas as pd
import numpy as np
from pathlib import Path
from scipy.interpolate import interp1d

# =============================================================================
# FUNCTIONS
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
    if 'x1_calc' in data.columns:
        x_calc = data['x1_calc']
    elif 'x1_approx' in data.columns:
        x_calc = data['x1_approx']
    else:
        raise ValueError("Data must contain either 'x1_calc' or 'x1_approx' column.")
    data['ALDS'] = np.abs(np.log(x_calc / data.x1)) * 100

    return data


# =============================================================================
# MAIN
# =============================================================================
if __name__ == '__main__':
    # ─── Define root and data directories ────────────────────────────────────
    ROOT = Path(__file__).resolve().parents[3]
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

    # ─── Extract unique system IDs from the experimental dataset ─────────────
    systems = data_exp['sys'].unique()

    # ─── Compute deviations for each system ──────────────────────────────────
    lle_vs_exp, dsp_vs_exp = [], []
    for i, system in enumerate(sorted(systems)):
        print(f"{system:04d}")

        # Filter out rows for this system
        calc_2010 = data_calc_2010[data_calc_2010.sys == system]
        calc_dsp = data_calc_dsp[data_calc_dsp.sys == system]
        exp = data_exp[data_exp.sys == system]

        # Pre‑process each subset before comparison
        df_lle = pre_process_dataframe(calc_2010)
        df_dsp = pre_process_dataframe(calc_dsp)
        df_exp = pre_process_dataframe(exp)

        # Approximate deviations between calculated and experimental curves
        deviation_lle = approx_calc_vs_exp(df_lle, df_exp)
        deviation_dsp = approx_calc_vs_exp(df_dsp, df_exp)
        lle_vs_exp.append(deviation_lle)
        dsp_vs_exp.append(deviation_dsp)

    # ─── Concatenate per‑system results into final DataFrames ────────────────
    lle_vs_exp = pd.concat(lle_vs_exp, ignore_index=True)
    dsp_vs_exp = pd.concat(dsp_vs_exp, ignore_index=True)

     # ─── Enrich with metadata before saving ────────────────────────────────────
    lle_vs_exp = add_metadata(lle_vs_exp)
    dsp_vs_exp = add_metadata(dsp_vs_exp)

    # ─── Write out the final deviation tables ─────────────────────────────────
    lle_vs_exp.to_csv("approx-SAC_2010.csv", sep=";", index=False)
    dsp_vs_exp.to_csv("approx-SAC_dsp.csv", sep=";", index=False)
