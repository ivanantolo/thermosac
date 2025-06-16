# === Imports & Setup ===
from pathlib import Path
import pandas as pd
import numpy as np

# === Control Panel ===
def main():
    global stats
    # stats = get_statistics(group='system', recalculate=True, save=True)
    # stats = get_statistics(group='class_pair', recalculate=True, save=True)
    # stats = get_statistics(member='component', recalculate=True, save=True)
    # stats = get_statistics(member='class', recalculate=True, save=True)
    stats = compute_total_aads_from_raw(save=True, restrict_by_direction=True)
    pass

def compute_total_aads_from_raw(save=True, file_path="stats_by_total_AAD.xlsx",
                                restrict_by_direction=False):
    models = {
        'COSMO-SAC-2010': lle_vs_exp[~lle_vs_exp['sys'].isin(null_dispersion.sys)],
        'COSMO-SAC-dsp': dsp_vs_exp,
        'COSMO-SAC-2010 (full)': lle_vs_exp
    }

    variable_pairs = [
        ('x1', 'x1_calc'),
        ('T', 'T_calc'),
        ('K1', 'K1_calc'),
        ('x1_L1', 'x1_calc'),  # phase = 'L1'
        ('x1_L2', 'x1_calc'),  # phase = 'L2'
    ]

    results = []

    for model, df in models.items():
        for var, calc_col in variable_pairs:
            if var.startswith('x1_L'):
                phase = var.split('_')[1]
                mask = df['phase'] == phase
                subset = df.loc[mask, ['x1', calc_col]].dropna()
                true_col = 'x1'
            else:
                subset = df[[var, calc_col]].dropna()
                true_col = var

            # Apply the optional filter
            subset = apply_direction_filter(subset, df, var, restrict_by_direction)

            if subset.empty:
                continue

            diff = subset[calc_col] - subset[true_col]
            ad = diff.mean()
            aad = diff.abs().mean()
            n = len(diff)

            results.append({
                'Model': model,
                'Variable': var,
                'AD': ad,
                'AAD': aad,
                'Datapoints': n
            })

    df_results = pd.DataFrame(results)

    if save:
        df_results.to_excel(file_path, index=False)

    return df_results




def compute_deviations(group):
    # Build the result in correct order
    result = {}
    used_indices = set()

    # Add deviations
    for var in ['x1', 'T', 'K1']:
        valid = group[[var, f"{var}_calc"]].dropna()
        diff = valid[f"{var}_calc"] - valid[var]
        result[(var, 'AD')] = diff.mean()
        result[(var, 'AAD')] = diff.abs().mean()
        result[(var, 'n')] = len(diff) if len(diff) > 0 else None

        # Track which rows were used
        used_indices.update(valid.index)

    # Unique number of datapoints effectively contributing to any deviation
    result[('Datapoints', 'N_eff')] = len(used_indices)

    # Total number of rows in group
    result[('Datapoints', 'N_tot')] = len(group)


    # Pre-fill with np.nan to keep numeric dtype
    for phase in ['L1', 'L2']:
        colname = f"x1_{phase}"
        result[(colname, 'AD')] = np.nan
        result[(colname, 'AAD')] = np.nan
        result[(colname, 'n')] = np.nan

    # Handle x1 split by phase: L1 and L2
    for phase in ['L1', 'L2']:
        phase_mask = group['phase'] == phase
        colname = f"x1_{phase}"
        if phase_mask.any():
            valid = group.loc[phase_mask, ['x1', 'x1_calc']].dropna()
            diff = valid['x1_calc'] - valid['x1']
            result[(colname, 'AD')] = diff.mean()
            result[(colname, 'AAD')] = diff.abs().mean()
            result[(colname, 'n')] = len(diff) if len(diff) > 0 else None

    ans = pd.Series(result)
    return pd.Series(result)

def get_statistics(group=None, member=None, recalculate=False, save=False, file_path=None):
    value = group or member
    file_path = (file_path or f"stats_by_{value}") + '.xlsx'

    if recalculate:
        # Exclude systems/components without dispersion parameter
        reduced = lle_vs_exp[~lle_vs_exp['sys'].isin(null_dispersion.sys)]

        stats = {}
        stats['COSMO-SAC-2010'] = get_stats(reduced, group, member)
        stats['COSMO-SAC-dsp'] = get_stats(dsp_vs_exp, group, member)
        stats['COSMO-SAC-2010 (full)'] = get_stats(lle_vs_exp, group, member)

        if save:
            save_to_excel(file_path, stats)
    else:
        stats = load_from_excel(file_path)

    return stats

def get_stats(calc_vs_exp, group=None, member=None):
    if bool(group):
        return get_group_stats(calc_vs_exp, group)
    elif bool(member):
        return get_member_stats(calc_vs_exp, member)
    else:
        raise ValueError('Select valid group or member statistics.')

def get_member_stats(calc_vs_exp, member='component'):
    # Step 1: Create separate DataFrames for c1/class1 and c2/class2, and rename columns
    df_c1 = calc_vs_exp[['c1', 'class1']].rename(columns={'c1': 'component', 'class1': 'class'})
    df_c2 = calc_vs_exp[['c2', 'class2']].rename(columns={'c2': 'component', 'class2': 'class'})

    # Step 2: Concatenate the two DataFrames
    comp_class = pd.concat([df_c1, df_c2])

    # Step 3: Drop duplicates to get unique component-class pairs
    comp_class = comp_class.drop_duplicates().reset_index(drop=True)

    members = comp_class[member].unique()
    col1, col2 = ['c1', 'c2'] if member == 'component' else ['class1', 'class2']

    # Dictionary to hold stats per component
    deviations = {}
    for memb in members:
        group = calc_vs_exp[(calc_vs_exp[col1] == memb) | (calc_vs_exp[col2] == memb)]
        deviation = compute_deviations(group)
        deviations[memb] = deviation

    deviations = pd.DataFrame.from_dict(deviations, orient='index')

    # Map the class info to the index of deviations
    if member == 'component':
        class_map = comp_class.set_index('component')['class']
        deviations.insert(0, ('Substance', 'class'), deviations.index.map(class_map))

    deviations.index.name = ('Substance', member)
    deviations = deviations.reset_index()
    deviations = drop_empty_stats(deviations)
    deviations.columns = pd.MultiIndex.from_tuples(deviations.columns, names=['var', 'stat'])

    return deviations


def get_group_stats(calc_vs_exp, group='system'):
    group_cols = {'system': 'sys', 'class_pair': 'class_ID'}
    group_col = group_cols[group]

    # Group and compute
    deviations = calc_vs_exp.groupby(group_col).apply(compute_deviations)

    # Finalize column MultiIndex names
    deviations = add_metadata(calc_vs_exp, deviations, group_col)
    deviations.columns = pd.MultiIndex.from_tuples(deviations.columns, names=['var', 'stat'])

    # Rename index and reset
    index_names = {'sys': ('System', 'ID'), 'class_ID': ('MainClass', 'ID')}
    index_name = index_names[group_col]
    deviations.index.name = index_name
    deviations = drop_empty_stats(deviations)
    deviations = deviations.reset_index()

    # Sort
    deviations = deviations.sort_values(by=[index_name], ignore_index=True)

    return deviations


# === Helper Functions ===
def drop_empty_stats(df):
    all_stat_cols = [col for col in df.columns if col[1] in ['AD', 'AAD']]
    stat_cols = [col for col in all_stat_cols if col[0] in ['x1', 'T', 'K1']]
    return df.dropna(subset=stat_cols, how='all')

def save_to_excel(file_path, stats):
    with pd.ExcelWriter(file_path, engine="openpyxl") as writer:
        for sheet_name, stat in stats.items():
            stat.to_excel(writer, sheet_name=sheet_name)

def load_from_excel(file_path):
    kwargs = dict(sheet_name=None, header=[0, 1], index_col=0)
    stats = pd.read_excel(file_path, **kwargs)
    return stats

def add_metadata(calc_vs_exp, deviations, groupby_col):
    """
    Adds metadata (system or class) to the deviations DataFrame using MultiIndex columns.

    Parameters:
    - calc_vs_exp: DataFrame containing metadata
    - deviations: DataFrame with MultiIndex index (groupby_col, ID)
    - groupby_col: str, either 'sys' or 'class_ID'

    Returns:
    - deviations DataFrame with metadata columns added
    """
    # Full metadata mapping
    meta_column_map = {
        'c1': ('Component', 'c1'),
        'c2': ('Component', 'c2'),
        'class_ID': ('MainClass', 'ID'),
        'class1': ('MainClass', 'class1'),
        'class2': ('MainClass', 'class2'),
    }

    # Pick relevant keys for the current groupby context
    if groupby_col == 'sys':
        selected_keys = ['c1', 'c2', 'class_ID', 'class1', 'class2']
    elif groupby_col == 'class_ID':
        selected_keys = ['class1', 'class2']
    else:
        raise ValueError(f"Unsupported groupby_col: {groupby_col}")

    # Extract and rename metadata
    metadata = calc_vs_exp.groupby(groupby_col)[selected_keys].first()
    metadata.columns = pd.MultiIndex.from_tuples([meta_column_map[col] for col in metadata.columns])

    # Join metadata into deviations
    deviations = deviations.join(metadata, how='left')

    # Reorder: place metadata columns at the beginning
    meta_cols = list(metadata.columns)
    other_cols = [col for col in deviations.columns if col not in meta_cols]
    deviations = deviations[meta_cols + other_cols]

    return deviations

def apply_direction_filter(subset, df, var, restrict_by_direction=True):
    """
    Optionally filters a subset by removing rows where the direction column is not NaN.
    For 'T', the direction column is assumed to be 'section_calc'; otherwise 'phase_calc'.

    Parameters:
        subset (pd.DataFrame): The subset of data to be filtered.
        df (pd.DataFrame): The original full DataFrame, needed for direction column.
        var (str): The variable name (e.g. 'T', 'x1', etc).
        restrict_by_direction (bool): Whether to apply the filter.

    Returns:
        pd.DataFrame: The filtered subset.
    """
    if not restrict_by_direction:
        return subset

    direction_col = 'phase_calc' if var == 'T' else 'section_calc'

    if direction_col in df.columns:
        # Merge direction info into the subset based on index
        if direction_col not in subset.columns:
            subset = subset.merge(
                df[[direction_col]],
                left_index=True, right_index=True, how='left'
            )
        # Keep rows where direction_col is NaN
        subset = subset[subset[direction_col].isna()]
        # Drop direction_col and NaNs in remaining cols
        subset = subset.drop(columns=[direction_col]).dropna()

    return subset



# === Run Main ===
if __name__ == "__main__":
    ROOT = Path('../..')
    DIR_STATS = ROOT / Path("statistics/calc_vs_exp")
    kwargs = dict(sheet_name=None, header=[0, 1], index_col=0)
    lle_vs_exp = pd.read_csv(DIR_STATS / "lle_vs_exp.csv", sep=';')
    dsp_vs_exp = pd.read_csv(DIR_STATS / "dsp_vs_exp.csv", sep=';')

    limit = 28
    # lle_vs_exp = lle_vs_exp[lle_vs_exp.sys <= limit]
    # dsp_vs_exp = dsp_vs_exp[dsp_vs_exp.sys <= limit]

    DIR_NULL = ROOT / Path('calculated/new_method/null_dispersion')
    null_dispersion = pd.read_excel(DIR_NULL / 'null_dispersion.xlsx', sheet_name='systems')

    main()
