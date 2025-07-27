#!/usr/bin/env python3
import pandas as pd
import numpy as np

# 1. Load data (semicolon-delimited)
dispersion = False
model = 'SAC_dsp' if dispersion else 'SAC_2010'
df = pd.read_csv(f'stats-{model}.csv', sep=';')

# 2. Classify each system as Aqueous if either component is water
is_water = df['c1'].str.contains('Water', case=False, na=False) | \
           df['c2'].str.contains('Water', case=False, na=False)
df['system_type'] = np.where(is_water, 'Aqueous', 'Nonaqueous')

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

results = {}
for grp in ['Nonaqueous', 'Aqueous']:
    results[grp] = compute_stats(df[df['system_type'] == grp])
results['Overall'] = compute_stats(df)

# ─── 4. Print table
print('-' * 44)
print(f'Model variant: COSMO-{model}')
print('-' * 44)
print(f"{'Group':<12}{'Systems':>10}{'Points':>10}{'%AALDS':>12}")
print('-' * 44)
for grp, (n_sys, n_pts, aalds) in results.items():
    a_str = f"{aalds:12.2f}" if not np.isnan(aalds) else f"{'n/a':>12}"
    print(f"{grp:<12}{n_sys:10d}{n_pts:10d}{a_str:12}")
