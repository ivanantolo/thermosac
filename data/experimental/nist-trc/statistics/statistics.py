#!/usr/bin/env python3
import pandas as pd
import numpy as np

# 1. Load data (semicolon-delimited)
df = pd.read_csv('stats-SAC_2010.csv', sep=';')
df = pd.read_csv('stats-SAC_dsp.csv', sep=';')

# 2. Classify each system as Aqueous if either component is water
is_water = df['c1'].str.contains('Water', case=False, na=False) | \
           df['c2'].str.contains('Water', case=False, na=False)
df['system_type'] = np.where(is_water, 'Aqueous', 'Nonaqueous')

def compute_stats(sub: pd.DataFrame):
    sub = sub.dropna(subset=['x1', 'x1_calc'])
    n_points  = len(sub)
    n_systems = sub['sys'].nunique()
    if n_points == 0:
        return n_systems, 0, np.nan
    devs = np.abs(np.log(sub['x1_calc'] / sub['x1']))
    aalds = devs.mean() * 100
    return n_systems, n_points, aalds

results = {}
for grp in ['Nonaqueous', 'Aqueous']:
    results[grp] = compute_stats(df[df['system_type'] == grp])
results['Overall'] = compute_stats(df)

# ─── 4. Print table
print(f"{'Group':<12}{'Systems':>10}{'Points':>10}{'%AALDS':>12}")
print('-' * 44)
for grp, (n_sys, n_pts, aalds) in results.items():
    a_str = f"{aalds:6.2f}" if not np.isnan(aalds) else "   n/a"
    print(f"{grp:<12}{n_sys:10d}{n_pts:10d}{a_str:12}")
