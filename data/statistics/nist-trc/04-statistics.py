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
    """
    Compute AALDS (%) using the PG&L convention: deviations are always taken for the
    dilute component. For each row, if the experimental x1 (x_ref) is > 0.5, switch
    to the complement (1 - x) for both reference and calculated values.

    Returns:
        (n_systems, n_points, aalds_percent)
    """
    # Determine which column to use for predicted x1
    if 'x1_calc' in sub.columns:
        x_col = 'x1_calc'
    elif 'x1_approx' in sub.columns:
        x_col = 'x1_approx'
    else:
        raise ValueError("Data must contain either 'x1_calc' or 'x1_approx' column.")

    # Drop rows with missing values in required columns
    sub = sub.dropna(subset=['x1', x_col])

    n_points = len(sub)
    n_systems = sub['sys'].nunique() if 'sys' in sub.columns else np.nan
    if n_points == 0:
        return n_systems, 0, np.nan

    # Extract experimental and calculated mole fractions of component 1
    x_ref = sub['x1'].to_numpy(dtype=float)
    x_calc = sub[x_col].to_numpy(dtype=float)

    # Map to the dilute component basis:
    # if experimental x_ref > 0.5, use complements for BOTH ref and calc
    use_complement = x_ref > 0.5
    x_ref_dil = np.where(use_complement, 1.0 - x_ref, x_ref)
    x_calc_dil = np.where(use_complement, 1.0 - x_calc, x_calc)

    # Numerical safety: clip away from 0 and 1 to avoid log(0) or division by ~0
    eps = 1e-12
    x_ref_dil = np.clip(x_ref_dil, eps, 1.0 - eps)
    x_calc_dil = np.clip(x_calc_dil, eps, 1.0 - eps)

    # Compute AALDS (%)
    devs = np.abs(np.log(x_calc_dil / x_ref_dil))
    aalds = float(devs.mean() * 100.0)

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
