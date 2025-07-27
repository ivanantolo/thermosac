"""
ex_05_EvaluateNISTBenchmark.py

This example script demonstrates a complete benchmark workflow for a selected binary system
from the NIST/TRC dataset using the COSMO-SAC model. It illustrates how the AALDS values
reported in Table 5 of the manuscript are derived in practice.

The script includes:
  - Initial LLE detection via GMixScanner
  - LLE curve tracing via adaptive binodal tracking
  - Approximate deviation analysis (via interpolation)
  - Optional exact deviation recalculation using ThermoSAC
  - Statistical evaluation using the AALDS metric
  - Visualization of predicted vs. experimental LLE data

This script focuses on a single system (ACETONITRILE + CYCLOHEXANE) and a single model
variant (COSMO-SAC-2010) for clarity and brevity. It serves as an entry-level demonstration
and blueprint for extending the benchmark to all 96 systems.

For full benchmark replication, refer to the scripts in:
  data/statistics/nist-trc/
    01_approx_deviation.py
    02_calc_vs_exp.py
    03_visualize.py
    04_statistics.py
"""

import numpy as np
import pandas as pd
from pathlib import Path

from thermosac import Component, Mixture, COSMOSAC
from thermosac.equilibrium.lle import GMixScanner, LLE
from thermosac.utils.spacing import spacing
from utils.helper import convert_name_to_system, filter_by_components
from utils.plotter import plot_context, plot_curve, plot_calc_vs_exp
from utils.statistics import pre_process_dataframe, approx_calc_vs_exp, add_metadata
from utils.statistics import get_deviation, calculate_log_deviation, print_aalds

DIR_PROFILES = Path("./data/profiles/UD/sigma3")
DIR_FIG = Path("./figures") / Path(__file__).stem.lower()
DIR_EXP = Path('./data/experimental/nist-trc')

if __name__ == "__main__":
    # Define system and model variant
    print("=" * 60)
    print("[1] Initializing system and model variant...")
    names = ["ACETONITRILE", "CYCLOHEXANE"]
    system = convert_name_to_system(names)
    dispersion = False  # COSMO-SAC-2010 (set to True for COSMO-SAC-dsp)

    # Initialize COSMO-SAC model
    print("=" * 60)
    print(f"[2] Loading σ-profiles and initializing {'COSMO-SAC-dsp' if dispersion else 'COSMO-SAC-2010'}...")
    mixture = Mixture(*[Component(name) for name in names])
    actmodel = COSMOSAC(mixture, dispersion=dispersion)
    actmodel._import_delaware(names, DIR_PROFILES)

    # Step 1: Initial LLE detection using GMixScanner
    print("=" * 60)
    print("[3] Running GMixScanner to identify initial binodal values...")
    temperatures = np.arange(100, 201, 20)
    mole_fractions = spacing(0, 1, 51, func_name="sigmoid", inflection=15)
    scanner = GMixScanner(actmodel, temperatures, mole_fractions)
    initial_values, gmix_curves = scanner.find_first_binodal()

    # Step 2: Trace the binodal using LLE class
    print("=" * 60)
    print("[4] Tracing full LLE curve from initial values...")
    lle = LLE(actmodel)
    init = filter_by_components(initial_values, names)
    T0, *x0 = init.iloc[0][['T', 'x1_L1', 'x1_L2']].values.T
    calc = lle.miscibility(T0, x0, dT0=10)
    calc['sys'] = system

    # Step 3: Load experimental data and approximate deviation
    print("=" * 60)
    print("[5] Loading experimental data...")
    data_exp = pd.read_excel(DIR_EXP / 'LLeDbPGL6ed96b.xlsx')
    exp = data_exp[data_exp.sys == system]
    df_calc = pre_process_dataframe(calc)
    df_exp = pre_process_dataframe(exp)

    print("  -> Computing **approximate deviations**...")
    deviation = approx_calc_vs_exp(df_calc, df_exp)

    print("  -> Computing **exact deviations** (recalculation)...")
    deviation = get_deviation(system, deviation, dispersion)  # exact recalculation
    add_metadata(deviation)

    # Step 4: Print AALDS statistics
    print("=" * 60)
    print("[6] Computing AALDS statistics...")
    stats = calculate_log_deviation(deviation)
    print_aalds(stats, dispersion)

    # Step 5: Plotting results
    print("=" * 60)
    print("[7] Generating visualization of predicted vs. experimental LLE curves...")
    model = 'COSMO-SAC-dsp' if dispersion else 'COSMO-SAC-2010'
    c = 'r' if dispersion else 'k'
    with plot_context(system, legend_loc='lower center', dispersion=dispersion) as ax:
        ax.plot(x0, [T0] * 2, 'go', mfc='w', ms=5, label='Initial values')
        plot_curve(ax, exp.drop(columns='x1_L2'), '.', c='C0', label='Phase 1 (NIST/TRC)')
        plot_curve(ax, exp.drop(columns='x1_L1'), '.', c='C1', label='Phase 2 (NIST/TRC)')
        plot_curve(ax, calc, '-', c=c, label=model)
        plot_calc_vs_exp(ax, deviation)

    print("=" * 60)
    print("Script completed successfully.\n")
