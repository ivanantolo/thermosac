''' Example: LLETracer (multi system)
'''
import numpy as np
import pandas as pd
from pathlib import Path

from thermosac import Component, Mixture, COSMOSAC
from utils.lle_generator import LLEGenerator
from utils.helper import initialize_cosmo_model
from utils.helper import plot_context, plot_curve, save_figure

DIR_PROFILES = Path("data/profiles/UD/sigma3")
DIR_INITS = Path("data/calculated/initial_values")
DIR_FIG = Path("./figures") / Path(__file__).stem.lower()

if __name__ == "__main__":
    dispersion = False
    model = 'SAC_dsp' if dispersion else 'SAC_2010'
    initial_values = pd.read_csv(DIR_INITS / f"init-{model}.csv", sep=';')
    systems = initial_values['sys'].unique() # Select all systems in init-{model}.csv
    systems = [3] # examplary subset of systems for testing

    initializer = initialize_cosmo_model
    initargs = dict(dispersion=dispersion)
    options = dict(initializer=initializer, initargs=initargs)

    # LLE Tracing (parallelized)
    lleGen = LLEGenerator(systems, initial_values, **options)
    miscibility = lleGen.calculate_miscibility(mode='sequential') # mode: sequential | parallel

    # Plot results
    for system, df in miscibility.groupby('sys'):
        with plot_context(system, 'lower center', dispersion) as ax:
            plot_curve(ax, df, '.-')
            save_figure(ax.get_figure(), system, dispersion, None, DIR_FIG, dpi=75)
