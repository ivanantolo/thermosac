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
