import numpy as np
import pandas as pd
import contextlib
import itertools
from scipy.signal import argrelextrema
from scipy.interpolate import CubicSpline, interp1d
from scipy.optimize import minimize, root_scalar
from concurrent.futures import ProcessPoolExecutor
from multiprocessing import Manager
from typing import Iterable, Union, Literal, Dict, Tuple

from ...actmodels import ActModel


class GMixScanner:

    def __init__(self,
                actmodel: ActModel,
                temperatures: Iterable[Union[float, int]],
                mole_fractions: Iterable[Union[float, int]],
                direction: Literal["up", "down"] = "up",
                mode: Literal["sequential", "parallel"] = "sequential",
                **kwargs) -> None:

        self.actmodel = actmodel
        self.temperatures = temperatures
        self.mole_fractions = mole_fractions

        # Default values (allow to use e.g. _process_gmix as standalone function)
        self.initializer = lambda actmodel: actmodel
        self.initargs = {'actmodel': self.actmodel}
        self.check_edges = False

# =============================================================================
# Public Methods
# =============================================================================
    def find_first_binodal(self, check_edges=True, gmix=None, keep_all_gmix=False):
        kwargs = dict(check_edges=check_edges, gmix=gmix, keep_all_gmix=keep_all_gmix)
        return self.find_all_binodal(stop_early=True, **kwargs)

    def find_all_binodal(self, mode='sequential', check_edges=True, keep_all_gmix=False,
                        stop_early=False, initializer=None, initargs=None, gmix=None):
        mode = 'sequential' if stop_early else mode
        self.check_edges = check_edges
        self.keep_all_gmix = keep_all_gmix
        args = self.temperatures
        process = lambda T: self._process_binodal(T, gmix=gmix)

        # Shared state for early stopping
        self.shared_state = Manager().dict() if stop_early else None

        # Set up initializer and initargs - needed for actmodel creation in 'parallel' mode
        self.__setup_actmodel(mode, initializer, initargs)
        config = (process, args, mode, stop_early, self.shared_state)
        with self.__manage_actmodel():
            results = self._run_process(*config)

        # Post-processing the results
        binodal, gmix = self._split_results(results)
        binodal = self._post_process_binodal(self, binodal)
        gmix = self._post_process_gmix(self, gmix)
        return binodal, gmix

# =============================================================================
# GMIX calculation
# =============================================================================
    def get_all_gmix(self, mode='sequential', check_edges=False,
                    initializer=None, initargs:dict=None, **kwargs):
        self.check_edges = check_edges
        args = self.temperatures
        process = self._process_gmix

        # Set up initializer and initargs - needed for actmodel creation in 'parallel' mode
        self.__setup_actmodel(mode, initializer, initargs)
        with self.__manage_actmodel():
            results = self._run_process(process, args, mode)

        # Post-processing the results
        gmix = self._split_results(results)
        gmix = self._post_process_gmix(self, gmix)
        return gmix

    def get_gmix(self, T, check_edges=False):
        try:
            gmix = self._compute_gmix(T, check_edges)
        except EdgesUpwardsError:
            # print(f'{sys:04d}: {T=:00d} - Edges are looking upwards')
            return None
        except GMixCalculationError:
            # print(f'{sys:04d}: {T=:00d} - GMIX could not be calculated.')
            return None
        except GMixAllNaNError:
            return None
        return gmix

    def _calculate_gmix(self, T, check_edges=False):
        x_values = self.mole_fractions

        # Enforce valid edge condition when check_edges is True
        if check_edges and not (x_values[0] == 0 and x_values[-1] == 1):
            raise ValueError("Check_edges only possible for mole fractions [0,..,1]")

        if check_edges:
            # Calculate 'gmix' for edge compositions to quickly check stability
            gmix_edges = self.actmodel.gmix(T, x_values[[0, 1, -2, -1]])
            # If any edge 'gmix' value is positive, the phase is unstable; skip further calculations
            if np.any(gmix_edges > 0):
                return None
            # Calculate 'gmix' for all other compositions
            gmix_inner = self.actmodel.gmix(T, x_values[2:-2])
            # Combine edge and inner 'gmix' values into a single array
            gmix = np.insert(gmix_edges, 2, gmix_inner, axis=0)
        else:
            # Calculate 'gmix' for all compositions directly
            gmix = self.actmodel.gmix(T, x_values)
        return gmix

    def _compute_gmix(self, T, check_edges=False):
        try:
            gmix = self._calculate_gmix(T, check_edges)
            if gmix is None:
                raise EdgesUpwardsError()
            if all(pd.isna(gmix[1:-1])):
                raise GMixAllNaNError(
                    "All gmix values are NaN except for the edge values (0 and 1). "
                    "This often indicates a missing dispersion parameter"
                    "'disp. e/kB [K]' in COSMO-SAC sigma profile."
                )
            return gmix
        except ValueError:
            raise GMixCalculationError()

# =============================================================================
# Internal / Helper Methods
# =============================================================================
    # Processing and Execution
    def _process_gmix(self, T, logging=True):
        self.actmodel = self.initializer(**self.initargs)
        gmix = self.get_gmix(T, self.check_edges)
        if logging:
            system = getattr(self, 'system', self.initargs.get('system', None))
            self._log_message(system, T, gmix)
        if gmix is None:
            return None
        x1 = self.mole_fractions
        c1, c2 = self.actmodel.mixture.names
        gmix = pd.DataFrame({'x1': x1, 'gmix': gmix})
        gmix[['c1', 'c2']] = self.actmodel.mixture.names
        gmix.insert(0, 'T', T)

        return gmix

    def _process_binodal(self, T, gmix=None):
        self.actmodel = self.initializer(**self.initargs)
        system = getattr(self, 'system', self.initargs.get('system', None))

        # Use the provided gmix if available; otherwise, calculate it
        if gmix is None:
            gmix = self._process_gmix(T, logging=False)
            if gmix is None:
                self._log_message(system, T, gmix)
                return (None, None)
        else:
            gmix = gmix[gmix['T'] == T]
            if gmix.empty:
                return (None, None)

        # Approximate LLE
        args = (gmix.x1.to_numpy(), gmix.gmix.to_numpy())
        res = self.estimate_lle_from_gmix(*args, logging=False)
        xB, yB, xS, yS = res

        # Check LLE status
        lle_found = xB is not None and xB.size > 0

        # Logging
        binodal = None
        add_msg = ''
        if lle_found:
            # Create binodal dataframe if LLE is found
            binodal = pd.DataFrame({'T': T}, index=[0])
            # Assign default phases to binodal: 'x1_L1' (leftmost) and 'x1_L2' (rightmost).
            # These represent phases for systems with a single LLE. They are compatible
            # with existing scripts, as the default (outer) phases remain unchanged.
            # The phases are selected as the first and last entries in the dataset.
            binodal[['x1_L1', 'x1_L2']] = xB[[0,-1]]
            binodal[['c1', 'c2']] = self.actmodel.mixture.names
            binodal[['x1_S1', 'x1_S2']] = xS[[0,-1]]
            binodal[['g_L1', 'g_L2']] = yB[[0,-1]]
            binodal[['g_S1', 'g_S2']] = yS[[0,-1]]
            # Handle cases with two separate LLEs, introducing two additional inner phases.
            # These are labeled with the "_inner" suffix to distinguish them. The inner
            # phases are positioned between the default phases, ensuring compatibility
            # while making the exception explicit.
            # Note: L1 still connects to L2 but here with suffix '_inner'.
            if len(xS) == 4:
                binodal[['x1_S2_inner', 'x1_S1_inner']] = xS[[1,2]]
                binodal[['g_S2_inner', 'g_S1_inner']] = yS[[1,2]]
                binodal['Note'] = 'Four spinodal points exist!'
                add_msg = "(4 spinodal points exist!)"
            if len(xB) == 4:
                binodal[['x1_L2_inner', 'x1_L1_inner']] = xB[[1,2]]
                binodal[['g_L2_inner', 'g_L1_inner']] = yB[[1,2]]
                binodal['Note'] = 'Two LLE: (L1--L2_inner) | (L1_inner--L2)'
                add_msg = "(2 LLE's exist!!)"

        self._log_message(system, T, gmix, binodal, add_msg)

        # Update shared state
        if self.shared_state is not None:
            self.shared_state['found_LLE'] = lle_found

        lle_not_found = (None, gmix) if self.keep_all_gmix else (None, None)
        return (binodal, gmix) if lle_found else lle_not_found

    @staticmethod
    def _run_process(process, args, mode='sequential', stop_early=False, shared_state=None):
        """
        A static method to run a process either in parallel or sequentially.
        :param process: Callable to process each argument
        :param args: Arguments to process
        :param mode: Execution mode ('parallel' or 'sequential')
        :param stop_early: Whether to stop early based on a shared state
        :param shared_state: Shared dictionary or object to track state across processes
        :return: List of results
        """
        if mode == 'parallel':
            # Parallel execution
            with ProcessPoolExecutor() as executor:
                results = list(executor.map(process, args))

        elif mode == 'sequential':
            # Sequential execution
            if stop_early:
                results = []
                for arg in args:
                    result = process(arg)
                    results.append(result)
                    if shared_state and shared_state.get('found_LLE', False):
                        break  # Exit loop if early stopping condition is met
            else:
                # Use list comprehension when stop_early=False
                results = [process(arg) for arg in args]
        else:
            raise ValueError("Mode should be either 'parallel' or 'sequential'")

        return results


    # Post-processing results
    @staticmethod
    def _split_results(results):
        """
        Dynamically handles the postprocessing of results by determining the structure
        of the elements and concatenating accordingly.

        Parameters:
        - results: list of results, where each element can be a DataFrame, a tuple of DataFrames, or None

        Returns:
        - If elements are single DataFrames, returns a concatenated DataFrame.
        - If elements are tuples/lists, returns a list of concatenated DataFrames for each index in the tuple/list.
        - Returns an empty DataFrame or list if no valid results are found.
        """
        # Filter out None values
        results = [res for res in results if res is not None]

        if not results:
            return pd.DataFrame()  # No valid results, return empty DataFrame

        # Check the structure of the first non-None result
        first_result = results[0]

        if isinstance(first_result, (tuple, list)):
            # Multiple components (e.g., tuple/list of DataFrames)
            num_components = len(first_result)
            concatenated_results = []

            for idx in range(num_components):
                extracted = [res[idx] for res in results if res[idx] is not None]
                filtered = [df.dropna(how='all', axis=1) for df in extracted]
                concatenated = pd.concat(filtered, ignore_index=True) if filtered else pd.DataFrame()
                concatenated_results.append(concatenated)

            return concatenated_results  # List of concatenated DataFrames

        elif isinstance(first_result, pd.DataFrame):
            # Single component (DataFrame)
            return pd.concat(results, ignore_index=True)

        else:
            raise ValueError("Unexpected result structure: only DataFrames or tuples/lists of DataFrames are supported.")

    @staticmethod
    def generate_columns(df, region, gmix=False):
        """
        Generates column sets dynamically based on the DataFrame columns.

        Parameters:
        - df: The input DataFrame to inspect.
        - region: Specifies the region ('L', 'S', or 'both').
        - gmix: Whether to include 'g_' columns.

        Returns:
        - A list of columns in the correct order.
        """

        # Validate region input
        if region not in ['L', 'S', 'both']:
            raise ValueError("Region must be 'L', 'S', or 'both'.")

        def reorder_columns(cols):
            """
            Reorder columns to place those with '_inner' between the first and second columns.
            If no '_inner' columns are present, the original order is preserved.
            """
            # Separate '_inner' columns and the rest
            inner_cols = [col for col in cols if '_inner' in col]
            non_inner_cols = [col for col in cols if '_inner' not in col]

            # Combine columns in the desired order
            return non_inner_cols[:1] + inner_cols + non_inner_cols[1:]

        # Start with 'T'
        columns = ['T']

        # Add all matching 'x1_' columns for the specified region(s)
        if region in ['L', 'both']:
            x1_l_columns = [col for col in df.columns if col.startswith('x1_L')]
            columns.extend(reorder_columns(x1_l_columns))

        if region in ['S', 'both']:
            x1_s_columns = [col for col in df.columns if col.startswith('x1_S')]
            columns.extend(reorder_columns(x1_s_columns))

        # Add 'c1' and 'c2'
        columns.extend(['c1', 'c2'])

        # Optionally add 'g_' columns for the specified region(s)
        if gmix:
            if region in ['L', 'both']:
                g_l_columns = [col for col in df.columns if col.startswith('g_L')]
                columns.extend(reorder_columns(g_l_columns))

            if region in ['S', 'both']:
                g_s_columns = [col for col in df.columns if col.startswith('g_S')]
                columns.extend(reorder_columns(g_s_columns))

        # Add 'Note' if it exists in the DataFrame
        if 'Note' in df.columns:
            columns.append('Note')

        return columns

    @classmethod
    def _post_process_binodal(cls, self, binodal):
        """
        Splits a binodal DataFrame into binodal, spinodal, binodal_g, and spinodal_g.

        Parameters:
        - self: The instance to assign attributes.
        - binodal: DataFrame containing binodal data.

        Returns:
        - The binodal DataFrame (filtered).
        """
        # Single-System Scanning
        if binodal.empty:
            columns =  ['T', 'x1_L1', 'x1_L2', 'c1', 'c2']
            binodal = pd.DataFrame(columns=columns, index=[0])
            binodal.loc[:, ['c1', 'c2']] = self.actmodel.mixture.names

        # Multi-System Scanning
        elif set(binodal.columns) == set(['sys','c1', 'c2']):
            columns =  ['sys', 'T', 'x1_L1', 'x1_L2', 'c1', 'c2']
            binodal[columns[1:4]] = np.nan
            binodal = binodal[columns]

        column_sets = {
            "binodal": cls.generate_columns(binodal, "L", gmix=False),
            "spinodal": cls.generate_columns(binodal, "S", gmix=False),
            "binodal_gmix": cls.generate_columns(binodal, "L", gmix=True),
            "spinodal_gmix": cls.generate_columns(binodal, "S", gmix=True),
            "binodal_full": cls.generate_columns(binodal, "both", gmix=True),
        }

        # Add 'sys' column to each column set if it exists
        if 'sys' in binodal.columns:
            column_sets = {key: ['sys'] + cols for key, cols in column_sets.items()}

        # Assign filtered DataFrames to self
        for attr, cols in column_sets.items():
            setattr(self, attr, binodal[cols])

        return self.binodal

    @classmethod
    def _post_process_gmix(cls, self, gmix):

        # Single-System Scanning
        if gmix.empty:
            columns =  ['T', 'x1', 'gmix', 'c1', 'c2']
            gmix = pd.DataFrame(columns=columns, index=[0])
            gmix.loc[:, ['c1', 'c2']] = self.actmodel.mixture.names

        # Multi-System Scanning
        elif set(gmix.columns) == set(['sys','c1', 'c2']):
            columns =  ['sys', 'T', 'x1', 'gmix', 'c1', 'c2']
            gmix[['T', 'x1', 'gmix']] = np.nan
            gmix = gmix[columns]

        # Ensure the DataFrame has the correct columns
        self.gmix = gmix.copy()

        # Explicitly delete gmix to free memory
        del gmix

        return self.gmix


    # Logging
    @classmethod
    def _log_message(cls, system, T, gmix=None, binodal=None, add_msg=''):
        """
        Centralized logging method.

        Parameters:
        - system: System identifier (int or None)
        - T: Temperature value (float or int)
        - gmix: GMIX result (can be None for failure or an object for success)
        - binodal: Binodal data (can be None if no LLE was found)
        """
        # Format the system identifier
        sys = f'sys={system:04d}:' if system is not None else ''

        # Base message
        msg = f'\r{sys} {T=}K'

        # Add GMIX status
        if gmix is None:
            msg += ' - Failed to calculate GMIX.'
        else:
            msg += ' - Successfully calculated GMIX.'

        # Add binodal/LLE status
        msg += '' if binodal is None else ' - LLE found! '

        # Add any additional message
        msg += add_msg

        # Print the message
        print(msg, flush=True)

# =============================================================================
# Binodal & Spinodal approximation
# =============================================================================
    @staticmethod
    def estimate_lle_from_gmix(x1, gmix, logging=True):
        # Initialize results to None
        xB, yB, xS, yS = (None, None, None, None)

        # Transform data names and approximate derivatives
        x, y = x1, gmix
        dy = np.gradient(y, x)
        ddy = np.gradient(dy, x)
        f_y = interp1d(x, y, kind='linear', bounds_error=True)
        f_dy = interp1d(x, dy, kind='linear', bounds_error=True)

        # Spinodal
        try:
            # Attempt to calculate the spinodal points
            xS = GMixScanner._approx_spinodal(x, ddy)

            # Validate spinodal points
            if len(xS) < 2 or len(xS) % 2 != 0:
                if logging:
                    if len(xS) % 2 != 0:
                        print("Error: xS must contain an even number of values.")
                    else:
                        print("Error: xS and yS must contain at least two values.")
                return xB, yB, xS, yS
        except Exception as e:
            # Handle the error in spinodal calculation
            if logging:
                print(f"Error in approx_spinodal: {e}")
            return xB, yB, xS, yS  # Return None for both if spinodal calculation fails

        # Binodal
        try:
            # Attempt to calculate the binodal points
            xB = GMixScanner._approx_binodal(xS, f_y, f_dy)
        except Exception as e:
            # Handle the error in binodal calculation
            if logging:
                print(f"Error in approx_binodal: {e}")
            # Return spinodal results (xB, yB remain None)
            return xB, yB, xS, f_y(xS)

        yS, yB = f_y(xS), f_y(xB)

        return xB, yB, xS, yS

    @staticmethod
    def _approx_spinodal(x, ddy):
        xS = find_root(x, ddy)
        return xS

    @classmethod
    def _approx_binodal(cls, xS, f_y, f_dy):

        def process_combination(values):
            """General function to process combinations."""
            xS_pair, limits = values[[1, 2]], values[[0, -1]]
            return cls.alternating_tangents(xS_pair, f_y, f_dy, limits)

        # Define the borders of each region (left, right and middle)
        borders = {
            'left': [0, xS[0]],
            **({'middle': [xS[1], xS[2]]} if len(xS) == 4 else {}),
            'right': [xS[-1], 1],
        }

        # Generate combinations
        combinations = {}
        for comb in itertools.combinations(borders.keys(), 2):
            combinations[comb] = np.array(borders[comb[0]] + borders[comb[1]])

        edges = ('left', 'right')

        if len(xS) == 4:
            # Handle middle parts
            middle_combinations = {k: v for k, v in combinations.items() if 'middle' in k}
            binodals = {comb: process_combination(values)
                        for comb, values in middle_combinations.items()}

            # Check if middle parts are separate
            two_separate_lle = cls._check_two_separate_lle(binodals)
            two_separate_lle = True

            if two_separate_lle:
                # Use only middle parts
                xB = np.concatenate(list(binodals.values()))
            else:
                # Compute tangents for edges
                xB = process_combination(combinations[edges])
        else:
            # Directly compute tangents for edges if no middle parts
            xB = process_combination(combinations[edges])

        # Remove NaN values
        xB = xB[~np.isnan(xB)]

        return xB

    @staticmethod
    def alternating_tangents(xS, f_y, f_dy, limits=(0, 1), tolerance=1e-5,
                             method='brentq', max_iterations=10):
        """
        Compute alternating tangents until convergence on both sides.

        Parameters:
            xS (list): Initial guesses for the tangent points [left, right].
            f_y (function): Function for the y-values.
            f_dy (function): Function for the derivative of y.
            limits (tuple): Bounds for the root finding.
            tolerance (float): Convergence tolerance for the tangent points.
            method (str): Method to use for root_scalar (default: 'brentq').
            max_iterations (int): Maximum number of iterations (default: 10).

        Returns:
            results (array): Final tangent points [left, right].
        """
    # =============================================================================
        def fobj(x, x0):
            """Objective function for tangent root finding."""
            return f_dy(x) - (f_y(x) - f_y(x0)) / (x - x0)
    # =============================================================================
        # Initialize convergence and results
        convergence = [False, False]  # Track convergence for left (0) and right (1)
        results = [*xS]
        brackets = np.split(np.sort((*limits, *xS)), 2)

        i = 0
        while not all(convergence):  # Continue until both sides converge

            if i >= max_iterations:  # Break if the maximum number of iterations is reached
                return np.array([np.nan] * 2)

            fixed_side = i % 2  # Alternate between sides (0 for left, 1 for right)
            variable_side = 1 - fixed_side

            # Skip calculation for already converged sides
            if convergence[variable_side]:
                i += 1
                continue

            x0 = results[fixed_side]  # Fixed point for the tangent
            bracket = brackets[variable_side]  # Bracket for root finding

            # Solve for the next tangent point using find_tangent
            new_tangent_point = find_tangent(f_y, f_dy, x0, bracket, method)

            # Handle failure in tangent finding
            if np.isnan(new_tangent_point):
                return np.array([np.nan] * 2)

            # Check convergence for the variable side
            delta = abs(new_tangent_point - results[variable_side])
            convergence[variable_side] = delta < tolerance
            results[variable_side] = new_tangent_point  # Update result for the side

            i += 1  # Increment iteration

        return np.array(results)

    @staticmethod
    def _check_two_separate_lle(binodals: Dict[Tuple[str, str], np.ndarray]) -> bool:
        """
        Checks if the two regions defined by 'left-middle' and 'middle-right' are separate.

        Args:
            binodals (Dict[Tuple[str, str], np.ndarray]): A dictionary where keys are tuples
            of specific region names (e.g., ('left', 'middle'), ('middle', 'right')), and values
            are NumPy arrays defining the ranges.

        Returns:
            bool: True if the regions are separate, otherwise False.
        """
        # Extract the ranges of interest
        left_middle = binodals.get(('left', 'middle'))
        middle_right = binodals.get(('middle', 'right'))

        # Return True if either range is missing or contains only NaN
        if left_middle is None or middle_right is None:
            return False
        if np.all(np.isnan(left_middle)) or np.all(np.isnan(middle_right)):
            return False

        # Check if the ranges are separate
        return max(left_middle) < min(middle_right)


# =============================================================================
# Parallelization and ActModel Management
# =============================================================================
    def __setup_actmodel(self, mode, initializer=None, initargs=None):
        """
        Set up the initializer and initargs for creating the actmodel.

        In both sequential and parallel modes, the actmodel is created using the initializer
        with initargs as arguments: `actmodel = initializer(**initargs)`.

        For parallel mode:
            - Sets up an initializer and initargs for dynamically creating the actmodel
              in worker processes.

        For sequential mode:
            - Sets up a dummy initializer that directly uses the existing self.actmodel.
            - The initializer returns the existing actmodel without reinitialization.

        Parameters:
            mode (str): Execution mode, either 'parallel' or 'sequential'.
            initializer (callable, optional): Function to initialize the actmodel.
            initargs (dict, optional): Arguments to pass to the initializer.

        Raises:
            ValueError: If required initializer and initargs are not provided for parallel mode.
        """
        if mode == 'parallel':
            if initializer is None or initargs is None:
                raise ValueError("For parallel mode, 'initializer' and 'initargs' are required.")
            # Store initializer and initargs for parallel mode
            self.initializer = initializer
            self.initargs = initargs
        elif mode == 'sequential':
            # Use a dummy initializer to directly return the existing actmodel
            self.initializer = lambda actmodel: actmodel
            self.initargs = {'actmodel': self.actmodel}
        else:
            raise ValueError("Mode must be either 'parallel' or 'sequential'.")

    @contextlib.contextmanager
    def __manage_actmodel(self):
        """
        Context manager to temporarily delete and recreate self.actmodel.
        Deletes self.actmodel before processing and recreates it afterward.
        Necessary for parallel computing to get rid of 'actmodel' in 'self'
        as it cannot be pickled. After processing return to the original state.
        """
        try:
            # Delete self.actmodel
            del self.actmodel
            yield
        finally:
            # Recreate self.actmodel using the initializer and initargs
            self.actmodel = self.initializer(**self.initargs)


# =============================================================================
# Error Handling
# =============================================================================
class EdgesUpwardsError(Exception):
    """
    Exception raised when the calculated edge compositions of a phase are unstable.

    Specifically, this indicates that the edge `gmix` values are positive, which
    suggests the phase is not stable at the given temperature and composition.
    """
    pass

class GMixCalculationError(Exception):
    """
    Exception raised when a `gmix` calculation fails due to invalid inputs,
    numerical errors, or unexpected issues during computation.

    This error is generally used to indicate issues unrelated to stability
    but rather computational problems within the activity model.
    """
    pass

class GMixAllNaNError(Exception):
    """
    Exception raised when all gmix values are NaN except for the edge values
    (0 and 1). This often indicates a missing dispersion parameter or other
    model input issues.

    This error helps identify cases where the gmix calculation fails entirely
    due to model setup issues.
    """
    pass


# =============================================================================
# Non-Class functions
# =============================================================================
def linear_function(x1, x2, y1, y2):
    def func(x):
        x = np.asarray(x)
        slope = (y2 - y1) / (x2 - x1)
        intercept = y1 - slope * x1
        return slope * x + intercept
    return func

def find_root(x, y):
    sign_changes = np.diff(np.sign(y))
    i_root = np.where(sign_changes != 0)[0]

    # Interpolate for more precise result
    interp = lambda x0, x1, y0, y1: x0 - y0 * (x1 - x0) / (y1 - y0)
    args = lambda i, x, y: (x[i], x[i + 1], y[i], y[i + 1])
    x_root = [interp(*args(i, x, y)) for i in i_root]

    return np.sort(x_root)

def find_tangent(f_y, f_dy, x0, bracket, method='brentq'):
    """
    Find the tangent point on one side.

    Parameters:
        f_y (function): Function for the y-values.
        f_dy (function): Function for the derivative of y.
        x0 (float): Fixed point for the tangent.
        bracket (tuple): Bracket for root finding.
        method (str): Method to use for root_scalar.

    Returns:
        float: The root (tangent point) found within the bracket, or np.nan on failure.
    """
    def fobj(x):
        return f_dy(x) - (f_y(x) - f_y(x0)) / (x - x0)

    try:
        solution = root_scalar(fobj, bracket=bracket, method=method)
        if solution.converged:
            return solution.root
    except ValueError:
        pass

    return np.nan


# =============================================================================
# Polish binodals
# =============================================================================
## Objective function: minimize the difference in slopes
def objective(params, f_dy):
    x1, x2 = params
    slope1 = f_dy(x1)
    slope2 = f_dy(x2)
    tangent_diff = slope1 - slope2
    return tangent_diff**2

## Constraints for the optimization
def tangent_constraint(params, f_y, f_dy):
    x1, x2 = params
    slope = f_dy(x1)  # Common slope
    y_diff = f_y(x2) - f_y(x1)
    x_diff = x2 - x1
    return slope - y_diff / x_diff  # Ensure tangent condition

def approx_lle(f_y, f_dy, bounds):
    # # Bounds for x1 and x2
    # bounds = [xL_bounds, xR_bounds]

    # Initial guess for x1 and x2
    initial_guess = [np.avg(bounds[0]), np.avg(bounds[1])]
    ## Optimization
    fobj = lambda params: objective(params, f_dy)
    constraint = lambda params: tangent_constraint(params, f_y, f_dy)
    constraints = {'type': 'eq', 'fun': constraint}
    options = dict(bounds=bounds, constraints=constraints)
    result = minimize(fobj, initial_guess, **options)
    xB = result.x
    return xB

# # LLE
# inits = np.split(xB, len(xB) // 2)
# binodal = np.concatenate([lle.binodal(T, x0) for x0 in inits if not np.isnan(x0).any()])
# for x in binodal:
#     ax.axvline(x, color='k', ls=':')
# print(system, T, *binodal, *actmodel.mixture.names)
