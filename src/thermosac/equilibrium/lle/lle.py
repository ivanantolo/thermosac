import numpy as np
import pandas as pd
import traceback
from scipy.optimize import least_squares
from typing import Union, Type, Literal, Dict

from ...mixtures import Mixture
from ...actmodels import ActModel

class LLE:
    def __init__(self,
                actmodel: Union[ActModel, Type[ActModel]],
                ) -> None:
        self.actmodel = actmodel
        self.mixture = actmodel.mixture
        self._validate_arguments()

    def fobj_binodal(self, x1, T):
        # Equilibrium: Isoactivity criterion (aL1 - aL2 = 0)
        x = np.array([x1, 1-x1])
        activity = self.actmodel.activity(T, x)
        equilibrium = np.diff(activity, axis=1)
        return equilibrium.ravel() # reshape from (2,1) --> (2,)

    def fobj_spinodal(self, x1, T):
        x = np.array([x1, 1-x1])
        return self.actmodel.thermofac(T, x)

    def binodal(self, T, x0=None):
        if x0 is None:
            x0 = [0.1, 0.999]
        kwargs = dict(bounds=(0,1), ftol=1e-15, xtol=1e-15)
        res = least_squares(self.fobj_binodal, x0, args=(T,), **kwargs)
        return res.x

    def spinodal(self, T, x0=None):
        if x0 is None:
            x0 = self.binodal(T, x0)
        kwargs = dict(bounds=(0,1), ftol=1e-15, xtol=1e-15)
        res = least_squares(self.fobj_spinodal, x0, args=(T,), **kwargs)
        return res.x

# =============================================================================
# TODO: (1) Add some "approx_initial_values" function based on gmix
# TODO: (2) Overall improve this code to match the SLE code
# =============================================================================
    def solve_lle(self, T: float, x0: np.ndarray) -> Dict[str, np.ndarray]:
        """ Solve for liquid-liquid equilibrium (LLE) at a given temperature and initial composition. """
        binodal = {'x': self.binodal(T, x0)}
        spinodal = {'x': np.array( [np.nan] * len(binodal['x']) )}
        if self.calculate_spinodal:
            spinodal = {'x': self.spinodal(T, x0)}
        output = [f"{value:.20f}" for value in binodal['x']]

        if self.is_valid_Mw:
            binodal['w'] = self.actmodel._convert(binodal['x'])
            spinodal['w'] = self.actmodel._convert(spinodal['x'])
            output += [f"{value:.20f}" for value in binodal['w']]

        names = (f"{c.name[:10]}" for c in self.mixture)
        # print(f"{T:.6f}", *output, *names)
        return binodal, spinodal


    def miscibility(self,
                    T: float,
                    x0: np.ndarray,
                    dT0: float,
                    # dT settings
                    exponent: float = 1,
                    max_dT: float = np.inf,
                    max_T: float = 1000,
                    use_constant_dT = False,
                    use_initial_dT0 = False,
                    # dx settings
                    max_gap: float = 0.01,
                    max_gap_retries: int = 5,
                    max_change = 0.06, # maximum change of mole fraction in each phase
                    max_change_retries: int = 3,
                    x0_type: Literal['mole', 'weight'] = 'mole',
                    max_gap_type: Literal['mole', 'weight'] = 'mole',
                    use_dynamic_x0: bool = True,
                    check_curve_direction: bool = True,
                    # Iteration settings
                    max_iteration = np.inf,
                    skip_first_iteration: bool = False,
                    # Other settings
                    calculate_spinodal: bool = False,
                    print_traceback: bool = True,
                    ) -> pd.DataFrame:
        """ Calculate miscibility """

        res = {'binodal':[], 'spinodal':[]}
        var = 'x' if max_gap_type == 'mole' else 'w'
        self.is_valid_Mw = self.is_valid_numpy_array(self.mixture.Mw)
        self.calculate_spinodal = calculate_spinodal

        # Check for valid molar masses
        if x0_type == 'weight' and self.is_valid_Mw:
            x0 = self.convert_to_mole_fractions(x0, self.mixture.Mw)
        elif x0_type == 'weight':
            raise ValueError("Molar masses are not available for "
                    "conversion from weight to mole fraction.")

        self.error_occurred = False
        try:
            # First iteration step
            if skip_first_iteration:
                binodal = {var: x0}
                spinodal = {var: [None, None]}
            else:
                binodal, spinodal = self.solve_lle(T, x0)
            gap = np.diff(binodal[var])[0]
            res['binodal'].append({'T':T, **binodal})
            res['spinodal'].append({'T':T, **spinodal})

            retries_entered = False
            first_iteration = True
            iteration = 0
            # Subsequent iteration steps
            while gap > max_gap and T <= max_T and iteration <= max_iteration:
                if not retries_entered:
                    # For the first iteration, use dT0 if the flag is set
                    if first_iteration and use_initial_dT0:
                        dT = dT0
                        first_iteration = False  # Ensure this block runs only once
                    else:
                        dT = dT0 if use_constant_dT else min(dT0 * gap ** exponent, max_dT)

                retries = 0  # Reset retries for each new T += dT
                last_adjustment_reason = None

                while True:
                    # Calculate the binodal, spinodal, gap, and change
                    iteration += 1
                    x0 = self.determine_x0(res, var, dT, use_dynamic_x0, check_curve_direction)
                    T += dT
                    print(T, *x0)
                    binodal, spinodal = self.solve_lle(T, x0)
                    gap = np.diff(binodal[var])[0]
                    change = np.abs(res['binodal'][-1][var] - binodal[var])
                    big_change = np.any(change > max_change)

                    if gap < max_gap or big_change:
                        # Revert T before adjusting dT
                        T -= dT

                        # Identify the current adjustment reason
                        if gap < max_gap:
                            current_adjustment_reason = "gap_limit"
                        else: # big_change
                            current_adjustment_reason = "big_change"

                        # Reset retries if the adjustment reason has changed
                        if current_adjustment_reason != last_adjustment_reason:
                            retries = 0
                        last_adjustment_reason = current_adjustment_reason

                        # Check if gap is decreasing
                        decreasing_gap = self._check_decreasing_gap(res, var)
                        retries_entered = True if decreasing_gap else False

                        if current_adjustment_reason == "gap_limit":
                            reason = f"Gap limit exceeded ({max_gap}): gap={gap:.3f}"
                            dT_factor = 0.7 if decreasing_gap else 0.3
                            max_retries = max_gap_retries
                        else: # current_adjustment_reason == "big_change"
                            reason = f"Big change occurred (dx > {max_change})"
                            dT_factor = 0.7 if decreasing_gap else 0.4
                            max_retries = max_change_retries

                        dT = min(dT * dT_factor, max_dT)
                        retries += 1

                        msg = f"___Retry ({retries}/{max_retries}): {reason}."
                        if retries < max_retries:
                            msg += f" Decrease dT by {dT_factor} (T={T+dT:.4f}K)."
                        print(msg)

                        if retries >= max_retries:
                            break
                    else:
                        # If neither condition is met, break the retry loop
                        break

                if gap < max_gap:
                    break

                if retries >= max_change_retries and big_change:
                    # Final adjustment to T and recalculate binodal, spinodal before appending
                    # x0 = self.determine_x0(res, var, dT, use_dynamic_x0, check_curve_direction)
                    T += dT
                    binodal, spinodal = self.solve_lle(T, x0)
                    print(f"--Final adjustment after max retries: Adjusting temperature to {T+dT:.4f} with dT = {dT:.4f}.")

                # Append to res
                res['binodal'].append({'T':T, **binodal})
                res['spinodal'].append({'T':T, **spinodal})
        except Exception as e:
            self.error_occurred = True
            self.error_exception = e
            if print_traceback:
                traceback.print_exc()
            print(f"An error occurred during LLE calculation: {e}")
        except KeyboardInterrupt as e:
            self.error_occurred = True
            self.error_exception = e
            print("Processing interrupted by user.")
        finally:
            # Define column names
            binodal_columns = ['T', 'x1_L1', 'x1_L2']
            spinodal_columns = ['T', 'x1_S1', 'x1_S2']
            if self.is_valid_Mw:
                binodal_columns += ['w1_L1', 'w1_L2']
                spinodal_columns += ['w1_S1', 'w1_S2']

            # Convert lists to DataFrames
            res = {k:self.flatten_dict_values(v) for k,v in res.items()}
            res['binodal'] = pd.DataFrame(res['binodal'], columns=binodal_columns)
            res['spinodal'] = pd.DataFrame(res['spinodal'], columns=spinodal_columns)
            res = pd.merge(res['binodal'], res['spinodal'], on='T')
            # Drop empty spinodal columns
            res = res.dropna(axis=1, how='all')
            res[['c1', 'c2']] = self.mixture.names
            print(f'\nFinished {self.mixture}\n')

            return res

# =============================================================================
# AUXILLIARY FUNCTIONS
# =============================================================================
    def _validate_arguments(self):
        """Validate the arguments for the LLE class."""
        # TODO: Insert case where both actmodel and mixture are provided (check if acmodel.mixture == mixture, if not raise warning)
        if isinstance(self.actmodel, ActModel):
            # If actmodel is an instance of ActModel
            self.mixture: Mixture = self.mixture or self.actmodel.mixture
        elif isinstance(self.actmodel, type) and issubclass(self.actmodel, ActModel):
            # If actmodel is a class (subclass of ActModel)
            if self.mixture is None:
                raise ValueError("Please provide a valid mixture:Mixture.")
            self.actmodel: ActModel = self.actmodel(self.mixture)
        else:
            # If actmodel is neither an instance nor a subclass of ActModel
            err = "'actmodel' must be an instance or a subclass of 'ActModel'"
            raise ValueError(err)

    def is_valid_numpy_array(self, arr: np.ndarray) -> bool:
        """Check if a numpy array contains only numbers and no None values."""
        if not isinstance(arr, np.ndarray):
            return False
        if arr.dtype == object:  # Check if the array contains objects (which could include None)
            return not np.any(arr == None)
        else:
            return np.issubdtype(arr.dtype, np.number)

    @staticmethod
    def _extract(dictionary: dict):
        return [val for vals in dictionary.values() for val in vals]

    @staticmethod
    def _check_decreasing_gap(res, var):
        if len(res['binodal']) >= 2:
            previous_gap = np.diff(res['binodal'][-1][var])[0]
            second_previous_gap = np.diff(res['binodal'][-2][var])[0]
            return second_previous_gap > previous_gap
        return False

    @staticmethod
    def determine_x0(res, var, dT, use_dynamic_x0, check_curve_direction=False):
        """
        Determine the initial x0 values for the next iteration.
        :param res: Dictionary containing previous binodal results
        :param dT: Temperature step (can be positive or negative)
        :param use_dynamic_x0: Boolean flag to enable/disable dynamic x0 adjustment
        :param check_curve_direction: Boolean flag to enable/disable curve direction check
        :return: numpy array containing the new x0 values
        """
        if not use_dynamic_x0 or len(res['binodal']) < 2:
            return res['binodal'][-1]['x']

        num_points = len(res['binodal'])
        x1, x2 = np.array([r['x'] for r in res['binodal'][-2:]])
        T1, T2 = [r['T'] for r in res['binodal'][-2:]]
        dx_dT = (x2 - x1) / (T2 - T1)

        # Calculate x0_new regardless of curve direction
        if num_points >= 3:
            x0_prev = res['binodal'][-3]['x']
            T0 = res['binodal'][-3]['T']
            d2x_dT2 = ((x2 - x1) / (T2 - T1) - (x1 - x0_prev) / (T1 - T0)) / ((T2 - T0) / 2)
            x0_new = x2 + dx_dT * dT + 0.5 * d2x_dT2 * dT**2
        else:
            x0_new = x2 + dx_dT * dT

        if check_curve_direction:
            # Check if the curves are moving away from each other
            if dT > 0:
                left_moving_left = dx_dT[0] < 0
                right_moving_right = dx_dT[1] > 0
            else:  # dT < 0
                left_moving_left = dx_dT[0] > 0
                right_moving_right = dx_dT[1] < 0

            # Use x2 values for sides moving closer to promote convergence
            if not left_moving_left:
                x0_new[0] = x2[0]
            if not right_moving_right:
                x0_new[1] = x2[1]

        # Ensure values are within [0, 1] range
        return np.clip(x0_new, 0, 1)

    @staticmethod
    def flatten_dict_values(dict_list):
        """
        Converts a list of dictionaries into a list of lists, flattening arrays and iterables into individual elements.

        Parameters:
        - dict_list: list of dictionaries to be converted

        Returns:
        - A list of lists with flattened values
        """
        return [
            [
                item.tolist() if isinstance(item, np.ndarray) else item
                for value in row.values()
                for item in (value if isinstance(value, (list, tuple, np.ndarray)) else [value])
            ]
            for row in dict_list
        ]
