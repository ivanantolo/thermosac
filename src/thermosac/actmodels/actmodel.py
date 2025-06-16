import numpy as np
import numbers
from functools import wraps
from numpy.typing import NDArray
from typing import Literal

from ..mixtures import Mixture
from ..utils.helpers import convert

# Temporarily suppress the warning for divide-by-zero errors in NumPy
old_settings = np.seterr(divide='ignore')

class ActModel:

    def __init__(self, mixture: Mixture):
        self.mixture = mixture

    def lngamma(self, T, x):
        raise NotImplementedError("lngamma() hasn't been implemented yet.")

    def dlngamma(self, T, x):
        # Only binary case
        def f(x1):
            x = np.array([x1, 1-x1])
            return self.lngamma(T, x)#[0]
        h, x = 0.0001, x[0]
        # h, x = 1e-12, x[0]
        x1 = np.clip(x-h, 0, 1)
        x2 = np.clip(x+h, 0, 1)
        dy = (f(x2)-f(x1))/(2*h)
        # Revert direction of dy2_dx2 --> dy2_dx1
        dy[1] = dy[1][::-1]
        return f(x), dy

    def activity(self, T, x):
        act = np.log(x) + self.lngamma(T, x)
        act[(act == np.inf) | (act == -np.inf)] = np.nan
        return act

    def mu(self, T, x):
        mu0 = self.mixture.mu0
        # mu0 = np.array([ [mu01, mu02] ]).T
        mu = mu0 + np.log(x) + self.lngamma(T, x)
        mu[(mu == np.inf) | (mu == -np.inf)] = np.nan
        return mu

    def gmix(self, T, x, ignore_sum_check=False):
        """
        Calculate the gmix value for the input composition `x` at temperature `T`.

        Parameters:
        ----------
        T : float
            Temperature at which the gmix calculation is performed.
        x : array-like
            Input composition values representing mole fractions or concentrations of the mixture components.
            The length of `x` should match the expected number of components defined by `self.mixture.nc`.
        ignore_sum_check : bool, optional, default=False
            Determines how to handle input values when the sum of all elements in `x` is approximately equal to 1.
            - If False, the method treats the input values that sum to 1 as if they represent a complete set of
              normalized mole fractions (e.g., `[x1, x2, ..., xn]` where sum is 1), and processes them accordingly.
              A warning message will be shown to indicate this behavior, and the input is treated as if `[x, 1 - x]`
              transformation was already applied.
            - If True, the method ignores the sum check and treats the values in `x` as independent, allowing
              the transformation to `[x, 1 - x]` or other expected forms, even when the sum of elements is approximately 1.

        Returns:
        -------
        gmix : float or ndarray
            The calculated gmix values based on the input composition and temperature. Returns a scalar if `x`
            is a scalar input; otherwise, returns an array.

        Notes:
        -----
        - The parameter `self.mixture.nc` determines the expected number of components in the input `x`. If the
          number of elements in `x` does not match `self.mixture.nc`, the function will handle this discrepancy based
          on internal logic, potentially transforming `x` accordingly.

        """
        is_scalar = np.isscalar(x)
        # Convert input as needed
        x = self._convert_input(x, ignore_sum_check)
        # Create mask to identify columns that don't contain 0 or 1
        mask = np.any((x != 0) & (x != 1), axis=0)
        # Apply the mask to filter x
        _x = x[:, mask]
        # Calculate gmix for the  x values
        _gmix = _x * (np.log(_x) + self.lngamma(T, _x))
        _gmix = np.sum(_gmix, axis=0)
        # Initialize gmix array with zeros
        gmix = np.zeros(1 if x.ndim == 1 else x.shape[1])
        # Fill gmix with calculated values where the mask is True
        gmix[mask] = _gmix
        return gmix[0] if is_scalar else gmix

    def thermofac(self, T, x):
        _, dlngamma = self.dlngamma(T, x)
        return 1 + x[0]*dlngamma[0]

    def _thermofac_slow(self, T, x):
        def f(x1):
            x = np.array([x1, 1-x1])
            return self.lngamma(T, x)[0]
        h, x = 0.0001, x[0]
        dy = (f(x+h)-f(x-h))/(2*h)
        return 1 + x * dy

# =============================================================================
# Wrapper functions (Decorators)
# =============================================================================
    @staticmethod
    def vectorize(func):
        ''' Intended vor ActModels where only single mole fractions can be
        handled, like e.g. COSMO-SAC. This function vectorizes the lngamma()
        to make it work with arrays of mole fractions.
        '''
        @wraps(func)  # This preserves the original function's signature and attributes
        def wrapper(self, T, x):
            # Convert input to appropriate format
            x = self._convert_input(x)
            # Process based on the dimensionality of x
            if x.ndim == 1:
                return func(self, T, x)
            elif x.ndim == 2:
                results = [func(self, T, x[:, col]) for col in range(x.shape[1])]
                return self._process_vectorized_results(results)
            else:
                raise ValueError("Input must be either a scalar, 0D, 1D or 2D array")
        return wrapper


# =============================================================================
# Auxilliary functions
# =============================================================================
    def _convert_input(self, x, ignore_sum_check=False):
        """Converts input to a 1-dim ndarray if it's a number or 0-dim ndarray."""
        if isinstance(x, numbers.Number) or (isinstance(x, np.ndarray) and x.ndim == 0):
            # Convert scalar or 0-d array into [x, 1 - x]
            return np.array([float(x), 1 - float(x)])

        elif isinstance(x, list) or (isinstance(x, np.ndarray) and x.ndim == 1):
            # Convert to ndarray for consistent handling
            x = np.array(x)

            # Check if x has exactly two elements
            if len(x) == self.mixture.nc:
                # Check if the sum of the two values is approximately equal to 1
                if np.isclose(x.sum(), 1, atol=1e-8):
                    # Get the class name and method name dynamically
                    class_name = self.__class__.__name__
                    method_name = '_convert_input'
                    # Handle based on the ignore_sum_check flag
                    if not ignore_sum_check:
                        # Display a warning message to the user
                        print(
                            f"Warning from {class_name}.{method_name}: The input values x={x} sum approximately to 1. "
                            "This input will be treated as if it were in the form [x, 1 - x], and only x will be used. "
                            "If these values are intended to be separate, set ignore_sum_check=True."
                        )
                        # Assume it's already in the form [x, 1 - x], return as-is
                        return x
                # Treat as independent values and convert to [x, 1 - x] format
                return np.array([x, 1 - x])

            # If not exactly two elements, treat normally
            elif len(x) != self.mixture.nc:
                # Convert to ndarray for consistent handling
                return np.array([x, 1 - x])

        # Return x unmodified if none of the conditions match
        return x

    def _convert(self,
                x : NDArray[np.float64],
                to : Literal['weight', 'mole'] ='weight'
                ) -> NDArray[np.float64]:
        """
        Convert the fraction of a binary mixture between mole fraction and weight fraction.

        This method is designed for internal use with binary mixtures, where the mixture is defined by two components.
        It uses the 'convert' function to perform the conversion by creating an array with the fractions of both
        components and the molecular weights from the mixture's attributes.

        Parameters:
            x (NDArray[np.float64]): The mole or weight fraction of the first component of the mixture.
                                    If converting 'to' weight, 'x' represents mole fractions; if converting 'to' mole,
                                    'x' represents weight fractions. This should be a single value or a 1D array of values.
            to (Literal['weight', 'mole'], optional): The target type for the conversion. Defaults to 'weight'.
                                                    Use 'weight' to convert mole fractions to weight fractions,
                                                    and 'mole' to convert weight fractions to mole fractions.

        Returns:
            NDArray[np.float64]: The converted fraction(s) of the first component in the same shape as 'x'.
                                If 'x' is a single value, the return will be a single converted value;
                                if 'x' is a 1D array, the return will be a 1D array of converted values.

        Example:
            >>> mixture = Mixture(components=[component1, component2], Mw=np.array([18.01528, 46.06844]))
            >>> sle = SLE(mix=mixture)
            >>> x_mole_fraction = np.array([0.4])  # Mole fraction of the first component
            >>> x_weight_fraction = sle._convert(x_mole_fraction, to='weight')
            >>> print(x_weight_fraction)
            array([0.01373165])
        """
        return convert(x=np.array([x, 1-x], dtype=np.float64), Mw=self.mixture.Mw, to=to)[0]


    def _process_vectorized_results(self, results):
        ''' Helper function to process the results of a vectorized function,
        handling cases with multiple return values (tuples). '''

        # Check if the function returns multiple outputs (tuple)
        if isinstance(results[0], tuple):
            # If there are multiple outputs, handle them separately
            num_outputs = len(results[0])
            transposed_results = []
            for i in range(num_outputs):
                # Collect i-th result from all tuples
                transposed_results.append(np.array([res[i] for res in results]).T)
            return tuple(transposed_results)
        else:
            # If only one output, handle it directly
            return np.array(results).T
