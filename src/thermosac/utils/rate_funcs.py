'''
A selection of rate functions, i.e., speed curves for animations.
(1) https://docs.manim.community/en/stable/reference/manim.utils.rate_functions.html
(2) https://easings.net/
'''

import numpy as np

# Indicate which objects should be imported with "from spacing import *"
__all__ = ["easeOutExpo", "easeOutPoly", "linear", "quadratic", "cubic",
           "circle", "root", "smooth"]

# =============================================================================
# Define Spacing Functions
# =============================================================================
def easeOutExpo(x, base=2, power=-10):
    return 1 - base**(power * x)

def easeOutPoly(x, n=1.5):
    return 1 - (1 - x)**n

def linear(x: np.ndarray) -> np.ndarray:
    """Linear function for spacing."""
    return easeOutPoly(x, n=1)

def quadratic(x: np.ndarray) -> np.ndarray:
    """Quadratic function for spacing."""
    return easeOutPoly(x, n=2)

def cubic(x: np.ndarray) -> np.ndarray:
    """Cubic function for spacing."""
    return easeOutPoly(x, n=3)

def circle(x: np.ndarray, n: float = 2, R: float = 1, x0: float = 1, a: float = 1) -> np.ndarray:
    """Circle function for spacing."""
    return a * (R**n - (x - x0)**n)**(1/n)

def root(x: np.ndarray, n: float = 2) -> np.ndarray:
    """Root function for spacing."""
    return x**(1/n)

def sigmoid(x: float) -> float:
    r"""Returns the output of the logistic function.

    The logistic function, a common example of a sigmoid function, is defined
    as :math:`\frac{1}{1 + e^{-x}}`.

    References
    ----------
    - https://en.wikipedia.org/wiki/Sigmoid_function
    - https://en.wikipedia.org/wiki/Logistic_function
    """
    return 1.0 / (1 + np.exp(-x))

def smooth(t: float, inflection: float = 10.0, center: float = 0.5) -> float:
    error = sigmoid(-inflection / 2)
    err = (sigmoid(inflection * (t - center)) - error) / (1 - 2 * error)
    return np.minimum(np.maximum(err, 0), 1)
