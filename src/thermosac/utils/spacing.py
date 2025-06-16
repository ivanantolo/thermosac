"""
Spacing module for generating custom-spaced sequences of numbers.

This module provides various spacing functions and a main `spacing` function
to generate sequences with different distributions, including constant arc length.
"""

from typing import Callable, Dict, List, Union, Tuple, Literal
import numpy as np
from thermosac.utils import rate_funcs as rf

# Indicate which objects should be imported with "from spacing import *"
__all__ = ["spacing"]

# =============================================================================
# Define Spacing Functions
# =============================================================================

SPACING_FUNCTIONS: Dict[str, Callable] = {
    'circle': rf.circle,
    'easeOutExpo': rf.easeOutExpo,
    'poly': rf.easeOutPoly,
    'root': rf.root,
    'linear': rf.linear,
    'quadratic': rf.quadratic,
    'cubic': rf.cubic,
    'sigmoid': rf.smooth,
}

FuncNameType = Literal[tuple(SPACING_FUNCTIONS.keys())]

def spacing(
    min_value: float,
    max_value: float,
    num: int,
    func_name: FuncNameType = 'linear',
    reverse: bool = False,
    log_scale: bool = False,
    trim: bool = True,
    constant_arc_length: bool = False,
    **kwargs
) -> np.ndarray:
    """
    Generate a sequence of numbers with custom spacing.

    Args:
        min_value (float): The minimum value of the sequence.
        max_value (float): The maximum value of the sequence.
        num (int): The number of points to generate.
        func_name (str): The name of the spacing function to use. Default is 'linear'.
        reverse (bool): If True, reverse the sequence. Default is False.
        log_scale (bool): If True, use logarithmic scaling. Default is False.
        trim (bool): If True, trim and rescale the output for certain functions. Default is True.
        constant_arc_length (bool): If True, use constant arc length spacing. Default is False.
        **kwargs: Additional arguments to pass to the spacing function.

    Returns:
        np.ndarray: An array of spaced values.

    Raises:
        ValueError: If invalid input parameters are provided.
    """
    if num <= 0:
        raise ValueError("'num' must be a positive integer")
    if min_value >= max_value:
        raise ValueError("'min_value' must be less than 'max_value'")

    if log_scale:
        min_value, max_value = 10**min_value, 10**max_value

    if constant_arc_length:
        x = constant_arc_length_spacing(num, func_name, **kwargs)
    else:
        x = np.linspace(0, 1, num)

    y = evaluate_function(x, func_name, trim=trim, constant_arc_length=constant_arc_length, **kwargs)
    y = (1 - y)[::-1] if reverse else y

    return min_value + (max_value - min_value) * y

def constant_arc_length_spacing(num: int, func_name: str, **kwargs) -> np.ndarray:
    """
    Generate x values for constant arc length spacing.

    Args:
        num (int): The number of points to generate.
        func_name (str): The name of the spacing function to use.
        **kwargs: Additional arguments to pass to the spacing function.

    Returns:
        np.ndarray: An array of x values with approximately constant arc length spacing.
    """
    func = SPACING_FUNCTIONS.get(func_name)
    if func is None:
        raise ValueError(f"Function with name '{func_name}' not found")

    def arc_length(x1: float, x2: float, num_points: int = 100) -> float:
        x = np.linspace(x1, x2, num_points)
        y = func(x, **kwargs)
        dx = np.diff(x)
        dy = np.diff(y)
        return np.sum(np.sqrt(dx**2 + dy**2))

    x_values = [0]
    total_length = arc_length(0, 1)
    target_length = total_length / (num - 1)

    while len(x_values) < num - 1:
        current_x = x_values[-1]
        next_x = current_x
        current_length = 0
        while current_length < target_length and next_x < 1:
            next_x += 0.01  # Small step size for better accuracy
            current_length = arc_length(current_x, next_x)

        # Overshoot and interpolate
        if next_x < 1:
            prev_x = next_x - 0.01
            prev_length = arc_length(current_x, prev_x)
            t = (target_length - prev_length) / (current_length - prev_length)
            interpolated_x = prev_x + t * 0.01
            x_values.append(interpolated_x)
        else:
            x_values.append(1)

    # Ensure the last point is exactly 1
    x_values.append(1)

    return np.array(x_values)

# Update the evaluate_function to ensure all functions start at (0,0) and end at (1,1)
def evaluate_function(x: np.ndarray, func_name: str, trim: bool = True, constant_arc_length: bool = False, **kwargs) -> np.ndarray:
    """
    Evaluate a spacing function on given input values.

    Args:
        x (np.ndarray): Input values.
        func_name (str): Name of the function to evaluate.
        trim (bool): If True, trim and rescale the output for certain functions.
        constant_arc_length (bool): If True, x is assumed to be arc length spaced.
        **kwargs: Additional arguments to pass to the function.

    Returns:
        np.ndarray: Function output.

    Raises:
        ValueError: If the specified function is not found.
    """
    func = SPACING_FUNCTIONS.get(func_name)
    if func is None:
        raise ValueError(f"Function with name '{func_name}' not found")

    y = func(x, **kwargs)

    # Ensure the function starts at (0,0) and ends at (1,1)
    y = (y - y[0]) / (y[-1] - y[0])

    trim_funcs = ('root', 'circle')
    if func_name in trim_funcs and trim and not constant_arc_length:
        x_extended = np.linspace(0, 1, len(x) + 1)
        y = func(x_extended, **kwargs)[1:]
        y = (y - y[0]) / (y[-1] - y[0])

    return y

# =============================================================================
# Data Preparation Functions
# =============================================================================

def generate_values(num_points, trim, constant_arc_length, specific_func=None):
    """Generate values for all or a specific spacing function."""
    x = np.linspace(0, 1, num_points)
    options = dict(trim=trim, constant_arc_length=constant_arc_length)
    y_values = {}
    x_values = {}

    if specific_func is None or specific_func == 'all':
        for name in SPACING_FUNCTIONS.keys():
            if constant_arc_length:
                x = constant_arc_length_spacing(num_points, name)
            y_values[name] = evaluate_function(x, name, **options)
            x_values[name] = x
    else:
        if constant_arc_length:
            x = constant_arc_length_spacing(num_points, specific_func)
        y_values[specific_func] = evaluate_function(x, specific_func, **options)
        x_values[specific_func] = x

    return x_values, y_values

def sort_functions(values: Dict[str, np.ndarray]) -> List[str]:
    """Sort functions based on their behavior."""
    sort1 = {k: (y[1], y[2]) for k, y in values.items()}
    sort1_keys_sorted = sorted(sort1, key=lambda k: sort1[k][0])

    change_index = len(sort1_keys_sorted) - 1
    for i, k in enumerate(sort1_keys_sorted[:-1]):
        if sort1[k][1] > sort1[sort1_keys_sorted[i + 1]][1]:
            change_index = i + 1
            break

    first_part = sort1_keys_sorted[:change_index]
    second_part = sorted(sort1_keys_sorted[change_index:], key=lambda k: sort1[k][1], reverse=True)
    return first_part + second_part

def generate_colors(num_colors: int) -> List[str]:
    """Generate a list of colors for plotting."""
    import matplotlib.pyplot as plt
    import matplotlib.colors
    default_colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
    additional_colors = plt.cm.viridis(np.linspace(0, 1, num_colors - len(default_colors)))
    additional_colors = [matplotlib.colors.to_hex(color) for color in additional_colors]
    return default_colors + additional_colors

def prepare_data(specific_func: Union[str, None] = None,
                 trim: bool = True, constant_arc_length: bool = False,
                 num_points: int = 11
                 ) -> Tuple[np.ndarray, Dict[str, np.ndarray], Dict[str, np.ndarray], List[str]]:
    """Prepare data for plotting."""

    x_values, y_values = generate_values(num_points, trim, constant_arc_length)

    if specific_func is not None and specific_func != 'all':
        return x_values, y_values

    sorted_keys = sort_functions(y_values)
    sorted_values = {k: y_values[k] for k in sorted_keys}
    diffs = {k: np.diff(y) for k, y in sorted_values.items()}
    colors = generate_colors(num_points)

    return x_values, sorted_values, diffs, colors

# =============================================================================
# Visualization Functions
# =============================================================================

def plot_bar_chart(trim: bool = True, constant_arc_length: bool = False):
    """Plot a bar chart comparing different spacing functions."""
    import matplotlib.pyplot as plt
    options = dict(trim=trim, constant_arc_length=constant_arc_length)
    x_values, y_values, diffs, colors = prepare_data(**options)
    fig, ax = plt.subplots()
    ax.set_ylim(0, 1)
    for idx, (name, dy) in enumerate(diffs.items()):
        bottom = 0
        for i, segment in enumerate(dy):
            bar = ax.bar(name, segment, bottom=bottom, color=colors[i])
            bar_width = bar[0].get_width()
            if idx > 0:
                prev_bottom = sum(list(diffs.values())[idx - 1][:i])
                w = bar_width / 2
                x, y = [(idx-1)+w, idx-w], [prev_bottom, bottom]
                ax.plot(x, y, ls=':', c='black', lw=.8)
            bottom += segment
    ax.set_xticks(range(len(diffs.keys())))
    ax.set_xticklabels(ax.get_xticklabels(), rotation=60)
    plt.title(f"Spacing Functions Comparison ({'Constant Arc Length' if constant_arc_length else 'Linear Spacing'})")
    plt.tight_layout()
    plt.show()

def plot_specific_function(name: str, trim: bool = True, constant_arc_length: bool = False):
    """Plot a specific spacing function."""
    import matplotlib.pyplot as plt
    options = dict(trim=trim, constant_arc_length=constant_arc_length)
    x_values, y_values = prepare_data(name, **options)[:2]
    for name, y in y_values.items():
        fig, ax = plt.subplots()
        x = x_values[name]
        ax.plot(x, y, '.-', label=name)
        for xx, yy in zip(x, y):
            ax.axhline(yy, xmin=0, xmax=xx, ls=':')
        ax.set_ylim(0, 1)
        ax.set_xlim(0, 1)
        ax.legend(loc='lower right')
        ax.set_aspect('equal', 'box')
        plt.title(f"{name} Function ({'Constant ds' if constant_arc_length else 'Constant dx'})")
        plt.tight_layout()
        plt.show()

# =============================================================================
# Main Function
# =============================================================================

def main():
    import phaseq.utils.plot_settings as ps
    ps.reset_rcParams()

    """Main function for demonstrating the spacing module."""
    # Constant dx
    plot_bar_chart(trim=False, constant_arc_length=False)
    plot_specific_function('all', trim=False, constant_arc_length=False)

    # Constant ds (arclength)
    # plot_bar_chart(trim=False, constant_arc_length=True)
    # plot_specific_function('all', trim=False, constant_arc_length=True)

if __name__ == '__main__':
    main()
