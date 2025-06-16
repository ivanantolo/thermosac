import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.interpolate import CubicHermiteSpline
from scipy.optimize import fsolve

def log_outliers(system, df, original_length, log_list):
    """
    Logs the outlier details for a given system.

    Parameters:
    - system: str
        The name of the system being processed.
    - df: pandas.DataFrame
        The DataFrame containing processed outlier information.
    - original_length: int
        The total number of records in the original dataset.
    - log_list: list
        A list to which the outlier log information will be appended.

    The function appends a log entry to the `log_list` containing:
    - The system name.
    - The original dataset length.
    - The count of outliers for each 'Outlier' column in the DataFrame.
    """
    # Select all columns starting from 'Outlier'
    outlier_columns_df = df.loc[:, 'Outlier':]

    # Compute the total count of outliers for each column
    outlier_counts = outlier_columns_df.sum()

    # Prepare the log entry for the current system
    log_entry = [system, original_length]

    # Append the number of outliers for each 'Outlier' column
    for col in outlier_columns_df.columns:
        log_entry.append(outlier_counts[col])

    # Append the log entry to the log list
    log_list.append(log_entry)

def save_outlier_log(df, outlier_log, filename='log_outlier', save=True):
    """
    Saves the log of outliers as a CSV file.

    Parameters:
        df (pd.DataFrame): The DataFrame containing the data with 'Outlier' columns.
        outlier_log (list): The list of processed outlier data to save.
        filename (str): The name of the output CSV file. Default is 'log_outlier.csv'.

    Returns:
        None
    """
    # Identify columns starting from 'Outlier' onward
    outlier_columns = df.loc[:, 'Outlier':]

    # Create column headers for the log
    columns = ['System', 'Points']
    columns.extend(f'{col}' for col in outlier_columns.columns)  # Add counts
    # columns.extend(f'#{col}_percentage' for col in outlier_columns.columns)  # Uncomment for percentages

    # Convert outlier columns to integers for visual feedback
    df[outlier_columns.columns] = df[outlier_columns.columns].astype(int)

    # Create a DataFrame for the outlier log using the generated column headers
    log_df = pd.DataFrame(outlier_log, columns=columns)
    if 'DC+BC' in log_df.columns and 'DC+BC_slope' in log_df.columns:
        log_df['Difference'] = log_df['DC+BC'] - log_df['DC+BC_slope']

    # Save the DataFrame to a CSV file
    if save:
        # log_df.to_csv(filename+'.csv', sep=';', index=False)
        log_df.to_excel(filename+'.xlsx', index=False)

    return log_df


# Updated save_plot to use separated concerns
def generate_and_save_plot(df, sys, save=False):
    """
    Generates and optionally saves a plot for the given DataFrame.

    Parameters:
        df (pd.DataFrame): Input DataFrame with an 'Outlier' column.
        sys (int): System identifier for the plot title and file naming.
        save (bool): Whether to save the plot to a file.
    """
    fig, ax = create_plot(df, sys)
    if save:
        save_plot(fig, df, sys)

    return ax

def zoom_in(df):
    # Find the index of the first occurrence of a True outlier
    first_outlier_index = df[df['Outlier']].index.min()
    if np.isnan(first_outlier_index):
        return df
    idx = first_outlier_index - 4
    if len(df) - idx < 5:
        idx = first_outlier_index - 5
    return df.loc[idx:]

def plot_data(df, x1, x2, y1, y2, ax, outlier=False, options=None):
    """
    Plots data on the provided Axes object.

    Parameters:
        df (pd.DataFrame): Input DataFrame.
        x1, x2, y1, y2 (str): Column names for the x and y axes.
        ax (matplotlib.axes.Axes): The Axes object to plot on.
        outlier (bool): Whether to plot as outliers.
        options (dict): Plot style options.
    """
    label = 'Outlier' if outlier else 'LLE'
    options = options or (
        {'c': 'r', 'marker': '.', 'ls': 'none'} if outlier
        else {'c': 'k', 'mfc': 'w', 'marker': 'o', 'ls': '-'}
    )

    ax.plot(df[x1], df[y1], label=label, **options)
    ax.plot(df[x2], df[y2], **options)

def create_plot(df, sys):
    """
    Creates a plot for the given DataFrame and returns the figure and axes.

    Parameters:
        df (pd.DataFrame): Input DataFrame with an 'Outlier' column.
        sys (int): System identifier for the plot title.

    Returns:
        tuple: The created figure and axes objects.
    """
    # Set plot background color
    color_rgb = (84 / 255, 130 / 255, 53 / 255)
    fig, ax = plt.subplots(facecolor=color_rgb)

    # Focus on the relevant section of the DataFrame
    df_zoomed = zoom_in(df)

    # Plot the main data
    x1, x2, y1, y2 = 'T*', 'T*', 'xL1*', 'xL2*'
    plot_data(df_zoomed, x1, x2, y1, y2, ax)

    # Highlight outliers
    if df_zoomed['Outlier'].any():
        outliers_df = df_zoomed[df_zoomed['Outlier']]
        plot_data(outliers_df, x1, x2, y1, y2, ax, outlier=True)

    # Set plot labels, title, and grid
    ax.set_ylabel('x1')
    ax.set_xlabel('T')
    ax.set_title(f'Sys={sys:04d}')
    plt.grid(True)

    # Style the plot
    ax.set_facecolor(color_rgb)
    ax.tick_params(colors='white')
    ax.xaxis.label.set_color('white')
    ax.yaxis.label.set_color('white')
    ax.title.set_color('white')

    # Style the legend
    legend = ax.legend()
    legend.get_frame().set_facecolor('gray')
    legend.get_frame().set_edgecolor('white')

    return fig, ax

def save_plot(fig, df, sys):
    """
    Saves the provided plot to a file based on the presence of outliers.

    Parameters:
        fig (matplotlib.figure.Figure): The figure to save.
        df (pd.DataFrame): Input DataFrame with an 'Outlier' column.
        sys (int): System identifier for the file name.
    """
    save_directory = 'figures'
    no_outlier_dir = f'{save_directory}/no_outlier'
    os.makedirs(save_directory, exist_ok=True)

    if not df['Outlier'].any():
        os.makedirs(no_outlier_dir, exist_ok=True)
        file_path = f'{no_outlier_dir}/{sys:04d}.png'
    else:
        # num_outliers = df['Outlier'].sum()
        # file_path = f'{save_directory}/N={num_outliers}_{sys:04d}.png'
        file_path = f'{save_directory}/{sys:04d}.png'

    fig.savefig(file_path, dpi=50)
    plt.close(fig)


# =============================================================================
# 01_fine_tune
# =============================================================================
def plot_lle_curve(df, out1, out2, sys, diff, save=True):
    """
    Plots the LLE curve for a given system and highlights the outliers.

    Parameters:
        df (DataFrame): The main dataframe for the system.
        out1 (DataFrame): Outliers for DC+BC.
        out2 (DataFrame): Outliers for DC+BC_slope.
        sys (int): System identifier.
        diff (int): Difference identifier.
        save (bool): Whether to save the plot or display it.
    """
    color_rgb = (84/255, 130/255, 53/255)
    fig, ax = plt.subplots(facecolor=color_rgb)

    def plot(ax, df, x1, x2, y1, y2, options, label):
        x1, x2, y1, y2 = 'x1_L1', 'x1_L2', 'T', 'T'
        ax.plot(df[x1], df[y1], label=label, **options)
        ax.plot(df[x2], df[y2], **options)

    x1, x2, y1, y2 = 'x1_L1', 'x1_L2', 'T', 'T'

    # Plot the main curve
    options = dict(c='k', mfc='w', marker='o', ls='-')
    plot(ax, df, x1, x2, y1, y2, options, label='LLE')

    # Plot outliers DC+BC
    options = dict(c='r', marker='.', ls='none')
    plot(ax, out1, x1, x2, y1, y2, options, label='DC+BC')

    # Plot outliers DC+BC_slope
    options = dict(c='orange', marker='.', ls='none')
    plot(ax, out2, x1, x2, y1, y2, options, label='DC+BC_slope')

    # Customize plot appearance
    ax.set_xlabel('x')
    ax.set_ylabel('T')
    ax.set_title(f'Sys={sys:04d}')
    ax.grid(True)

    # Set text and background colors
    ax.set_facecolor(color_rgb)
    ax.tick_params(colors='white')
    ax.xaxis.label.set_color('white')
    ax.yaxis.label.set_color('white')
    ax.title.set_color('white')

    # Customize legend
    legend = ax.legend()
    legend.get_frame().set_facecolor('gray')
    legend.get_frame().set_edgecolor('white')

    # Save or display the plot
    if save:
        # Create folder for saving plots
        folder = f'figures/diff={diff:02d}'
        os.makedirs(folder, exist_ok=True)
        plt.savefig(f'{folder}/{sys:04d}.png', dpi=50)
        plt.close(fig)
    else:
        plt.show()

    return ax


# =============================================================================
# 02
# =============================================================================
def approximate(lle, npoints=15, ndata=5, part='upper', ax=None):
    # Calculate derivatives at endpoints (you can specify these values)
    ascending = part == 'lower'
    x1, y1, dydx_1 = derivatives(lle, x='x1_L1', ndata=ndata, ascending=ascending)
    x2, y2, dydx_2 = derivatives(lle, x='x1_L2', ndata=ndata, ascending=ascending)
    x = np.array([x1, x2])
    y = np.array([y1, y2])
    dydx = np.array([dydx_1, dydx_2])
    spline = CubicHermiteSpline(x, y, dydx)

    # Generate fine points to sample the curve
    t_fine = np.linspace(x[0], x[-1], 100)  # Fine sampling for detecting maximum
    smooth_curve_fine = spline(t_fine, 0)
    if ax is not None:
        ax.plot(t_fine, smooth_curve_fine, 'r-')
    wiggly_curve = check_second_derivative_sign_change(t_fine, spline)

    # Find index of the local maximum
    if part == 'upper':
        max_idx = np.argmax(smooth_curve_fine)
    else:
        max_idx = np.argmin(smooth_curve_fine)
    x_max = t_fine[max_idx]

    half_points = (npoints - 1) // 2 + 1 # Add +1 to compensate for exluding edges
    t_left = np.linspace(x[0], x_max, half_points + 1)[:-1]  # Includes the maximum
    t_right = np.linspace(x_max, x[-1], half_points + 1)[1:]  # Skip the maximum to avoid duplication

    # Get the selected points
    t_left, t_right = adjust_x_pair(spline, t_left, t_right)
    t = np.concatenate((t_left, [x_max], t_right))
    smooth_curve = spline(t, 0)
    return t, smooth_curve, wiggly_curve

def check_second_derivative_sign_change(x, spline):
    """
    Check if the second derivative of the curve changes sign.
    """
    # Compute second derivatives
    second_derivative = spline.derivative(2)(x)

    # Check if the second derivative changes sign
    if np.all(second_derivative > 0) or np.all(second_derivative < 0):
        return False  # No sign change, smooth curvature
    else:
        return True  # Sign change, inflection points present

def adjust_x_pair(spline, x1, x2):
    x2 = x2[::-1]
    # Get corresponding y values for the x values
    y1 = spline(x1, 0)
    y2 = spline(x2, 0)

    # Find the common y value
    common_y = (y1 + y2) / 2

    # Adjust x1 and x2 to match this common y value
    def equation(x_var, target_y):
        return spline(x_var) - target_y

    new_x1 = fsolve(equation, x1, args=(common_y))
    new_x2 = fsolve(equation, x2, args=(common_y))[::-1]

    return new_x1, new_x2

def derivatives(data, x='x1_L1', ndata=5, ascending=False):
    df = data.copy().sort_values(by='T', ascending=ascending).reset_index(drop=True)
    x, y = df.loc[:,[x, 'T']].iloc[:ndata,:].values.T
    fit = np.polyfit(x, y, deg=2)
    f = np.poly1d(fit)
    dydx = f.deriv(m=1)(x[0])
    return x[0], y[0], dydx

def plot_system_data(sys, df, outliers):
    """
    Plots the system data and outliers with a customized appearance.

    Parameters:
        sys: Identifier for the system being plotted.
        df: DataFrame for the main data points.
        outliers: DataFrame for outliers.
        x1, x2, y1, y2: Column names for x and y values.
    """
    # Define figure and axis with a custom background color
    color_rgb = (84 / 255, 130 / 255, 53 / 255)
    fig, ax = plt.subplots(facecolor=color_rgb)
    ax.set_facecolor(color_rgb)

    x1, x2, y1, y2 = 'x1_L1', 'x1_L2', 'T', 'T'

    def plot(ax, df, x1, x2, y1, y2, options, label=None):
        ax.plot(df[x1], df[y1], label=label, **options)
        ax.plot(df[x2], df[y2], **options)

    # Plot the main curve
    options = dict(c='k', mfc='w', marker='o', ls='-')
    plot(ax, df, x1, x2, y1, y2, options, label='LLE')

    # Plot outliers
    options = dict(c='r', marker='.', ls=':', mec='k', zorder=10)
    plot(ax, outliers.iloc[1:], x1, x2, y1, y2, options, label='Outlier')
    options.pop('marker', None)
    options['zorder'] = 1
    plot(ax, outliers.iloc[:2], x1, x2, y1, y2, options, label=None)

    # Customize plot appearance
    ax.set_xlabel('x')
    ax.set_ylabel('T')
    ax.set_title(f'Sys={sys:04d}')
    ax.grid(True)

    # Set text and background colors
    ax.tick_params(colors='white')
    ax.xaxis.label.set_color('white')
    ax.yaxis.label.set_color('white')
    ax.title.set_color('white')

    # Customize legend
    legend = ax.legend()
    legend.get_frame().set_facecolor('gray')
    legend.get_frame().set_edgecolor('white')

    return fig, ax

def save_fig(sys, fig, wiggly_curve=False):
    # Create folder
    DIR_FIG = 'figures'
    DIR_FIG += "/wiggly" if wiggly_curve else ""
    if not os.path.exists(DIR_FIG):
        os.makedirs(DIR_FIG)

    # Save the plot with low DPI
    plt.savefig(f'{DIR_FIG}/{sys:04d}.png', dpi=50)
    plt.close(fig)


# =============================================================================
# 03
# =============================================================================
def save_adjustments(sys, fig, adjustment):
    # Create folder
    DIR_FIG = 'figures/'
    if isinstance(adjustment, str):
        DIR_FIG += f"{adjustment}"
    else:
        DIR_FIG += f"adjustment={adjustment:02d}"
    if not os.path.exists(DIR_FIG):
        os.makedirs(DIR_FIG)

    # Save the plot with low DPI
    fig.savefig(f'{DIR_FIG}/{sys:04d}.png', dpi=50)
    plt.close(fig)

# =============================================================================
# 04
# =============================================================================
def plot(ax, df, options, label):
    x1, x2, y1, y2 = 'x1_L1', 'x1_L2', 'T', 'T'
    ax.plot(df[x1], df[y1], label=label, **options)
    ax.plot(df[x2], df[y2], **options)

def save_recalc(sys, fig, res, log_file, log_name, T_range, remove_points,
                adjustment, close_fig=True, log=True):
    # Create folder
    DIR_FIG = 'figures'
    # DIR_FIG += f"recalc={adjustment:02d}"
    if not os.path.exists(DIR_FIG):
        os.makedirs(DIR_FIG)

    file_name = f'{DIR_FIG}/{sys:04d}'
    res.to_excel(f'{file_name}'+'.xlsx', index=False)

    if log:
        txt = f"T={T_range}, {remove_points=}"
        log_file.loc[log_file['System'] == sys, 'Recalculate'] = txt
        log_file.loc[log_file['System'] == sys, 'Adjustment'] = adjustment
        log_file.to_excel(log_name, index=False)

    # Save the plot with low DPI
    fig.savefig(f'{file_name}'+'.png', dpi=50)
    if close_fig:
        plt.close(fig)

def recalculate_binodal(ax, df, temperatures, lle, remove_points=0):
    recalc = []
    x0 = df.iloc[-1][['x1_L1', 'x1_L2']]
    for T in temperatures:
        binodal = lle.binodal(T, x0)
        options = dict(c='k', mfc='w', marker='o', ls='none')
        x0 = binodal
        recalc.append((T, *binodal))

    recalc = recalc[:-remove_points] if remove_points > 0 else recalc
    recalc = pd.DataFrame(recalc, columns=['T', 'x1_L1', 'x1_L2'])
    recalc[['sys','c1','c2']] = [i for i in df.loc[0, ['sys', 'c1', 'c2']]]
    options = dict(c='k', mfc='w', marker='o', ls='-')
    plot(ax, recalc, options, label='Recalc')
    recalc['Note'] = 'Recalculation'
    return recalc

def generate_approximation(df, recalc=None, npoints=7, ndata=3, ax=None):
    if recalc is not None and not recalc.empty:
        df = pd.concat([df.copy(), recalc], ignore_index=True)
    t, smooth, wiggly_curve = approximate(df, npoints, ndata)
    if ax is not None:
        ax.plot(t[1:-1], smooth[1:-1], 'o-', c='cyan', mec='k', label='Approx')
        ax.plot(t[:2], smooth[:2], '-', c='cyan', mec='k', zorder=0)
        ax.plot(t[-2:], smooth[-2:], '-', c='cyan', mec='k', zorder=0)

    t, smooth = t[1:-1], smooth[1:-1]
    n = round(len(t) / 2)
    approx = np.column_stack((smooth[:n], t[:n], t[n-1:][::-1]))
    approx = pd.DataFrame(approx, columns=['T', 'x1_L1', 'x1_L2'])
    approx[['sys','c1','c2']] = [i for i in df.loc[0, ['sys', 'c1', 'c2']]]
    approx['Note'] = 'Approximation'
    return approx, wiggly_curve


# =============================================================================
# 05 - FINALIZE
# =============================================================================
# Define a function to apply the subsequent steps
def process_data(df):
    df = df.drop(columns=["Recalculate", "Note", "Points"])
    df = df.sort_values(by=['System']).reset_index(drop=True)
    df['Adjustment'] = df['Adjustment'].fillna(0).astype(int)
    return df

def get_approx(data, outliers, plot=False, max_T=1000):
    systems = outliers['System'].unique()
    # systems = [2106]
    ax = None
    res = []
    for i, sys in enumerate(systems):
        dfs = data[data.sys==sys]
        outlier = outliers[outliers['System']==sys]['Outlier'].iloc[0]
        adjustment = outliers[outliers['System']==sys]['Adjustment'].iloc[0]
        _outlier = outlier + adjustment
        df = dfs.iloc[:-_outlier].tail(3) if _outlier > 0 else dfs.copy().tail(4)
        df = df.reset_index(drop=True)
        if plot:
            out = dfs.tail(_outlier + 1)
            fig, ax = plot_system_data(sys, df, out)
        dfs = dfs.copy().iloc[:-_outlier] if _outlier > 0 else dfs.copy()

        # Approximate
        max_temp = dfs['T'].max()
        if max_temp < max_T:
            approx, wiggly_curve = generate_approximation(df, ax=ax)
            msg = '- is wiggly' if wiggly_curve else ''
            print(f"{i+1:02d}:{sys:04d} processed. {msg}")
            # Save results
            res.append(pd.concat([dfs, approx], ignore_index=True))
        else:
            print(f"{i+1:02d}:{sys:04d} skipped approximation due to high temperature.")
            res.append(dfs)
    return pd.concat(res, ignore_index=True)
