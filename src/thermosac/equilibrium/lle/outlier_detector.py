import numpy as np
import pandas as pd

class LLEOutlierDetector:

    def detect_outliers(self, df, methods=['DC', 'BC']):
        """
        Detects outliers based on specified methods.

        Parameters:
        - df (DataFrame): The input DataFrame.
        - methods (list): List of methods to apply for outlier detection.
                          Available methods: ['DC', 'BC', 'TA'].
                          Default is ['DC', 'BC'].

        Returns:
        - DataFrame: The DataFrame with outlier columns processed and combined.
        """
        # Validate methods input
        valid_methods = {'DC', 'BC', 'TA'}
        invalid_methods = [method for method in methods if method not in valid_methods]
        if invalid_methods:
            raise ValueError(f"Invalid methods specified: {invalid_methods}. Valid methods are {valid_methods}")

        deriv = self.gradient
        df['dx1'] = deriv(df['xL1*'], df['T*'])
        df['dx2'] = deriv(df['xL2*'], df['T*'])
        df['ddx1'] = deriv(df['dx1'], df['T*'])
        df['ddx2'] = deriv(df['dx2'], df['T*'])

        # Apply Direction Changes (DC)
        if 'DC' in methods:
            DC_slope_1 = self.check_direction_change(df['dx1'], min_value=0.1)
            DC_slope_2 = self.check_direction_change(df['dx2'], min_value=0.1)
            DC_curvature_1 = self.check_direction_change(df['ddx1'], min_value=1)
            DC_curvature_2 = self.check_direction_change(df['ddx2'], min_value=1)

            direction_change_slope = DC_slope_1 | DC_slope_2
            direction_change_curvature = DC_curvature_1 | DC_curvature_2
            df['DC'] = direction_change_slope | direction_change_curvature

        # Apply Big Changes (BC)
        if 'BC' in methods:
            BC_slope_1 = self.check_big_change(df['dx1'], factor=3, min_value=1)
            BC_slope_2 = self.check_big_change(df['dx2'], factor=3, min_value=1)
            BC_curvature_1 = self.check_big_change(df['ddx1'], factor=8, min_value=10)
            BC_curvature_2 = self.check_big_change(df['ddx2'], factor=8, min_value=10)

            big_change_slope = BC_slope_1 | BC_slope_2
            big_change_curvature = BC_curvature_1 | BC_curvature_2
            df['BC'] = big_change_slope | big_change_curvature
            df['BC_slope'] = big_change_slope

        # Apply Taylor Approximation (TA)
        if 'TA' in methods:
            df['TA'] = compute_taylor_and_outliers(df, 'xL1*', 'xL2*', 'T*')

        # Post-process outliers
        return self.post_process_outliers(df)


    @staticmethod
    def check_direction_change(col, min_value):
        sign_change = ~(np.sign(col).diff().fillna(0) == 0)
        greater_than_threshold = abs(col) > min_value
        return sign_change & greater_than_threshold

    @staticmethod
    def check_big_change(col, factor, min_value):
        prev_values = col.shift(1)
        factor_change = abs(col / prev_values)
        big_change = factor_change >= factor
        greater_than_threshold = (abs(col) > min_value)
        return big_change & greater_than_threshold

    def post_process_outliers(self, df):
        # Define outlier columns
        outlier_columns = ['Outlier', 'DC', 'BC', 'BC_slope', 'TA']

        # Combine outlier conditions dynamically using df.get()
        df['Outlier'] = False  # Initialize the 'Outlier' column
        for col in outlier_columns[1:]:  # Skip 'Outlier' in the loop
            df['Outlier'] |= df.get(col, False)  # Dynamically combine conditions

        # Propagate all outlier-related columns dynamically if they exist
        for col in [c for c in outlier_columns if c in df.columns]:
            df = self.propagate_outliers(df, outlier_column=col)

        # Reorder DataFrame columns: existing columns first,
        # followed by outlier columns in order
        df = df[
            [col for col in df.columns if col not in outlier_columns]
            + [col for col in outlier_columns if col in df.columns]
        ]

        # Check if DC and BC columns exist and create combined columns
        if 'DC' in df.columns and 'BC' in df.columns:
            df['DC+BC'] = df['DC'] | df['BC']
        if 'DC' in df.columns and 'BC_slope' in df.columns:
            df['DC+BC_slope'] = df['DC'] | df['BC_slope']

        return df

    @staticmethod
    def propagate_outliers(df, outlier_column='Outlier', sort_by='T*'):
        """
        Propagates True values in the outlier column to all rows below the first True outlier,
        based on increasing values in the specified column (e.g., temperature).

        Parameters:
        df (DataFrame): Input DataFrame containing the outlier information.
        outlier_column (str): The column name that contains the outlier flags (default is 'Outlier').
        sort_by (str): The column name by which to sort the DataFrame (default is 'T*').

        Returns:
        DataFrame: The DataFrame with propagated outliers.
        """
        # Make a copy of the DataFrame to avoid modifying the original one
        df = df.copy()

        # Sort the DataFrame by the specified column to ensure increasing order
        df = df.sort_values(by=sort_by).reset_index(drop=True)

        # Find the index of the first occurrence of a True outlier
        first_outlier_index = df[df[outlier_column]].index.min()

        # If a True outlier is found, set all rows below the first outlier to True
        if not pd.isna(first_outlier_index):
            df.loc[df.index >= first_outlier_index, outlier_column] = True

        return df

    @staticmethod
    def select_part(df, part='upper'):
        # Get original xmin and xmax
        xmin = df['x1_L1'].min()
        xmax = df['x1_L2'].max()

        # Find the temperatures corresponding to xmin and xmax
        T_xmin = df[df['x1_L1'] == xmin]['T'].values[0]
        T_xmax = df[df['x1_L2'] == xmax]['T'].values[0]

        # Filter the DataFrame based on T_threshold
        if part == 'upper':
            T_threshold = max(T_xmin, T_xmax)
            df = df[df['T'] >= T_threshold]
        elif part == 'lower':
            T_threshold = min(T_xmin, T_xmax)
            df = df[df['T'] <= T_threshold]
        else:
            raise ValueError("Invalid part. Select 'upper' or 'lower'.")

        return df

    @staticmethod
    def normalize_data(df):
        df = df.copy()
        ''' Normalize both axes to be within [0, 1] '''
        xmin = df['x1_L1'].min()
        xmax = df['x1_L2'].max()
        ymin = df['T'].min()
        ymax = df['T'].max()

        df['T*'] = (df['T'] - ymin) / (ymax - ymin)
        df['xL1*'] = (df['x1_L1'] - xmin) / (xmax - xmin)
        df['xL2*'] = (df['x1_L2'] - xmin) / (xmax - xmin)
        # df['gap'] = df['x1_L2'] - df['x1_L1']
        # df['gap*'] = df['xL2*'] - df['xL1*']

        return df

    @staticmethod
    def filter_data(df, threshold=0.2, min_required_points=10, filter_type='upper', column='T*'):
        """
        Filters a DataFrame to retain rows within the upper or lower threshold
        of a specified column, ensuring a minimum number of rows are included.

        Parameters:
        - df (DataFrame): The input DataFrame.
        - threshold (float): The maximum deviation threshold (e.g., 0.2 for top or bottom 20%).
        - min_required_points (int): The minimum number of points to retain after filtering.
        - filter_type (str): 'upper' to filter for the top values, 'lower' for the bottom values.
        - column (str): The column to apply the filter on. Default is 'T*'.

        Returns:
        - DataFrame: The filtered DataFrame.
        """
        if column not in df.columns:
            raise ValueError(f"The DataFrame must contain a column named '{column}'.")

        # Determine the boundary based on filter type
        boundary = 1 - threshold if filter_type == 'upper' else threshold
        ascending = filter_type == 'lower'

        # Filter the DataFrame based on the boundary
        df_filtered = df[df[column] >= boundary] if filter_type == 'upper' else df[df[column] <= boundary]

        # Ensure at least the minimum required points are retained
        if len(df_filtered) < min_required_points:
            df_filtered = df.sort_values(by=column, ascending=ascending).head(min_required_points)

        # Return the sorted DataFrame
        return df_filtered.sort_values(by=column, ascending=True)


    @staticmethod
    def gradient(y, x, method='backward', ascending=True):
        if len(x) != len(y):
            raise ValueError("x and y must have the same length")
        # Reverse the arrays if direction is 'decreasing'
        if not ascending:
            x = x[::-1]
            y = y[::-1]
        # Calculate the forward difference slopes
        with np.errstate(invalid='ignore'):
            dy = np.diff(y)
        dx = np.diff(x)
        slopes = np.divide(dy, dx, out=np.full_like(dy, np.inf), where=dx != 0)
        if method == 'forward':
            # slopes = np.append(slopes, slopes[-1])
            slopes = np.append(slopes, np.nan)
        elif method =='backward':
            # slopes = np.insert(slopes, 0, slopes[0])
            slopes = np.insert(slopes, 0, np.nan)
        else:
            raise ValueError("Invalid method. Use 'forward' or 'backward'.")
        # Reverse the arrays if direction is 'decreasing'
        if not ascending:
            x = x[::-1]
            y = y[::-1]
            slopes = slopes[::-1]
        return slopes



# =============================================================================
# Other outlier detection algorithms
# =============================================================================
def detect_outliers_moving_avg(df, column, window_size=3, threshold=1.8):
    """
    Detects outliers based on a moving average and returns a boolean mask.

    Parameters:
    df (DataFrame): Input DataFrame containing the dataset.
    column (str): The column name to calculate the moving average on.
    window_size (int): The size of the moving window (default is 3).
    threshold (float): The factor threshold for detecting outliers (default is 2).

    Returns:
    Series: A boolean Series where True represents an outlier.
    """
    # Create a copy of the DataFrame to avoid modifying the original one
    df = df.copy()

    # Find the first valid (non-NaN) index in the column
    first_valid_index = df[column].first_valid_index()

    # Calculate the moving average
    weights = np.ones(window_size) / window_size
    moving_avg = np.convolve(df[column].dropna(), weights, mode='valid')

    # Discard the last value of the moving average to align it starting at index window_size
    moving_avg = moving_avg[:-1]

    # Create a new column for Moving_Average with NaN for initial rows
    df['Moving_Average'] = np.nan

    # Calculate where the moving average values should start (based on the first valid index)
    start_index = df.index.get_loc(first_valid_index) + window_size

    # Assign the moving average values to the correct indices
    df.loc[df.index[start_index]:df.index[start_index + len(moving_avg) - 1], 'Moving_Average'] = moving_avg

    # Create a Ratio column, compute the ratio where the moving average is non-zero
    df['Ratio'] = np.nan
    valid_moving_avg = df['Moving_Average'] != 0  # Avoid division by zero
    df.loc[valid_moving_avg, 'Ratio'] = df[column] / df['Moving_Average']

    # Generate the outlier boolean mask based on ratio exceeding the threshold
    outlier = abs(df['Ratio']) > threshold

    return outlier

def taylor_approximation(df, ycol, xcol):
    """
    Compute the Taylor series approximation for a given column in a DataFrame using the
    first and second derivatives.

    Parameters:
    -----------
    df : pd.DataFrame
        The input DataFrame containing the data. Must include columns for temperature ('T'),
        the target column for approximation, and its first and second derivatives.
    col : str
        The column name for which to compute the Taylor approximation. The function will
        automatically identify the derivative columns by appending 'dx' and 'ddx' with the
        appropriate phase (last character of `col`).

    Returns:
    --------
    pd.Series
        A pandas Series containing the Taylor approximations for each point, using values
        from the previous temperature as the expansion point.
    """
    def find_first_number_right_to_left(s):
        import re
        # Find all digits and reverse them
        digits = re.findall(r'\d', s)
        # Return the last digit (first from the right)
        return digits[-1] if digits else None

    df = df.copy()
    phase = find_first_number_right_to_left(ycol)
    x = df[xcol]
    y = df[ycol]
    dy = df['dx' + phase]
    ddy = df['ddx' + phase]
    # dddy = df['ddx' + phase]

    # Step 1: Calculate delta_T (difference in temperature relative to the previous T)
    dx = x - x.shift(1)

    # Step 2: Shift previous values for col, dx1, and ddx1
    y0 = y.shift(1)
    dy = dy.shift(1)
    ddy = ddy.shift(1)
    # ddy = dddy.shift(1)

    # Step 3: Compute the Taylor series approximation in a vectorized way
    approx = y0 + dy * dx + 0.5 * ddy * dx**2# + (1/6) * dddy * dx**3

    return approx

# def compute_taylor_and_outliers(df, col1, col2, xcol, ratio_threshold=5, deviation_threshold=0.01):
def compute_taylor_and_outliers(data, col1, col2, xcol, ratio_threshold=4, deviation_threshold=0.005):
    df = data.copy()

    # Step 1: Compute Taylor approximations for both columns
    df['taylor1'] = taylor_approximation(df, col1, xcol)
    df['taylor2'] = taylor_approximation(df, col2, xcol)
    # data['taylor1'] = df['taylor1']
    # data['taylor2'] = df['taylor2']

    # Step 2: Compute deviations based on Taylor approximations
    df['deviation1'] = abs(df[col1] - df['taylor1'])
    df['deviation2'] = abs(df[col2] - df['taylor2'])

    df['ratio1'] = df['deviation1'] / df['deviation1'].shift(1)
    df['ratio2'] = df['deviation2'] / df['deviation2'].shift(1)

    # Step 3: Detect outliers based on deviation ratios and deviation thresholds
    df['outlier1'] = (df['ratio1'] > ratio_threshold) & (df['deviation1'] > deviation_threshold)
    df['outlier2'] = (df['ratio2'] > ratio_threshold) & (df['deviation2'] > deviation_threshold)

    # Handle NaN in the first row (fill NaN with False, no inplace)
    df['outlier1'] = df['outlier1'].fillna(False)
    df['outlier2'] = df['outlier2'].fillna(False)

    # Return a combination of both outliers
    return df['outlier1'] | df['outlier2']
