import pandas as pd
import numpy as np

from thermosac import LLE, GMixScanner

class LLEGenerator:
    def __init__(self,
                 systems,
                 initial_values,
                 initializer,  # Mandatory initializer
                 initargs: dict,  # Mandatory initialization arguments
                 **kwargs) -> None:

        self.args = self.generate_args(systems, initial_values)
        self.two_separate_LLE = ()
        if "x1_L2_inner" in initial_values.columns:
            two_lle = initial_values['x1_L2_inner'].notna()
            self.two_separate_LLE = initial_values[two_lle]['sys'].unique()

        # Set up initializer and initargs - 'actmodel = initializer(**initargs)'
        self.initializer = initializer
        self.initargs = initargs

    def calculate_miscibility(self, mode='sequential', **settings):
        self.settings = settings
        args = self.args
        process = self._process_miscibility
        results = GMixScanner._run_process(process, args, mode)
        miscibility = pd.concat(results)
        miscibility = self.post_process_miscibility(miscibility)
        return miscibility

    def _process_miscibility(self, arg):
        system, T, x0 = arg
        actmodel = self.initializer(system=system, **self.initargs)
        lle = LLE(actmodel)
        dT0 = 30 if 'WATER' in actmodel.mixture.names else 10
        dT0 = self.settings.pop('dT0', dT0)
        res = lle.miscibility(T, x0, dT0, **self.settings)
        res.insert(0, 'sys', system)
        return res

    def post_process_miscibility(self, miscibility):
        # Step 1: Split systems into two groups based on 'self.two_separate_LLE'.
        sys_with_two_lle = miscibility[miscibility['sys'].isin(self.two_separate_LLE)]
        if sys_with_two_lle.empty:
            self.miscibility = self.multi_combination_sorting(miscibility)
            return self.miscibility

        # Step 2: Reset the index, find split points, and split the DataFrame into blocks.
        df = sys_with_two_lle.reset_index(drop=False)
        idx = df[df['index'] == 0].index[1:]  # Identify split points.
        df = df.drop(columns=['index'])
        splits = self.split_dataframe(df, idx)

        # Step 3: Concatenate alternating splits into left and right parts.
        df_L = pd.concat(splits[::2], ignore_index=True)
        df_R = pd.concat(splits[1::2], ignore_index=True)

        # Step 4: Dynamically create renaming mappings for `cols_L` and `cols_R`.
        check = lambda col: col.startswith("x1_L") or col.startswith("x1_S")
        cols = [col for col in df.columns if check(col)]
        # Columns for two LLE: left=(L1--L2_inner) | right=(L1_inner--L2)
        cols_L = {col: f"{col}_inner" if col.endswith("2") else col for col in cols}
        cols_R = {col: f"{col}_inner" if col.endswith("1") else col for col in cols}

        # Step 5: Rename columns and sort each group by temperature ('T').
        df_L = df_L.rename(columns=cols_L).sort_values(by='T')
        df_R = df_R.rename(columns=cols_R).sort_values(by='T')

        # Step 6: Combine the processed left and right DataFrames.
        sys_with_two_lle = pd.concat([df_L, df_R]).reset_index(drop=True)

        # Step 7: Append unprocessed systems, sort by `sys`, and finalize the result.
        sys_with_one_lle = miscibility[~miscibility['sys'].isin(self.two_separate_LLE)]
        res = pd.concat([sys_with_two_lle, sys_with_one_lle]).sort_values(by='sys')
        xL_cols = [col for col in res.columns if col.startswith('x1_L')]
        xS_cols = [col for col in res.columns if col.startswith('x1_S')]
        columns = ['sys', 'T'] + xL_cols + ['c1', 'c2'] + xS_cols
        self.miscibility = self.multi_combination_sorting(res[columns])

        # Return the final processed miscibility DataFrame.
        return self.miscibility

    @classmethod
    def multi_combination_sorting(cls, df):
        """
        Processes a DataFrame by dynamically excluding irrelevant x1_L columns based on combinations,
        sorting within each combination by T, and ensuring the final DataFrame is sorted by sys,
        combination order, and T.

        Parameters:
            df (pd.DataFrame): The input DataFrame containing columns like 'sys', 'T', 'x1_L1', etc.
            combinations (list of tuple): List of column combinations to process.

        Returns:
            pd.DataFrame: The sorted and processed DataFrame.
        """

        combinations = cls.generate_combinations(df)
        dfs_processed = []

        # Process each combination
        for idx, cols in enumerate(combinations):
            # Identify columns to exclude dynamically
            cols_to_exclude = [col for col in df.columns
                               if col.startswith('x1_L') and col not in cols]

            # Drop excluded columns temporarily
            df_subset = df.drop(columns=cols_to_exclude).dropna(subset=cols).sort_values(by='T')

            # Add a temporary column for combination order
            df_subset['combination_order'] = idx
            dfs_processed.append(df_subset)

        # Ensure there is data to process
        if not dfs_processed:
            raise ValueError("No valid data found for the given combinations.")

        # Step 1: Concatenate all processed DataFrames
        df_combined = pd.concat(dfs_processed)

        # Step 2: Sort by 'sys', 'combination_order', and 'T'
        df_sorted = df_combined.sort_values(by=['sys', 'combination_order', 'T'])

        # Step 3: Drop the temporary 'combination_order' column
        df_sorted = df_sorted.drop(columns=['combination_order'])

        # Step 4: Reset the index of the final DataFrame
        df_final = df_sorted.reset_index(drop=True)

        # Ensure the final DataFrame has the same column order as the original DataFrame
        df_final = df_final.reindex(columns=df.columns)

        return df_final

    @staticmethod
    def generate_combinations(df):
        """
        Generate valid combinations of L1 and L2 columns from the given DataFrame.

        Parameters:
            df (pd.DataFrame): The input DataFrame containing columns like 'x1_L1', 'x1_L2', etc.

        Returns:
            list of tuple: A list of valid (L1, L2) combinations.
        """
        # Function to check if a column starts with x1_L or x1_S
        def is_valid_column(col):
            return col.startswith("x1_L") or col.startswith("x1_S")

        # Identify all valid x1_L and x1_S columns
        cols = [col for col in df.columns if is_valid_column(col)]

        # Separate columns into L1 and L2 groups
        L1_cols = [col for col in cols if col.endswith("L1") or col.endswith("L1_inner")]
        L2_cols = [col for col in cols if col.endswith("L2") or col.endswith("L2_inner")]

        # Generate combinations: L1 and L2 columns paired according to the rules
        combinations = [
            (l1, l2)
            for l1 in L1_cols
            for l2 in L2_cols
            if l1 < l2 and not (l1.endswith("_inner") and l2.endswith("_inner"))
        ]

        return combinations

    @staticmethod
    def generate_args(systems, initial_values):
        args = []

        for system in systems:
            # Filter initial values for the current system
            inits = initial_values[initial_values['sys'] == system]

            # Extract T value
            T = inits['T'].iloc[0]

            # Extract and clean initial values
            x_init = inits.filter(like='x1_L', axis=1).values.flatten()
            x_init = x_init[~np.isnan(x_init)]

            # Pair adjacent values as x0 and append arguments
            for i in range(0, len(x_init) - 1, 2):
                x0 = x_init[[i, i + 1]]
                args.append((system, T, x0))

        return args

    @staticmethod
    def split_dataframe(df, idx):
        """
        Splits a DataFrame at specified indices into multiple chunks.

        Parameters:
        df (pd.DataFrame): The DataFrame to split.
        idx (list): List of indices where the DataFrame should be split.

        Returns:
        list: A list of DataFrame chunks.
        """
        # Add start and end points
        split_points = [0, *idx, len(df)]

        # Create slices using start and end variables
        splits = [df.iloc[start:end]
                  for start, end in zip(split_points[:-1], split_points[1:])]

        return splits
