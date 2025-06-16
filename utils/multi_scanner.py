from typing import Iterable, Union
from thermosac.equilibrium import GMixScanner

class GMixMultiScan:
    def __init__(self,
                 systems: Iterable[int],
                 temperatures: Iterable[Union[float, int]],
                 mole_fractions: Iterable[Union[float, int]],
                 initializer,  # Mandatory initializer
                 initargs: dict,  # Mandatory initialization arguments
                 **kwargs) -> None:

        self.systems = systems
        self.temperatures = temperatures
        self.mole_fractions = mole_fractions
        # Set up initializer and initargs - 'actmodel = initializer(**initargs)'
        self.initializer = initializer
        self.initargs = initargs

# =============================================================================
# Public Methods
# =============================================================================
    def find_first_binodal(self, mode='parallel', check_edges=True, gmix=None,
                           keep_all_gmix=False):
        args = self.systems
        process = self._process_first_binodal
        config = (process, args, mode, check_edges, gmix, keep_all_gmix)
        binodal, gmix = self._execute(*config)
        binodal = GMixScanner._post_process_binodal(self, binodal)
        gmix = GMixScanner._post_process_gmix(self, gmix)
        return binodal, gmix

    def find_all_binodal(self, mode='parallel', check_edges=True, gmix=None,
                         keep_all_gmix=False):
        args = ((sys, T) for sys in self.systems for T in self.temperatures)
        process = self._process_binodal
        config = (process, args, mode, check_edges, gmix, keep_all_gmix)
        binodal, gmix = self._execute(*config)
        binodal = GMixScanner._post_process_binodal(self, binodal)
        gmix = GMixScanner._post_process_gmix(self, gmix)
        return binodal, gmix

    def get_all_gmix(self, mode='parallel', check_edges=False):
        args = ((sys, T) for sys in self.systems for T in self.temperatures)
        process = self._process_gmix
        config = (process, args, mode, check_edges)
        gmix = self._execute(*config)
        return gmix

# =============================================================================
# Processing and Execution
# =============================================================================
    def _process_first_binodal(self, arg):
        system = arg
        args = (system, self.temperatures, 'find_first_binodal')
        binodal, gmix = self._initialize_and_scan(*args)
        return binodal, gmix

    def _process_gmix(self, arg):
        system, T = arg
        args = (system, [T], 'get_all_gmix')
        gmix = self._initialize_and_scan(*args)
        return gmix

    def _process_binodal(self, arg):
        system, T = arg
        args = (system, [T], 'find_all_binodal')
        binodal, gmix = self._initialize_and_scan(*args)
        return binodal, gmix

    def _execute(self, process, args, mode, check_edges, gmix, keep_all_gmix):
        self.check_edges = check_edges
        self.gmix = gmix
        self.keep_all_gmix = keep_all_gmix
        results = GMixScanner._run_process(process, args, mode)
        return GMixScanner._split_results(results)

# =============================================================================
# Internal / Helper Methods
# =============================================================================
    def _initialize_and_scan(self, system, temperatures, scan_method):
        """
        Helper method to initialize the activity model and scanner,
        then invoke the specified scanning method dynamically.
        """
        actmodel = self.initializer(system=system, **self.initargs)
        scanner = GMixScanner(actmodel, temperatures, self.mole_fractions)
        scanner.system = system

        # Dynamically invoke the scan method on the scanner
        scan_func = getattr(scanner, scan_method)
        gmix = None if self.gmix is None else self.gmix[self.gmix.sys == system]
        args = dict(check_edges=self.check_edges, gmix=gmix)
        result = scan_func(**args, keep_all_gmix=self.keep_all_gmix)
        if hasattr(scanner, 'binodal_full'):
            gmix = result[-1]
            result = (scanner.binodal_full, gmix)
        return self._insert_system(system, result)

    @staticmethod
    def _insert_system(system, result):
        """
        Inserts 'sys' at the start of the result, whether it's a single value or a tuple.
        """
        if isinstance(result, tuple):
            return tuple([safe_insert(r, 0, 'sys', system) for r in result])
        else:
            safe_insert(result, 0, 'sys', system)
            return result


def safe_insert(df, loc, column, value):
    """
    Safely inserts a column into a DataFrame if it doesn't already exist.

    Parameters:
        df (pd.DataFrame): The DataFrame to insert into.
        loc (int): The position to insert the column.
        column (str): The name of the column to insert.
        value: The values to insert into the column.

    Returns:
        None: The DataFrame is modified in place.
    """
    if column not in df.columns:
        df.insert(loc, column, value)
    return df
