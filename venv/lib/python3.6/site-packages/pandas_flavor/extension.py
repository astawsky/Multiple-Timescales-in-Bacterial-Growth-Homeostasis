from .register import register_dataframe_accessor
from functools import wraps
#def get_functions(module):


def get_dataframe_accessor(name, functions):
    """Returns a dataframe accessor with the given name and methods.
    """
    @register_dataframe_accessor(name)
    class DataFrameAccessor:

        def __init__(self, df):
            self._df = df

    # Add functions to accessor
    for f in functions:
        # Build a method from a function
        @wraps(f)
        def method(self, *args, **kwargs):
            return f(self._df, *args, **kwargs)

        # Set the function name
        setattr(DataFrameAccessor, f.__name__, method)

    return DataFrameAccessor


def get_series_accessor(name, functions):
    pass