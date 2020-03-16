class CustomError(Exception):
    """Base class for exceptions in this module."""
    pass


class InsufficientDataError(CustomError):
    """Exception raised for input datasets not containing sufficient data, for
    example, the dataset could have only a single pixel that has valid non-fill
    values. Alternatively, non-fill values may appear in only a single row or
    single column.

    """
    def __init__(self, message):
        self.message = message
