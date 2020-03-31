class CustomError(Exception):
    """Base class for exceptions in this module."""
    def __init__(self, exception_type, message, exit_status):
        self.exception_type = exception_type
        self.exit_status = exit_status
        self.message = message


class InsufficientDataError(CustomError):
    """Exception raised for input datasets not containing sufficient data, for
    example, the dataset could have only a single pixel that has valid non-fill
    values. Alternatively, non-fill values may appear in only a single row or
    single column.

    """
    def __init__(self, message):
        super().__init__('InsufficientDataError', message, None)


class InsufficientProjectionInformation(CustomError):
    """Exception raised when there is no projection information in an input
    granule, and the granule's collection does not have default projection
    information in the global MaskFill configuration file.

    """
    def __init__(self, dataset_name):
        super().__init__('InsufficientProjectionInformation',
                         ('Cannot find projection information for dataset: '
                          f'{dataset_name}.'),
                         None)


class InternalError(CustomError):
    """This Exception is used as a default for when a standard (non-MaskFill
    specific) exception is raised. This Exception is used in output error
    messages.

    """
    def __init__(self, message='An internal error occurred.'):
        super().__init__('InternalError', message, None)


class InvalidParameterValue(CustomError):
    """Exception raised when an input parameter cannot be parsed as expected.
    This Exception is used in output error messages.

    """
    def __init__(self, message='Incorrect parameter specified for given dataset(s).'):
        super().__init__('InvalidParameterValue', message, 1)


class MissingCoordinateDataset(CustomError):
    """Exception raised when a science dataset refers to a coordinate dataset,
    within its `coordinates` attribute, that is not present in the granule. This
    Exception is used in output error messages.

    """
    def __init__(self, file_name, dataset):
        super().__init__('MissingCoordinateDataset',
                         f'Cannot find "{dataset}" in "{file_name}".', 4)


class MissingParameterValue(CustomError):
    """Exception raised when a required input parameter is not given to MaskFill.
    This Exception is used in output error messages.

    """
    def __init__(self, message='No parameter value(s) specified for given dataset(s).'):
        super().__init__('MissingParameterValue', message, 2)


class NoMatchingData(CustomError):
    """This Exception is used in output error messages. Currently is not raised
    within MaskFill.

    """
    def __init__(self, message='No data found that matched the subset constraints.'):
        super().__init__('NoMatchingData', message, 3)
