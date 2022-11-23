class CustomError(Exception):
    """Base class for exceptions in this module."""
    def __init__(self, exception_type, message, exit_status=None):
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
        super().__init__('InsufficientDataError', message)


class InsufficientProjectionInformation(CustomError):
    """Exception raised when there is no projection information in an input
    granule, and the granule's collection does not have default projection
    information in the global MaskFill configuration file.

    """
    def __init__(self, dataset_name):
        super().__init__('InsufficientProjectionInformation',
                         ('Cannot find projection information for dataset: '
                          f'{dataset_name}.'), 5)


class InternalError(CustomError):
    """This Exception is used as a default for when a standard (non-MaskFill
    specific) exception is raised. This Exception is used in output error
    messages.

    """
    def __init__(self, message='An internal error occurred.'):
        super().__init__('InternalError', message)


class InvalidParameterValue(CustomError):
    """Exception raised when an input parameter cannot be parsed as expected.
    This Exception is used in output error messages.

    """
    def __init__(self, message='Incorrect parameter specified for given dataset(s).'):
        super().__init__('InvalidParameterValue', message, 1)


class InvalidMetadata(CustomError):
    """ Exception raised when metadata contained in a dataset's attributes is
        invalid. For example, if a reference to a grid mapping or coordinate
        variable has a relative path that suggests that the origin of the
        reference is more deeply nested than it actually is. For example:

        Referee: '/group1/science_variable'
        Reference: '../../grid_mapping'

    """
    def __init__(self, dataset, attribute_name, attribute_value, message=None):
        combined_message = (f'Invalid metadata in {dataset}: {attribute_name}='
                            f'"{attribute_value}"')

        if message is not None:
            combined_message = ': '.join([combined_message, message])

        super().__init__('InvalidMetadata', combined_message, 6)


class MissingCoordinateDataset(CustomError):
    """ Exception raised when a science dataset refers to a coordinate dataset,
        within its `coordinates` attribute, that is not present in the granule.
        This Exception is used in output error messages.

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
