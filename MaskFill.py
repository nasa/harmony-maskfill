#!/usr/bin/env python
""" Executable which creates a mask filled version of a data file using a shapefile.
    Applies a fill value to the data file in the areas outside of the given shapes.
    Writes a log file to the current working directory.

    Input parameters:
        --FILE_URLS: Path to a GeoTIFF or HDF5 file

        --BOUNDINGSHAPE: Path to a shapefile or the native GeoJson string
            (shp, kml, geojson, etc.)

        --OUTPUT_DIR: (optional) Path to the output directory
            where the mask filled file will be written.
            If not provided, the current working directory will be used.

        --MASK_GRID_CACHE: (optional) Value determining how the mask arrays
            used in the mask fill are cached and used.
            Valid values: ignore_and_delete  | ignore_and_save | use_cache
                          | use_cache_delete | MaskGrid_Only

            ignore_and_delete - ignore any existing MaskGrid (create new)
                and delete the MaskGrid after processing
            input file (default if MaskGridCache not specified)
            ignore_and_save - save the MaskGrid in output directory
                and continue (ignore any existing)
            use_cache | use_and_save - use any existing cache value
                and save/preserve MaskGrid in output directory
            use_cache_delete - use any existing MaskGrid, but delete after processing
            MaskGrid_Only - ignore and save, but no MaskFill processing

            If not provided, the value 'ignore_and_delete' will be used.

        --DEFAULT_FILL: (optional) The default fill value for the mask fill
            if no other fill values are provided.
            If not provided, the value -9999 will be used.

        --DEBUG: (optional) If True, changes the log level to DEBUG from the
            default INFO. If not provided, the value will be False.
"""
from argparse import ArgumentParser, Namespace
from typing import Dict, Union
import logging
import os
import re
import uuid

from pymods.exceptions import (InsufficientProjectionInformation,
                               InternalError, InvalidMetadata,
                               InvalidParameterValue, MissingCoordinateDataset,
                               MissingParameterValue, NoMatchingData,
                               UnknownCollectionShortname)
import GeotiffMaskFill
import H5MaskFill


DEFAULT_FILL_VALUE = -9999
DEFAULT_MASK_GRID_CACHE = 'ignore_and_delete'
OUTPUT_EXCEPTIONS = (InsufficientProjectionInformation, InvalidMetadata,
                     InvalidParameterValue, MissingCoordinateDataset,
                     MissingParameterValue, NoMatchingData,
                     UnknownCollectionShortname)


def mask_fill(input_file: str, shape_file: str, output_dir: str,
              mask_grid_cache: str, fill_value: Union[int, float],
              logger: logging.Logger) -> str:
    """ Performs a mask fill on the given data file using RQS agent call input
        parameters.

        Returns:
            str: The full path of the masked output file
    """
    # Perform mask fill according to input file type
    input_extension = os.path.splitext(input_file)[1].lower()

    if input_extension == '.tif':
        # GeoTIFF case
        logger.info(f'Performing mask fill with GeoTIFF {input_file} and '
                    f'shapefile {shape_file}')
        output_file = GeotiffMaskFill.produce_masked_geotiff(
            input_file, shape_file, output_dir, output_dir,
            mask_grid_cache, fill_value, logger
        )
    elif input_extension == '.h5':
        # HDF5 case
        logger.info(f'Performing mask fill with HDF5 file {input_file} '
                    f'and shapefile {shape_file}')
        output_file = H5MaskFill.produce_masked_hdf(
            input_file, shape_file, output_dir, output_dir,
            mask_grid_cache, fill_value, logger
        )

    return output_file


def get_sdps_logger(output_dir: str, debug: bool) -> logging.Logger:
    """ Configures the logger for the MaskFill process, setting the log level
        to INFO. The logger will output to both the console via a
        `logging.StreamHandler` object, and to a file via a
        `logging.FileHandler` object.

    """
    logging_formatter = logging.Formatter('%(levelname)s: %(message)s')

    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(logging_formatter)

    file_handler = logging.FileHandler(get_log_file_path(output_dir))
    file_handler.setFormatter(logging_formatter)

    logger = logging.getLogger('MaskFill')

    if debug:
        logger.setLevel(logging.DEBUG)
    else:
        logger.setLevel(logging.INFO)

    # Remove any stale handlers, to avoid duplicate messages to the terminal
    for handler in logger.handlers:
        handler.close()

    logger.handlers.clear()

    # Add fresh handlers
    logger.addHandler(stream_handler)
    logger.addHandler(file_handler)

    return logger


def get_log_file_path(output_dir: str) -> str:
    """ Gets the file path to the log file for the mask fill process.
        The file will be in the output directory with filename 'mask_fill.log'.

        Returns:
            str: The log file path
    """
    return os.path.join(output_dir, 'mask_fill.log')


def debug_bool(debug_input: Union[bool, str]) -> bool:
    """ Parse the input string parameter for the `debug` argument, as received
        by the `argparse.ArgumentParser` instance, and convert it to a boolean.

    """
    if isinstance(debug_input, bool):
        debug_boolean = debug_input
    elif (
            isinstance(debug_input, str)
            and debug_input.strip('\'').lower() in ('true', 't')
    ):
        debug_boolean = True
    else:
        debug_boolean = False

    return debug_boolean


def get_input_parameters() -> Namespace:
    """ Gets the input parameters using an `argparse.ArgumentParser`.
        If no input is given for certain parameters, a default value is stored.

        Returns:
            `argparse.Namespace`: An object containing all of the input
            parameters values

    """
    parser = ArgumentParser()

    parser.add_argument(
        '--FILE_URLS', dest='input_file',
        help='Name of the input file to mask fill'
    )
    parser.add_argument(
        '--BOUNDINGSHAPE', dest='shape_file',
        help='Shapefile or native GeoJSON with which to perform the mask fill'
    )
    parser.add_argument(
        '--OUTPUT_DIR', dest='output_dir', default=os.getcwd(),
        help='Name of the output directory to put the output file',
    )
    parser.add_argument(
        '--IDENTIFIER', dest='identifier', default='',
        help='Identifier of the request used to determine output directory'
    )
    parser.add_argument(
        '--MASK_GRID_CACHE', dest='mask_grid_cache',
        help='ignore_and_delete | ignore_and_save | use_cache | use_cache_delete | MaskGrid_Only',
        default=DEFAULT_MASK_GRID_CACHE
    )
    parser.add_argument(
        '--DEFAULT_FILL', dest='fill_value', help='Fill value for mask fill',
        default=DEFAULT_FILL_VALUE
    )
    parser.add_argument(
        '--DEBUG', default=False, dest='debug', type=debug_bool,
        help='If True, changes the log level to DEBUG from the default INFO'
    )

    return parser.parse_args()


def check_shapefile_geojson(shape_file: str, output_dir: str) -> str:
    """ Checks if the input is a native GeoJson, and if so, creates a temporary
        file in the cache directory and returns it. Otherwise it just returns
        the shape_file value that was passed in, which is a path.

    """
    # Ensure that a shape file is given (either as a file path or raw GeoJSON).
    if shape_file is None:
        raise MissingParameterValue('A shapefile is required for the mask '
                                    'fill utility')

    if re.search(r'{.+}', shape_file):
        # We have a native geojson string passed in
        unique_filename = f'{output_dir}/shape_{str(uuid.uuid4())}.geojson'
        with open(unique_filename, 'w') as new_shape_file:
            new_shape_file.write(shape_file)

        shape_file_name = unique_filename

    else:
        # Otherwise return input shape_file path. It's existence is checked in
        # validate_input_parameters
        shape_file_name = shape_file

    return shape_file_name


def validate_input_parameters(input_file: str, shape_file: str,
                              output_dir: str, fill_value: Union[float, int],
                              logger: logging.Logger) -> str:
    """ Ensures that all required input parameters exist, and that all given
        parameters are valid. If not, raises an `InvalidParameterValue`
        custom exception. This takes place separately to the
        `argparse.ArgumentParser`, as parsing errors caught there default to
        a `SystemExit`.

        If parameters are all valid, return the path for the shape file.

        The shape file path is returned, as the input argument may have been
        raw GeoJSON. During the validation this is checked and, if raw GeoJSON
        was supplied, a temporary shape file is created. It is the path to this
        file that should be returned.

    """
    # Ensure that an input file is given
    if input_file is None:
        raise MissingParameterValue('An input data file is required for the '
                                    'mask fill utility')

    # Ensure the input file is a valid file type
    if not os.path.splitext(input_file)[1].lower() in ('.tif', '.h5'):
        raise InvalidParameterValue('The input data file must be a GeoTIFF or '
                                    'HDF5 file type')

    # Ensure that all given paths exist
    for file_path in {input_file, shape_file, output_dir}:
        if not os.path.exists(file_path):
            raise MissingParameterValue(f'The path {file_path} does not exist')

    # Ensure that fill_value is numerical
    if not isinstance(fill_value, (float, int)):
        raise InvalidParameterValue('The default fill value must be a number')

    logger.debug('All input parameters are valid')


def format_parameters(parameters: Namespace) -> Dict:
    """ Removes any single quotes around the given parameters. But retains any
        internal to a string parameter.

        Args:
            params (argparse.Namespace): The input parameters

        Returns:
            Dictionary of single-quote stripped parameters
    """
    def get_parameter_value(parameter_value):
        if parameter_value is not None and isinstance(parameter_value, str):
            output_parameter_value = parameter_value.strip('\'')
        else:
            output_parameter_value = parameter_value

        return output_parameter_value

    return {parameter_name: get_parameter_value(parameter_value)
            for parameter_name, parameter_value
            in vars(parameters).items()}


def get_xml_error_response(output_dir: str, raised_exception: Exception) -> str:
    """ Returns an XML error response corresponding to the input exit status,
        error message, and code.

        If no code is given, the default code will be InternalError; if no
        error message is given, the default error message will be
        "An internal error occurred."

        Args:
            output_dir: Where the products and log of MaskFill should be saved.
            raised_exception: An Exception describing the reason for the error;
                if not a MaskFill custom exception, defaults to InternalError.

                * Exit status 1: InvalidParameterValue
                * Exit status 2: MissingParameterValue
                * Exit status 3: NoMatchingData
                * Exit status 4: MissingCoordinateDataset
                * Exit status 5: InsufficientProjectionInformation
                * Exit status 6: InvalidMetadata
                * Exit status 7: UnknownCollectionShortname

        Returns:
            str: An ESI standard XML error response
    """
    if not isinstance(raised_exception, OUTPUT_EXCEPTIONS):
        raised_exception = InternalError(repr(raised_exception))

    return f"""<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
    <iesi:Exception
        xmlns:iesi="http://eosdis.nasa.gov/esi/rsp/i"
        xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance"
        xmlns:esi="http://eosdis.nasa.gov/esi/rsp"
        xmlns:ssw="http://newsroom.gsfc.nasa.gov/esi/rsp/ssw"
        xmlns:eesi="http://eosdis.nasa.gov/esi/rsp/e"
        xsi:schemaLocation="http://eosdis.nasa.gov/esi/rsp/i
        http://newsroom.gsfc.nasa.gov/esi/8.1/schemas/ESIAgentResponseInternal.xsd">
        <Code>{raised_exception.exception_type}</Code>
        <Message>
                {raised_exception.message}
                MaskFillUtility failed with code {raised_exception.exit_status}
                Log file path: {get_log_file_path(output_dir)}
        </Message>
    </iesi:Exception>"""


def get_xml_success_response(input_file: str, shape_file: str,
                             output_file: str) -> str:
    """ Returns an XML response containing the input file, shape file, and
        output file paths for the process.

        Args:
            input_file (str): The path to the input HDF5 or GeoTIFF file used
                in the process
            shape_file (str): The path to the input shapefile used in the process
            output_file (str): The path to the output file produced by the process

        Returns:
            str: An ESI standard XML string for normal (successful) completion
    """
    return f"""<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
    <ns2:agentResponse xmlns:ns2="http://eosdis.nasa.gov/esi/rsp/i">
        <downloadUrls>
            {output_file}
        </downloadUrls>
        <processInfo>
            <message>
                INFILE = {input_file},
                SHAPEFILE = {shape_file},
                OUTFILE = {output_file}
            </message>
        </processInfo>
    </ns2:agentResponse>"""


def maskfill_sdps() -> str:
    """ Performs a mask fill on the given data file using RQS agent call input
        parameters.

        Prints:
            str: An ESI standard XML string for either normal (successful)
            completion, including the download-URL for accessing the output
            file, or an exception response if necessary.
    """
    # Parse parameters from input argument string.
    argument_namespace = get_input_parameters()
    parameters = format_parameters(argument_namespace)

    # Identify the output subdirectory for MaskFill, ensuring it exists.
    output_dir = os.path.join(parameters['output_dir'], parameters['identifier'])
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    logger = get_sdps_logger(output_dir, parameters['debug'])

    try:
        # If the shape file is raw GeoJSON, write it to a file, and use that.
        # This happens before other parameter validation, to ensure a shape
        # file exists
        shape_file = check_shapefile_geojson(parameters['shape_file'],
                                             output_dir)

        validate_input_parameters(parameters['input_file'], shape_file,
                                  output_dir, parameters['fill_value'], logger)

        # Use MaskFill on the input file.
        output_file = mask_fill(parameters['input_file'], shape_file,
                                output_dir, parameters['mask_grid_cache'],
                                parameters['fill_value'], logger)

        response = get_xml_success_response(parameters['input_file'],
                                            shape_file, output_file)
    except Exception as exception:
        logger.exception(exception)
        response = get_xml_error_response(output_dir, exception)

    return response


if __name__ == '__main__':
    sdps_response = maskfill_sdps()
    print(sdps_response)
