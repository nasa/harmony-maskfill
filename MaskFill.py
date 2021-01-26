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
            default INFO.
"""
from argparse import ArgumentParser, Namespace
from typing import Tuple, Union
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


def mask_fill() -> str:
    """ Performs a mask fill on the given data file using RQS agent call input
        parameters.

        Returns:
            str: An ESI standard XML string for either normal (successful)
            completion, including the download-URL for accessing the output
            file, or an exception response if necessary.
    """
    try:
        # Parse, format and validate input parameters
        args = get_input_parameters()
        input_file, shape_file, output_dir, identifier, \
            mask_grid_cache, fill_value, debug = format_parameters(args)

        output_dir = os.path.join(output_dir, identifier)
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        configure_logger(output_dir)

        validate_input_parameters(input_file, shape_file, output_dir,
                                  fill_value, debug)

        # Perform mask fill according to input file type
        input_extension = os.path.splitext(input_file)[1].lower()

        if input_extension == '.tif':
            # GeoTIFF case
            logging.info(f'Performing mask fill with GeoTIFF {input_file} and '
                         f'shapefile {shape_file}')
            output_file = GeotiffMaskFill.produce_masked_geotiff(
                input_file, shape_file, output_dir, output_dir,
                mask_grid_cache, fill_value
            )
        elif input_extension == '.h5':
            # HDF5 case
            logging.info(f'Performing mask fill with HDF5 file {input_file} '
                         f'and shapefile {shape_file}')
            output_file = H5MaskFill.produce_masked_hdf(
                input_file, shape_file, output_dir, output_dir,
                mask_grid_cache, fill_value
            )

        return get_xml_success_response(input_file, shape_file, output_file)
    except Exception as exception:
        return get_xml_error_response(output_dir, exception)


def configure_logger(output_dir: str) -> None:
    """ Configures the logger for the mask fill process, setting the log level
        to INFO.

    """
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    logger.addHandler(logging.FileHandler(get_log_file_path(output_dir)))
    logging.info('Logger configured')


def get_log_file_path(output_dir: str) -> str:
    """ Gets the file path to the log file for the mask fill process.
        The file will be in the output directory with filename 'mask_fill.log'.

        Returns:
            str: The log file path
    """
    return os.path.join(output_dir, 'mask_fill.log')


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
        '--DEBUG', dest='debug',
        help='If True, changes the log level to DEBUG from the default INFO'
    )

    logging.info('Parsed input parameters')
    return parser.parse_args()


def check_shapefile_geojson(shape_file: str, output_dir: str) -> str:
    """ Checks if the input is a native GeoJson, and if so, creates a temporary
        file in the cache directory and returns it. Otherwise it just returns
        the shape_file value that was passed in, which is a path.

    """
    if (re.search(r'{.+}', shape_file)):
        # We have a native geojson string passed in
        unique_filename = f"{output_dir}/shape_{str(uuid.uuid4())}.geojson"
        with open(unique_filename, "w") as new_shape_file:
            new_shape_file.write(shape_file)

        return unique_filename

    else:
        # Otherwise return input shape_file path. It's existence is checked in
        # validate_input_parameters
        return shape_file


def validate_input_parameters(input_file: str, shape_file: str,
                              output_dir: str, fill_value: Union[float, int],
                              debug: str) -> None:
    """ Ensures that all required input parameters exist, and that all given
        parameters are valid. If not, raises an `InvalidParameterValue`
        custom exception. Otherwise, returns `None`.

    """
    # Ensure that an input file and a shape file are given
    required_files = [[input_file, "An input data file"],
                      [shape_file, "A shapefile"]]

    for file_type in required_files:
        if file_type[0] is None:
            raise MissingParameterValue(f'{file_type[1]} is required for the '
                                        'mask fill utility')

    # Ensure the input file and shape file are valid file types
    if not os.path.splitext(input_file)[1].lower() in [".tif", ".h5"]:
        raise InvalidParameterValue('The input data file must be a GeoTIFF or '
                                    'HDF5 file type')

    # Check if the shapefile may be geojson input
    shape_file = check_shapefile_geojson(shape_file, output_dir)

    # Ensure that all given paths exist
    for path in {input_file, shape_file, output_dir}:
        if not os.path.exists(path):
            raise MissingParameterValue(f'The path {path} does not exist')

    # Ensure that fill_value is a float
    if not isinstance(fill_value, (float, int)):
        raise InvalidParameterValue('The default fill value must be a number')

    # Set logging level to DEBUG if the input parameter is true
    if debug is not None and debug.lower() == 'true':
        logging.getLogger().setLevel(logging.DEBUG)

    # If no issues are found, return None
    logging.debug('All input parameters are valid')
    # return None


def format_parameters(params: Namespace) -> Tuple[str]:
    """ Removes any single quotes around the given parameters.

        Args:
            params (argparse.Namespace): The input parameters

        Returns:
            generator: Producing quote stripped parameters
    """
    parameters = (params.input_file, params.shape_file, params.output_dir,
                  params.identifier, params.mask_grid_cache, params.fill_value,
                  params.debug)

    # Remove any single quotes around parameters
    return (param.replace("\'", "")
            if param is not None and isinstance(param, str)
            else param
            for param in parameters)


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

    logging.exception(raised_exception.message)

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
    xml_response = f"""<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
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

    logging.debug('Process completed successfully')
    logging.debug(f'Output file: {output_file}')
    return xml_response


if __name__ == '__main__':
    response = mask_fill()
    print(response)
