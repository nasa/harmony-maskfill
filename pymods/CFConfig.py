""" Utilities for reading and interpreting the CF configuration file
    Allows processing of hdf-5 files that do not fully follow the CF conventions
    Where the configuration file provides the missing information.
"""
from typing import List, Optional, Union
import json
import os
import re

from numpy import bytes_
import h5py


def readConfigFile():
    """ Read the config json file
        Args:
            configFile(string): config file path
    """
    global config

    maskfill_directory = os.path.abspath(os.sep.join([
        os.path.dirname(os.path.abspath(__file__)),
        os.pardir
    ]))
    config_file_path = os.sep.join([maskfill_directory, 'data',
                                    'MaskFillConfig.json'])

    with open(config_file_path) as file_handler:
        configString = file_handler.read()

    configStringWoComments = removeComments(configString)
    config = json.loads(configStringWoComments)


def removeComments(text):
    """ Remove c-style comments.
        Args:
            txt(string): blob of text with comments (can include newlines)
        Return:
            text with comments removed
    """
    pattern = r"""
                        ##  --------- COMMENT ---------
       /\*              ##  Start of /* ... */ comment
       [^*]*\*+         ##  Non-* followed by 1-or-more *'s
       (                ##
         [^/*][^*]*\*+  ##
       )*               ##  0-or-more things which don't start with /
                        ##    but do end with '*'
       /                ##  End of /* ... */ comment
     |                  ##  -OR-  various things which aren't comments:
       (                ##
                        ##  ------ " ... " STRING ------
         "              ##  Start of " ... " string
         (              ##
           \\.          ##  Escaped char
         |              ##  -OR-
           [^"\\]       ##  Non "\ characters
         )*             ##
         "              ##  End of " ... " string
       |                ##  -OR-
                        ##
                        ##  ------ ' ... ' STRING ------
         '              ##  Start of ' ... ' string
         (              ##
           \\.          ##  Escaped char
         |              ##  -OR-
           [^'\\]       ##  Non '\ characters
         )*             ##
         '              ##  End of ' ... ' string
       |                ##  -OR-
                        ##
                        ##  ------ ANYTHING ELSE -------
         .              ##  Anything other char
         [^/"'\\]*      ##  Chars which doesn't start a comment, string
       )                ##    or escape
    """
    regex = re.compile(pattern, re.VERBOSE | re.MULTILINE | re.DOTALL)
    noncomments = [m.group(2) for m in regex.finditer(text) if m.group(2)]
    return "".join(noncomments)


def getShortName(input_file_object: Union[h5py.File, str]) -> str:
    """ Get product short name using config json file
        Args:
            input_file(string): input file path
        Returns:
            shortname(string): product short name
    """
    shortnamePaths = config["ShortNamePath"]
    if isinstance(input_file_object, str):
        input_file = h5py.File(input_file_object, 'r')
    else:
        input_file = input_file_object

    for path in shortnamePaths:
        if path.endswith("/"):
            path = path[:-1]

        shortnamePath = path.rpartition("/")[0]
        label = path.rpartition("/")[2]
        if shortnamePath in input_file:
            if label in input_file[shortnamePath].attrs:
                shortName = input_file[shortnamePath].attrs[label]
                break

    if isinstance(shortName, (bytes, bytes_)):
        shortName = shortName.decode()

    return shortName


def get_grid_epsg_code(short_name: Union[bytes, str],
                       dataset_name: str) -> Optional[str]:
    """ Get grid mapping data, if present, for CF-Compliance
        Args:
            short_name: collection short name (e.g. SPL3FTP).
            dataset_name: string identifier for specific dataset.
        Return:
            grid mapping information, or None if absent.
    """
    if not isinstance(short_name, str):
        short_name = short_name.decode()

    for collection_pattern, collection in config['Grid_Mapping_Group'].items():
        if re.match(collection_pattern, short_name):
            for dataset_pattern, epsg_code in collection.items():
                if re.match(dataset_pattern, dataset_name):
                    return epsg_code

    return None


def get_dataset_config_fill_value(short_name: str, dataset_name: str):
    """ Check MaskFill global configuration object for predefined FillValues.
        These are known data issues, where the FillValue attribute in a dataset
        does not correspond to the used value.

        Args:
            short_name: Product short name. e.g. "SPL3FTP"
            dataset_name
        Return:
            value, or None if there is no corresponding value in the configuration
            file.
    """
    if not isinstance(short_name, str):
        short_name = short_name.decode()

    config_fill_values = config['Corrected_Fill_Value']

    for key, value in config_fill_values.items():
        if re.match(key, short_name):
            if dataset_name in config_fill_values[key]:
                return config_fill_values[key][dataset_name]
            else:
                return None

    return None


def get_dataset_exclusions() -> List[str]:
    """ Pull MaskFill dataset exclusion values from configuration data """
    dataset_exclusions = config['maskfill_dataset_exclusions']
    return dataset_exclusions


def get_collection_coordinate_variables(geotiff_path: str) -> List[str]:
    """ Given the file name, find the longest matching key in the
        `coordinate_exclusions_full_paths` group of the configuration file.
        Return the resulting list, or an empty list if there are no matches.

    """
    exclusions = config['collection_coordinate_variables']
    matching_collection_key = ''

    for collection in exclusions:
        if (
                (collection in geotiff_path)
                and (len(collection) > len(matching_collection_key))
        ):
            matching_collection_key = collection

    return exclusions.get(matching_collection_key, [])
