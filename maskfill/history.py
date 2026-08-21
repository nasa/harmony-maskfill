"""This module contains functions for updating or creating a history attribute
to maintain file provenance.

"""

import hashlib
import json
import os
from datetime import datetime, timezone
from typing import Dict, List, Union

import h5py

# Values needed for history_json attribute
HISTORY_JSON_SCHEMA = (
    'https://harmony.earthdata.nasa.gov/schemas/history/0.1.0/history-v0.1.0.json'
)
PROGRAM = 'Harmony Maskfill'
PROGRAM_REF = 'https://github.com/nasa/harmony-maskfill'


def update_history_metadata(
    input_file: str,
    shape_file: str,
    fill_value: Union[int, float, None],
    bounding_box: List,
) -> None:
    """Update the history-related metadata of an HDF5 or NetCDF4 output file.

    This method opens the target file and updates two forms of history
    metadata:

    • The 'history_json' attribute is updated by appending a new structured
      record describing the current Maskfill operation, including the
      timestamp, program name, version, request URL, and processing parameters.

    • The human‑readable `history` (or `History`) global attribute is updated
      by appending a new line summarizing the Maskfill execution. If no
      history attribute exists, a new one is created.

    History metadata can only be created or modified for HDF5 and NetCDF4
    files; GeoTIFFs do not support these attributes.

    Parameters
    ----------
    input_file : str
        Path to the output HDF5 or NetCDF4 file whose history metadata
        should be updated.
    shape_file : str
        Path to the user‑provided shape file.
    fill_value : int, float, or None
        The fill value applied to masked pixels during Maskfill processing.
    bounding_box : List
        bounding box in the format of [W, S, E, N]

    Returns
    -------
    None
        This function modifies the file in place and does not return a value.

    """
    with h5py.File(input_file, 'a') as h5_input_file:
        history_attribute_name, existing_history = read_history_attrs(h5_input_file)

        request_url = get_request_url_attribute(h5_input_file)

        maskfill_parameters = get_maskfill_parameters(
            shape_file, fill_value, bounding_box
        )

        # Create new history_json attribute and append existing_history
        new_history_json_record = create_history_json_record(
            request_url, maskfill_parameters
        )

        output_history_json = read_history_json_attrs(h5_input_file)

        # Append `history_json_record` to the existing `history_json` array:
        output_history_json.append(new_history_json_record)

        # Update existing `history_json` array:
        h5_input_file.attrs['history_json'] = json.dumps(output_history_json)

        history_parameters = dict(new_history_json_record['parameters'])

        # Create a new history for Harmony Maskfill
        new_history_line = ' '.join(
            [
                new_history_json_record['date_time'],
                new_history_json_record['program'],
                new_history_json_record['version'],
                json.dumps(history_parameters),
            ]
        )

        # Append new Harmony Maskfill history to existing history
        output_history = '\n'.join(filter(None, [existing_history, new_history_line]))

        # Update history attribute with new Harmony Maskfill entry
        h5_input_file.attrs[history_attribute_name] = output_history


def read_history_attrs(h5_input_file: h5py.File) -> tuple[str, str | None]:
    """Retrieve the history-related global attribute from an HDF5
    or NetCDF4 file.

    This function checks for the presence of either a `History` or `history`
    attribute in the file’s global attributes. If found, the attribute value is
    returned as a UTF‑8 string (decoding from bytes when necessary). If neither
    attribute exists, the function returns a default attribute name of
    `"history"` and a value of `None`.

    Parameters
    ----------
    h5_input_file : h5py.File
        An open HDF5 or NetCDF4 file object from which history attributes
        should be read.

    Returns
    -------
    tuple[str, str | None]
        A tuple containing:
        • The attribute name used for history (`"History"` or `"history"`).
        • The existing history string, or `None` if no history attribute
          exists.

    """
    if 'History' in h5_input_file.attrs:
        history_attribute_name = 'History'
        existing_history = h5_input_file.attrs['History']
        # Convert bytes to string if needed
        if isinstance(existing_history, bytes):
            existing_history = existing_history.decode('utf-8')
    elif 'history' in h5_input_file.attrs:
        history_attribute_name = 'history'
        existing_history = h5_input_file.attrs['history']
        # Convert bytes to string if needed
        if isinstance(existing_history, bytes):
            existing_history = existing_history.decode('utf-8')
    else:
        history_attribute_name = 'history'
        existing_history = None

    return history_attribute_name, existing_history


def read_history_json_attrs(h5_input_file: h5py.File) -> List:
    """
    Retrieve and normalize the `history_json` global attribute from an
    HDF5 or NetCDF4 file.

    This function checks whether the file contains a `history_json`
    attribute. If present, the JSON content is parsed and returned as a
    list of history records. The attribute may contain either a single
    JSON object or a list of objects; in both cases, the return value is
    normalized to a list. If the attribute does not exist, an empty list
    is returned.

    Parameters
    ----------
    h5_input_file : h5py.File
        An open HDF5 or NetCDF4 file object from which the
        `history_json` attribute should be read.

    Returns
    -------
    List
        A list of parsed `history_json` records. Returns an empty list
        when the file does not contain a `history_json` attribute.
    """
    output_history_json = []

    if 'history_json' in h5_input_file.attrs:
        old_history_json = json.loads(h5_input_file.attrs['history_json'])
        if isinstance(old_history_json, list):
            output_history_json = old_history_json
        else:
            # Single `history_json_record` element.
            output_history_json = [old_history_json]

    return output_history_json


def get_request_url_attribute(h5_input_file: h5py.File) -> str:
    """Extract the request URL from the file's `history_json` attribute.

    This function reads the `history_json` global attribute—if present—and
    attempts to extract the `request_url` value from its `parameters` field.
    The method supports both dictionary- and list-based parameter structures.
    If a request URL is found, any query string (text after '?') is removed.
    If no valid request URL is available, the function returns the file's
    own filename as a fallback.

    Parameters
    ----------
    h5_input_file : h5py.File
        An open HDF5 or NetCDF4 file object from which the request URL
        should be extracted.

    Returns
    -------
    str
        The extracted request URL without query parameters, or the file's
        filename if no request URL is present.

    """
    if 'history_json' not in h5_input_file.attrs:
        return h5_input_file.filename

    history_json = json.loads(h5_input_file.attrs['history_json'])

    if isinstance(history_json, list):
        history_json = history_json[0]

    parameters = history_json.get('parameters')

    if isinstance(parameters, dict):
        return parameters.get('request_url', h5_input_file.filename)

    if isinstance(parameters, list):
        for item in parameters:
            if isinstance(item, dict) and 'request_url' in item:
                request_url = item['request_url'].split('?', 1)[0]
                return request_url

    return h5_input_file.filename


def create_history_json_record(granule_url: str, maskfill_parameters: dict) -> Dict:
    """Create a serializable dictionary for the `history_json` global
    attribute in the merged output NetCDF-4 file.

    This function assembles a serializable dictionary capturing metadata
    about the current Maskfill operation. The record includes the execution
    timestamp, program name, version, processing parameters, and the source
    granule URL.

    Parameters
    ----------
    input_history : str or list
        The existing history from the file
    granule_url : str
        The URL of the input granule from which the output file was derived.
        Stored in the `derived_from` field.
    maskfill_parameters : dict
        A dictionary of Maskfill processing parameters to include in the
        `parameters` field of the history record.

    Returns
    -------
    Dict
        A fully populated dictionary representing a `history_json` record,
        ready to be serialized and written to the output file.

    """
    history_json_record = {
        '$schema': HISTORY_JSON_SCHEMA,
        'date_time': datetime.utcnow().replace(tzinfo=timezone.utc).isoformat(),
        'program': PROGRAM,
        'version': get_semantic_version(),
        'parameters': maskfill_parameters,
        'derived_from': granule_url,
        'program_ref': PROGRAM_REF,
    }

    return history_json_record


def get_semantic_version() -> str:
    """Retrieve the semantic version string for the application.

    This function reads the `service_version.txt` file located in the
    `docker/` directory relative to the current module and returns its
    contents as a semantic version string. If the file is empty or the
    version cannot be determined, a placeholder value of
    "[version not found]" is returned.

    Parameters
    ----------
    None

    Returns
    -------
    str
        The semantic version number read from `service_version.txt`, or
        "[version not found]" if the file is empty.

    """
    current_directory = os.path.dirname(os.path.abspath('__file__'))
    path = os.path.join(current_directory, 'docker/service_version.txt')
    with open(path, encoding='utf-8') as file_handler:
        semantic_version = file_handler.read().strip()
        if not semantic_version:
            return '[version not found]'
        return semantic_version


def get_maskfill_parameters(
    shape_file: str, fill_value: Union[int, float, None], bounding_box: List
) -> Dict:
    """Build and return the parameter dictionary used for a Maskfill operation.

    This function collects the inputs relevant to Maskfill—such as the hashed
    shape file identifier, mask grid cache, fill value, and the source
    granule URL—and assembles them into a standardized parameter dictionary.

    Parameters
    ----------
    shape_file : str
        Path to the shape file used for spatial masking. If provided, a
        SHA‑224 hash of the file path is stored as `shape_file_hash`.
    fill_value : int or float
        The fill value applied to masked pixels.
    bounding_box : List
        bounding box in the format of [W, S, E, N]

    Returns
    -------
    Dict
        A dictionary containing the Maskfill processing parameters to be
        recorded in the `history_json` metadata.

    """
    maskfill_parameters = {}

    if bounding_box:
        maskfill_parameters['bbox'] = bounding_box
    else:
        mask_id = hashlib.sha224(f'{shape_file}'.encode()).hexdigest()
        maskfill_parameters['shape_file_hash'] = mask_id

    if fill_value:
        maskfill_parameters['fill_value'] = fill_value

    return maskfill_parameters
