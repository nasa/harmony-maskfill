""" Utility functions to retrieve and interpret Grid Projection meta data
    from HDF-5 input files, including usage of CFConfig file
"""
import logging
import re
from typing import Dict, List, Optional, Tuple, Union

import affine
import numpy as np
from h5py import Dataset
from pyproj import CRS, Proj

from pymods import CFConfig
from pymods.exceptions import (InsufficientDataError,
                               InsufficientProjectionInformation,
                               MissingCoordinateDataset)


def get_hdf_proj4(h5_dataset: Dataset, shortname: str) -> str:
    # TODO: have this function call get_shortname internally
    """ Returns the proj4 string corresponding to the coordinate reference
    system of the HDF5 dataset. Currently logic:

    * Check for DIMENSIONS_LIST attribute on dataset. If present, check the
      units of the first dimension dataset for "degrees". If present, return
      geographic.
    * Next check for the grid_mapping attribute. If present return this variable.
    * Finally, check the global MaskFill configuration file for default
      projection information.

        Args:
             h5_dataset (h5py._hl.dataset.Dataset): The HDF5 dataset
             shortname: The collection shortname for the file
        Returns:
            str: The proj4 string corresponding to the given dataset
    """
    dimensions = get_dimension_datasets(h5_dataset)

    if dimensions is not None:
        units = dimensions[0].attrs.get('units', b'').decode()

        if 'degrees' in units:
            logging.debug(f'Dataset {h5_dataset.name} has dimensions with '
                          'units "degrees". Using Geographic coordinates.')
            return '+proj=longlat +ellps=WGS84 +datum=WGS84 +no_defs'

    grid_mapping_name = h5_dataset.attrs.get('grid_mapping', None)

    if grid_mapping_name is not None:
        logging.debug(f'Dataset {h5_dataset.name} has grid_mapping attribute,'
                      'using this to derive projection information.')
        return get_proj4(h5_dataset.file[grid_mapping_name])

    # Projection absent in granule; get information from MaskFill configuration.
    grid_mapping = _get_grid_mapping_data(shortname, h5_dataset.name)

    if grid_mapping is None:
        raise InsufficientProjectionInformation(h5_dataset.name)
    else:
        logging.debug(f'Dataset {h5_dataset.name} has no projection '
                      'information; using default projection information for the '
                      f'{shortname} collection.')
        return get_proj4(grid_mapping)


def _get_short_name(h5_dataset: Dataset) -> str:
    """ Get collection shortname for the hdf5 dataset (file)
        redefined here for convenience and clarification  """
    return CFConfig.getShortName(h5_dataset.file)


def _get_grid_mapping_data(short_name: Union[bytes, str],
                           dataset_name: str) -> Optional[Dict[str, str]]:
    """ Get the grid_mapping data (attributes, values) assigned to the CF
        grid_mapping variable

        Redefined here for convenience and clarification
    """
    return CFConfig.get_grid_mapping_data(short_name, dataset_name)


def get_lon_lat_datasets(h5_dataset: Dataset) -> Tuple[Dataset, Dataset]:
    """ Finds the lat/lon datsets corresponding to the given HDF5 dataset.
        Args: h5_dataset (h5py._hl.dataset.Dataset): The HDF5 dataset
        Returns:
             tuple: x coordinate dataset (longitude), y coordinate dataset (latitude);
                   both datasets are of type h5py._hl.dataset.Dataset
    """
    h5_file = h5_dataset.file
    coordinate_list = re.split('[, ]', h5_dataset.attrs['coordinates'].decode())

    for coordinate in coordinate_list:
        try:
            if 'lat' in coordinate:
                latitude = h5_file[coordinate]

            if 'lon' in coordinate:
                longitude = h5_file[coordinate]
        except KeyError:
            raise MissingCoordinateDataset(h5_file.filename, coordinate)

    return longitude, latitude


def get_dimension_datasets(h5_dataset: Dataset) -> Optional[Tuple[Dataset, Dataset]]:
    """ Finds the dimension scales datasets corresponding to the given HDF5 dataset.
        Args:
             h5_dataset (h5py._hl.dataset.Dataset): The HDF5 dataset
        Returns:
            tuple: x coordinate dataset, y coordinate dataset;
                   both datasets are of type h5py._hl.dataset.Dataset
    """
    h5_file = h5_dataset.file
    dim_list = h5_dataset.attrs.get('DIMENSION_LIST', None)

    if dim_list is not None:
        for ref in dim_list:
            dim = h5_file[ref[0]]
            if len(dim[:]) == h5_dataset.shape[0]:
                y = dim

            if len(dim[:]) == h5_dataset.shape[1]:
                x = dim

        return x, y
    else:
        return None


def get_proj4(grid_mapping: Dataset) -> str:
    """ Returns the proj4 string corresponding to a grid mapping dataset.
        Args:
            grid_mapping (h5py._hl.dataset.Dataset):
                A dataset containing CF parameters for a coordinate reference system
        Returns:
            str: The proj4 string corresponding to the grid mapping
    """
    if isinstance(grid_mapping, dict):
        cf_parameters = grid_mapping
    else:
        cf_parameters = dict(grid_mapping.attrs)
        decode_bytes(cf_parameters)

    crs_dict = CRS.from_cf(cf_parameters).to_dict()
    if 'standard_parallel' in cf_parameters:
        crs_dict['lat_ts'] = cf_parameters['standard_parallel']

    return CRS.from_dict(crs_dict).to_proj4()


def decode_bytes(dictionary):
    """ Decodes all byte values in the dictionary.
        Args:
            dictionary (dict): A dictionary whose values may be byte objects
    """
    for key, value in dictionary.items():
        if isinstance(value, bytes):
            dictionary[key] = value.decode()


def get_transform(h5_dataset: Dataset) -> affine.Affine:
    """ Determines the transform from the index coordinates of the HDF5 dataset
        to projected coordinates (meters) in the coordinate reference frame of the HDF5 dataset.
        See https://pypi.org/project/affine/ for more information.

        Reference corner points are (x_0, y_0) = (x[0][0], y[0][0]) and
        (x_N, y_M) = (x[-1][-1], y[-1][-1]).

        The projected coordinates of the corner pixels, x and y in metres, are
        used directly, as they are the centre of the pixels, as expected by
        the Affine transformation matrix.

        Args:
            h5_dataset (h5py._hl.dataset.Dataset): The given HDF5 dataset
        Returns:
            affine.Affine: A transform mapping from image coordinates to world coordinates
    """
    if 'DIMENSION_LIST' in h5_dataset.attrs:
        # CF compliant case with projected coordinates defined as dimensions
        cell_width, cell_height = get_cell_size_from_dimensions(h5_dataset)
        x_0, _, y_0, _ = get_corner_points_from_dimensions(h5_dataset)
    else:
        # Dimensions not defined, assume Geographic dimensions defined by
        # lat/lon coordinate references
        cell_width, cell_height = get_cell_size_from_lat_lon(h5_dataset)
        x_0, _, y_0, _ = get_corner_points_from_lat_lon(h5_dataset)

        x_0 -= cell_width / 2.0
        y_0 -= cell_height / 2.0

    return affine.Affine(cell_width, 0, x_0, 0, cell_height, y_0)


def get_transform_information(h5_dataset: Dataset) -> str:
    """ Determine the attributes of an HDF-5 dataset that will be used to
        determine the Affine transformation between pixel indices and
        projected coordinates. This function doesn't actually derive the
        tranform itself, in an effort to minimise computationally intensive
        operations.

    """
    dimension_list = h5_dataset.attrs.get('DIMENSION_LIST', None)
    if dimension_list is not None:
        h5_file = h5_dataset.file
        dimension_names = ', '.join([h5_file[reference[0]].name
                                     for reference in dimension_list])
        output_string = f'DIMENSION_LIST: {dimension_names}'
    else:
        output_string = f'coords: {h5_dataset.attrs["coordinates"].decode()}'

    return output_string


def get_cell_size_from_dimensions(h5_dataset: Dataset) -> Tuple[int, int]:
    """ Gets the cell height and width of the gridded HDF5 dataset in the dataset's dimension scales.
        Note: For Affine matrix, the cell height will be negative when the
              projected y metres increase downwards.
        Args:
             h5_dataset (h5py._hl.dataset.Dataset): The HDF5 dataset
        Returns:
            tuple: cell width, cell height
    """
    x, y = get_dimension_arrays(h5_dataset)
    cell_width, cell_height = x[1] - x[0], y[1] - y[0]
    return cell_width, cell_height


def get_cell_size_from_lat_lon(h5_dataset: Dataset) -> Tuple[float, float]:
    """ Gets the cell height and width of the gridded HDF5 dataset from the dataset's
            latitude and longitude coordinate datasets.

        Note: the cell height can be negative when projected y metres of data
        increases downwards.

        Args:
             h5_dataset (h5py._hl.dataset.Dataset): The HDF5 dataset

        Returns:
            tuple: cell width, cell height
    """
    x, y = get_lon_lat_arrays(h5_dataset)
    x_0, x_N, y_0, y_M = get_corner_points_from_lat_lon(h5_dataset)
    cell_height = (y_M - y_0) / (len(y) - 1)
    cell_width = (x_N - x_0) / (len(x) - 1)
    return cell_width, cell_height


def get_corner_points_from_dimensions(h5_dataset: Dataset) \
        -> Tuple[float, float, float, float]:  # projected meters, ul_x, x_max, y_min, ul_y
    """ Finds the min and max locations in both coordinate axes of the dataset.
        Args:
             h5_dataset (h5py._hl.dataset.Dataset): The HDF5 dataset
        Returns:
            tuple: x_0, x_N, y 0, y M
    """
    x, y = get_dimension_arrays(h5_dataset)
    cell_width, cell_height = get_cell_size_from_dimensions(h5_dataset)
    x_0, x_N = x[0] - cell_width / 2.0, x[-1] + cell_width / 2.0
    y_0, y_M = y[0] - cell_height / 2.0, y[-1] + cell_height / 2.0
    return x_0, x_N, y_0, y_M


def get_corner_points_from_lat_lon(h5_dataset: Dataset) \
        -> Tuple[float, float, float, float]:  # degrees, West, East, North, South
    """ Finds the min and max locations in both coordinate axes of the dataset.
        Args:
             h5_dataset (h5py._hl.dataset.Dataset): The HDF5 dataset
        Returns:
            tuple: x min, x max, y min, y max
    """
    shortname = _get_short_name(h5_dataset)
    proj4_str = _get_grid_mapping_data(shortname, h5_dataset.name)
    p = Proj(get_proj4(proj4_str))

    lon, lat = get_lon_lat_datasets(h5_dataset)
    lon_fill_value, lat_fill_value = get_lon_lat_fill_values(h5_dataset)

    if len(lon.shape) == 3:
        # If there are 3 dimensions, select the first for corner point location.
        # Using lon, assuming lat is the same.
        # TODO: refactor MaskFill so each band in a 3-D dataset is reprojected
        # and masked separately, instead of using coordinate data only from
        # one band.
        logging.debug('lat/lon for {h5_dataset.name} is 3-D, using first band for coordinates.')
        band = 0
    else:
        band = None

    lower_left_tuple = tuple(0 for _ in lat.shape[-2:])
    upper_right_tuple = tuple(extent - 1 for extent in lat.shape[-2:])

    if band is not None:
        lon_corner_values = [lon[band][lower_left_tuple], lon[band][upper_right_tuple]]
        lat_corner_values = [lat[band][lower_left_tuple], lat[band][upper_right_tuple]]
    else:
        lon_corner_values = [lon[lower_left_tuple], lon[upper_right_tuple]]
        lat_corner_values = [lat[lower_left_tuple], lat[upper_right_tuple]]

    if (
        dataset_all_fill_value(lon, None, band)
        or dataset_all_fill_value(lat, None, band)
    ):
        # The longitude or latitude arrays are entirely fill values
        raise InsufficientDataError('{lon.name} or {lat.name} have no valid data.')
    elif lon_fill_value in lon_corner_values or lat_fill_value in lat_corner_values:
        # At least one of the top right or bottom left have a fill value in
        # either (or both) the latitude and longitude arrays.
        # Get indices of the lower left and upper right points with valid coordinates
        lower_left_indices, upper_right_indices = get_valid_coordinates_extent(
            lat, lon, lat_fill_value, lon_fill_value, lower_left_tuple,
            upper_right_tuple, band
        )

        x_ll_data, y_ll_data = p(lon[lower_left_indices], lat[lower_left_indices])
        x_ur_data, y_ur_data = p(lon[upper_right_indices], lat[upper_right_indices])

        # Derive pixel scales from pixels with valid coordiantes
        pixel_scale_x, pixel_scale_y = get_pixel_size_from_data_extent(
            x_ll_data, y_ll_data, x_ur_data, y_ur_data, lower_left_indices,
            upper_right_indices
        )

        if lower_left_tuple == tuple(lower_left_indices[-2:]):
            # The bottom left corner of the array has valid lon and lat data
            x1, y1 = p(lon_corner_values[0], lat_corner_values[0])
        else:
            x1 = extrapolate_coordinate(lon, lon_fill_value, h5_dataset.name,
                                        x_ll_data, lower_left_indices, -1,
                                        lower_left_tuple, pixel_scale_x, band)

            y1 = extrapolate_coordinate(lat, lat_fill_value, h5_dataset.name,
                                        y_ll_data, lower_left_indices, -2,
                                        lower_left_tuple, pixel_scale_y, band)

        if upper_right_tuple == tuple(upper_right_indices[-2:]):
            # The upper right corner of the array has valid lon and lat data
            x2, y2 = p(lon_corner_values[1], lat_corner_values[1])
        else:
            x2 = extrapolate_coordinate(lon, lon_fill_value, h5_dataset.name,
                                        x_ur_data, upper_right_indices, -1,
                                        upper_right_tuple, pixel_scale_x, band)

            y2 = extrapolate_coordinate(lat, lat_fill_value, h5_dataset.name,
                                        y_ur_data, upper_right_indices, -2,
                                        upper_right_tuple, pixel_scale_y, band)
    else:
        # The bottom left and top right both have valid latitudes and longitudes
        x1, y1 = p(lon_corner_values[0], lat_corner_values[0])
        x2, y2 = p(lon_corner_values[1], lat_corner_values[1])

    return x1, x2, y1, y2


def euclidean_distance(point_one: Tuple[int], point_two: Tuple[int]) -> float:
    """Get the Euclidean distance between two points with an arbitrary number
    of dimensions. (Both points should have the same number of dimensions.

    """
    if len(point_one) == len(point_two):
        return np.sqrt(sum([(coordinate_one - point_two[ind])**2.0
                            for ind, coordinate_one
                            in enumerate(point_one)]))
    else:
        return None


def get_dimension_arrays(h5_dataset: Dataset) \
        -> Tuple[List[float], List[float]]:  # projected meters - x-coordinates, y-coordinates
    """ Gets the dimension scales arrays of the HDF5 dataset.
        Args:
             h5_dataset (h5py._hl.dataset.Dataset): The HDF5 dataset
        Returns:
            tuple: The x coordinate array and the y coordinate array
    """
    x, y = get_dimension_datasets(h5_dataset)
    return x[:], y[:]


def get_lon_lat_arrays(h5_dataset: Dataset) \
        -> Tuple[List[float], List[float]]:  # degrees - lat, lon
    """ Gets the lat/lon arrays of the HDF5 dataset.
        Args:
             h5_dataset (h5py._hl.dataset.Dataset): The HDF5 dataset
        Returns:
            tuple: The x coordinate array (longitude) and the y coordinate
               array (latitude), if the data are 3-dimensional, return the
               first band.
    """
    x, y = get_lon_lat_datasets(h5_dataset)
    if len(x.shape) == 2:
        return x[0], y[:, 0]
    elif len(x.shape) == 3:
        return x[0][0], y[0][:, 0]


def get_lon_lat_fill_values(h5_dataset: Dataset) -> Tuple[float, float]:
    """Gets the fill values for the latitude and longitude datasets.
        Args:
            h5_dataset: The H5 dataset.
        Returns:
            tuple: The fill value attributes of longitude and latitude arrays

    """
    lon_dataset, lat_dataset = get_lon_lat_datasets(h5_dataset)

    return get_fill_value(lon_dataset, None), get_fill_value(lat_dataset, None)


def _get_config_fill_value(h5_dataset: Dataset):
    """Checks the global configuration object to see if an overriding fill
    value is present for the dataset in question. If so, that value is returned,
    otherwise, a None is returned.

    """
    short_name = _get_short_name(h5_dataset)
    return CFConfig.get_dataset_config_fill_value(short_name, h5_dataset.name)


def get_fill_value(h5_dataset: Dataset, default_fill_value: float) -> float:
    """ Returns the fill value for the given HDF5 dataset.
        If the HDF5 dataset has no fill value, returns the given default fill value.

        Note: It is not possible to  access the fill value for some longitude and
        latitude datasets via h5_dataset.attrs['_FillValue']. However, in these
        instances, the fill value can be accessed via the Dataset.fillvalue
        class attribute. Accessing via Dataset.attrs is preferable, where
        possible, as it handles some datatypes, such as UBYTE, better than the
        fillvalue class attribute.

        Args:
            h5_dataset (h5py._hl.dataset.Dataset): The given HDF5 dataset
            default_fill_value (float): The default value which is returned if
                no fill value is found in the dataset
        Returns:
            float: The fill value
    """
    config_fill_value = _get_config_fill_value(h5_dataset)

    if config_fill_value is not None:
        logging.debug(f'The dataset {h5_dataset.name} has a known incorrect fill '
                      f'value. Using {config_fill_value} instead.')
        return config_fill_value

    fill_value_attribute = h5_dataset.attrs.get('_FillValue')

    if fill_value_attribute is not None:
        return fill_value_attribute
    elif h5_dataset.fillvalue != b'':
        return h5_dataset.fillvalue
    else:
        logging.info(f'The dataset {h5_dataset.name} does not have a fill value, '
                     f'so the default fill value {default_fill_value} will be used')
        return default_fill_value


def get_valid_coordinates_extent(latitude: np.array, longitude: np.array,
                                 lat_fill_value: float, lon_fill_value: float,
                                 lower_left_tuple: Tuple[int],
                                 upper_right_tuple: Tuple[int],
                                 band: Optional[int] = None) -> Tuple[Tuple[int]]:
    """ Find the indices of the bottom-left-most and top-right-most points that
        do not have a fill value in either the latitude or longitude arrays.

        Args:
            latitude: h5py.Dataset (2 or 3 dimensions).
            longitude: h5py.Dataset (2 or 3 dimensions). The value for a given
                set of indices (e.g. longitude[0][0]) pairs with the
                latitude value at the same array location (e.g. latitude[0][0]).
            lat_fill_value: Either the value taken directly from the latitude
                Dataset object, or the global default fill value.
            lon_fill_value: Either the value taken directly from the latitude
                Dataset object, or the global default fill value.
            lower_left_tuple: The indices of the bottom left corner of the
                latitide and longitude arrays: (0, 0) or (0, 0, 0)
            upper_right_tuple: The indices of the top right corner of the
                latitude and longitude arrays, two or three dimensioned.
        Returns:
            lower_left_data: The indices of the element closest to the
                bottom-left corner of the array with a non-fill value in both
                the latitude and longitude arrays.
            upper_right_data: The indices of the element closest to the
                top-right corner of the array with a non-fill value in both
                the latitude and longitude arrays.
    """
    if band is not None:
        # Three dimensional case, only considering the specified band.
        lat_array = latitude[band][:]
        lon_array = longitude[band][:]
    else:
        # Two dimensional case
        lat_array = latitude[:]
        lon_array = longitude[:]

    # Create lists of indices matching criteria, where i and j are
    # array indices: ([i_0, i_1, i_2], [j_0, j_1, j_2])
    data_indices_lat = np.where(lat_array != lat_fill_value)
    data_indices_lon = np.where(lon_array != lon_fill_value)

    # Convert into tuples of these indices: {[i_0, j_0], [i_1, j_1], [i_2, j_2]}
    # More infomation on zip: https://docs.python.org/3.6/library/functions.html#zip
    zipped_lat_indices = {item for item in zip(*data_indices_lat)}
    zipped_lon_indices = {item for item in zip(*data_indices_lon)}
    valid_lat_and_lon = list(zipped_lat_indices.intersection(zipped_lon_indices))

    # Calculate the euclidean distance between the indices (not coordinates)
    # of the lower left corner and each point with valid latitude and longitude.
    diff_from_lower_left = [euclidean_distance(lower_left_tuple, coordinates)
                            for coordinates
                            in valid_lat_and_lon]

    lower_left_data = valid_lat_and_lon[diff_from_lower_left.index(min(diff_from_lower_left))]

    # Choose the indices which maximize the area between the
    # lower left corner and upper right corner
    area_from_lower_left = [(x - lower_left_data[0]) * (y - lower_left_data[1])
                            for x, y in valid_lat_and_lon]

    upper_right_data = valid_lat_and_lon[area_from_lower_left.index(max(area_from_lower_left))]

    if band is not None:
        # Prepend band on to output indices.
        return (band, ) + lower_left_data, (band, ) + upper_right_data
    else:
        return lower_left_data, upper_right_data


def get_pixel_size_from_data_extent(x_lower_left: np.float32, y_lower_left: np.float32,
                                    x_upper_right: np.float32, y_upper_right: np.float32,
                                    lower_left_indices: Tuple[int],
                                    upper_right_indices: Tuple[int]) -> Tuple[np.float32]:
    """ Take the bottom left most and upper right most points that have data
        both the longitude and latitude arrays, and derive the pixel scale in
        each direction. If all valid data is contained in a single row or column,
        it is assumed that the pixels are square, and that the pixel scale in
        the perpendicular direction can be applied.

        Args:
            x_lower_left: Projected x coordinate of the pixel nearest the bottom
                left of the data array with a valid latitude and longitude.
            y_lower_left: Projected y coordinate of the pixel nearest the bottom
                left of the data array with a valid latitude and longitude.
            x_upper_right: Projected x coordinate of the pixel nearest the top
                right of the data array with a valid latitude and longitude.
            y_upper_right: Projected y coordinate of the pixel nearest the top
                right of the data array with a valid latitude and longitude.
            lower_left_indices: A tuple of array indices for the pixel nearest
                the bottom left of the data array with valid latitude and
                longitude, corresponding to x_lower_left and y_lower_left,
                e.g. (0, 0) or (0, 0, 0).
            upper_right_indices: A tuple of array indices for the pixel nearest
                the top right of the data array with valid latitude and
                longitude, corresponding to x_upper_right and y_upper_right,
                e.g. (1023, 1023) or (1023, 1023, 1)
        Returns:
            pixel_scale_x:
            pixel_scale_y:

    """
    if upper_right_indices[-1] != lower_left_indices[-1]:
        pixel_scale_x = ((x_upper_right - x_lower_left)
                         / (upper_right_indices[-1] - lower_left_indices[-1]))
    else:
        pixel_scale_x = 0.0

    if upper_right_indices[-2] != lower_left_indices[-2]:
        pixel_scale_y = ((y_upper_right - y_lower_left)
                         / (upper_right_indices[-2] - lower_left_indices[-2]))
    else:
        pixel_scale_y = 0.0

    if pixel_scale_x == 0.0 and pixel_scale_y == 0.0:
        raise InsufficientDataError('Only a single, unmasked data point in '
                                    'coordinates. Unable to calculate corner points.')
    elif pixel_scale_x == 0.0:
        raise InsufficientDataError('Only a single, unmasked column of data. '
                                    'Unable to calculate x pixel size.')
    elif pixel_scale_y == 0.0:
        raise InsufficientDataError('Only a single, unmasked row of data. '
                                    'Unable to calculate y pixel size.')

    return pixel_scale_x, pixel_scale_y


def extrapolate_coordinate(coordinate_dataset: Dataset, coordinate_fill_value: float,
                           dataset_name: str, reference_value: float,
                           reference_indices: Tuple[int], coordinate_dim: int,
                           target_indices: Tuple[int], pixel_scale: float,
                           band: Optional[int] = None) -> np.float64:
    """ Extrapolate data in one dimension from a point with a known value to
        another point in an array.

        Args:
            coordinate_dataset: An h5py Dataset for either longtiude or latitude.
            coordinate_fill_value: The fill value for the coordinate dataset.
            dataset_name: The name of the h5py Dataset the refers to the coordinates
                being extrapolated.
            reference_value: Value of the coordinate at a reference point.
            reference_indices: Indices within the coordinate dataset of the
                reference point.
            coordinate_dim: Which of the dimensions in the coordinate array
                that corresponds to a change in that coordinate:
                latitude: -2
                longitude: -1
            target_indices: The indices of the point being extrapolated to.
            pixel_scale: The change in the coordinate dimension per pixel.
        Returns:
            extrapolated_value: The value of the coordinate, latitude or
                longitude, at the target point.

    """
    if band is not None:
        band_dataset = coordinate_dataset[band]
    else:
        band_dataset = coordinate_dataset

    if band_dataset[target_indices] == coordinate_fill_value:
        logging.info(f'{dataset_name}: Detected fill value in '
                     f'{coordinate_dataset.name} at {target_indices}.')
        return reference_value + ((target_indices[coordinate_dim]
                                   - reference_indices[coordinate_dim]) * pixel_scale)
    else:
        return coordinate_dataset[target_indices]


def dataset_all_fill_value(dataset: Dataset, default_fill_value: float,
                           band: Optional[int] = None) -> bool:
    """ Check if an HDF5 dataset only contains a fill value.

        Args:
            dataset: An HDF-5 object containing an np.array and a fill value
                attribute.
            default_fill_value: The fill value to check for if there is no
                assigned fill value on the dataset.
            band: If the data are three dimensional, the band to use.
        Returns:
            is_filled: boolean
    """
    fill_value = get_fill_value(dataset, default_fill_value)

    if band is not None:
        return np.all(dataset[band][:] == fill_value)
    else:
        return np.all(dataset[:] == fill_value)


def dataset_all_outside_valid_range(dataset: Dataset) -> bool:
    """ Check if an HDF-5 dataset contains only values outside of the range
        specified by the valid_min and valid_max attributes, if they are
        present.

        Args:
            dataset: An HDF-5 object containing an np.array and optional
                valid_min and valid_max attributes.
        Returns:
            all_out_of_range: boolean
    """
    valid_min = dataset.attrs.get('valid_min', None)
    valid_max = dataset.attrs.get('valid_max', None)
    dataset_array = dataset[()]

    if valid_min is not None and valid_max is not None:
        return not np.any((dataset_array <= valid_max) & (dataset_array >= valid_min))
    elif valid_min is not None:
        return np.all([dataset_array < valid_min])
    elif valid_max is not None:
        return np.all([dataset_array > valid_max])
    else:
        return False
