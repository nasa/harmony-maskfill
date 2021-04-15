""" Utility functions to retrieve and interpret Grid Projection meta data
    from HDF-5 input files, including usage of CF-Conventions configuration
    file.
"""
from logging import Logger
from typing import Dict, List, Optional, Tuple, Union
import re

import affine
import numpy as np
from h5py import Dataset, Reference
from pyproj import CRS, Proj
from pyproj.exceptions import CRSError

from pymods.cf_config import CFConfigH5
from pymods.exceptions import (InsufficientDataError,
                               InsufficientProjectionInformation,
                               InvalidMetadata, MissingCoordinateDataset)


def get_hdf_proj4(h5_dataset: Dataset, cf_config: CFConfigH5,
                  logger: Logger) -> str:
    """ Returns the proj4 string corresponding to the coordinate reference
    system of the HDF5 dataset. Current logic:

    * Check for DIMENSION_LIST attribute on dataset. If present, check the
      units of the first dimension dataset for "degrees". If present, return
      geographic.
    * Next check for the grid_mapping attribute. If present return this variable.
    * Finally, check the global MaskFill configuration file for default
      projection information.

        Args:
             h5_dataset (h5py.Dataset): The HDF5 dataset.
             cf_config: A configuration object with collection defaults.
        Returns:
            str: The proj4 string corresponding to the given dataset
    """
    dimensions = get_dimension_datasets(h5_dataset)

    if dimensions is not None:
        units = dimensions[0].attrs.get('units', b'').decode()

        if 'degrees' in units:
            logger.debug(f'Dataset {h5_dataset.name} has dimensions with '
                         'units "degrees". Using Geographic coordinates.')
            return '+proj=longlat +ellps=WGS84 +datum=WGS84 +no_defs'

    grid_mapping_name = get_grid_mapping_name(h5_dataset)

    if grid_mapping_name is not None:
        logger.debug(f'Dataset {h5_dataset.name} has grid_mapping data, '
                     'using this to derive projection information.')
        return get_crs(h5_dataset.file[grid_mapping_name]).to_proj4()

    # Projection absent in granule; get information from MaskFill configuration.
    grid_mapping = cf_config.get_dataset_grid_mapping_attributes(h5_dataset.name)

    if grid_mapping is None:
        raise InsufficientProjectionInformation(h5_dataset.name)
    else:
        logger.debug(f'Dataset {h5_dataset.name} has no projection '
                     'information; using default projection information for '
                     f'the {cf_config.shortname} collection.')
        return get_crs(grid_mapping).to_proj4()


def get_grid_mapping_name(h5_dataset: Dataset) -> Optional[str]:
    """ Check the associated metadata for a science variable, and extract the
        `grid_mapping` attribute. Account for any use of the extended grid
        mapping name (see CF-Conventions, 1.8, section 5.6) and resolve any
        relative variable paths.

    """
    grid_mapping_attribute = h5_dataset.attrs.get('grid_mapping', None)

    if isinstance(grid_mapping_attribute, Reference):
        grid_mapping_name = h5_dataset.file[grid_mapping_attribute].name
    elif grid_mapping_attribute is not None:
        if isinstance(grid_mapping_attribute, (bytes, np.bytes_)):
            grid_mapping_attribute = grid_mapping_attribute.decode()

        # Splitting based on a colon, will eliminate any issues from the
        # grid mapping being of the extended format.
        grid_mapping_name = resolve_relative_dataset_path(
            h5_dataset, grid_mapping_attribute.split(':')[0]
        )
    else:
        grid_mapping_name = None

    return grid_mapping_name


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
            qualified_coordinate = resolve_relative_dataset_path(h5_dataset,
                                                                 coordinate)
            if 'lat' in coordinate:
                latitude = h5_file[qualified_coordinate]

            if 'lon' in coordinate:
                longitude = h5_file[qualified_coordinate]
        except InvalidMetadata:
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


def get_crs(grid_mapping: Union[Dataset, Dict]) -> CRS:
    """ Returns the proj4 string corresponding to a grid mapping dataset.
        Args:
            grid_mapping (h5py.Dataset):
                A dataset containing CF parameters for a coordinate reference
                system. This can also be an entry from the MaskFill
                configuration file.
        Returns:
            str: The proj4 string corresponding to the grid mapping
    """
    if isinstance(grid_mapping, dict):
        # Given a grid mapping dictionary from the MaskFill configuration file.
        cf_parameters = grid_mapping
    else:
        cf_parameters = get_dataset_attributes(grid_mapping)

    try:
        # pyproj==2.3.1 doesn't handle standard_parallel properly.
        # pyproj~=3.0 does. Upgrading should simplify the following lines:
        # crs = CRS.from_cf(cf_parameters)
        crs_dict = CRS.from_cf(cf_parameters).to_dict()
        if 'standard_parallel' in cf_parameters:
            crs_dict['lat_ts'] = cf_parameters['standard_parallel']

        crs = CRS.from_dict(crs_dict)
    except CRSError:
        if 'srid' in cf_parameters:
            crs = CRS(cf_parameters['srid'])
        else:
            raise InsufficientProjectionInformation(grid_mapping.name)

    return crs


def get_dataset_attributes(h5_dataset: Dataset) -> Dict:
    """ Retrieve all attributres for an HDF-5 dataset. Any values that are
        `bytes` are decoded during the list comprehension.

    """
    return {attribute_key: (attribute_value.decode()
                            if isinstance(attribute_value, bytes)
                            else attribute_value)
            for attribute_key, attribute_value
            in h5_dataset.attrs.items()}


def get_transform(h5_dataset: Dataset, cf_config: CFConfigH5,
                  logger: Logger) -> affine.Affine:
    """ Determines the transform from the index coordinates of the HDF5 dataset
        to projected coordinates (meters) in the coordinate reference frame of the HDF5 dataset.
        See https://pypi.org/project/affine/ for more information.

        Reference corner points are (x_0, y_0) = (x[0][0], y[0][0]) and
        (x_n, y_m) = (x[-1][-1], y[-1][-1]).

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
        x_0, x_n, y_0, y_m = get_corner_points_from_lat_lon(h5_dataset,
                                                            cf_config, logger)
        cell_width, cell_height = get_cell_size_from_lat_lon_extents(h5_dataset,
                                                                     x_0, x_n,
                                                                     y_0, y_m)

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
             h5_dataset (h5py.Dataset): The HDF5 dataset
        Returns:
            tuple: cell width, cell height
    """
    x, y = get_dimension_arrays(h5_dataset)
    cell_width, cell_height = x[1] - x[0], y[1] - y[0]
    return cell_width, cell_height


def get_cell_size_from_lat_lon_extents(h5_dataset: Dataset, x_0: float,
                                       x_n: float, y_0: float,
                                       y_m: float) -> Tuple[float, float]:
    """ Gets the cell height and width of the gridded HDF-5 dataset from the
        dataset's latitude and longitude coordinate datasets and their extents.

        Note: the cell height can be negative when projected y metres of data
        increases downwards.

        Args:
            h5_dataset (h5py.Dataset): The HDF5 dataset
            x_0, x_n, y_0, y_m: Minimum and maximum projected x and y values.
        Returns:
            tuple: (cell width, cell height)
    """
    x, y = get_lon_lat_arrays(h5_dataset)
    cell_height = (y_m - y_0) / (len(y) - 1)
    cell_width = (x_n - x_0) / (len(x) - 1)
    return cell_width, cell_height


def get_corner_points_from_dimensions(h5_dataset: Dataset) \
        -> Tuple[float, float, float, float]:  # projected meters, ul_x, x_max, y_min, ul_y
    """ Finds the min and max locations in both coordinate axes of the dataset.
        Args:
             h5_dataset (h5py.Dataset): The HDF5 dataset
        Returns:
            tuple: x_0, x_n, y 0, y M
    """
    x, y = get_dimension_arrays(h5_dataset)
    cell_width, cell_height = get_cell_size_from_dimensions(h5_dataset)
    x_0, x_n = x[0] - cell_width / 2.0, x[-1] + cell_width / 2.0
    y_0, y_m = y[0] - cell_height / 2.0, y[-1] + cell_height / 2.0
    return x_0, x_n, y_0, y_m


def get_corner_points_from_lat_lon(h5_dataset: Dataset, cf_config: CFConfigH5,
                                   logger: Logger) -> Tuple[float]:
    """ Finds the min and max locations in both coordinate axes of the dataset.
        Args:
             h5_dataset (h5py.Dataset): The HDF5 dataset
        Returns:
            tuple: x min, x max, y min, y max
    """
    grid_mapping_attributes = cf_config.get_dataset_grid_mapping_attributes(h5_dataset.name)
    projection = Proj(get_crs(grid_mapping_attributes))

    lon, lat = get_lon_lat_datasets(h5_dataset)
    lon_fill_value = get_fill_value(lon, cf_config, logger, None)
    lat_fill_value = get_fill_value(lat, cf_config, logger, None)

    if len(lon.shape) == 3:
        # If there are 3 dimensions, select the first for corner point location.
        # Using lon, assuming lat is the same.
        # TODO: refactor MaskFill so each band in a 3-D dataset is reprojected
        # and masked separately, instead of using coordinate data only from
        # one band.
        logger.debug(f'lat/lon for {h5_dataset.name} is 3-D, using first '
                     'band for coordinates.')
        band = 0
    else:
        band = None

    lower_left_tuple = tuple(0 for _ in lat.shape[-2:])
    upper_right_tuple = tuple(extent - 1 for extent in lat.shape[-2:])

    if band is not None:
        lon_values = lon[band]
        lat_values = lat[band]
        lon_corner_values = [lon_values[lower_left_tuple], lon_values[upper_right_tuple]]
        lat_corner_values = [lat_values[lower_left_tuple], lat_values[upper_right_tuple]]
    else:
        lon_values = lon[:]
        lat_values = lat[:]
        lon_corner_values = [lon_values[lower_left_tuple], lon_values[upper_right_tuple]]
        lat_corner_values = [lat_values[lower_left_tuple], lat_values[upper_right_tuple]]

    if (
        dataset_all_fill_value(lon, cf_config, logger, None, band)
        or dataset_all_fill_value(lat, cf_config, logger, None, band)
    ):
        # The longitude or latitude arrays are entirely fill values
        raise InsufficientDataError(f'{lon.name} or {lat.name} have no valid data.')
    elif lon_fill_value in lon_corner_values or lat_fill_value in lat_corner_values:
        # Find all pixels with a valid associated latitude and longitude:
        valid_indices = np.where((lon_values != lon_fill_value)
                                 & (lat_values != lat_fill_value))

        # Get the extent in each direction, projected x and projected y:
        x_0, x_n = get_projected_coordinate_extent(projection, lat_values,
                                                   lon_values, 1, valid_indices)
        y_0, y_m = get_projected_coordinate_extent(projection, lat_values,
                                                   lon_values, 0, valid_indices)
    else:
        # The bottom left and top right both have valid latitudes and longitudes
        x_0, y_0 = projection(lon_corner_values[0], lat_corner_values[0])
        x_n, y_m = projection(lon_corner_values[1], lat_corner_values[1])

    return x_0, x_n, y_0, y_m


def get_projected_coordinate_extent(projection: Proj, latitude: np.ndarray,
                                    longitude: np.ndarray,
                                    array_dimension: int,
                                    valid_indices: Tuple) -> Tuple:
    """ Calculate the maximum and minimum projected coordinate in a single
        dimension of coordinate grid. This method assumes that the data are
        gridded, and that all points in the same row have the same projected
        y value, while all points in the same column have the same projected
        x value. This function will only be called if there are fill values in
        the coordinate arrays in the (0, 0) and (M, N) corner points.

        The first and last rows (or columns) with valid coordinate data are
        found. Then the associated latitudes and longitudes are projected into
        the coordinate references system (CRS) of the grid. The pixel scale in
        that dimension is then calculated, before extrapolating from the
        minimum and maximum valid coordinates to those of the first and last
        rows (or columns) in the array.

    """
    if array_dimension == 0:
        projection_dim = 1
        dimension_string, line_type = ('y', 'row')
    else:
        projection_dim = 0
        dimension_string, line_type = ('x', 'column')

    # Find the first instances of the minimum and maximum in the valid data
    # array. valid_indices = ([i_0, i_1, ..., i_N], [j_0, j_1, ..., j_N])
    # The minimum/maximum values are not retrieved directly as we need both
    # the i and j index of the point in the minimum and maximum row or column
    # with valid coordinate data.
    argmin_valid_index = valid_indices[array_dimension].argmin()
    argmax_valid_index = valid_indices[array_dimension].argmax()

    longitude_min = longitude[valid_indices[0][argmin_valid_index],
                              valid_indices[1][argmin_valid_index]]
    longitude_max = longitude[valid_indices[0][argmax_valid_index],
                              valid_indices[1][argmax_valid_index]]
    latitude_min = latitude[valid_indices[0][argmin_valid_index],
                            valid_indices[1][argmin_valid_index]]
    latitude_max = latitude[valid_indices[0][argmax_valid_index],
                            valid_indices[1][argmax_valid_index]]

    min_point = projection(longitude_min, latitude_min)
    max_point = projection(longitude_max, latitude_max)

    if argmax_valid_index != argmin_valid_index:
        index_of_max_point = valid_indices[array_dimension][argmax_valid_index]
        index_of_min_point = valid_indices[array_dimension][argmin_valid_index]
        pixel_size = (
            max_point[projection_dim] - min_point[projection_dim]
        ) / (index_of_max_point - index_of_min_point)
    else:
        raise InsufficientDataError(f'Only a single, unmasked {line_type} '
                                    'of data. Unable to calculate '
                                    f'{dimension_string} pixel size.')

    min_valid_index = valid_indices[array_dimension][argmin_valid_index]
    max_valid_index = valid_indices[array_dimension][argmax_valid_index]
    lower_corner = min_point[projection_dim] - (min_valid_index * pixel_size)
    upper_corner = max_point[projection_dim] + (
        (latitude.shape[array_dimension] - 1 - max_valid_index) * pixel_size
    )

    return lower_corner, upper_corner


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
             h5_dataset (h5py.Dataset): The HDF5 dataset
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


def get_fill_value(h5_dataset: Dataset, cf_config: CFConfigH5, logger: Logger,
                   default_fill_value: Optional[float]) -> Optional[float]:
    """ Returns the fill value for the given HDF5 dataset.
        If the HDF5 dataset has no fill value, returns the given default fill value.

        Note: It is not possible to  access the fill value for some longitude and
        latitude datasets via h5_dataset.attrs['_FillValue']. However, in these
        instances, the fill value can be accessed via the Dataset.fillvalue
        class attribute. Accessing via Dataset.attrs is preferable, where
        possible, as it handles some datatypes, such as UBYTE, better than the
        fillvalue class attribute.

        Args:
            h5_dataset (h5py.Dataset): The given HDF5 dataset
            default_fill_value (float): The default value which is returned if
                no fill value is found in the dataset
        Returns:
            float: The fill value
    """
    config_fill_value = cf_config.get_dataset_fill_value(h5_dataset.name)

    if config_fill_value is not None:
        logger.debug(f'The dataset {h5_dataset.name} has a known incorrect fill '
                     f'value. Using {config_fill_value} instead.')
        return config_fill_value

    fill_value_attribute = h5_dataset.attrs.get('_FillValue')

    if fill_value_attribute is not None:
        return fill_value_attribute
    elif h5_dataset.fillvalue != b'':
        return h5_dataset.fillvalue
    else:
        logger.info(f'The dataset {h5_dataset.name} does not have a fill value, '
                    f'so the default fill value {default_fill_value} will be used')
        return default_fill_value


def dataset_all_fill_value(dataset: Dataset, cf_config: CFConfigH5,
                           logger: Logger, default_fill_value: float,
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
    fill_value = get_fill_value(dataset, cf_config, logger, default_fill_value)

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


def resolve_relative_dataset_path(h5_dataset: Dataset,
                                  relative_path: str) -> str:
    """ Given a relative path within a granule, resolve an absolute path given
        the location of the variable making the reference. For example, a
        variable might refer to a grid_mapping variable, or a coordinate
        variable in the CF-Convention metadata attributes.

        Finally, the resolved path is checked, to ensure it exists in the
        granule. If not, an exception will be raised.

    """
    referee_location = h5_dataset.parent.name
    referee_pieces = referee_location.split('/')[1:]
    relative_first_piece = relative_path.split(':')[0]

    if relative_path.startswith('/'):
        # If the path starts with a slash, assume it is absolute
        resolved_path = relative_path
    elif not relative_path.startswith('../'):
        # If a path doesn't indicate nesing, first check if there is a variable
        # matching the name in the same group as the referee, otherwise assume
        # the variable reference is from the root group.
        reference_in_group = '/'.join([referee_location, relative_path])

        if reference_in_group in h5_dataset.file:
            resolved_path = reference_in_group
        else:
            resolved_path = f'/{relative_path}'
    else:
        # The path begins with '../', so resolve the path
        try:
            working_path = relative_path
            while working_path.startswith('../'):
                working_path = working_path[3:]
                del referee_pieces[-1]

            resolved_path = '/'.join([''] + referee_pieces + [working_path])
        except IndexError:
            # This exception will be raised if the relative path claims to be
            # more nested than the referee actually is.
            # e.g.: "/group1/variable" has a reference: "../../other_variable".
            raise InvalidMetadata(h5_dataset.name,
                                  'grid_mapping or coordinate',
                                  relative_path,
                                  'Relative path has incorrect nesting')

    if resolved_path not in h5_dataset.file:
        raise InvalidMetadata(h5_dataset.name, 'grid_mapping or coordinate',
                              relative_path, 'Variable reference not in file ')

    return resolved_path
