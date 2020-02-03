""" Utility functions to retrieve and interpret Grid Projection meta data
    from HDF-5 input files, including usage of CFConfig file
"""
import logging
import re
from typing import Dict, Tuple, List

import affine
from h5py import Dataset
from pyproj import CRS, Proj

from pymods import CFConfig


def get_hdf_proj4(h5_dataset: Dataset, shortname: str) -> str:
    # TODO: have this function call get_shortname internally
    """ Returns the proj4 string corresponding to the coordinate reference system of the HDF5 dataset.
        Args:
             h5_dataset (h5py._hl.dataset.Dataset): The HDF5 dataset
             shortname: The collection shortname for the file
        Returns:
            str: The proj4 string corresponding to the given dataset
    """
    objname = h5_dataset.name

    grid_mapping = _get_grid_mapping_data(_get_grid_mapping_group(shortname, objname))
    if not grid_mapping:
        dimensions = get_dimension_datasets(h5_dataset)
        units = dimensions[0].attrs['units'].decode()

        if 'degrees' in units:
            # Geographic proj4 string
            return "+proj=longlat +ellps=WGS84 +datum=WGS84 +no_defs"

        if h5_dataset.attrs.__contains__('grid_mapping'):
            grid_mapping_name = h5_dataset.attrs['grid_mapping']
            grid_mapping = h5_dataset.file[grid_mapping_name]

    return get_proj4(grid_mapping)

def _get_short_name(h5_dataset: Dataset) -> str:
    """ Get collection shortname for the hdf5 dataset (file)
        redefined here for convenience and clarification  """
    return CFConfig.getShortName(h5_dataset.file)

def _get_grid_mapping_group(shortname: str, objname: str) -> str:
    """ Get the grid_mapping name (projection name), used as CF grid_mapping variable name
        Redefined here for convenience and clarification """
    return CFConfig.getGridMappingGroup(shortname, objname)

def _get_grid_mapping_data(projection: str) -> Dict[str, str]:
    """ Get the grid_mapping data (attributes, values) assigned to the CF grid_mapping variable
        Redefined here for conveneience and clarification """
    return CFConfig.getGridMappingData(projection)

def get_lat_lon_datasets(h5_dataset: Dataset) -> Tuple[Dataset, Dataset]:
    """ Finds the lat/lon datsets corresponding to the given HDF5 dataset.
        Args: h5_dataset (h5py._hl.dataset.Dataset): The HDF5 dataset
        Returns:
             tuple: x coordinate dataset, y coordinate dataset;
                   both datasets are of type h5py._hl.dataset.Dataset
    """
    file = h5_dataset.file
    coordinate_list = re.split('[, ]', h5_dataset.attrs['coordinates'].decode())

    for coordinate in coordinate_list:
        if 'lat' in coordinate:
            latitude = file[coordinate]
        if 'lon' in coordinate:
            longitude = file[coordinate]

    return longitude, latitude

def get_dimension_datasets(h5_dataset: Dataset) -> Tuple[Dataset, Dataset]:
    """ Finds the dimension scales datasets corresponding to the given HDF5 dataset.
        Args:
             h5_dataset (h5py._hl.dataset.Dataset): The HDF5 dataset
        Returns:
            tuple: x coordinate dataset, y coordinate dataset;
                   both datasets are of type h5py._hl.dataset.Dataset
    """
    file = h5_dataset.file
    dim_list = h5_dataset.attrs['DIMENSION_LIST']

    for ref in dim_list:
        dim = file[ref[0]]
        if len(dim[:]) == h5_dataset.shape[0]: y = dim
        if len(dim[:]) == h5_dataset.shape[1]: x = dim

    return x, y

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
        if isinstance(value, bytes): dictionary[key] = value.decode()


def get_transform(h5_dataset:Dataset) -> affine.Affine:
    """ Determines the transform from the index coordinates of the HDF5 dataset
        to projected coordinates (meters) in the coordinate reference frame of the HDF5 dataset.
        See https://pypi.org/project/affine/ for more information.
        Args:
            h5_dataset (h5py._hl.dataset.Dataset): The given HDF5 dataset
        Returns:
            affine.Affine: A transform mapping from image coordinates to world coordinates
    """
    # CF compliant case with projected coordinates defined as dimensions
    if 'DIMENSION_LIST' in h5_dataset.attrs:
        cell_width, cell_height = get_cell_size_from_dimensions(h5_dataset)
        x_min, x_max, y_min, y_max = get_corner_points_from_dimensions(h5_dataset)
    # Dimensions not defined, assume Geographic dimensions defined by lat/lon coordinate references
    else:
        cell_width, cell_height = get_cell_size_from_lat_lon(h5_dataset)
        x1, x2, y1, y2 = get_corner_points_from_lat_lon(h5_dataset)

        x_min, x_max = min(x1, x2) - cell_width / 2, max(x1, x2) + cell_width / 2
        y_min, y_max = min(y1, y2) + cell_height / 2, max(y1, y2) - cell_height / 2

    return affine.Affine(cell_width, 0, x_min, 0, cell_height, y_max)


def get_cell_size_from_dimensions(h5_dataset: Dataset) -> Tuple[int, int]:
    """ Gets the cell height and width of the gridded HDF5 dataset in the dataset's dimension scales.
        Note: For Affine matrix, the cell height is expected to be negative
              because the row indices of image data increase downwards.
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
        Note: the cell height is expected to be negative because the row indices of image data increase downwards.

        Args:
             h5_dataset (h5py._hl.dataset.Dataset): The HDF5 dataset

        Returns:
            tuple: cell width, cell height
    """
    x, y = get_lat_lon_arrays(h5_dataset)
    x_min, x_max, y_min, y_max = get_corner_points_from_lat_lon(h5_dataset)
    cell_height = (y_max - y_min)/(len(y) - 1)
    cell_width = (x_max - x_min)/(len(x) - 1)
    return cell_width, cell_height


def get_corner_points_from_dimensions(h5_dataset: Dataset) \
        -> Tuple[float, float, float, float]:  # projected meters, ul_x, x_max, y_min, ul_y
    """ Finds the min and max locations in both coordinate axes of the dataset.
        Args:
             h5_dataset (h5py._hl.dataset.Dataset): The HDF5 dataset
        Returns:
            tuple: x min, x max, y min, y max
    """
    x, y = get_dimension_arrays(h5_dataset)
    cell_width, cell_height = get_cell_size_from_dimensions(h5_dataset)

    x_min, x_max = x[0] - cell_width / 2, x[-1] + cell_width / 2
    y_min, y_max = y[-1] + cell_height / 2, y[0] - cell_height / 2

    return x_min, x_max, y_min, y_max


def get_corner_points_from_lat_lon(h5_dataset: Dataset) \
        -> Tuple[float, float, float, float]:  # degrees, West, East, North, South
    """ Finds the min and max locations in both coordinate axes of the dataset.
        Args:
             h5_dataset (h5py._hl.dataset.Dataset): The HDF5 dataset
        Returns:
            tuple: x min, x max, y min, y max
    """
    lat, lon = get_lat_lon_datasets(h5_dataset)

    shortname = _get_short_name(h5_dataset)
    proj4_str = _get_grid_mapping_data(_get_grid_mapping_group(shortname, h5_dataset.name))
    p = Proj(get_proj4(proj4_str))

    if len(lat.shape) == 2:
        x1, y1 = p(lat[0][0], lon[0][0])
        x2, y2 = p(lat[-1][-1], lon[-1][-1])

    if len(lat.shape) == 3:
        x1, y1 = p(lat[0][0][0], lon[0][0][0])
        x2, y2 = p(lat[-1][-1][-1], lon[-1][-1][-1])

    return x1, x2, y1, y2


def get_dimension_arrays(h5_dataset: Dataset) \
        -> Tuple[List[float],List[float]]:  # projected meters - x-coordinates, y-coordinates
    """ Gets the dimension scales arrays of the HDF5 dataset.
        Args:
             h5_dataset (h5py._hl.dataset.Dataset): The HDF5 dataset
        Returns:
            tuple: The x coordinate array and the y coordinate array
    """
    x, y = get_dimension_datasets(h5_dataset)
    return x[:], y[:]

def get_lat_lon_arrays(h5_dataset: Dataset) \
        -> Tuple[List[float],List[float]]:  # degrees - lat, lon
    """ Gets the lat/lon arrays of the HDF5 dataset.
        Args:
             h5_dataset (h5py._hl.dataset.Dataset): The HDF5 dataset
        Returns:
            tuple: The x coordinate array and the y coordinate array
    """
    x, y = get_lat_lon_datasets(h5_dataset)
    if len(x.shape) == 2: return x[0], y[:,0]
    elif len(x.shape) == 3: return x[0][0], y[0][:,0]

    return x[:], y[:]

def get_fill_value(h5_dataset: Dataset, default_fill_value: float) -> float:
    """ Returns the fill value for the given HDF5 dataset.
        If the HDF5 dataset has no fill value, returns the given default fill value.
        Args:
            h5_dataset (h5py._hl.dataset.Dataset): The given HDF5 dataset
            default_fill_value (float): The default value which is returned if no fill value is found in the dataset
        Returns:
            float: The fill value
    """
    if h5_dataset.attrs.__contains__('_FillValue'): return h5_dataset.attrs['_FillValue']
    logging.info(f'The dataset {h5_dataset.name} does not have a fill value, '
                 f'so the default fill value {default_fill_value} will be used')
    return default_fill_value
