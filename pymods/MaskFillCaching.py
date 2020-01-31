""" Utility functions to support MaskFill cache processing
    To cache the masking array used to apply fill values (outside mask)
"""
import hashlib
import logging
import os

import numpy as np

from pymods import MaskFillUtil

mask_grid_cache_values = ['ignore_and_delete',
                          'ignore_and_save',
                          'use_cache',
                          'use_and_save',
                          'use_cache_delete',
                          'maskgrid_only']


def get_cached_mask_array(data, shape_path, cache_dir, mask_grid_cache):
    """
        Returns the corresponding cached mask array if it exists, and None otherwise.
        Args:
            data (obj): The path to a GeoTIFF file (str) or an HDF5 dataset (h5py._hl.dataset.Dataset)
            shape_path (str): Path to a shape file used to create the mask array for the mask fill
            cache_dir (str): The path to the cache directory where cached mask array files are stored
            mask_grid_cache (str): Value determining whether to use previously cached mask arrays
                                   and whether to cache newly created mask arrays
        Returns:
            numpy.ndarray: The mask array
    """
    mask_array_path = get_mask_array_path(data, shape_path, cache_dir)

    mask_array = None
    if 'use' in mask_grid_cache and os.path.exists(mask_array_path):
        mask_array = np.load(mask_array_path)

    return mask_array


def get_mask_array_id(data, shape_path):
    """ Gets the id of the mask array corresponding to the given data and shape file.
        Args:
            data (obj): The path to a GeoTIFF file (str) or an HDF5 dataset (h5py._hl.dataset.Dataset)
            shape_path (str): Path to a shape file used to create the mask array for the mask fill
        Returns:
            str: The mask id
    """
    # GeoTIFF case
    if isinstance(data, str) and data.lower().endswith('.tif'):
        mask_id = MaskFillUtil.get_geotiff_mask_array_id(data, shape_path)
    # HDF5 case
    else:
        mask_id = MaskFillUtil.get_h5_mask_array_id(data, shape_path)

    return mask_id


def create_mask_array_id(proj_string, transform, dataset_shape, shape_file_path):
    """ Creates an id corresponding to the given shapefile, projection information, and shape of a dataset,
        which determine the mask array for the dataset.
        Args:
            proj_string (str): A proj4 string which describes the projection information of the data
            transform (affine.Affine): The affine transform corresponding to the data
            dataset_shape (tuple): The shape of the data
            shape_file_path (str): The path to the given shapefile
        Returns:
            str: The mask id
    """
    mask_id = proj_string + str(transform) + str(dataset_shape) + shape_file_path

    # Hash the mask id and return
    mask_id = hashlib.sha224(mask_id.encode()).hexdigest()
    return mask_id


def get_mask_array_path(data, shape_path, cache_dir):
    """ Returns the path to cached mask array file corresponding to the given data and shapefile. If the file does not
        exist, the at which path it would reside is returned.
        Args:
            data (obj): The path to a GeoTIFF file (str) or an HDF5 dataset (h5py._hl.dataset.Dataset)
            shape_path (str): Path to a shape file used to create the mask array for the mask fill
            cache_dir (str): The path to the directory where the mask arrays are cached
        Returns:
            str: The path to the mask array file
    """
    mask_id = get_mask_array_id(data, shape_path)
    mask_array_path = get_mask_array_path_from_id(mask_id, cache_dir)
    return mask_array_path


def get_mask_array_path_from_id(mask_id, cache_dir):
    """ Returns the path to the file containing a mask array corresponding to the given mask_id and cache directory.
        Args:
            mask_id (str): An id corresponding to the desired mask array file
            cache_dir (str): The directory where mask array files are cached
        Returns:
            str: The path to the mask array file
    """
    return os.path.join(cache_dir, mask_id + ".npy")


def cache_mask_array(mask_array, data, shape_path, cache_dir, mask_grid_cache):
    """ Caches the mask array corresponding to the given data and shapefile as a .npy file in the cache directory,
        if the mask grid cache value allows.
        Args:
            mask_array (numpy.ndarray): The mask array to be cached
            data (obj): The path to a GeoTIFF file (str) or an HDF5 dataset (h5py._hl.dataset.Dataset)
            shape_path (str): Path to a shape file used to create the mask array for the mask fill
            cache_dir (str): The path to the cache directory where cached mask array files are stored
            mask_grid_cache (str): Value determining whether to use previously cached mask arrays
                                   and whether to cache newly created mask arrays
    """
    if 'save' in mask_grid_cache or mask_grid_cache == 'maskgrid_only':
        mask_array_path = get_mask_array_path(data, shape_path, cache_dir)
        np.save(mask_array_path, mask_array)


def cache_mask_arrays(mask_arrays, cache_dir, mask_grid_cache):
    """ Caches all of the given mask arrays as a .npy file in the cache directory, if the mask grid cache value allows.
        Args:
            mask_arrays (dict): A dictionary mapping from mask ids to the corresponding mask arrays
            cache_dir (str): The path to the cache directory where cached mask array files are stored
            mask_grid_cache (str): Value determining whether to use previously cached mask arrays
                                   and whether to cache newly created mask arrays
    """
    # Save mask arrays if the mask_grid_cache value requires
    if 'delete' not in mask_grid_cache:
        for mask_id, mask_array in mask_arrays.items():
            mask_array_path = get_mask_array_path_from_id(mask_id, cache_dir)
            np.save(mask_array_path, mask_array)

        logging.debug('Cached all mask arrays')
