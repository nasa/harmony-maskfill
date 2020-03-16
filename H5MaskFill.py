""" Creates a mask filled version of the given HDF5 file using the given shapefile. Outputs the new HDF5 file to the
    given output directory.

    Args:
        h5_dataset (h5py._hl.dataset.Dataset): The given HDF5 dataset
        shape_path (str): Path to a shape file used to create the mask array for the mask fill
        output_dir (str): The path to the output directory
        cache_dir (str): The path to a cache directory
        mask_grid_cache (str): Value determining whether to use previously cached mask arrays and whether to cache newly
                               created mask arrays
        default_fill_value (float): The default fill value for the mask fill if no other fill values are provided

    Returns:
        str: The path to the output HDF5 file
"""
import logging
import os
import shutil

import numpy as np

from pymods import H5GridProjectionInfo, MaskFillUtil, CFConfig, MaskFillCaching
from pymods.MaskFillUtil import apply_2D, get_h5_mask_array_id, process_h5_file

mask_grid_cache_values = ['ignore_and_delete',
                          'ignore_and_save',
                          'use_cache',
                          'use_and_save',
                          'use_cache_delete',
                          'maskgrid_only']


def produce_masked_hdf(hdf_path, shape_path, output_dir, cache_dir, mask_grid_cache, default_fill_value):
    mask_grid_cache = mask_grid_cache.lower()
    saved_mask_arrays = dict()

    CFConfig.readConfigFile(get_config_file_path())
    shortname = CFConfig.getShortName(hdf_path)

    if mask_grid_cache == 'maskgrid_only':
        process_h5_file(hdf_path, mask_fill, shape_path, cache_dir, mask_grid_cache,
                        default_fill_value, saved_mask_arrays, shortname)
    else:
        new_file_path = MaskFillUtil.get_masked_file_path(hdf_path, output_dir)
        shutil.copy(hdf_path, new_file_path)
        logging.debug(f'Created output file: {new_file_path}')
        process_h5_file(new_file_path, mask_fill, shape_path, cache_dir, mask_grid_cache, default_fill_value,
                        saved_mask_arrays, shortname)

    MaskFillCaching.cache_mask_arrays(saved_mask_arrays, cache_dir, mask_grid_cache)

    if mask_grid_cache != 'maskgrid_only':
        return MaskFillUtil.get_masked_file_path(hdf_path, output_dir)


def get_config_file_path():
    """
        Returns:
            str: The path to the MaskFillConfig.json file.
    """
    current_file_path = os.path.abspath(__file__)
    scripts_directory = os.path.dirname(current_file_path)
    data_directory = os.path.join(scripts_directory, "data")
    config_file_path = os.path.join(data_directory, "MaskFillConfig.json")

    return config_file_path


def mask_fill(h5_dataset, shape_path, cache_dir, mask_grid_cache,
              default_fill_value, saved_mask_arrays, shortname):
    """ Replaces the data in the HDF5 dataset with a mask filled version of the data.

        Args:
            h5_dataset (h5py._hl.dataset.Dataset): The given HDF5 dataset
            shape_path (str): Path to a shape file used to create the mask array for the mask fill
            cache_dir (str): The path to a cache directory
            mask_grid_cache (str): Value determining how the mask arrays used in the mask fill are created and cached
            default_fill_value (float): The default fill value for the mask fill if no other fill values are provided
            saved_mask_arrays (?):
            shortname (str):
    """

    # Ensure dataset has at least two dimensions and can be mask filled
    if len(h5_dataset.shape) < 2 or not h5_dataset.attrs.__contains__('coordinates'):
        logging.debug(f'The dataset {h5_dataset.name} is not two dimensional '
                      'or does not contain coordinates attribute, and cannot '
                      'be mask filled')
        return
    elif H5GridProjectionInfo.dataset_all_fill_value(h5_dataset, default_fill_value):
        logging.debug(f'The dataset {h5_dataset.name} only contains fill value,'
                      ' so there is no need to mask fill.')
        return

    # Get the mask array corresponding to the HDF5 dataset and the shapefile
    mask_array = get_mask_array(h5_dataset, shape_path, cache_dir, mask_grid_cache, saved_mask_arrays, shortname)

    # Perform mask fill and write the new mask filled data to the dataset,
    # unless the mask_grid_cache value only requires us to create a mask array
    if mask_grid_cache != 'maskgrid_only':
        fill_value = H5GridProjectionInfo.get_fill_value(h5_dataset, default_fill_value)
        mask_filled_data = apply_2D(h5_dataset[:], MaskFillUtil.mask_fill_array, mask_array, fill_value)
        h5_dataset.write_direct(mask_filled_data)

        # Get all values in mask_filled_data excluding the fill value
        unfilled_data = mask_filled_data[np.where(mask_filled_data != fill_value)]

        # Update statistics in the h5_dataset
        if h5_dataset.attrs.__contains__('observed_max'):
            h5_dataset.attrs.modify('observed_max', max(unfilled_data))

        if h5_dataset.attrs.__contains__('observed_min'):
            h5_dataset.attrs.modify('observed_min', min(unfilled_data))

        if h5_dataset.attrs.__contains__('observed_mean'):
            h5_dataset.attrs.modify('observed_mean', np.mean(unfilled_data))

        logging.debug(f'Mask filled the dataset {h5_dataset.name}')


def get_mask_array(h5_dataset, shape_path, cache_dir, mask_grid_cache, saved_mask_arrays, shortname):
    """ Gets the mask array corresponding the HDF5 file and shape file from a set of saved mask arrays or the cache directory.
        If the mask array file does not already exist, it is created and added to the set of saved mask arrays.

        Args:
            h5_dataset (h5py._hl.dataset.Dataset): The given HDF5 dataset
            shape_path (str): The path to the shapefile used to create the mask array
            cache_dir (str): The path to the directory where the mask array file is cached

        Returns:
            numpy.ndarray: The mask array
    """
    # Get the mask id which corresponds to the mask required for the HDF5 dataset and shapefile
    mask_id = get_h5_mask_array_id(h5_dataset, shape_path, shortname)

    # If the required mask array is in the set of saved mask arrays, get and return the mask array from the set
    if mask_id in saved_mask_arrays:
        return saved_mask_arrays[mask_id]

    # If the required mask array has already been created and cached, and the mask_grid_cache value allows the use of
    # cached arrays, read in the cached mask array from the file
    mask_array_path = get_mask_array_path(mask_id, cache_dir)
    if 'use' in mask_grid_cache and os.path.exists(mask_array_path):
        mask_array = np.load(mask_array_path)

    # Otherwise, create the mask array
    else:
        mask_array = create_mask_array(h5_dataset, shape_path, shortname)

    # Save and return the mask array
    saved_mask_arrays[mask_id] = mask_array
    return mask_array


def get_mask_array_path(mask_id, cache_dir):
    """ Returns the path to the file containing a mask array corresponding to the given HDF5 dataset.

        Args:
            h5_dataset (h5py._hl.dataset.Dataset): The given HDF5 dataset
            cache_dir (str): The directory in which mask arrays are cached

        Returns:
            str: The path to the mask array file
    """
    return os.path.join(cache_dir, mask_id + ".npy")


def create_mask_array(h5_dataset, shape_path, shortname):
    """ Creates a mask array corresponding to the HDF5 dataset and shape file

        Args:
            h5_dataset (h5py._hl.dataset.Dataset): The given HDF5 dataset
            shape_path (str): The path to the shapefile used to create the mask array

        Returns:
            numpy.ndarray: The mask array
    """
    proj4 = H5GridProjectionInfo.get_hdf_proj4(h5_dataset, shortname)
    shapes = MaskFillUtil.get_projected_shapes(proj4, shape_path)
    transform = H5GridProjectionInfo.get_transform(h5_dataset)

    return MaskFillUtil.get_mask_array(shapes, h5_dataset.shape[-2:], transform)
