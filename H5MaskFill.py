""" Creates a mask filled version of the given HDF5 file using the given
    shapefile. Outputs the new HDF5 file to the given output directory.

    Args:
        h5_dataset (h5py._hl.dataset.Dataset): The given HDF5 dataset
        shape_path (str): Path to a shape file used to create the mask array
            for the mask fill
        output_dir (str): The path to the output directory
        cache_dir (str): The path to a cache directory
        mask_grid_cache (str): Value determining whether to use previously
            cached mask arrays and whether to cache newly created mask arrays
        default_fill_value (float): The default fill value for the mask fill
            if no other fill values are provided

    Returns:
        str: The path to the output HDF5 file
"""

import logging
import os
import shutil
from typing import Dict, Set

import numpy as np
import h5py

from pymods import (H5GridProjectionInfo, MaskFillUtil, CFConfig,
                    MaskFillCaching)
from pymods.MaskFillUtil import apply_2D, get_h5_mask_array_id, process_h5_file

mask_grid_cache_values = ['ignore_and_delete',
                          'ignore_and_save',
                          'use_cache',
                          'use_and_save',
                          'use_cache_delete',
                          'maskgrid_only']


def produce_masked_hdf(hdf_path: str, shape_path: str, output_dir: str,
                       cache_dir: str, mask_grid_cache: str,
                       default_fill_value: float) -> str:
    """ This is the main wrapper function that is called from MaskFill.py
        when processing an HDF-5 file. This deals primarily with instantiating
        a dictionary-based cache for masks, based on their coordinate
        information, placing the output file in the correct location
        and caching masks if required.

        The path to the output file is returned.

    """
    mask_grid_cache = mask_grid_cache.lower()
    saved_mask_arrays = dict()

    CFConfig.readConfigFile()
    shortname = CFConfig.getShortName(hdf_path)
    exclusion_set = get_exclusions(hdf_path)

    if mask_grid_cache == 'maskgrid_only':
        process_h5_file(hdf_path, mask_fill, shape_path, cache_dir,
                        mask_grid_cache, default_fill_value, saved_mask_arrays,
                        shortname, exclusion_set)
    else:
        new_file_path = MaskFillUtil.get_masked_file_path(hdf_path, output_dir)
        shutil.copy(hdf_path, new_file_path)
        logging.debug(f'Created output file: {new_file_path}')
        process_h5_file(new_file_path, mask_fill, shape_path, cache_dir,
                        mask_grid_cache, default_fill_value, saved_mask_arrays,
                        shortname, exclusion_set)

    MaskFillCaching.cache_mask_arrays(saved_mask_arrays, cache_dir,
                                      mask_grid_cache)

    if mask_grid_cache != 'maskgrid_only':
        return MaskFillUtil.get_masked_file_path(hdf_path, output_dir)


def get_exclusions(h5_file_path: str) -> Set[str]:
    """ Get the set of dataset exclusions from coordinates and config file.

    Args:
        h5_dataset: h5py data object for dataset within hdf5 file
    Returns:
        set of dataset names that should not be processed by MaskFill.
    """
    with h5py.File(h5_file_path, mode='r') as input_file:
        exclusion_set = get_coordinates(input_file)
        exclusion_set.update(CFConfig.get_dataset_exclusions())

    return exclusion_set


def mask_fill(h5_dataset: h5py.Dataset,
              shape_path: str, cache_dir: str,
              mask_grid_cache: str, default_fill_value: float,
              saved_mask_arrays: Dict[str, np.ndarray],
              shortname: str, exclusions_set: Set[str]):
    """ Replaces the data in the HDF5 dataset with a mask filled version of the
        data. This function is applied to each dataset via the apply_2D
        mechanism.

        Args:
            h5_dataset (h5py._hl.dataset.Dataset): The given HDF5 dataset
            shape_path: The path to a masking shape file (Polygon)
            cache_dir (str): The path to a cache directory
            mask_grid_cache (str): Value determining how the mask arrays used
                in the mask fill are created and cached
            default_fill_value (float): The default fill value for the mask
                fill if no other fill values are provided
            saved_mask_arrays (dict): Previously calculated masks, saved via a
                key derived from dataset coordinates (or dimensions), dataset
                shape, shape file and projection.
            shortname (str): Reference to the collection shortname. Used
                primarily in detecting default configuration values.
    """
    # Test if h5_dataset matches within list of exclusions for maskfill
    split_dataset_name = h5_dataset.name.split('/')
    for exclusion in exclusions_set:
        # Check if any part of the dataset hierarchy matches the exclusion.
        if exclusion in split_dataset_name:
            logging.debug(f'{exclusion} in {h5_dataset.name} is in exclusion '
                          'list and will not be mask filled.')
            return

    # Ensure dataset has at least two dimensions and can be mask filled
    if len(h5_dataset.shape) < 2 or not h5_dataset.attrs.__contains__('coordinates'):
        logging.debug(
            f'The dataset {h5_dataset.name} is not two dimensional '
            f'or does not contain coordinates attribute, and cannot be mask filled')
        return
    elif H5GridProjectionInfo.dataset_all_fill_value(h5_dataset, default_fill_value):
        logging.debug(f'The dataset {h5_dataset.name} only contains fill value,'
                      ' so there is no need to mask fill.')
        return

    # Get the mask array corresponding to the HDF5 dataset and the shapefile
    mask_array = get_mask_array(h5_dataset, shape_path, cache_dir,
                                mask_grid_cache, saved_mask_arrays, shortname)

    # Perform mask fill and write the new mask filled data to the dataset,
    # unless the mask_grid_cache value only requires us to create a mask array
    if mask_grid_cache != 'maskgrid_only':
        fill_value = H5GridProjectionInfo.get_fill_value(h5_dataset,
                                                         default_fill_value)
        mask_filled_data = apply_2D(h5_dataset[:],
                                    MaskFillUtil.mask_fill_array,
                                    mask_array, fill_value)

        h5_dataset.write_direct(mask_filled_data)

        # If the dataset attributes contain observed statistics, update them.
        statistics = ['observed_max', 'observed_min', 'observed_mean']

        if any(statistic in h5_dataset.attrs.keys()
               for statistic in statistics):
            logging.debug(f'Updating statistics for {h5_dataset.name}')

            # Get all values in mask_filled_data excluding the fill value
            unfilled_data = mask_filled_data[np.not_equal(mask_filled_data,
                                                          fill_value)]

            if unfilled_data.size > 0:
                observed_max = np.max(unfilled_data)
                observed_min = np.min(unfilled_data)
                observed_mean = np.mean(unfilled_data)
            else:
                observed_max = observed_min = observed_mean = fill_value

            # Update statistics in the h5_dataset
            if h5_dataset.attrs.__contains__('observed_max'):
                h5_dataset.attrs.modify('observed_max', observed_max)

            if h5_dataset.attrs.__contains__('observed_min'):
                h5_dataset.attrs.modify('observed_min', observed_min)

            if h5_dataset.attrs.__contains__('observed_mean'):
                h5_dataset.attrs.modify('observed_mean', observed_mean)

        logging.debug(f'Mask filled the dataset {h5_dataset.name}')


def get_mask_array(h5_dataset: h5py.Dataset, shape_path: str,
                   cache_dir: str, mask_grid_cache: str,
                   saved_mask_arrays: Dict[str, np.ndarray],
                   shortname: str) -> np.ndarray:
    """ Gets the mask array corresponding the HDF5 file and shape file from a
        set of saved mask arrays or the cache directory.
        If the mask array file does not already exist, it is created
        and added to the set of saved mask arrays.

        Args:
            h5_dataset (h5py._hl.dataset.Dataset): The given HDF5 dataset
            shape_path (str): The path to the shapefile used to create the
                mask array
            cache_dir (str): The path to the directory where the mask array
                file is cached
            mask_grid_cache (str): directive for mask grid cache handling
            saved_mask_arrays: Shared array for caching mask grids
            shortname (str): the collection shortname for the h5 file.

        Returns:
            numpy.ndarray: The mask array
    """
    # Get the mask id which corresponds to the mask required for the HDF5
    # dataset and shapefile
    mask_id = get_h5_mask_array_id(h5_dataset, shape_path, shortname)

    # If the required mask array is in the set of saved mask arrays,
    # get and return the mask array from the set
    if mask_id in saved_mask_arrays:
        logging.debug(f'{h5_dataset.name}: Retrieving saved mask.')
        return saved_mask_arrays[mask_id]

    # If the required mask array has already been created and cached, and the
    # mask_grid_cache value allows the use of cached arrays, read in the
    # cached mask array from the file
    mask_array_path = get_mask_array_path(mask_id, cache_dir)
    if 'use' in mask_grid_cache and os.path.exists(mask_array_path):
        logging.debug(f'{h5_dataset.name}: Loading cached mask.')
        mask_array = np.load(mask_array_path)
    # Otherwise, create the mask array
    else:
        logging.debug(f'{h5_dataset.name}: Creating new mask.')
        mask_array = create_mask_array(h5_dataset, shape_path, shortname)

    # Save and return the mask array
    saved_mask_arrays[mask_id] = mask_array
    return mask_array


def get_mask_array_path(mask_id: str, cache_dir: str) -> str:
    """ Returns the path to the file containing a mask array corresponding to
        the given HDF5 dataset.
        Args:
            h5_dataset (h5py._hl.dataset.Dataset): The given HDF5 dataset
            cache_dir (str): The directory in which mask arrays are cached
        Returns:
            str: The path to the mask array file
    """
    return os.path.join(cache_dir, mask_id + ".npy")


def create_mask_array(h5_dataset: h5py.Dataset, shape_path: str,
                      shortname: str) -> np.ndarray:
    """ Creates a mask array corresponding to the HDF5 dataset and shape file
        Args:
            h5_dataset (h5py._hl.dataset.Dataset): The given HDF5 dataset
            shape_path (str): The path to the shapefile used to create the
                mask array

        Returns:
            numpy.ndarray: The mask array
    """
    proj4 = H5GridProjectionInfo.get_hdf_proj4(h5_dataset, shortname)
    transform = H5GridProjectionInfo.get_transform(h5_dataset)

    return MaskFillUtil.get_mask_array(shape_path, proj4, h5_dataset.shape[-2:], transform)


def get_coordinates(input_file: h5py.File) -> Set[str]:
    """ Gets the coordinate reference datasets within the input_file.
    :param input_file: H5Py data object for HDF5 input data file, follows CF conventions
    :return: set of coordinate reference paths found in data file.
    """
    coords = set()
    # loop through datasets in file by name
    for item_name in input_file:
        group_or_dataset = input_file[item_name]
        if isinstance(group_or_dataset, h5py.Group):
            # recursive get coordinates for group
            sub_coords = get_coordinates(group_or_dataset)
            coords.update(sub_coords)
        elif hasattr(group_or_dataset, 'attrs') \
                and 'coordinates' in group_or_dataset.attrs:
            coordinates = group_or_dataset.attrs['coordinates']
            if isinstance(coordinates, bytes):
                coordinates = coordinates.decode()
            coordinate_list = coordinates.split(' ')

            for coordinate in coordinate_list:
                coords.add(coordinate)

    return coords
