import os
import shutil
import numpy as np
import h5py
from pymods import H5GridProjectionInfo, MaskFill, CFConfig, MaskFillCaching
import logging

mask_grid_cache_values = ['ignore_and_delete',
                          'ignore_and_save',
                          'use_cache',
                          'use_and_save',
                          'use_cache_delete',
                          'maskgrid_only']

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
def produce_masked_hdf(hdf_path, shape_path, output_dir, cache_dir, mask_grid_cache, default_fill_value):
    mask_grid_cache = mask_grid_cache.lower()
    saved_mask_arrays = dict()

    CFConfig.readConfigFile(get_config_file_path())
    shortname = CFConfig.getShortName(hdf_path)

    if mask_grid_cache == 'maskgrid_only':
        process_file(hdf_path, mask_fill, shape_path, cache_dir, mask_grid_cache, default_fill_value, saved_mask_arrays,
                     shortname)
    else:
        new_file_path = MaskFill.get_masked_file_path(hdf_path, output_dir)
        shutil.copy(hdf_path, new_file_path)
        logging.debug(f'Created output file: {new_file_path}')
        process_file(new_file_path, mask_fill, shape_path, cache_dir, mask_grid_cache, default_fill_value,
                     saved_mask_arrays, shortname)

    MaskFillCaching.cache_mask_arrays(saved_mask_arrays, cache_dir, mask_grid_cache)

    if mask_grid_cache != 'maskgrid_only': return MaskFill.get_masked_file_path(hdf_path, output_dir)


"""
    Returns:
        str: The path to the MaskFillConfig.json file. 
"""
def get_config_file_path():
    current_file_path = os.path.abspath(__file__)
    pymods_directory = os.path.dirname(current_file_path)
    scripts_directory = os.path.dirname(pymods_directory)
    data_directory = os.path.join(scripts_directory, "data")
    config_file_path = os.path.join(data_directory, "MaskFillConfig.json")

    return config_file_path


""" Performs the given process on all datasets in the HDF5 file.

    Args:
        file_path (str): The path to the input HDF5 file
        process (function): The process to be performed on the datasets in the file
        *args: The arguments passed to the process
"""
def process_file(file_path, process, *args):
    def process_children(obj, process, *args):
        for name, child in obj.items():
            # Process the children of a group
            if isinstance(child, h5py._hl.group.Group):
                process_children(child, process, *args)
            # Process datasets
            elif isinstance(child, h5py._hl.dataset.Dataset):
                process(child, *args)

    with h5py.File(file_path, mode='r+') as file:
        process_children(file, process, *args)


""" Replaces the data in the HDF5 dataset with a mask filled version of the data. 

    Args:
        h5_dataset (h5py._hl.dataset.Dataset): The given HDF5 dataset
        shape_path (str): Path to a shape file used to create the mask array for the mask fill
        cache_dir (str): The path to a cache directory 
        mask_grid_cache (str): Value determining how the mask arrays used in the mask fill are created and cached
        default_fill_value (float): The default fill value for the mask fill if no other fill values are provided
"""
def mask_fill(h5_dataset, shape_path, cache_dir, mask_grid_cache, default_fill_value, saved_mask_arrays, shortname):
    # Ensure dataset has at least two dimensions and can be mask filled
    if len(h5_dataset.shape) < 2 or not h5_dataset.attrs.__contains__('coordinates'):
        logging.debug(
            f'The dataset {h5_dataset.name} is not two dimensional or does not contain coordinates attribute, and cannot be mask filled')
        return

    # Get the mask array corresponding to the HDF5 dataset and the shapefile
    mask_array = get_mask_array(h5_dataset, shape_path, cache_dir, mask_grid_cache, saved_mask_arrays, shortname)

    # Perform mask fill and write the new mask filled data to the h5_dataset,
    # unless the mask_grid_cache value only requires us to create a mask array
    if mask_grid_cache != 'maskgrid_only':
        fill_value = H5GridProjectionInfo.get_fill_value(h5_dataset, default_fill_value)
        mask_filled_data = process_multiple_dimensions(h5_dataset[:], MaskFill.mask_fill_array, mask_array, fill_value, name = h5_dataset.name)
        h5_dataset.write_direct(mask_filled_data)

        # Get all values in mask_filled_data excluding the fill value
        unfilled_data = mask_filled_data[mask_filled_data != fill_value]

        # Update statistics in the h5_dataset
        if h5_dataset.attrs.__contains__('observed_max'): h5_dataset.attrs.modify('observed_max', max(unfilled_data))
        if h5_dataset.attrs.__contains__('observed_min'): h5_dataset.attrs.modify('observed_min', min(unfilled_data))
        if h5_dataset.attrs.__contains__('observed_mean'): h5_dataset.attrs.modify('observed_mean', np.mean(unfilled_data))

        logging.debug(f'Mask filled the dataset {h5_dataset.name}')


""" Recursively mask fills datasets with two or more dimensions by mask filling each 2D array within the dataset.

    Args:
        data (numpy.ndarray): The data array to be mask filled
        mask_array (numpy.ndarray): The mask array which will be applied to data
        fill_value (float): Value used to fill in the masked values when necessary
        
    Returns:
        numpy.ndarray: The mask filled array
"""
def process_multiple_dimensions(data, process, *args, name=""):
    # 2D Case
    if len(data.shape) == 2:
        return process(data, *args)
        return data

    # For more than two dimensions, mask fill each dimension recursively
    for i in range(len(data)): data[i] = process_multiple_dimensions(data[i], process, *args, name=name)
    return data


""" Gets the mask array corresponding the HDF5 file and shape file from a set of saved mask arrays or the cache directory.
    If the mask array file does not already exist, it is created and added to the set of saved mask arrays.

    Args:
        h5_dataset (h5py._hl.dataset.Dataset): The given HDF5 dataset
        shape_path (str): The path to the shapefile used to create the mask array
        cache_dir (str): The path to the directory where the mask array file is cached

    Returns:
        numpy.ndarray: The mask array
"""
def get_mask_array(h5_dataset, shape_path, cache_dir, mask_grid_cache, saved_mask_arrays, shortname):
    # Get the mask id which corresponds to the mask required for the HDF5 dataset and shapefile
    mask_id = get_mask_array_id(h5_dataset, shape_path, shortname)

    # If the required mask array is in the set of saved mask arrays, get and return the mask array from the set
    if mask_id in saved_mask_arrays: return saved_mask_arrays[mask_id]

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


""" Creates an id corresponding to the given shapefile, projection information, and shape of a dataset,
    which determine the mask array for the dataset.  

    Args:
        h5_dataset (h5py._hl.dataset.Dataset): The given HDF5 dataset
        shape_path (str): Path to a shape file used to create the mask array for the mask fill

    Returns:
        str: The id 
"""
def get_mask_array_id(h5_dataset, shape_path, shortname):
    # The mask array is determined by the CRS of the dataset, the dataset's transform, the shape of the dataset,
    # and the shapes used in the mask
    proj_string = H5GridProjectionInfo.get_hdf_proj4(h5_dataset, shortname)
    transform = H5GridProjectionInfo.get_transform(h5_dataset)
    dataset_shape = h5_dataset[:].shape

    return MaskFillCaching.create_mask_array_id(proj_string, transform, dataset_shape, shape_path)


""" Returns the path to the file containing a mask array corresponding to the given HDF5 dataset.

    Args:
        h5_dataset (h5py._hl.dataset.Dataset): The given HDF5 dataset
        cache_dir (str): The directory in which mask arrays are cached

    Returns:
        str: The path to the mask array file 
"""
def get_mask_array_path(mask_id, cache_dir):
    return os.path.join(cache_dir, mask_id + ".npy")


""" Creates a mask array corresponding to the HDF5 dataset and shape file

    Args:
        h5_dataset (h5py._hl.dataset.Dataset): The given HDF5 dataset
        shape_path (str): The path to the shapefile used to create the mask array

    Returns:
        numpy.ndarray: The mask array
"""
def create_mask_array(h5_dataset, shape_path, shortname):
    proj4 = H5GridProjectionInfo.get_hdf_proj4(h5_dataset, shortname)
    shapes = MaskFill.get_projected_shapes(proj4, shape_path)
    raster_arr = h5_dataset[:]
    transform = H5GridProjectionInfo.get_transform(h5_dataset)

    return MaskFill.get_mask_array(shapes, raster_arr.shape[-2:], transform)

