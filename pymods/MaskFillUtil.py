""" Utility functions to support MaskFill processing
"""
import os

import geopandas as gpd
import h5py
import numpy as np
from rasterio import features  # as in geographic features, shapes, polygons

def get_projected_shapes(proj4, shape_path):
    """ Projects the shapes in the given shapefile to a new coordinate reference system.
        Args:
            proj4 (string): The proj4 string representing the new coordinate reference system
            shape_path (string): The path to the shape file which is to be converted
        Returns:
            geopandas.geoseries.GeoSeries: A geopandas GeoSeries containing the projected shape information.
    """
    try:
        shape_gdf = gpd.read_file(shape_path)  # Load shape file as geopandas data frame
    except:
        raise ValueError('Shape data cannot be read from the given shapefile.')

    projected_gdf = shape_gdf.to_crs(proj4)  # Project data frame to new coordinate reference system
    return projected_gdf['geometry']


def get_mask_array(shapes, out_shape, transform):
    """ Rasterizes the given shapes to create a mask array.
        Args:
            shapes (geopandas.geoseries.GeoSeries): The shapes to be rasterized
            out_shape (tuple): The shape of the resultant mask array
            transform (affine.Affine): A transform mapping from image coordinates to world coordinates
        Returns:
            numpy.ndarray: A numpy array representing the rasterized shapes
    """
    mask = features.rasterize(shapes=shapes, default_value=0, fill=1, out_shape=out_shape,
                              dtype=np.uint8, transform=transform, all_touched=True)
    return mask


def mask_fill_array(raster_arr, mask_array, fill_value):
    """ Performs a mask fill on raster_arr using the mask mask_array
        Args:
            raster_arr (numpy.ndarray): The array to be mask filled
            mask_array (numpy.ndarray): The mask array which will be applied to raster_arr
            fill_value (float): Value used to fill in the masked values when necessary
        Returns:
            numpy.ndarray: The mask filled array
    """
    out_image = np.ma.array(raster_arr, mask=mask_array, fill_value=fill_value)
    return out_image.filled()


def get_masked_file_path(original_file_path, output_dir):
    """ Returns the path to the mask filled output file.
        Args:
            original_file_path (str): The original file which is to be mask filled
            output_dir (str): The directory to which the output file will be written
        Returns:
            str: The path to the mask filled version of the original file

    """
    base_name = os.path.basename(original_file_path)
    file_name, extension = os.path.splitext(base_name)
    return os.path.join(output_dir, file_name + "_mf" + extension)


def process_h5_file(file_path, process, *args):
    """ Performs the given process on all datasets in the HDF5 file.
        Args:
            file_path (str): The path to the input HDF5 file
            process (function): The process to be performed on the datasets in the file
            *args: The arguments passed to the process
    """

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


def apply_2D(data, process, *args):  # , name = ""  ??? - unused!
    """ Recursively applies a 2D process to datasets with two or more dimensions,
        Always applies the process to the last 2 dimensions of the dataset,
        iterating through any lower dimensions and processing up through n-2 dimensions.
        E.g., a dataset with coordinates dimensions: (time, lat, lon),
            will apply to lat, lon dimensions, for each time entry.
        Args:
            data (numpy.ndarray): The data array to be processed
            process: The process to be applied to data
            *args: tuple of parameters being passed to process
        Returns:
            numpy.ndarray: The processed array
    """
    # 2D Case
    if len(data.shape) == 2:
        return process(data, *args)
        return data

    # For more than two dimensions, mask fill each dimension recursively
    for i in range(len(data)):
        data[i] = apply_2D(data[i], process, *args)

    return data
