""" Executable which overwrites corrupted coordinate arrays for Polar LAEA data in HDF5 files with correct coordinates.

    Input parameters:
        --FILE_URLS: Path to an HDF5 file 

        --OUTPUT_DIR: (optional) Path to the output directory where the uncorrupted file will be written.
            If not provided, the current working directory will be used.
"""
import argparse
import os
import shutil

import affine
import numpy as np
from pyproj import Transformer, CRS

from pymods import MaskFillUtil

polar_proj = CRS.from_proj4("+ellps=WGS84 +datum=WGS84 +proj=laea +lat_0=90")
lat_long_proj = CRS.from_epsg(4326)
proj_trans = Transformer.from_crs(polar_proj, lat_long_proj)  # Transform from laea to lat/long coordinates

lat_array, long_array = None, None


def uncorrupt_coordinates(h5_dataset):
    """ If the dataset contains latitude or longitude coordinates, overwrites the corrupted data with uncorrupted versions.

        Args:
            h5_dataset (h5py._hl.dataset.Dataset): The given HDF5 dataset
    """
    if 'Polar' in h5_dataset.parent.name:
        if 'latitude' in h5_dataset.name:
            data = h5_dataset[:]
            uncorrupt_data = pymods.MaskFillUtil.apply_2D(data, get_latitude)
            h5_dataset.write_direct(uncorrupt_data)

        elif 'longitude' in h5_dataset.name:
            data = h5_dataset[:]
            uncorrupt_data = pymods.MaskFillUtil.apply_2D(data, get_longitude)
            h5_dataset.write_direct(uncorrupt_data)

def get_latitude(dataset):
    """ Returns the uncorrupted latitude coordinate array corresponding to the dataset.

        Args:
            h5_dataset (h5py._hl.dataset.Dataset): The given HDF5 dataset

        Returns:
            numpy.ndarray: The latitude array
    """
    if lat_array is None: get_lat_long_arrays(dataset)
    return lat_array


def get_longitude(dataset):
    """ Returns the uncorrupted longitude coordinate array corresponding to the dataset.

        Args:
            h5_dataset (h5py._hl.dataset.Dataset): The given HDF5 dataset

        Returns:
            numpy.ndarray: The longitude array
    """
    if long_array is None: get_lat_long_arrays(dataset)
    return long_array


def get_lat_long_arrays(dataset):
    """ Calculates the uncorrupted latitude and longitude coordinate arrays corresponding to the dataset.

        Args:
            h5_dataset (h5py._hl.dataset.Dataset): The given HDF5 dataset
    """
    lat_arr, long_arr = np.zeros(dataset.shape), np.zeros(dataset.shape)
    trans = get_transform(dataset)
    rows, cols = dataset.shape

    # Get x and y projected meters corresponding to each row and column in image array
    for i in range(rows):
        for j in range(cols):
            # Projected meters of center of cell
            x_meters, y_meters = trans * (i + 0.5, j + 0.5)

            # Get lat/long values from projected meters
            lat, long = proj_meters_to_lat_long(x_meters, y_meters)

            lat_arr[i][j] = lat
            long_arr[i][j] = long

    global lat_array; global long_array
    lat_array, long_array = lat_arr, long_arr

def get_transform(dataset):
    """ Returns a transform from image indices in the dataset to projected meters.

        Args:
            h5_dataset (h5py._hl.dataset.Dataset): The given HDF5 dataset

        Returns:
            affine.Affine: A transform mapping from image index values to projected meters world coordinates
    """
    x_min, y_max = -9000000, 9000000

    if dataset.shape == (500, 500): cell_width, cell_height = 36000, -36000
    if dataset.shape == (2000, 2000): cell_width, cell_height = 9000, -9000
    if dataset.shape == (6000, 6000): cell_width, cell_height = 3000, -3000

    return affine.Affine(cell_width, 0, x_min, 0, cell_height, y_max)

def proj_meters_to_lat_long(x, y):
    """ Converts projected meter values to latitude/longitude coordinates.

        Args:
            x (float): The projected meters value along the x-axis
            y (float): The projected meters value along the y-axis

        Returns:
            tuple: latitude, longitude
    """
    return proj_trans.transform(x, y)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--FILE_URLS', dest='input_file', help='Name of the input file to uncorrupt')
    parser.add_argument('--OUTPUT_DIR', dest='output_dir', help='Name of the output directory to put the output file',
                        default=os.getcwd())
    parser = parser.parse_args()

    input_file_path = parser.input_file
    new_file_path = MaskFillUtil.get_masked_file_path(input_file_path, parser.output_dir)
    shutil.copy(input_file_path, new_file_path)

    pymods.MaskFillUtil.process_h5_file(new_file_path, uncorrupt_coordinates)
    print("Output file:", new_file_path)