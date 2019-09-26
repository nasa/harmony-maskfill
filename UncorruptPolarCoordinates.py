import affine
import argparse
import numpy as np
import os
import shutil
from pyproj import Transformer, CRS
from pymods import H5MaskFill
from pymods import MaskFill

""" Executable which overwrites corrupted coordinate arrays for Polar LAEA data in HDF5 files with correct coordinates.

    Input parameters:
        --FILE_URLS: Path to an HDF5 file 

        --OUTPUT_DIR: (optional) Path to the output directory where the uncorrupted file will be written.
            If not provided, the current working directory will be used.
"""


polar_proj = CRS.from_proj4("+ellps=WGS84 +datum=WGS84 +proj=laea +lat_0=90")
lat_long_proj = CRS.from_epsg(4326)
proj_trans = Transformer.from_crs(polar_proj, lat_long_proj)  # Transform from laea to lat/long coordinates

lat_array, long_array = None, None


""" If the dataset contains latitude or longitude coordinates, overwrites the corrupted data with uncorrupted versions. 
    
    Args:
        h5_dataset (h5py._hl.dataset.Dataset): The given HDF5 dataset
"""
def uncorrupt_coordinates(h5_dataset):
    if 'Polar' in h5_dataset.parent.name:
        if 'latitude' in h5_dataset.name:
            data = h5_dataset[:]
            uncorrupt_data = H5MaskFill.process_multiple_dimensions(data, get_latitude)
            h5_dataset.write_direct(uncorrupt_data)

        elif 'longitude' in h5_dataset.name:
            data = h5_dataset[:]
            uncorrupt_data = H5MaskFill.process_multiple_dimensions(data, get_longitude)
            h5_dataset.write_direct(uncorrupt_data)

""" Returns the uncorrupted latitude coordinate array corresponding to the dataset.

    Args:
        h5_dataset (h5py._hl.dataset.Dataset): The given HDF5 dataset
        
    Returns:
        numpy.ndarray: The latitude array
"""
def get_latitude(dataset):
    if lat_array is None: get_lat_long_arrays(dataset)
    return lat_array


""" Returns the uncorrupted longitude coordinate array corresponding to the dataset.

    Args:
        h5_dataset (h5py._hl.dataset.Dataset): The given HDF5 dataset

    Returns:
        numpy.ndarray: The longitude array
"""
def get_longitude(dataset):
    if long_array is None: get_lat_long_arrays(dataset)
    return long_array


""" Calculates the uncorrupted latitude and longitude coordinate arrays corresponding to the dataset.

    Args:
        h5_dataset (h5py._hl.dataset.Dataset): The given HDF5 dataset
"""
def get_lat_long_arrays(dataset):
    lat_arr, long_arr = np.zeros(dataset.shape), np.zeros(dataset.shape)
    trans = get_transform(dataset)
    rows, cols = dataset.shape

    # Get x and y projected meters corresponding to each row and column in image array
    for i in range(rows):
        for j in range(cols):
            x_meters_left, y_meters_top = trans * (i, j)  # Projected meters of top-left corner of cell
            x_meters_right, y_meters_bottom = trans * (i + 1, j + 1)  # Projected meters of bottom-right corner of cell

            # Projected meters of center of cell
            x_meters, y_meters = (x_meters_left + x_meters_right) / 2, (y_meters_top + y_meters_bottom) / 2

            # Get lat/long values from projected meters
            lat, long = proj_meters_to_lat_long(x_meters, y_meters)

            lat_arr[i][j] = lat
            long_arr[i][j] = long

    global lat_array; global long_array
    lat_array, long_array = lat_arr, long_arr

""" Returns a transform from image indices in the dataset to projected meters.

    Args:
        h5_dataset (h5py._hl.dataset.Dataset): The given HDF5 dataset
    
    Returns:
        affine.Affine: A transform mapping from image index values to projected meters world coordinates
"""
def get_transform(dataset):
    x_min, y_max = -9000000, 9000000

    if dataset.shape == (500, 500): cell_width, cell_height = 36000, -36000
    if dataset.shape == (2000, 2000): cell_width, cell_height = 9000, -9000
    if dataset.shape == (6000, 6000): cell_width, cell_height = 3000, -3000

    return affine.Affine(cell_width, 0, x_min, 0, cell_height, y_max)

""" Converts projected meter values to latitude/longitude coordinates.

    Args:
        x (float): The projected meters value along the x-axis
        y (float): The projected meters value along the y-axis
    
    Returns:
        tuple: latitude, longitude
"""
def proj_meters_to_lat_long(x, y):
    return proj_trans.transform(x, y)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--FILE_URLS', dest='input_file', help='Name of the input file to uncorrupt')
    parser.add_argument('--OUTPUT_DIR', dest='output_dir', help='Name of the output directory to put the output file',
                        default=os.getcwd())
    parser = parser.parse_args()

    input_file = parser.input_file
    new_file_path = MaskFill.get_masked_file_path(input_file, parser.output_dir)
    shutil.copy(input_file, new_file_path)

    H5MaskFill.process_file(new_file_path, uncorrupt_coordinates)
    print("Output file:", new_file_path)