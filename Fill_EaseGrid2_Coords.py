""" Executable which overwrites incomplete and/or corrupted coordinate arrays
    for Polar LAEA data in HDF5 files, with recomputed, correct coordinates.

    Input parameters:
        --FILE_URLS: Path to an HDF5 file

        --OUTPUT_DIR: (optional) Path to the output directory where the corrected file will be written.
            If not provided, the current working directory will be used.
"""
import argparse
import os
import shutil

import numpy as np

from pymods import MaskFillUtil
from pymods.Ease2Grid import \
    Ease2Grid, \
    Ease2GridResolution, \
    Ease2GridType

lat_array, long_array = [[], []], [[], []]  # (2) 2D ([x, y]) grids


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--FILE_URLS', dest='input_file', help='Name of the input file to fix')
    parser.add_argument('--OUTPUT_DIR', dest='output_dir', help='Name of the output directory to put the output file',
                        default=os.getcwd())
    parser = parser.parse_args()

    input_file_path = parser.input_file
    new_file_path = MaskFillUtil.get_masked_file_path(input_file_path, parser.output_dir)
    shutil.copy(input_file_path, new_file_path)

    MaskFillUtil.process_h5_file(new_file_path, fill_ease2grid_coords)
    print("Output file:", new_file_path)


def fill_ease2grid_coords(h5_dataset):
    """ If the dataset contains latitude or longitude coordinates, overwrites any
        corrupted data with re-computed, correct values.
        Args:
            h5_dataset (h5py._hl.dataset.Dataset): The given HDF5 dataset
    """
    if 'Polar' in h5_dataset.parent.name:
        if 'latitude' in h5_dataset.name:
            data = h5_dataset[:]
            corrected_data = MaskFillUtil.apply_2D(data, get_latitude)
            h5_dataset.write_direct(corrected_data)

        elif 'longitude' in h5_dataset.name:
            data = h5_dataset[:]
            corrected_data = MaskFillUtil.apply_2D(data, get_longitude)
            h5_dataset.write_direct(corrected_data)


def get_latitude(h5_dataset):
    """ Returns the re-computed latitude coordinate array corresponding to the dataset.
        Args:
            h5_dataset (h5py._hl.dataset.Dataset): The given HDF5 dataset
        Returns:
            numpy.ndarray: The latitude array
    """
    if lat_array is None:
        get_lat_long_arrays(h5_dataset)

    return lat_array


def get_longitude(h5_dataset):
    """ Returns the uncorrupted longitude coordinate array corresponding to the dataset.
        Args:
            h5_dataset (h5py._hl.dataset.Dataset): The given HDF5 dataset
        Returns:
            numpy.ndarray: The longitude array
    """
    if long_array is None:
        get_lat_long_arrays(h5_dataset)

    return long_array


def get_lat_long_arrays(h5_dataset):
    """ Re-Calculates the latitude and longitude coordinate arrays corresponding to the dataset.
        Args:
            h5_dataset (h5py._hl.dataset.Dataset): The given HDF5 dataset
    """
    lat_arr, long_arr = np.zeros(h5_dataset.shape), np.zeros(h5_dataset.shape)
    ease2grid = Ease2Grid(Ease2GridResolution.r_36K,
                          Ease2GridType.t_polar)
    rows, cols = h5_dataset.shape
    global lat_array, long_array

    # Get x and y projected meters corresponding to each row and column in image array
    for i in range(rows):
        for j in range(cols):
            # Projected meters of center of cell
            x_meters, y_meters = ease2grid.affine_transform * (i + 0.5, j + 0.5)

            # Get lat/long values from projected meters
            lat, long = ease2grid.crs_transform(x_meters, y_meters)

            lat_array[i][j] = lat
            long_array[i][j] = long

    return lat_array, long_array


if __name__ == "__main__":
    main()
