import affine
import numpy as np
import shutil
import time
from pyproj import Transformer, CRS
from pymods import H5MaskFill
from pymods import MaskFill


polar_proj = CRS.from_proj4("+ellps=WGS84 +datum=WGS84 +proj=laea +lat_0=90")
lat_long_proj = CRS.from_epsg(4326)
proj_trans = Transformer.from_crs(polar_proj, lat_long_proj)

lat_array, long_array = None, None


""" Overwrites corrupted latitude and longitude coordinate data with uncorrupted versions. """
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

def get_latitude(dataset):
    if lat_array is None: get_lat_long_arrays(dataset)
    return lat_array

def get_longitude(dataset):
    if long_array is None: get_lat_long_arrays(dataset)
    return long_array

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

""" Transform from image coordinates to projected meters."""
def get_transform(dataset):
    x_min, y_max = -9000000, 9000000

    if dataset.shape == (500, 500): cell_width, cell_height = 36000, -36000
    if dataset.shape == (2000, 2000): cell_width, cell_height = 9000, -9000
    if dataset.shape == (6000, 6000): cell_width, cell_height = 3000, -3000

    t = affine.Affine(cell_width, 0, x_min, 0, cell_height, y_max)
    print(t)
    return t

def proj_meters_to_lat_long(x, y):
    return proj_trans.transform(x, y)


if __name__ == '__main__':
    file_path = "/Users/jrao/Downloads/SMAP_L3_FT_P_20180618_R16010_001.h5"
    new_file_path = MaskFill.get_masked_file_path(file_path, "/Users/jrao/Documents/")
    shutil.copy(file_path, new_file_path)

    start_time = time.time()
    H5MaskFill.process_file(new_file_path, uncorrupt_coordinates)
    print("Total time:", time.time() - start_time)