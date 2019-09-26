import geopandas as gpd
from rasterio import features
import numpy as np
import os


""" Projects the shapes in the given shapefile to a new coordinate reference system.

    Args:
        proj4 (string): The proj4 string representing the new coordinate reference system
        shape_path (string): The path to the shape file which is to be converted

    Returns: 
        geopandas.geoseries.GeoSeries: A geopandas GeoSeries containing the projected shape information.
"""
def get_projected_shapes(proj4, shape_path):
    try:
        shape_gdf = gpd.read_file(shape_path)  # Load shape file as geopandas data frame
    except:
        raise ValueError('Shape data cannot be read from the given shapefile.')

    projected_gdf = shape_gdf.to_crs(proj4)  # Project data frame to new coordinate reference system
    return projected_gdf['geometry']


""" Rasterizes the given shapes to create a mask array.

    Args:
        shapes (geopandas.geoseries.GeoSeries): The shapes to be rasterized
        out_shape (tuple): The shape of the resultant mask array
        transform (affine.Affine): A transform mapping from image coordinates to world coordinates

    Returns:
        numpy.ndarray: A numpy array representing the rasterized shapes
"""
def get_mask_array(shapes, out_shape, transform):
    mask = features.rasterize(shapes=shapes, default_value=0, fill=1, out_shape=out_shape,
                              dtype=np.uint8, transform=transform, all_touched=True)
    return mask


""" Performs a mask fill on raster_arr using the mask mask_array

    Args:
        raster_arr (numpy.ndarray): The array to be mask filled
        mask_array (numpy.ndarray): The mask array which will be applied to raster_arr
        fill_value (float): Value used to fill in the masked values when necessary

    Returns:
        numpy.ndarray: The mask filled array
"""
def mask_fill_array(raster_arr, mask_array, fill_value):
    out_image = np.ma.array(raster_arr, mask=mask_array, fill_value=fill_value)
    return out_image.filled()


""" Returns the path to the mask filled output file.

    Args:
        original_file_path (str): The original file which is to be mask filled
        output_dir (str): The directory to which the output file will be written

    Returns: 
        str: The path to the mask filled version of the original file

"""
def get_masked_file_path(original_file_path, output_dir):
    base_name = os.path.basename(original_file_path)
    file_name, extension = os.path.splitext(base_name)
    return os.path.join(output_dir, file_name + "_mf" + extension)


