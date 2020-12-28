""" Utility functions to support MaskFill processing
"""
from typing import Tuple
import hashlib
import os

from affine import Affine
from geopandas import GeoSeries
from osgeo import osr
from pyproj import Transformer, CRS
from rasterio import features  # as in geographic features, shapes, polygons
from shapely.geometry import Polygon
import gdal
import geopandas as gpd
import h5py
import numpy as np
import rasterio

from pymods import H5GridProjectionInfo
from pymods.cf_config import CFConfigH5


def get_mask_array(shape_path, proj4, out_shape, transform):
    """ Rasterizes the intersection of the given shapes and the bounding box of
        the data to create a mask array.

        Args:
            shape_path (string): The path to the shape file
            proj4 (string): The proj4 string corresponding to the target CRS
            out_shape (tuple): The shape of the resultant mask array
            transform (affine.Affine): A transform mapping from image coordinates
            to world coordinates
        Returns:
            numpy.ndarray: A numpy array representing the rasterized shapes
    """
    epsg = CRS(proj4).to_epsg()
    bounded_shape_gdf = get_bounded_shape(shape_path, epsg, proj4, out_shape, transform)

    # Project data frame to new coordinate reference system
    projected_gdf = bounded_shape_gdf.to_crs(proj4)
    shapes = projected_gdf['geometry']

    # Rasterize the bounded and projected shapes into the mask array
    if shapes.is_empty.empty:
        return np.ones(out_shape)

    mask = features.rasterize(shapes=shapes, default_value=0, fill=1, out_shape=out_shape,
                              dtype=np.uint8, transform=transform, all_touched=True)
    return mask


def get_bounded_shape(shape_path, epsg, proj4, out_shape, transform):
    """ Creates a geodataframe (in geographic coordinates) for the shapes in the shape file.
    Bounds the shapes by the geographic extent of the data.

            Args:
                shape_path (string): The path to the shape file
                epsg (int): The epsg code of the data
                proj4 (string): The proj4 string corresponding to the target CRS
                out_shape (tuple): The shape of the resultant mask array
                transform (affine.Affine): A transform mapping from image coordinates
                to world coordinates
            Returns:
                geodataframe: The bounded shape geodataframe
        """

    if epsg is not None:
        # Get geographic extent of data using the EPSG code
        minx, miny, maxx, maxy = CRS(epsg).area_of_use.bounds

    # Transform all indices in the data array to geographic coordinates
    # and get min/max lat/lon values
    else:
        y_vals, x_vals = np.indices(out_shape)
        indices = [x_vals.flatten(), y_vals.flatten()]

        # Decompose the affine transform
        transform_matrix = np.array(transform).reshape(3, 3)
        A, b = transform_matrix[:2, :2], transform_matrix[:2, 2:]
        crs_coors = np.matmul(A, indices) + b  # coordinates in the original CRS

        to_geo_trans = Transformer.from_proj(proj4, 'EPSG:4326')
        y_geo, x_geo = to_geo_trans.transform(crs_coors[0, :], crs_coors[1, :])

        x_geo = x_geo[np.isfinite(x_geo)]
        y_geo = y_geo[np.isfinite(y_geo)]

        minx, maxx = np.min(x_geo), np.max(x_geo)
        miny, maxy = np.min(y_geo), np.max(y_geo)

    # Create bounding box in geographic coordinates
    bbox = [(minx, miny), (maxx, miny), (maxx, maxy), (minx, maxy)]
    bbox_gdf = gpd.GeoDataFrame(geometry=GeoSeries(Polygon(bbox), crs={'init': 'epsg:4326'}))

    # Intersect shapes with bounding box
    shape_gdf = gpd.read_file(shape_path)
    bounded_shape_gdf = gpd.overlay(shape_gdf, bbox_gdf, how='intersection')
    return bounded_shape_gdf


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


def get_h5_mask_array_id(h5_dataset: h5py.Dataset, shape_path: str,
                         cf_config: CFConfigH5) -> str:
    """ Creates an ID corresponding to the given shape file, projection
        information, pixel-to-projected coordinates Affine transformation
        inputs, and shape of a dataset, which determine the mask array for
        the dataset.

        Args:
            h5_dataset: The given HDF5 dataset
            shape_path: Path to a shape file used to create the mask array
                for the mask fill.
            shortname: The short form name of the granule collection.

        Returns:
            str: A string ID generated via a hashing algorithm, based upon the
                a combined input string of the shape file path, dataset shape,
                dataset projection and Affine transformation.
    """
    proj_string = H5GridProjectionInfo.get_hdf_proj4(h5_dataset, cf_config)
    transform = H5GridProjectionInfo.get_transform_information(h5_dataset)
    dataset_shape = h5_dataset[:].shape

    return create_mask_array_id(proj_string, transform, dataset_shape, shape_path)


def get_geotiff_mask_array_id(geotiff_path: str, shape_path: str) -> str:
    """ Creates an ID corresponding to the given shape file, projection
    information and shape of a dataset, which determine the mask array for the
    dataset.

    Args:
        geotiff_path (str): Path to the GeoTIFF dataset.
        shape_path (str): Path to a shape file used to create the mask array
            for the mask fill.

        Returns:
            str: A string ID generated via a hashing algorithm, based upon the
                a combined input string of the shape file path, dataset shape,
                dataset projection and Affine transformation.
    """
    # The mask array is determined by the CRS of the dataset, the dataset's
    # transform, the shape of the dataset, and the shapes used in the mask.
    proj_string = get_geotiff_proj4(geotiff_path)
    dataset_shape, transform = get_geotiff_info(geotiff_path)

    return create_mask_array_id(proj_string, transform, dataset_shape, shape_path)


def create_mask_array_id(proj_string, transform, dataset_shape, shape_file_path):
    """ Creates an id corresponding to the given shapefile, projection information, and shape of a dataset,
        which determine the mask array for the dataset.
        Args:
            proj_string (str): A proj4 string which describes the projection information of the data
            transform (affine.Affine): The affine transform corresponding to the data
            dataset_shape (tuple): The shape of the data
            shape_file_path (str): The path to the given shapefile
        Returns:
            str: A string ID generated via a hashing algorithm, based upon the
                a combined input string of the shape file path, dataset shape,
                dataset projection and Affine transformation.
    """
    mask_id = proj_string + str(transform) + str(dataset_shape) + shape_file_path

    # Hash the mask id and return
    mask_id = hashlib.sha224(mask_id.encode()).hexdigest()
    return mask_id


def get_geotiff_proj4(geotiff_path):
    """ Returns the proj4 string corresponding to the coordinate reference
    system of the GeoTIFF file.

    Args:
        geotiff_path (str): The path to the GeoTIFF file

    Returns:
        str: The proj4 string corresponding to the given file.
    """
    data = gdal.Open(geotiff_path)
    proj_text = data.GetProjection()
    srs = osr.SpatialReference()
    srs.ImportFromWkt(proj_text)
    return srs.ExportToProj4()


def get_geotiff_info(geotiff_path: str) -> Tuple[Tuple[int], Affine]:
    """ Retuns the shape and transform of the given GeoTIFF.

    Args:
        geotiff_path (str): The path to the GeoTIFF

    Returns:
        tuple: The shape (tuple) and transform (affine.Affine) corresponding to
            the GeoTIFF.
    """
    with rasterio.open(geotiff_path) as raster:
        shape = raster.read(1).shape
        transform = raster.transform

    return shape, transform


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
            if isinstance(child, h5py.Group):
                process_children(child, process, *args)
            # Process datasets
            elif isinstance(child, h5py.Dataset):
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
