""" Utility functions to support MaskFill processing """
from logging import Logger
from typing import Tuple, Union
from warnings import catch_warnings, simplefilter
import hashlib
import os

from affine import Affine
from geopandas import GeoDataFrame, GeoSeries
from h5py import Dataset, File as H5File, Group
from osgeo import gdal
from pyproj import Transformer, CRS
from rasterio.features import rasterize
from shapely.geometry import Polygon
import geopandas as gpd
import numpy as np
import rasterio

from pymods import H5GridProjectionInfo
from pymods.cf_config import CFConfigH5


def get_mask_array(shape_path: str, crs: CRS, out_shape: Tuple[int, int],
                   transform: Affine) -> np.ndarray:
    """ Rasterizes the intersection of the given shapes and the bounding box of
        the data to create a mask array.

        Args:
            shape_path (string): The path to the shape file
            crs (pyproj.CRS): The target CRS
            out_shape (tuple): The shape of the resultant mask array
            transform (affine.Affine): A transform mapping from image coordinates
            to world coordinates
        Returns:
            numpy.ndarray: A numpy array representing the rasterized shapes
    """
    bounded_shape_gdf = get_bounded_shape(shape_path, crs, out_shape,
                                          transform)

    # Project data frame to new coordinate reference system
    projected_gdf = bounded_shape_gdf.to_crs(crs)
    shapes = projected_gdf['geometry']

    # Rasterize the bounded and projected shapes into the mask array
    if shapes.is_empty.empty:
        return np.ones(out_shape)

    return rasterize(shapes=shapes, default_value=0, fill=1,
                     out_shape=out_shape, dtype=np.uint8, transform=transform,
                     all_touched=True)


def get_bounded_shape(shape_path: str, crs: CRS, out_shape: Tuple[int, int],
                      transform: Affine) -> gpd.GeoDataFrame:
    """ Creates a geodataframe (in geographic coordinates) for the shapes in
        the shape file. Bounds the shapes by the geographic extent of the data.

            Args:
                shape_path (string): The path to the shape file
                crs (pyproj.CRS): CRS of the target grid.
                out_shape (tuple): The shape of the resultant mask array
                transform (affine.Affine): A transform mapping from image coordinates
                to world coordinates
            Returns:
                geodataframe: The bounded shape geodataframe

        `pyproj.CRS` will not necessarily populate an Area Of Use for non-EPSG
        defined CRS objects, so if a CRS has an EPSG code, this will need to be
        used to instantiate a new CRS with an area of use.

    """
    epsg = crs.to_epsg()

    if epsg is not None and not should_ignore_pyproj_bounds(crs):
        # Get geographic extent of data using the EPSG code
        minx, miny, maxx, maxy = CRS(epsg).area_of_use.bounds

    else:
        # Transform all indices in the data array to geographic coordinates
        # and get min/max lat/lon values
        y_vals, x_vals = np.indices(out_shape)
        indices = [x_vals.flatten(), y_vals.flatten()]

        # Decompose the affine transform
        transform_matrix = np.array(transform).reshape(3, 3)
        A, b = transform_matrix[:2, :2], transform_matrix[:2, 2:]
        crs_coors = np.matmul(A, indices) + b  # coordinates in the original CRS

        to_geo_trans = Transformer.from_crs(crs, 4326)
        y_geo, x_geo = to_geo_trans.transform(crs_coors[0, :], crs_coors[1, :])

        x_geo = x_geo[np.isfinite(x_geo)]
        y_geo = y_geo[np.isfinite(y_geo)]

        minx, maxx = np.min(x_geo), np.max(x_geo)
        miny, maxy = np.min(y_geo), np.max(y_geo)

    # Create bounding box in geographic coordinates
    bbox = [(minx, miny), (maxx, miny), (maxx, maxy), (minx, maxy)]
    bbox_gdf = GeoDataFrame(geometry=GeoSeries(Polygon(bbox), crs='EPSG:4326'))

    # Intersect shapes with bounding box
    shape_gdf = gpd.read_file(shape_path)
    bounded_shape_gdf = gpd.overlay(shape_gdf, bbox_gdf, how='intersection')
    return bounded_shape_gdf


def should_ignore_pyproj_bounds(crs: CRS) -> bool:
    """ A function to check if the CRS is a projection that should not use the
        bounds defined by the `pyproj.CRS` object. For example, strict UTM
        zones are constrained to 6 degree-wide longitudinal regions, but users
        may have shapes that transcend these regions.

        `pyproj` gives a warning when converting CRS objects to a Proj4 string,
        however, we do not require the additional information that may be lost,
        so those warnings are suppressed in this function.

    """
    with catch_warnings():
        simplefilter('ignore')
        proj4_string = crs.to_proj4()

    return '+proj=utm' in proj4_string


def mask_fill_array(raster_arr: np.ndarray, mask_array: np.ndarray,
                    fill_value: float) -> np.ndarray:
    """ Performs a mask fill on raster_arr using the mask mask_array
        Args:
            raster_arr: The array to be masked.
            mask_array: The mask array which will be applied to raster_arr
            fill_value (float): Value used to fill in the masked values when necessary
        Returns:
            numpy.ndarray: The mask filled array
    """

    out_image = np.ma.array(raster_arr, mask=mask_array, fill_value=fill_value)
    return out_image.filled()


def get_masked_file_path(original_file_path: str, output_dir: str) -> str:
    """ Returns the path to the mask filled output file.
        Args:
            original_file_path (str): The original file which is to be mask filled
            output_dir (str): The directory to which the output file will be written
        Returns:
            str: The path to the mask filled version of the original file

    """
    base_name = os.path.basename(original_file_path)
    file_name, extension = os.path.splitext(base_name)
    return os.path.join(output_dir, f'{file_name}_mf{extension}')


def get_h5_mask_array_id(h5_dataset: Dataset, crs: CRS,
                         shape_path: str) -> str:
    """ Creates an ID corresponding to the given shape file, projection
        information, pixel-to-projected coordinates Affine transformation
        inputs, and shape of a dataset, which determine the mask array for
        the dataset.

        Returns:
            str: A string ID generated via a hashing algorithm, based upon the
                a combined input string of the shape file path, dataset shape,
                dataset projection and Affine transformation.
    """
    transform_info = H5GridProjectionInfo.get_transform_information(h5_dataset)
    dataset_shape = h5_dataset[:].shape

    return create_mask_array_id(crs, transform_info, dataset_shape, shape_path)


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
    crs = get_geotiff_crs(geotiff_path)
    dataset_shape, transform = get_geotiff_info(geotiff_path)

    return create_mask_array_id(crs, transform, dataset_shape, shape_path)


def create_mask_array_id(crs: CRS, transform: Union[str, Affine],
                         dataset_shape: Tuple[int],
                         shape_file_path: str) -> str:
    """ Creates an ID corresponding to the given shapefile, projection
        information (CRS and affine transformation), and shape of a dataset.

        Args:
            crs: A `pyproj.CRS` object describing the projection of the data.
            transform:
              - GeoTIFF (affine.Affine): The affine transform for the grid.
              - HDF-5 (str): A string describing the affine transform.
            dataset_shape (tuple): The shape of the data array.
            shape_file_path (str): The path to the given shapefile.
        Returns:
            str: A string ID generated via a hashing algorithm, based upon the
                a combined input string of the shape file path, dataset shape,
                dataset projection and Affine transformation.
    """
    mask_id = (f'{crs.to_string()}{str(transform)}{str(dataset_shape)}'
               f'{shape_file_path}')

    return hashlib.sha224(mask_id.encode()).hexdigest()


def get_geotiff_crs(geotiff_path: str) -> CRS:
    """ Returns a `pyproj.crs.Crs` object that is constructed from the Well
        Known Text (WKT) representation of the GeoTIFF projection information.

    """
    data = gdal.Open(geotiff_path)
    wkt_string = data.GetProjection()
    return CRS.from_wkt(wkt_string)


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
            if isinstance(child, Group):
                process_children(child, process, *args)
            # Process datasets
            elif isinstance(child, Dataset):
                process(child, *args)

    with H5File(file_path, mode='r+') as file:
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

    # For more than two dimensions, mask fill each dimension recursively
    for i in range(len(data)):
        data[i] = apply_2D(data[i], process, *args)

    return data
