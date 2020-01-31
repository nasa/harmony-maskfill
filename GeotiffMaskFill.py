""" Provides Maskfill support for GeoTIFF files
"""
import logging

import gdal
import numpy as np
import rasterio
import rasterio.mask
from osgeo import gdal_array
from osgeo import osr

from pymods import MaskFillUtil, MaskFillCaching


def produce_masked_geotiff(geotiff_path, shape_path, output_dir, cache_dir, mask_grid_cache, default_fill_value):
    """ Performs a mask fill on the given GeoTIFF using the shapes in the given shapefile.
        Writes the resulting GeoTIFF to output_dir.
        Args:
            geotiff_path (str): The path to the GeoTIFF
            shape_path (str): The path to the shape file
            output_dir (str): The path to the output directory
            default_fill_value (float): The fill value used for the mask fill if the GeoTIFF has no fill value
        Returns:
            str: The path to the output GeoTIFF file
    """
    mask_grid_cache = mask_grid_cache.lower()
    mask_array = get_mask_array(geotiff_path, shape_path, cache_dir, mask_grid_cache)

    if mask_grid_cache == 'maskgrid_only':
        return None

    # Perform mask fill
    raster_arr, fill_value = gdal_array.LoadFile(geotiff_path), get_fill_value(geotiff_path, default_fill_value)
    out_image = MaskFillUtil.mask_fill_array(raster_arr, mask_array, fill_value)
    out_image = np.array([out_image])

    # Output file with proper metadata
    output_path = MaskFillUtil.get_masked_file_path(geotiff_path, output_dir)
    out_meta = rasterio.open(geotiff_path).meta.copy()
    out_meta.update({"driver": "GTiff", "height": out_image.shape[1], "width": out_image.shape[2]})
    with rasterio.open(output_path, "w", **out_meta) as dest:
        dest.write(out_image)

    logging.debug('Successfully created masked GeoTIFF and updated metadata')

    return MaskFillUtil.get_masked_file_path(geotiff_path, output_dir)


def get_mask_array(geotiff_path, shape_path, cache_dir, mask_grid_cache):
    """ Gets the mask array corresponding the GeoTIFF file and shape file from the cache directory (if the mask grid cache
        value allows).
        If the mask array file does not already exist, it is created and cached (if the mask grid cache value allows).
        Args:
            geotiff_path (str): The path to the GeoTIFF
            shape_path (str): The path to the shape file
            cache_dir (str): The path to the cache directory
            default_fill_value (float): The fill value used for the mask fill if the GeoTIFF has no fill value
        Returns:
            numpy.ndarray: The mask array
    """
    mask_array = MaskFillCaching.get_cached_mask_array(geotiff_path, shape_path, cache_dir, mask_grid_cache)

    if mask_array is None:
        mask_array = create_mask_array(geotiff_path, shape_path)
        MaskFillCaching.cache_mask_array(mask_array, geotiff_path, shape_path, cache_dir, mask_grid_cache)

    return mask_array


def create_mask_array(geotiff_path, shape_path):
    """ Rasterizes the shapes in the given shape file to create a mask array for the given GeoTIFF.
        Args:
            geotiff_path (str): The GeoTIFF for which a mask array will be created
            shape_path (str): The path to the shape file which will be rasterized
        Returns:
            numpy.ndarray: A numpy array representing the rasterized shapes from the shape file
    """
    projected_shapes = MaskFillUtil.get_projected_shapes(get_geotiff_proj4(geotiff_path), shape_path)
    raster = rasterio.open(geotiff_path)
    return MaskFillUtil.get_mask_array(projected_shapes, raster.read(1).shape, raster.transform)


def get_geotiff_proj4(geotiff_path):
    """ Returns the proj4 string corresponding to the coordinate reference system of the GeoTIFF file.
        Args:
            geotiff_path (str): The path to the GeoTIFF file
        Returns:
            str: The proj4 string corresponding to the given file
    """
    data = gdal.Open(geotiff_path)
    proj_text = data.GetProjection()
    srs = osr.SpatialReference()
    srs.ImportFromWkt(proj_text)
    return srs.ExportToProj4()


def get_fill_value(geotiff_path, default_fill_value):
    """ Returns the fill value for the given GeoTIFF.
        If the GeoTIFF has no fill value, returns the given default fill value.
        Args:
            geotiff_path (str): The path to a GeoTIFF file
            default_fill_value (float): The default value which is returned if no fill value is found in the GeoTIFF
        Returns:
            float: The fill value
    """
    raster = gdal.Open(geotiff_path)
    fill_value = raster.GetRasterBand(1).GetNoDataValue()

    if fill_value is None:
        logging.info(f'The GeoTIFF does not have a fill value, '
                     f'so the default fill value {default_fill_value} will be used')
        return default_fill_value
    return fill_value


def get_mask_array_id(geotiff_path, shape_path):
    """ Creates an id corresponding to the given shapefile, projection information, and shape of a dataset,
        which determine the mask array for the dataset.
        Args:
            geotiff_path (str): The given path to a geotiff file
            shape_path (str): Path to a shape file used to create the mask array for the mask fill
        Returns:
            str: The id
    """
    # The mask array is determined by the CRS of the dataset, the dataset's transform, the shape of the dataset,
    # and the shapes used in the mask
    proj_string = get_geotiff_proj4(geotiff_path)
    dataset_shape, transform = get_geotiff_info(geotiff_path)

    return MaskFillCaching.create_mask_array_id(proj_string, transform, dataset_shape, shape_path)


def get_geotiff_info(geotiff_path):
    """ Returns the shape and transform of the given GeoTIFF
        Args:
            geotiff_path (str): The path to the GeoTIFF
        Returns:
            tuple: The shape (tuple) and transform (affine.Affine) corresponding to the GeoTIFF"""
    raster = rasterio.open(geotiff_path)
    shape = raster.read(1).shape
    transform = raster.transform

    return shape, transform
