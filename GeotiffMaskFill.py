""" Provides Maskfill support for GeoTIFF files """
from logging import Logger
from os.path import basename
from shutil import copyfile
from typing import List
import re

import gdal
import numpy as np
import rasterio
import rasterio.mask
from osgeo import osr

from pymods import MaskFillUtil
from pymods.cf_config import CFConfigGeotiff
from pymods.MaskFillCaching import (cache_geotiff_mask_array,
                                    get_geotiff_cached_mask_array)


def produce_masked_geotiff(geotiff_path: str, shape_path: str, output_dir: str,
                           cache_dir: str, mask_grid_cache: str,
                           default_fill_value: float, logger: Logger) -> str:
    """ Performs a mask fill on the given GeoTIFF using the shapes in the giveni
        shape file. If the variable is a coordinate, or EASE-2 grid index, then
        the input file is copied directly to the output, unmasked.

        Writes the resulting GeoTIFF to output_dir.

        Args:
            geotiff_path (str): The path to the GeoTIFF
            shape_path (str): The path to the shape file
            output_dir (str): The path to the output directory
            default_fill_value (float): The fill value used for the mask fill
                if the GeoTIFF has no fill value

        Returns:
            str: The path to the output GeoTIFF file

    """
    mask_grid_cache = mask_grid_cache.lower()
    cf_config = CFConfigGeotiff(geotiff_path)
    exclusions = [convert_variable_path(exclusion_path)
                  for exclusion_path in cf_config.get_file_exclusions()]

    mask_array = get_mask_array(geotiff_path, shape_path, cache_dir,
                                mask_grid_cache)

    output_path = MaskFillUtil.get_masked_file_path(geotiff_path, output_dir)

    if mask_grid_cache == 'maskgrid_only':
        return None

    if variable_should_be_masked(geotiff_path, exclusions):
        input_dataset = gdal.Open(geotiff_path)

        fill_value = get_fill_value(input_dataset, default_fill_value, logger)
        compression = (input_dataset.GetMetadata('IMAGE_STRUCTURE')
                       .get('COMPRESSION', None))

        # Raster band indices in gdal.Dataset are 1-based, range is 0-based.
        out_image = np.array([
            MaskFillUtil.mask_fill_array(
                input_dataset.GetRasterBand(band + 1).ReadAsArray(),
                mask_array,
                fill_value
            )
            for band in range(input_dataset.RasterCount)
        ])

        # Output file with updated metadata
        out_meta = rasterio.open(geotiff_path).meta.copy()
        out_meta.update({'compress': compression,
                         'driver': 'GTiff',
                         'height': out_image.shape[1],
                         'width': out_image.shape[2]})

        with rasterio.open(output_path, 'w', **out_meta) as dest:
            dest.write(out_image)

        logger.debug('Successfully created masked GeoTIFF and updated '
                     'metadata')
    else:
        # Copy input file to output path
        logger.debug(f'{geotiff_path} matches coordinate exclusion rule; '
                     f'copying input to {output_path}')
        copyfile(geotiff_path, output_path)

    return output_path


def get_mask_array(geotiff_path: str, shape_path: str, cache_dir: str,
                   mask_grid_cache: str) -> np.ndarray:
    """ Gets the mask array corresponding the GeoTIFF file and shape file from
        the cache directory (if the mask grid cache value allows).

        If the mask array file does not already exist, it is created and cached
        (if the mask grid cache value allows).

        Args:
            geotiff_path (str): The path to the GeoTIFF
            shape_path (str): The path to the shape file
            cache_dir (str): The path to the cache directory
            default_fill_value (float): The fill value used for the mask fill
                if the GeoTIFF has no fill value

        Returns:
            numpy.ndarray: The mask array

    """
    mask_array = get_geotiff_cached_mask_array(geotiff_path, shape_path,
                                               cache_dir, mask_grid_cache)

    if mask_array is None:
        mask_array = create_mask_array(geotiff_path, shape_path)
        cache_geotiff_mask_array(mask_array, geotiff_path, shape_path,
                                 cache_dir, mask_grid_cache)

    return mask_array


def create_mask_array(geotiff_path: str, shape_path: str) -> np.ndarray:
    """ Rasterizes the shapes in the given shape file to create a mask array
        for the given GeoTIFF.

        Args:
            geotiff_path (str): The GeoTIFF for which a mask array will be created
            shape_path (str): The path to the shape file which will be rasterized
        Returns:
            numpy.ndarray: A numpy array representing the rasterized shapes
                from the shape file

    """
    raster = rasterio.open(geotiff_path)
    proj4 = get_geotiff_proj4(geotiff_path)

    return MaskFillUtil.get_mask_array(shape_path, proj4, raster.read(1).shape,
                                       raster.transform)


def get_geotiff_proj4(geotiff_path: str) -> str:
    """ Returns the proj4 string corresponding to the coordinate reference
        system of the GeoTIFF file.

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


def get_fill_value(geotiff_dataset: gdal.Dataset, default_fill_value: float,
                   logger: Logger) -> float:
    """ Returns the fill value for the given GeoTIFF.
        If the GeoTIFF has no fill value, returns the given default fill value.

        Args:
            geotiff_path (str): The path to a GeoTIFF file
            default_fill_value (float): The default value which is returned if
                no fill value is found in the GeoTIFF

        Returns:
            float: The fill value

    """
    fill_value = geotiff_dataset.GetRasterBand(1).GetNoDataValue()

    if fill_value is None:
        logger.info('The GeoTIFF does not have a fill value, so the default '
                    f'fill value {default_fill_value} will be used')
        return default_fill_value

    return fill_value


def variable_should_be_masked(geotiff_path: str,
                              exclusions: List[str]) -> bool:
    """ Compare the GeoTIFF file name, which will contain the full variable
        path (with underscores in place of forward slashes), to the full list
        of collection variables that should be excluded from masking.

        Any potential directory structure is omitted, in case this causes a
        false positive match.
    """
    geotiff_base_path = basename(geotiff_path)

    return not any([re.search(exclusion, geotiff_base_path)
                    for exclusion in exclusions])


def convert_variable_path(variable_path: str) -> str:
    """ Take a full variable path, e.g. `/group_one/group_two/variable_name`,
        and convert it to be compatible with a GeoTIFF file name for that
        variable, e.g. `_group_one_group_two_variable_name`.

    """
    return variable_path.replace('/', '_').replace('.', '_')
