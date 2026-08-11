#!/usr/bin/env python
""" Applies a fill value to the areas of a data file that fall outside
    a set of shapes defined by a shapefile.
"""
import logging
import os

from maskfill import geotiff_maskfill
from maskfill import h5_maskfill
from maskfill import history


DEFAULT_MASK_GRID_CACHE = 'ignore_and_delete'


def mask_fill(
    input_file: str,
    collection_short_name: str,
    shape_file: str,
    output_dir: str,
    mask_grid_cache: str,
    fill_value: float | None,
    logger: logging.Logger,
    bounding_box=None,
) -> str:
    """Performs a masking operation on the given data file using the supplied shape.

    Fill value are applied to all pixels outside the shape and the masked output
    is written to `output_dir`.

    Returns:
        str: The full path of the masked output file

    """
    # Perform mask fill according to input file type
    input_extension = os.path.splitext(input_file)[1].lower()

    if input_extension == '.tif':
        # GeoTIFF case
        logger.info(f'Performing mask fill with GeoTIFF {input_file} and '
                    f'shapefile {shape_file}')
        output_file = geotiff_maskfill.produce_masked_geotiff(
            input_file,
            collection_short_name,
            shape_file,
            output_dir,
            output_dir,
            mask_grid_cache,
            fill_value,
            logger,
        )
    elif input_extension in ('.h5', '.hdf5', '.nc4'):
        # HDF5 or NetCDF-4 case (HOSS produces NetCDF-4 files)
        logger.info('Performing mask fill with HDF5/NetCDF-4 file '
                    f'{input_file} and shapefile {shape_file}')
        output_file = h5_maskfill.produce_masked_hdf(
            input_file,
            collection_short_name,
            shape_file,
            output_dir,
            output_dir,
            mask_grid_cache,
            fill_value,
            logger,
        )

        history.update_history_metadata(
            output_file,
            shape_file,
            fill_value,
            bounding_box
        )

    return output_file
