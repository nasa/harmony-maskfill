from logging import getLogger
from os import mkdir
from os.path import isdir
from shutil import copy, rmtree
from unittest import TestCase

from osgeo import gdal

from maskfill.GeotiffMaskFill import (
    convert_variable_path,
    get_fill_value,
    get_geotiff_variable_type,
    variable_should_be_masked,
)


class TestGeotiffMaskfill(TestCase):
    """ Unit tests for GeotiffMaskfill branch. These are not end-to-end. """

    @classmethod
    def setUpClass(cls):
        """ Define test fixtures that can be reused between tests. """
        cls.logger = getLogger('tests')

    def setUp(self):
        """ Define test fixtures that shold be unique for each test. """
        self.output_dir = 'tests/output'
        mkdir(self.output_dir)

    def tearDown(self):
        """Clean up test artefacts after each test."""
        if isdir(self.output_dir):
            rmtree(self.output_dir)

    def test_convert_variable_path(self):
        """ Ensure that the full variable path regular expression is correctly
            converted to remove slashes and full stops.

        """
        self.assertEqual(convert_variable_path('/group/variable.band'),
                         '_group_variable_band')

    def test_variable_should_be_masked(self):
        """ Ensure the correct determination is made as to whether a variable
            should be masked or not, as indicated by its GeoTIFF file path and
            the available list of exclusions.

        """
        exclusions = ['latitude', 'longitude']
        test_args = [['No matches', 'SMAP_L3_variable.tif', exclusions, True],
                     ['A match', 'SMAP_L3_latitude.tif', exclusions, False],
                     ['No exclusions', 'SMAP_L3_latitude.tif', [], True]]

        for description, geotiff_path, excluding, should_exclude in test_args:
            with self.subTest(description):
                self.assertEqual(
                    variable_should_be_masked(geotiff_path, excluding),
                    should_exclude
                )

    def test_get_fill_value(self):
        """ Ensure the correct fill value is retrieved. The following places
            should be checked, in descending order of precedence:

            * In-file nodata value.
            * User-specified default fill value.
            * Variable type default fill value (e.g., -9999.0 for floats).

        """
        without_nodata = gdal.Open('tests/data/SMAP_L3_FT_P_banded_input.tif')
        user_supplied_fill = 1234.5
        infile_nodata_value = 5432.1

        # Create copy of GeoTIFF and add nodata value to it:
        with_nodata = copy(
            'tests/data/SMAP_L3_FT_P_banded_input.tif',
            self.output_dir
        )
        geotiff_nodata = gdal.Open(with_nodata)
        geotiff_nodata.GetRasterBand(1).SetNoDataValue(infile_nodata_value)
        geotiff_nodata.FlushCache()

        with self.subTest('Retrieves nodata value from GeoTIFF'):
            self.assertEqual(
                get_fill_value(geotiff_nodata, user_supplied_fill, self.logger),
                infile_nodata_value
            )

        with self.subTest('No nodata, retrieves end-user supplied default'):
            self.assertEqual(
                get_fill_value(without_nodata, user_supplied_fill, self.logger),
                user_supplied_fill
            )

        with self.subTest('No nodata or end-user fill value, uses type fill'):
            # GeoTIFF variable is a float32, so default fill is -9999.0
            self.assertEqual(
                get_fill_value(without_nodata, None, self.logger),
                -9999.0
            )

    def test_get_geotiff_variable_type(self):
        """ Ensure the correct string representation of a GeoTIFF variable is
            retrieved. If the GeoTIFF has no bands, a value of None should be
            returned.

            The SMAP_L3_FT_P_banded_input.tif has float32 data.

        """
        float32_geotiff = gdal.Open('tests/data/SMAP_L3_FT_P_banded_input.tif')
        self.assertEqual(
            get_geotiff_variable_type(float32_geotiff),
            'float32'
        )
