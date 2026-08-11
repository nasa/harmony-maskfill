from logging import getLogger
from os import makedirs
from os.path import join

from numpy import array, array_equal, where
from osgeo import gdal
import h5py

from maskfill.maskfill import (
    DEFAULT_MASK_GRID_CACHE,
    mask_fill,
)

from tests.utilities import MaskFillTestCase


class TestMaskFill(MaskFillTestCase):

    def setUp(self):
        super().setUp()
        self.logger = getLogger('maskfill.tests')
        self.input_geotiff_file = 'tests/data/SMAP_L4_SM_aup_input.tif'
        self.input_h5_file = 'tests/data/SMAP_L4_SM_aup_input.h5'
        self.input_short_name = 'SPL4SMAU'
        self.shape_file = 'tests/data/USA.geo.json'
        self.shape_file_south_pole = 'tests/data/south_pole.geo.json'
        self.output_geotiff_file = self.create_output_file_name(self.input_geotiff_file)
        self.output_h5_file = self.create_output_file_name(self.input_h5_file)
        self.output_geotiff_template = 'tests/data/SMAP_L4_SM_aup_output.tif'
        self.output_geotiff_template_south_pole = 'tests/data/SMAP_L3_FT_P_polar_3d_south_pole_output.tif'
        self.output_h5_template_south_pole = 'tests/data/SMAP_L3_FT_P_polar_3d_south_pole_output.h5'
        self.output_h5_template = 'tests/data/SMAP_L4_SM_aup_output.h5'
        self.input_corner_file = 'tests/data/SMAP_L3_FT_P_corners_input.h5'
        self.corner_short_name = 'SPL3FTP'
        self.output_corner_file = self.create_output_file_name(self.input_corner_file)
        self.output_corner_template = 'tests/data/SMAP_L3_FT_P_corners_output.h5'
        self.input_polar_h5_file = 'tests/data/SMAP_L3_FT_P_polar_3d_input.h5'
        self.input_polar_geo_file = 'tests/data/SMAP_L3_FT_P_polar_3d_input.tif'
        self.polar_short_name = 'SPL3FTP'
        self.output_polar_h5_file = self.create_output_file_name(self.input_polar_h5_file)
        self.output_polar_geo_file = self.create_output_file_name(self.input_polar_geo_file)
        self.output_polar_template = 'tests/data/SMAP_L3_FT_P_polar_3d_output.h5'
        self.input_comparison_geo = 'tests/data/SMAP_L4_SM_aup_comparison.tif'
        self.input_comparison_h5 = 'tests/data/SMAP_L4_SM_aup_comparison.h5'
        self.output_comparison_geo = self.create_output_file_name(self.input_comparison_geo)
        self.output_comparison_h5 = self.create_output_file_name(self.input_comparison_h5)

    def run_mask_fill(
        self,
        input_file: str,
        collection_short_name: str,
        shape_file: str,
        fill_value: float | None = None,
        mask_grid_cache: str = DEFAULT_MASK_GRID_CACHE,
    ) -> str:
        """Call `mask_fill` and return the path of the masked output file."""
        working_dir = join(self.output_dir, self.identifier)
        makedirs(working_dir, exist_ok=True)

        return mask_fill(
            input_file,
            collection_short_name,
            shape_file,
            working_dir,
            mask_grid_cache,
            fill_value,
            self.logger,
        )

    def test_mask_fill_h5(self):
        """A full test of the `mask_fill` utility using an HDF-5 input file.
        This checks the returned output path, and then compares the output file
        to a templated output by checking the expected datasets.

        """
        output_file = self.run_mask_fill(
            self.input_h5_file,
            self.input_short_name,
            self.shape_file,
        )

        self.assertEqual(output_file, self.output_h5_file)
        self.compare_h5_files(self.output_h5_template, self.output_h5_file)

    def test_mask_fill_geotiff(self):
        """A full test of the `mask_fill` utility using a GeoTIFF input file.
        This checks the returned output path, and then compares the output file
        to a templated output file by checking the dataset and metadata.

        """
        output_file = self.run_mask_fill(
            self.input_geotiff_file,
            self.input_short_name,
            self.shape_file,
        )

        self.assertEqual(output_file, self.output_geotiff_file)
        self.compare_geotiff_files(self.output_geotiff_template, self.output_geotiff_file)

    def test_mask_fill_h5_extrapolating_corner(self):
        """A full test of the `mask_fill` utility using an HDF-5 input file
        that has filled data in the upper right corner of the longitude and
        latitude arrays.

        """
        output_file = self.run_mask_fill(
            self.input_corner_file,
            self.corner_short_name,
            self.shape_file,
        )

        self.assertEqual(output_file, self.output_corner_file)
        self.compare_h5_files(self.output_corner_template, self.output_corner_file)

    def test_mask_fill_h5_polar_3d(self):
        """A full test of the `mask_fill` utility using an HDF-5 input file
        that contains SMAP L3 FTP polar data. These data are 3-dimensional,
        such that array indices [i, j, k] corresond to:

            - i: data band
            - j: projected x
            - k: projected y

        The data use the NSIDC EASE-2 polar standard grid.

        """
        output_file = self.run_mask_fill(
            self.input_polar_h5_file,
            self.polar_short_name,
            self.shape_file,
        )

        self.assertEqual(output_file, self.output_polar_h5_file)
        self.compare_h5_files(self.output_polar_template, self.output_polar_h5_file)

    def test_mask_fill_compare_h5_geo(self):
        """Run MaskFill over the same input data in both GeoTIFF and HDF-5
        format to ensure the output is consistent between the two methologies.

        """
        shape_file = 'tests/data/comparison.geo.json'

        response_h5 = self.run_mask_fill(
            self.input_comparison_h5,
            self.input_short_name,
            shape_file,
        )
        response_geo = self.run_mask_fill(
            self.input_comparison_geo,
            self.input_short_name,
            shape_file,
        )

        self.assertEqual(response_h5, self.output_comparison_h5)
        self.assertEqual(response_geo, self.output_comparison_geo)

        geo_dataset = gdal.Open(self.output_comparison_geo)
        geo_array = array(geo_dataset.ReadAsArray())

        h5_file = h5py.File(self.output_comparison_h5, 'r')
        h5_array = h5_file['Analysis_Data']['sm_profile_analysis'][:]
        h5_file.close()

        # Initial (fastest) check that the arrays match in size:
        self.assertEqual(h5_array.shape, geo_array.shape)

        # Next check that the same pixels are masked/unmasked
        good_geo = where(geo_array != -9999.0)
        good_h5 = where(h5_array != -9999.0)
        self.assertTrue(array_equal(good_h5[0], good_geo[0]))
        self.assertTrue(array_equal(good_h5[1], good_geo[1]))

        # Finally, check all pixel values are identical (slowest check)
        self.assertTrue(array_equal(h5_array, geo_array))

    def test_mask_fill_h5_default_fill(self):
        """ Ensure MaskFill can process a file that has no in-file fill value
            metadata, relying instead on default fill values that are selected
            based on the data type of each variable in the HDF-5 file.

        """
        input_file_path = 'tests/data/SMAP_L3_FT_P_fill_input.h5'
        output_file_path = self.create_output_file_name(input_file_path)

        output_file = self.run_mask_fill(
            input_file_path,
            self.input_short_name,
            self.shape_file,
            fill_value=None,
        )

        self.assertEqual(output_file, output_file_path)
        self.compare_h5_files(
            'tests/data/SMAP_L3_FT_P_fill_output.h5',
            output_file_path
        )

    def test_mask_fill_geo_float_default(self):
        """A full test of the `mask_fill` utility using a GeoTIFF input file.
        This specific test ensures that when an input GeoTIFF has floating point
        data and a missing nodata value, and the user does not specify a default
        fill value, a fill value is used determined by the data type. For this
        test, it should be -9999.0.

        """
        input_file_path = 'tests/data/SMAP_L3_FT_P_fill_float_input.tif'
        output_file_path = self.create_output_file_name(input_file_path)

        output_file = self.run_mask_fill(
            input_file_path,
            'SPL3FTP',
            self.shape_file,
            fill_value=None,
        )

        self.assertEqual(output_file, output_file_path)
        self.compare_geotiff_files(
            'tests/data/SMAP_L3_FT_P_fill_float_output.tif',
            output_file_path
        )

    def test_mask_fill_geo_uint_default(self):
        """A full test of the `mask_fill` utility using a GeoTIFF input file.
        This specific test ensures that when an input GeoTIFF has unsigned
        integer data and a missing nodata value, and the user does not specify a
        default fill value, a fill value is used determined by the data type.
        For this test, it should be 254.

        """
        input_file_path = 'tests/data/SMAP_L3_FT_P_fill_uint_input.tif'
        output_file_path = self.create_output_file_name(input_file_path)

        output_file = self.run_mask_fill(
            input_file_path,
            'SPL3FTP',
            self.shape_file,
            fill_value=None,
        )

        self.assertEqual(output_file, output_file_path)
        self.compare_geotiff_files(
            'tests/data/SMAP_L3_FT_P_fill_uint_output.tif',
            output_file_path
        )

    def test_mask_fill_south_pole(self):
        """ Test mask fill with a shapefile containing the south pole for
            both h5 and geotiff files.

        """
        with self.subTest('South Pole HDF-5 file format'):
            output_file = self.run_mask_fill(
                self.input_polar_h5_file,
                self.polar_short_name,
                self.shape_file_south_pole,
            )

            self.assertEqual(output_file, self.output_polar_h5_file)
            self.compare_h5_files(self.output_h5_template_south_pole,
                                  self.output_polar_h5_file)

        with self.subTest('South Pole GeoTIFF file format'):
            output_file = self.run_mask_fill(
                self.input_polar_geo_file,
                self.polar_short_name,
                self.shape_file_south_pole,
            )

            self.assertEqual(output_file, self.output_polar_geo_file)
            self.compare_geotiff_files(self.output_geotiff_template_south_pole,
                                       self.output_polar_geo_file)

    def test_mask_fill_geotiff_coordinates(self):
        """ Check that a GeoTIFF file that matches a coordinate pattern is
            copied without masking.

        """
        geotiff_base = ('SMAP_L3_FT_P_20180618_R16010_001_Freeze_Thaw_'
                        'Retrieval_Data_Global_longitude_Bands_1_488b73ed')

        input_name = f'tests/data/{geotiff_base}.tif'
        output_name = self.create_output_file_name(input_name)

        output_file = self.run_mask_fill(
            input_name,
            'SPL3FTP',
            self.shape_file
        )

        self.assertEqual(output_file, output_name)
        self.compare_geotiff_files(input_name, output_name)

    def test_mask_fill_geotiff_bands(self):
        """ Check that a GeoTIFF with multiple bands will successfully be
            processed by MaskFill.

        """
        base_name = 'SMAP_L3_FT_P_banded'
        input_name = f'tests/data/{base_name}_input.tif'
        template_output = f'tests/data/{base_name}_output.tif'
        test_output = self.create_output_file_name(input_name)
        shape_file = 'tests/data/WV.geo.json'

        output_file = self.run_mask_fill(
            input_name,
            'SPL3FTP',
            shape_file,
        )

        self.assertEqual(output_file, test_output)
        self.compare_geotiff_files(template_output, test_output)

    def test_mask_fill_geotiff_compression(self):
        """ Ensure that the compression of an input granule is preserved in the
            output from MaskFill.

        """
        input_file = 'tests/data/SMAP_L4_SM_aup_compression.tif'
        output_file_path = self.create_output_file_name(input_file)

        output_file = self.run_mask_fill(
            input_file,
            'SPL4SMAU',
            self.shape_file,
        )

        self.assertEqual(output_file, output_file_path)
        self.compare_geotiff_files(self.output_geotiff_template, output_file_path)

        geotiff_results = gdal.Open(output_file_path)
        compression = geotiff_results.GetMetadata('IMAGE_STRUCTURE').get('COMPRESSION', None)
        self.assertEqual(compression, 'LZW')

    def test_mask_fill_h5_dimension_list(self):
        """ Ensure a science variable with DIMENSION_LIST, but not coordinates
            metadata attributes will be masked.

        """
        input_file = 'tests/data/SMAP_L4_SM_aup_dimension_list_input.h5'
        shape_file = 'tests/data/afg_kite.geo.json'
        output_file_path = self.create_output_file_name(input_file)
        template_output = 'tests/data/SMAP_L4_SM_aup_dimension_list_output.h5'

        output_file = self.run_mask_fill(
            input_file,
            'SPL4SMAU',
            shape_file,
        )

        self.assertEqual(output_file, output_file_path)
        self.compare_h5_files(template_output, output_file_path)

    def test_mask_fill_h5_utm(self):
        """ Ensure an HDF-5 file can be correctly masked when the input file
            has a UTM grid.

        """
        input_file = 'tests/data/SMAP_L4_SM_aup_UTM_input.h5'
        shape_file = 'tests/data/COL.geo.json'
        output_file_path = self.create_output_file_name(input_file)
        template_output = 'tests/data/SMAP_L4_SM_aup_UTM_output.h5'

        output_file = self.run_mask_fill(
            input_file,
            'SPL4SMAU',
            shape_file,
        )

        self.assertEqual(output_file, output_file_path)
        self.compare_h5_files(template_output, output_file_path)

    def test_mask_fill_geo_utm(self):
        """ Ensure a GeoTIFF file can be correctly masked when the input file
            has a UTM grid.

        """
        input_file = 'tests/data/SMAP_L4_SM_aup_UTM_input.tif'
        shape_file = 'tests/data/COL.geo.json'
        output_file_path = self.create_output_file_name(input_file)
        template_output = 'tests/data/SMAP_L4_SM_aup_UTM_output.tif'

        output_file = self.run_mask_fill(
            input_file,
            'SPL4SMAU',
            shape_file,
        )

        self.assertEqual(output_file, output_file_path)
        self.compare_geotiff_files(template_output, output_file_path)

    def test_mask_fill_netcdf4_input(self):
        """ Ensure a NetCDF-4 file input (e.g., from HOSS) can be correctly
            masked using an example GPM/IMERG granule.

        """
        input_file = 'tests/data/GPM_3IMERGHH_input.nc4'
        shape_file = 'tests/data/USA.geo.json'
        output_file_path = self.create_output_file_name(input_file)
        template_output = 'tests/data/GPM_3IMERGHH_output.nc4'

        output_file = self.run_mask_fill(
            input_file,
            'GPM_3IMERGHH',
            shape_file,
        )

        self.assertEqual(output_file, output_file_path)
        self.compare_h5_files(template_output, output_file_path)
