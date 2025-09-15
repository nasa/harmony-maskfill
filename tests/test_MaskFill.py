from argparse import Namespace
from unittest.mock import patch

from numpy import array, array_equal, where
from osgeo import gdal
import h5py

from maskfill.MaskFill import (
    DEFAULT_MASK_GRID_CACHE,
    maskfill_sdps,
    get_xml_success_response,
)

from tests.utilities import MaskFillTestCase


class TestMaskFill(MaskFillTestCase):

    def setUp(self):
        super().setUp()
        self.input_geotiff_file = 'tests/data/SMAP_L4_SM_aup_input.tif'
        self.input_h5_file = 'tests/data/SMAP_L4_SM_aup_input.h5'
        self.shape_file = 'tests/data/USA.geo.json'
        self.shape_file_south_pole = 'tests/data/south_pole.geo.json'
        self.output_geotiff_file = self.create_output_file_name(self.input_geotiff_file)
        self.output_h5_file = self.create_output_file_name(self.input_h5_file)
        self.output_geotiff_template = 'tests/data/SMAP_L4_SM_aup_output.tif'
        self.output_geotiff_template_south_pole = 'tests/data/SMAP_L3_FT_P_polar_3d_south_pole_output.tif'
        self.output_h5_template_south_pole = 'tests/data/SMAP_L3_FT_P_polar_3d_south_pole_output.h5'
        self.output_h5_template = 'tests/data/SMAP_L4_SM_aup_output.h5'
        self.input_corner_file = 'tests/data/SMAP_L3_FT_P_corners_input.h5'
        self.output_corner_file = self.create_output_file_name(self.input_corner_file)
        self.output_corner_template = 'tests/data/SMAP_L3_FT_P_corners_output.h5'
        self.input_polar_h5_file = 'tests/data/SMAP_L3_FT_P_polar_3d_input.h5'
        self.input_polar_geo_file = 'tests/data/SMAP_L3_FT_P_polar_3d_input.tif'
        self.output_polar_h5_file = self.create_output_file_name(self.input_polar_h5_file)
        self.output_polar_geo_file = self.create_output_file_name(self.input_polar_geo_file)
        self.output_polar_template = 'tests/data/SMAP_L3_FT_P_polar_3d_output.h5'
        self.input_comparison_geo = 'tests/data/SMAP_L4_SM_aup_comparison.tif'
        self.input_comparison_h5 = 'tests/data/SMAP_L4_SM_aup_comparison.h5'
        self.output_comparison_geo = self.create_output_file_name(self.input_comparison_geo)
        self.output_comparison_h5 = self.create_output_file_name(self.input_comparison_h5)

        self.default_parameters = {'debug': 'false',
                                   'fill_value': -9999.0,
                                   'identifier': self.identifier,
                                   'input_file': self.input_h5_file,
                                   'mask_grid_cache': DEFAULT_MASK_GRID_CACHE,
                                   'output_dir': self.output_dir,
                                   'shape_file': self.shape_file}

    def create_parameters_namespace(self, parameters_dictionary):
        return Namespace(**parameters_dictionary)

    @patch('maskfill.MaskFill.get_input_parameters')
    def test_mask_fill_h5(self, mock_get_input_parameters):
        """A full test of the `mask_fill` utility using an HDF-5 input file,
        patching the reading of input parameters. This checks for a success
        message, and then compares the output file from `mask_fill` to a
        templated output by checking the expected datasets.

        """
        mock_get_input_parameters.return_value = self.create_parameters_namespace(
            self.default_parameters
        )

        response = maskfill_sdps()

        self.assertEqual(response, get_xml_success_response(self.input_h5_file,
                                                            self.shape_file,
                                                            self.output_h5_file))

        self.compare_h5_files(self.output_h5_template, self.output_h5_file)

    @patch('maskfill.MaskFill.get_input_parameters')
    def test_mask_fill_geotiff(self, mock_get_input_parameters):
        """A full test of the `mask_fill` utility using a GeoTIFF input file,
        patching the reading of input parameters. This checks for a success
        message, and then compares the output file from `mask_fill` to a
        templated output file by checking the dataset and metadata.

        """
        geotiff_parameters = self.default_parameters.copy()
        geotiff_parameters['input_file'] = self.input_geotiff_file

        mock_get_input_parameters.return_value = self.create_parameters_namespace(
            geotiff_parameters
        )
        response = maskfill_sdps()

        self.assertEqual(response, get_xml_success_response(self.input_geotiff_file,
                                                            self.shape_file,
                                                            self.output_geotiff_file))

        self.compare_geotiff_files(self.output_geotiff_template, self.output_geotiff_file)

    @patch('maskfill.MaskFill.get_input_parameters')
    def test_mask_fill_h5_extrapolating_corner(self, mock_get_input_parameters):
        """A full test of the `mask_fill` utility using an HDF-5 input file
        that has filled data in the upper right corner of the longitude and
        latitude arrays.

        """
        corner_parameters = self.default_parameters.copy()
        corner_parameters['input_file'] = self.input_corner_file

        mock_get_input_parameters.return_value = self.create_parameters_namespace(corner_parameters)
        response = maskfill_sdps()

        self.assertEqual(response, get_xml_success_response(self.input_corner_file,
                                                            self.shape_file,
                                                            self.output_corner_file))

        self.compare_h5_files(self.output_corner_template, self.output_corner_file)

    @patch('maskfill.MaskFill.get_input_parameters')
    def test_mask_fill_h5_polar_3d(self, mock_get_input_parameters):
        """A full test of the `mask_fill` utility using an HDF-5 input file
        that contains SMAP L3 FTP polar data. These data are 3-dimensional,
        such that array indices [i, j, k] corresond to:

            - i: data band
            - j: projected x
            - k: projected y

        The data use the NSIDC EASE-2 polar standard grid.

        """
        polar_parameters = self.default_parameters.copy()
        polar_parameters['input_file'] = self.input_polar_h5_file

        mock_get_input_parameters.return_value = self.create_parameters_namespace(polar_parameters)
        response = maskfill_sdps()

        self.assertEqual(response, get_xml_success_response(self.input_polar_h5_file,
                                                            self.shape_file,
                                                            self.output_polar_h5_file))

        self.compare_h5_files(self.output_polar_template, self.output_polar_h5_file)

    @patch('maskfill.MaskFill.get_input_parameters')
    def test_mask_fill_compare_h5_geo(self, mock_get_input_parameters):
        """Run MaskFill over the same input data in both GeoTIFF and HDF-5
        format to ensure the output is consistent between the two methologies.

        """
        shape_file = 'tests/data/comparison.geo.json'

        geotiff_parameters = self.default_parameters.copy()
        geotiff_parameters['input_file'] = self.input_comparison_geo
        geotiff_parameters['shape_file'] = shape_file

        h5_parameters = self.default_parameters.copy()
        h5_parameters['input_file'] = self.input_comparison_h5
        h5_parameters['shape_file'] = shape_file

        mock_get_input_parameters.side_effect = [
            self.create_parameters_namespace(h5_parameters),
            self.create_parameters_namespace(geotiff_parameters),
        ]

        response_h5 = maskfill_sdps()
        response_geo = maskfill_sdps()

        self.assertEqual(response_h5, get_xml_success_response(self.input_comparison_h5,
                                                               shape_file,
                                                               self.output_comparison_h5))
        self.assertEqual(response_geo, get_xml_success_response(self.input_comparison_geo,
                                                                shape_file,
                                                                self.output_comparison_geo))

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

    @patch('maskfill.MaskFill.get_input_parameters')
    def test_mask_fill_h5_default_fill(self, mock_get_input_parameters):
        """ Ensure MaskFill can process a file that has no in-file fill value
            metadata, relying instead on default fill values that are selected
            based on the data type of each variable in the HDF-5 file.

        """
        input_file_path = 'tests/data/SMAP_L3_FT_P_fill_input.h5'
        output_file_path = self.create_output_file_name(input_file_path)

        default_fill_parameters = self.default_parameters.copy()
        default_fill_parameters['input_file'] = input_file_path
        default_fill_parameters['fill_value'] = None

        mock_get_input_parameters.return_value = self.create_parameters_namespace(
            default_fill_parameters
        )
        response = maskfill_sdps()

        self.assertEqual(
            response,
            get_xml_success_response(
                input_file_path,
                self.shape_file,
                output_file_path
            )
        )

        self.compare_h5_files(
            'tests/data/SMAP_L3_FT_P_fill_output.h5',
            output_file_path
        )

    @patch('maskfill.MaskFill.get_input_parameters')
    def test_mask_fill_geo_float_default(self, mock_get_input_parameters):
        """A full test of the `mask_fill` utility using a GeoTIFF input file,
        patching the reading of input parameters. This specific test ensure
        that when an input GeoTIFF has floating point data and a missing nodata
        value, and the user does not specify a default fill value, a fill value
        is used determined by the data type. For this test, it should be
        -9999.0.

        """
        input_file_path = 'tests/data/SMAP_L3_FT_P_fill_float_input.tif'
        output_file_path = self.create_output_file_name(input_file_path)

        float_default_fill_parameters = self.default_parameters.copy()
        float_default_fill_parameters['input_file'] = input_file_path
        float_default_fill_parameters['fill_value'] = None

        mock_get_input_parameters.return_value = self.create_parameters_namespace(
            float_default_fill_parameters
        )
        response = maskfill_sdps()

        self.assertEqual(
            response,
            get_xml_success_response(
                input_file_path,
                self.shape_file,
                output_file_path
            )
        )

        self.compare_geotiff_files(
            'tests/data/SMAP_L3_FT_P_fill_float_output.tif',
            output_file_path
        )

    @patch('maskfill.MaskFill.get_input_parameters')
    def test_mask_fill_geo_uint_default(self, mock_get_input_parameters):
        """A full test of the `mask_fill` utility using a GeoTIFF input file,
        patching the reading of input parameters. This specific test ensure
        that when an input GeoTIFF has unsigned integer data and a missing
        nodata value, and the user does not specify a default fill value, a
        fill value is used determined by the data type. For this test, it
        should be 254.

        """
        input_file_path = 'tests/data/SMAP_L3_FT_P_fill_uint_input.tif'
        output_file_path = self.create_output_file_name(input_file_path)

        uint_default_fill_parameters = self.default_parameters.copy()
        uint_default_fill_parameters['input_file'] = input_file_path
        uint_default_fill_parameters['fill_value'] = None

        mock_get_input_parameters.return_value = self.create_parameters_namespace(
            uint_default_fill_parameters
        )
        response = maskfill_sdps()

        self.assertEqual(
            response,
            get_xml_success_response(
                input_file_path,
                self.shape_file,
                output_file_path
            )
        )

        self.compare_geotiff_files(
            'tests/data/SMAP_L3_FT_P_fill_uint_output.tif',
            output_file_path
        )

    @patch('maskfill.MaskFill.get_input_parameters')
    def test_mask_fill_south_pole(self, mock_get_input_parameters):
        """ Test mask fill with a shapefile containing the south pole for
            both h5 and geotiff files.

        """
        with self.subTest('South Pole HDF-5 file format'):
            h5_parameters = self.default_parameters.copy()
            h5_parameters['input_file'] = self.input_polar_h5_file
            h5_parameters['shape_file'] = self.shape_file_south_pole

            mock_get_input_parameters.return_value = self.create_parameters_namespace(
                h5_parameters
            )
            response = maskfill_sdps()

            self.assertEqual(
                response,
                get_xml_success_response(self.input_polar_h5_file,
                                         self.shape_file_south_pole,
                                         self.output_polar_h5_file)
            )

            self.compare_h5_files(self.output_h5_template_south_pole,
                                  self.output_polar_h5_file)

        with self.subTest('South Pole GeoTIFF file format'):
            geotiff_parameters = self.default_parameters.copy()
            geotiff_parameters['input_file'] = self.input_polar_geo_file
            geotiff_parameters['shape_file'] = self.shape_file_south_pole

            mock_get_input_parameters.return_value = self.create_parameters_namespace(
                geotiff_parameters
            )
            response = maskfill_sdps()

            self.assertEqual(
                response,
                get_xml_success_response(self.input_polar_geo_file,
                                         self.shape_file_south_pole,
                                         self.output_polar_geo_file)
            )

            self.compare_geotiff_files(self.output_geotiff_template_south_pole,
                                       self.output_polar_geo_file)

    @patch('maskfill.MaskFill.get_input_parameters')
    def test_mask_fill_geotiff_coordinates(self, mock_get_input_parameters):
        """ Check that a GeoTIFF file that matches a coordinate pattern is
            copied without masking.

        """
        geotiff_base = ('SMAP_L3_FT_P_20180618_R16010_001_Freeze_Thaw_'
                        'Retrieval_Data_Global_longitude_Bands_1_488b73ed')

        input_name = f'tests/data/{geotiff_base}.tif'
        output_name = self.create_output_file_name(input_name)

        geotiff_parameters = self.default_parameters.copy()
        geotiff_parameters['input_file'] = input_name

        mock_get_input_parameters.return_value = self.create_parameters_namespace(
            geotiff_parameters
        )

        response = maskfill_sdps()

        self.assertEqual(response, get_xml_success_response(input_name,
                                                            self.shape_file,
                                                            output_name))

        self.compare_geotiff_files(input_name, output_name)

    @patch('maskfill.MaskFill.get_input_parameters')
    def test_mask_fill_geotiff_bands(self, mock_get_input_parameters):
        """ Check that a GeoTIFF with multiple bands will successfully be
            processed by MaskFill.

        """
        base_name = 'SMAP_L3_FT_P_banded'
        input_name = f'tests/data/{base_name}_input.tif'
        template_output = f'tests/data/{base_name}_output.tif'
        test_output = (f'{self.output_dir}/{self.identifier}/{base_name}_'
                       'input_mf.tif')
        shape_file = 'tests/data/WV.geo.json'

        geotiff_parameters = self.default_parameters.copy()
        geotiff_parameters['input_file'] = input_name
        geotiff_parameters['shape_file'] = shape_file

        mock_get_input_parameters.return_value = self.create_parameters_namespace(
            geotiff_parameters
        )

        response = maskfill_sdps()

        self.assertEqual(response, get_xml_success_response(input_name,
                                                            shape_file,
                                                            test_output))
        self.compare_geotiff_files(template_output, test_output)

    @patch('maskfill.MaskFill.get_input_parameters')
    def test_mask_fill_geotiff_compression(self, mock_get_input_parameters):
        """ Ensure that the compression of an input granule is preserved in the
            output from MaskFill.

        """
        input_file = 'tests/data/SMAP_L4_SM_aup_compression.tif'
        output_file = f'{self.output_dir}/{self.identifier}/SMAP_L4_SM_aup_compression_mf.tif'
        geotiff_parameters = self.default_parameters.copy()
        geotiff_parameters['input_file'] = input_file

        mock_get_input_parameters.return_value = self.create_parameters_namespace(geotiff_parameters)
        response = maskfill_sdps()

        self.assertEqual(response, get_xml_success_response(input_file,
                                                            self.shape_file,
                                                            output_file))

        self.compare_geotiff_files(self.output_geotiff_template, output_file)

        geotiff_results = gdal.Open(output_file)
        compression = geotiff_results.GetMetadata('IMAGE_STRUCTURE').get('COMPRESSION', None)
        self.assertEqual(compression, 'LZW')

    @patch('maskfill.MaskFill.get_input_parameters')
    def test_mask_fill_h5_dimension_list(self, mock_get_input_parameters):
        """ Ensure a science variable with DIMENSION_LIST, but not coordinates
            metadata attributes will be masked.

        """
        input_file = 'tests/data/SMAP_L4_SM_aup_dimension_list_input.h5'
        shape_file = 'tests/data/afg_kite.geo.json'
        output_file = (f'{self.output_dir}/{self.identifier}/'
                       'SMAP_L4_SM_aup_dimension_list_input_mf.h5')
        template_output = 'tests/data/SMAP_L4_SM_aup_dimension_list_output.h5'

        parameters = self.default_parameters.copy()
        parameters['input_file'] = input_file
        parameters['shape_file'] = shape_file

        mock_get_input_parameters.return_value = self.create_parameters_namespace(parameters)
        response = maskfill_sdps()

        self.assertEqual(response, get_xml_success_response(input_file,
                                                            shape_file,
                                                            output_file))

        self.compare_h5_files(template_output, output_file)

    @patch('maskfill.MaskFill.get_input_parameters')
    def test_mask_fill_h5_utm(self, mock_get_input_parameters):
        """ Ensure an HDF-5 file can be correctly masked when the input file
            has a UTM grid.

        """
        input_file = 'tests/data/SMAP_L4_SM_aup_UTM_input.h5'
        shape_file = 'tests/data/COL.geo.json'
        output_file = (f'{self.output_dir}/{self.identifier}/'
                       'SMAP_L4_SM_aup_UTM_input_mf.h5')
        template_output = 'tests/data/SMAP_L4_SM_aup_UTM_output.h5'

        parameters = self.default_parameters.copy()
        parameters['input_file'] = input_file
        parameters['shape_file'] = shape_file

        mock_get_input_parameters.return_value = self.create_parameters_namespace(parameters)
        response = maskfill_sdps()

        self.assertEqual(response, get_xml_success_response(input_file,
                                                            shape_file,
                                                            output_file))

        self.compare_h5_files(template_output, output_file)

    @patch('maskfill.MaskFill.get_input_parameters')
    def test_mask_fill_geo_utm(self, mock_get_input_parameters):
        """ Ensure a GeoTIFF file can be correctly masked when the input file
            has a UTM grid.

        """
        input_file = 'tests/data/SMAP_L4_SM_aup_UTM_input.tif'
        shape_file = 'tests/data/COL.geo.json'
        output_file = (f'{self.output_dir}/{self.identifier}/'
                       'SMAP_L4_SM_aup_UTM_input_mf.tif')
        template_output = 'tests/data/SMAP_L4_SM_aup_UTM_output.tif'

        parameters = self.default_parameters.copy()
        parameters['input_file'] = input_file
        parameters['shape_file'] = shape_file

        mock_get_input_parameters.return_value = self.create_parameters_namespace(parameters)
        response = maskfill_sdps()

        self.assertEqual(response, get_xml_success_response(input_file,
                                                            shape_file,
                                                            output_file))

        self.compare_geotiff_files(template_output, output_file)

    @patch('maskfill.MaskFill.get_input_parameters')
    def test_mask_fill_netcdf4_input(self, mock_get_input_parameters):
        """ Ensure a NetCDF-4 file input (e.g., from HOSS) can be correctly
            masked using an example GPM/IMERG granule.

        """
        input_file = 'tests/data/GPM_3IMERGHH_input.nc4'
        shape_file = 'tests/data/USA.geo.json'
        output_file = (f'{self.output_dir}/{self.identifier}/'
                       'GPM_3IMERGHH_input_mf.nc4')
        template_output = 'tests/data/GPM_3IMERGHH_output.nc4'

        parameters = self.default_parameters.copy()
        parameters['input_file'] = input_file
        parameters['shape_file'] = shape_file

        mock_get_input_parameters.return_value = self.create_parameters_namespace(parameters)
        response = maskfill_sdps()

        self.assertEqual(response, get_xml_success_response(input_file,
                                                            shape_file,
                                                            output_file))

        self.compare_h5_files(template_output, output_file)
