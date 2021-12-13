from argparse import Namespace
from os.path import basename, isdir, join, splitext
from shutil import rmtree
from unittest import TestCase
from unittest.mock import patch

from numpy import array, array_equal, ndarray, where
from osgeo import gdal
import h5py

from MaskFill import (DEFAULT_FILL_VALUE, DEFAULT_MASK_GRID_CACHE,
                      maskfill_sdps, get_xml_success_response)


class TestMaskFill(TestCase):

    @classmethod
    def setUpClass(cls):
        cls.identifier = 'test'

    def setUp(self):
        self.input_geotiff_file = 'tests/data/SMAP_L4_SM_aup_input.tif'
        self.input_h5_file = 'tests/data/SMAP_L4_SM_aup_input.h5'
        self.output_dir = 'tests/output'
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

        self.default_parameters = {'debug': 'true',
                                   'fill_value': DEFAULT_FILL_VALUE,
                                   'identifier': self.identifier,
                                   'input_file': self.input_h5_file,
                                   'mask_grid_cache': DEFAULT_MASK_GRID_CACHE,
                                   'output_dir': self.output_dir,
                                   'shape_file': self.shape_file}

    def tearDown(self):
        """Clean up test artifacts after each test."""
        if isdir(self.output_dir):
            rmtree(self.output_dir)

    def create_parameters_namespace(self, parameters_dictionary):
        return Namespace(**parameters_dictionary)

    def create_output_file_name(self, input_file_name):
        output_root, output_extension = splitext(basename(input_file_name))
        output_basename = f'{output_root}_mf{output_extension}'
        return join(self.output_dir, self.identifier, output_basename)

    def compare_geotiff_files(self, file_one_name, file_two_name):
        """Check both files have the same number of bands, and that the data
        within those bands match. Also, ensure any file-level metadata is
        identical.

        :type file_one_name: str
        :type file_two_name: str

        """

        dataset_one = gdal.Open(file_one_name)
        dataset_two = gdal.Open(file_two_name)

        band_one = array(dataset_one.ReadAsArray())
        band_two = array(dataset_two.ReadAsArray())
        self.assertEqual(band_one.shape, band_two.shape)
        self.assertTrue(array_equal(band_one, band_two))
        self.assertEqual(dataset_one.GetMetadata(), dataset_two.GetMetadata())

    def compare_h5_files(self, file_one_name, file_two_name):
        """Check all Attributes, Datasets and Groups within two HDF-5 files are
        equal.

        :type file_one_name: str
        :type file_two_name: str

        """
        file_one = h5py.File(file_one_name, 'r')
        file_two = h5py.File(file_two_name, 'r')

        self.compare_h5_file_datasets(file_one, file_two)
        self.compare_h5_file_attributes(file_one, file_two)

        file_one.close()
        file_two.close()

    def compare_h5_file_attributes(self, file_one, file_two):
        """For both files, extract dictionaries of attributes. Check both
        dictionaries contain the same keys (attribute names). Then compare the
        values of each attribute to ensure equality. Note, the attribute names
        are the full path from the root of the file, so also contain the
        hierarchy (groups) that the attributes belong to. This ensures the
        attributes location within the file is also being compared.

        :type file_one: h5py.File
        :type file_two: h5py.File

        """
        file_one_attributes = self.extract_all_h5_attributes(file_one, {})
        file_two_attributes = self.extract_all_h5_attributes(file_two, {})

        self.assertEqual(list(file_one_attributes.keys()),
                         list(file_two_attributes.keys()))

        for attribute_name, attribute_value in file_one_attributes.items():
            if isinstance(attribute_value, ndarray):
                if attribute_name.endswith('_LIST'):
                    # This catches 'DIMENSION_LIST' and 'REFERENCE_LIST' metadata
                    for ref_index, ref_one in enumerate(attribute_value):
                        dataset_ref_one = file_one[ref_one[0]]
                        ref_two = file_two_attributes[attribute_name][ref_index][0]
                        dataset_ref_two = file_two[ref_two]
                        self.assertEqual(dataset_ref_one.name,
                                         dataset_ref_two.name)
                else:
                    self.assertTrue(
                        array_equal(attribute_value,
                                    file_two_attributes[attribute_name]),
                        attribute_name
                    )

            else:
                self.assertEqual(attribute_value,
                                 file_two_attributes[attribute_name],
                                 attribute_name)

    def compare_h5_file_datasets(self, object_one, object_two):
        """For both files, traverse through all Groups and Datasets. Ensure
        that each Group or Dataset is present in both files, and each Group
        has the same child Groups and Datasets. For each Dataset compare the
        values between the two files.

        :type object_one: h5py.Dataset, h5py.File or h5py.Group
        :type object_two: h5py.Dataset, h5py.File or h5py.Group

        """

        for object_one_name, object_one_value in object_one.items():
            self.assertIn(object_one_name, list(object_two.keys()))
            object_two_value = object_two[object_one_name]

            if isinstance(object_one_value, h5py.Dataset):
                self.assertEqual(object_one_value.shape, object_two_value.shape)
                if isinstance(object_one_value[()], ndarray):
                    self.assertTrue(array_equal(object_one_value[()],
                                                object_two_value[()]))
                else:
                    self.assertEqual(object_one_value[()], object_two_value[()])

            else:
                self.assertEqual(list(object_one.keys()), list(object_two.keys()))
                self.compare_h5_file_datasets(object_one_value, object_two_value)

    def extract_all_h5_attributes(self, h5py_object, attribute_dictionary):
        """Starting at the root given of an HDF-5 file, recursively extract all
        attributes to a Python dictionary. The keys should be the full path of
        the attribute, for example: '/Metadata/Source/L1C_TB/version'

        :type h5py_object: h5py.File or h5py.Group
        :rtype: dict

        """
        if h5py_object.name.startswith('/'):
            key_prefix = h5py_object.name
        else:
            key_prefix = f'/{h5py_object.name}'

        for attr_key, attr_value in h5py_object.attrs.items():
            attribute_dictionary[f'{key_prefix}/{attr_key}'] = attr_value

        for iterable_object in h5py_object.values():
            if isinstance(iterable_object, h5py.Dataset):
                for attr_key, attr_value in iterable_object.attrs.items():
                    attribute_dictionary[f'{iterable_object.name}/{attr_key}'] = attr_value

            else:
                self.extract_all_h5_attributes(iterable_object, attribute_dictionary)

        return attribute_dictionary

    @patch('MaskFill.get_input_parameters')
    def test_mask_fill_h5(self, mock_get_input_parameters):
        """A full test of the `mask_fill` utility using an HDF-5 input file,
        patching the reading of input parameters. This checks for a success
        message, and then compares the output file from `mask_fill` to a
        templated output by checking the expected datasets.

        """
        mock_get_input_parameters.return_value = self.create_parameters_namespace(self.default_parameters)

        response = maskfill_sdps()

        self.assertEqual(response, get_xml_success_response(self.input_h5_file,
                                                            self.shape_file,
                                                            self.output_h5_file))

        self.compare_h5_files(self.output_h5_template, self.output_h5_file)

    @patch('MaskFill.get_input_parameters')
    def test_mask_fill_geotiff(self, mock_get_input_parameters):
        """A full test of the `mask_fill` utility using a GeoTIFF input file,
        patching the reading of input parameters. This checks for a success
        message, and then compares the output file from `mask_fill` to a
        templated output file by checking the dataset and metadata.

        """
        geotiff_parameters = self.default_parameters.copy()
        geotiff_parameters['input_file'] = self.input_geotiff_file

        mock_get_input_parameters.return_value = self.create_parameters_namespace(geotiff_parameters)
        response = maskfill_sdps()

        self.assertEqual(response, get_xml_success_response(self.input_geotiff_file,
                                                            self.shape_file,
                                                            self.output_geotiff_file))

        self.compare_geotiff_files(self.output_geotiff_template, self.output_geotiff_file)

    @patch('MaskFill.get_input_parameters')
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

    @patch('MaskFill.get_input_parameters')
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

    @patch('MaskFill.get_input_parameters')
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

    @patch('MaskFill.get_input_parameters')
    def test_mask_fill_south_pole(self, mock_get_input_parameters):
        """ Test mask fill with a shapefile containing the south pole for
            both h5 and geotiff files.

        """

        # h5 file test
        h5_parameters = self.default_parameters.copy()
        h5_parameters['input_file'] = self.input_polar_h5_file
        h5_parameters['shape_file'] = self.shape_file_south_pole

        mock_get_input_parameters.return_value = self.create_parameters_namespace(h5_parameters)
        response = maskfill_sdps()

        self.assertEqual(response, get_xml_success_response(self.input_polar_h5_file,
                                                            self.shape_file_south_pole,
                                                            self.output_polar_h5_file))

        self.compare_h5_files(self.output_h5_template_south_pole, self.output_polar_h5_file)

        # Geotiff file test
        geotiff_parameters = self.default_parameters.copy()
        geotiff_parameters['input_file'] = self.input_polar_geo_file
        geotiff_parameters['shape_file'] = self.shape_file_south_pole

        mock_get_input_parameters.return_value = self.create_parameters_namespace(
            geotiff_parameters
        )
        response = maskfill_sdps()

        self.assertEqual(response, get_xml_success_response(self.input_polar_geo_file,
                                                            self.shape_file_south_pole,
                                                            self.output_polar_geo_file))

        self.compare_geotiff_files(self.output_geotiff_template_south_pole,
                                   self.output_polar_geo_file)

    @patch('MaskFill.get_input_parameters')
    def test_mask_fill_geotiff_coordinates(self, mock_get_input_parameters):
        """ Check that a GeoTIFF file that matches a coordinate pattern is
            copied without masking.

        """
        geotiff_base = ('SMAP_L3_FT_P_20180618_R16010_001_Freeze_Thaw_'
                        'Retrieval_Data_Global_longitude_Bands_1_488b73ed')

        input_name = f'tests/data/{geotiff_base}.tif'
        output_name = f'{self.output_dir}/{self.identifier}/{geotiff_base}_mf.tif'

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

    @patch('MaskFill.get_input_parameters')
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

    @patch('MaskFill.get_input_parameters')
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

    @patch('MaskFill.get_input_parameters')
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

    @patch('MaskFill.get_input_parameters')
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

    @patch('MaskFill.get_input_parameters')
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
