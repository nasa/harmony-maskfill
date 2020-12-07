from argparse import Namespace
from os import makedirs
from os.path import basename, isdir, isfile, join, splitext
from shutil import rmtree
from unittest import TestCase
from unittest.mock import patch

from MaskFill import (check_shapefile_geojson, DEFAULT_FILL_VALUE,
                      format_parameters, get_log_file_path,
                      get_xml_error_response, validate_input_parameters)
from pymods.exceptions import (InsufficientProjectionInformation,
                               InvalidParameterValue, MissingCoordinateDataset,
                               MissingParameterValue, NoMatchingData)


class TestMaskFill(TestCase):

    def setUp(self):
        self.identifier = 'test'
        self.input_geotiff_file = 'tests/data/SMAP_L4_SMAU_input.tif'
        self.input_h5_file = 'tests/data/SMAP_L4_SMAU_input.h5'
        self.output_dir = 'tests/output'
        self.shape_file = 'tests/data/USA.geo.json'

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

    def test_get_log_file_path(self):
        """A log file name is correctly appended to the output directory."""
        output_dirs = {'/this/is/a/path': '/this/is/a/path/mask_fill.log',
                       '/path/has/a/slash/': '/path/has/a/slash/mask_fill.log'}

        for output_dir, log_file in output_dirs.items():
            with self.subTest(output_dir=output_dir, log_file=log_file):
                self.assertEqual(get_log_file_path(output_dir), log_file)

    def test_format_parameters(self):
        """MaskFill.format_parameters handles the following cases:

        * Strings without single quotes are unchanged.
        * Strings *with* single quotes have those quotes stripped out.
        * Parameters that are of `None` type are unchanged.
        * Parameters that are of other, non-string type are unchanged.

        """
        parameter_names = ['debug', 'fill_value', 'identifier', 'input_file',
                           'mask_grid_cache', 'output_dir', 'shape_file']

        parameter_values = {'no-quotes': 'no-quotes',
                            'with-a-\'quote': 'with-a-quote',
                            None: None,
                            123: 123}

        for input_value, output_value in parameter_values.items():
            input_parameters = Namespace(**{parameter: input_value
                                            for parameter in parameter_names})
            formatted_parameters = list(format_parameters(input_parameters))

            with self.subTest(input_value=input_value):
                self.assertEqual(len(parameter_names), len(formatted_parameters))
                for parameter in formatted_parameters:
                    self.assertEqual(parameter, output_value)

    @patch('uuid.uuid4')
    def test_check_shapefile_geojson_path(self, mock_uuid4):
        """MaskFill.check_shapefile_geojson handles a shape file path."""
        self.assertEqual(self.shape_file,
                         check_shapefile_geojson(self.shape_file, self.output_dir))
        mock_uuid4.assert_not_called()

    @patch('uuid.uuid4')
    def test_check_shapefile_geojson_native_string(self, mock_uuid4):
        """MaskFill.check_shapefile_geojson handles a raw GeoJSON string."""
        makedirs(self.output_dir)
        test_uuid4 = '18045b77-5733-430f-a5f6-1547baea88d4'
        mock_uuid4.return_value = test_uuid4
        geojson_string = '{"type": "FeatureCollection", "features": []}'
        expected_output_shape_file = (f'{self.output_dir}/'
                                      f'shape_{test_uuid4}.geojson')

        shape_file = check_shapefile_geojson(geojson_string, self.output_dir)
        self.assertEqual(mock_uuid4.call_count, 1)
        self.assertEqual(shape_file, expected_output_shape_file)
        self.assertTrue(isfile(expected_output_shape_file))

        with open(expected_output_shape_file, 'r') as file_handler:
            saved_geojson_string = file_handler.read()

        self.assertEqual(saved_geojson_string, geojson_string)

    def test_validate_input_parameters_all_valid(self):
        """No exception is raised when all parameters are valid."""
        makedirs(self.output_dir)
        self.assertEqual(validate_input_parameters(self.input_h5_file,
                                                   self.shape_file,
                                                   self.output_dir,
                                                   DEFAULT_FILL_VALUE,
                                                   None), None)

    def test_validate_input_parameters_valid_extensions(self):
        """Ensure the expected input file extensions are valid."""
        makedirs(self.output_dir)

        for extension in ['h5', 'H5', 'tif', 'TIF']:
            with self.subTest(extension=extension):
                input_file = f'{self.output_dir}/input.{extension}'
                with open(input_file, 'w'):
                    pass

                self.assertEqual(validate_input_parameters(input_file,
                                                           self.shape_file,
                                                           self.output_dir,
                                                           DEFAULT_FILL_VALUE,
                                                           None), None)

    def test_validate_input_parameters_invalid_extension(self):
        """Ensure invalid input file extensions are detected."""
        expected_message = 'The input data file must be a GeoTIFF or HDF5 file type'
        makedirs(self.output_dir)

        input_file = f'{self.output_dir}/input.json'
        with open(input_file, 'w'):
            pass

        with self.assertRaises(InvalidParameterValue) as context:
            validate_input_parameters(input_file, self.shape_file,
                                      self.output_dir, DEFAULT_FILL_VALUE,
                                      None)
            self.assertEqual(context.exception.message, expected_message)

    def test_validate_input_parameters_no_input_file(self):
        """Validation should fail for either None or non-existant file."""
        expected_message = 'An input data file is required for the mask fill utility'
        makedirs(self.output_dir)

        input_file = None
        with self.assertRaises(MissingParameterValue) as context:
            validate_input_parameters(input_file, self.shape_file,
                                      self.output_dir, DEFAULT_FILL_VALUE,
                                      None)
            self.assertEqual(context.exception.message, expected_message)

    def test_validate_input_parameters_bad_input_file(self):
        """Validation should fail for either None or non-existant file."""
        makedirs(self.output_dir)

        input_file = 'not_a_real_file.h5'
        expected_message = f'The path {input_file} does not exist'

        with self.assertRaises(MissingParameterValue) as context:
            validate_input_parameters(input_file, self.shape_file,
                                      self.output_dir, DEFAULT_FILL_VALUE,
                                      None)
            self.assertEqual(context.exception.message, expected_message)

    def test_validate_input_parameters_bad_shape_file(self):
        """Validation should fail if the shape file doesn't exist."""
        makedirs(self.output_dir)
        expected_message = 'The path not_a_real_file does not exist'

        with self.assertRaises(MissingParameterValue) as context:
            validate_input_parameters(self.input_h5_file, 'not_a_real_file',
                                      self.output_dir, DEFAULT_FILL_VALUE,
                                      None)
            self.assertEqual(context.exception.message, expected_message)

    def test_validate_input_parameters_bad_output_dir(self):
        """Validation should fail if the output directory doesn't exist."""
        expected_message = f'The path {self.output_dir} does not exist'

        with self.assertRaises(MissingParameterValue) as context:
            validate_input_parameters(self.input_h5_file, self.shape_file,
                                      self.output_dir, DEFAULT_FILL_VALUE,
                                      None)
            self.assertEqual(context.exception.message, expected_message)

    def test_validate_input_parameters_valid_fill_value_type(self):
        """Validation should only pass for floats or integers."""
        makedirs(self.output_dir)

        for fill_value in [1, 1.234]:
            self.assertEqual(validate_input_parameters(self.input_h5_file,
                                                       self.shape_file,
                                                       self.output_dir,
                                                       fill_value,
                                                       None), None)

    def test_validate_input_parameters_invalid_fill_value_type(self):
        """Validation should fail for a string fill value."""
        expected_message = 'The default fill value must be a number'
        makedirs(self.output_dir)

        with self.assertRaises(InvalidParameterValue) as context:
            validate_input_parameters(self.input_h5_file, self.shape_file,
                                      self.output_dir, 'not a number',
                                      None)
            self.assertEqual(context.exception.message, expected_message)

    def test_get_xml_error_response(self):
        """Exceptions that are designed to be used in XML output should be
        directly placed in the output message.

        """
        test_exceptions = [InvalidParameterValue(), MissingParameterValue(),
                           NoMatchingData(),
                           MissingCoordinateDataset('file.hdf', 'dataset'),
                           InsufficientProjectionInformation('dataset')]

        for exception in test_exceptions:
            with self.subTest(exception.exception_type):
                xml_error = get_xml_error_response(self.output_dir, exception)

                expected_substrings = ['iesi:Exception',
                                       f'<Code>{exception.exception_type}</Code>',
                                       exception.message]
                for text in expected_substrings:
                    self.assertIn(text, xml_error)

    def test_get_xml_error_response_non_custom(self):
        """Exceptions that are not designed to be used in XML output should be
        replaced with an InternalError in the output message.

        """
        exception = KeyError('latitude')
        xml_error = get_xml_error_response(self.output_dir, exception)

        for text in ['iesi:Exception', '<Code>InternalError',
                     'KeyError(\'latitude\'']:
            self.assertTrue(text in xml_error)
