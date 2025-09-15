from argparse import Namespace
from logging import DEBUG, FileHandler, getLogger, INFO, StreamHandler
from os import makedirs
from os.path import basename, isfile, join, splitext
from shutil import rmtree
from unittest import TestCase
from unittest.mock import ANY, patch

from maskfill.MaskFill import (
    check_shapefile_geojson,
    debug_bool,
    DEFAULT_MASK_GRID_CACHE,
    format_parameters,
    get_log_file_path,
    get_sdps_logger,
    get_xml_error_response,
    get_xml_success_response,
    maskfill_sdps,
    validate_input_parameters,
)
from maskfill.exceptions import (
    InsufficientProjectionInformation,
    InvalidMetadata,
    InvalidParameterValue,
    MissingCoordinateDataset,
    MissingParameterValue,
    NoMatchingData,
)


class TestMaskFill(TestCase):

    @classmethod
    def setUpClass(cls):
        cls.identifier = 'test'
        cls.logger = getLogger(cls.identifier)
        cls.input_geotiff_file = 'tests/data/SMAP_L4_SM_aup_input.tif'
        cls.input_h5_file = 'tests/data/SMAP_L4_SM_aup_input.h5'
        cls.output_dir = 'tests/output'
        cls.shape_file = 'tests/data/USA.geo.json'

    def setUp(self):
        """ Create the output directory for each test. """
        makedirs(self.output_dir)

    def tearDown(self):
        """Clean up test artifacts after each test."""
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
        * Strings surrounded by single quotes have those quotes stripped out.
        * Strings with internal single quotes retain those quotes.
        * Parameters that are of `None` type are unchanged.
        * Parameters that are of other, non-string type are unchanged.

        """
        parameter_values = {'no-quotes': 'no-quotes',
                            '\'external-quotes\'': 'external-quotes',
                            'with-a-\'quote': 'with-a-\'quote',
                            None: None,
                            123: 123}

        for input_value, output_value in parameter_values.items():
            with self.subTest(input_value=input_value):
                input_parameters = Namespace(**{'parameter_name': input_value})
                output_parameters = format_parameters(input_parameters)

                self.assertEqual(output_parameters['parameter_name'],
                                 output_value)

    @patch('uuid.uuid4')
    def test_check_shapefile_geojson_path(self, mock_uuid4):
        """MaskFill.check_shapefile_geojson handles a shape file path."""
        self.assertEqual(self.shape_file,
                         check_shapefile_geojson(self.shape_file, self.output_dir))
        mock_uuid4.assert_not_called()

    @patch('uuid.uuid4')
    def test_check_shapefile_geojson_native_string(self, mock_uuid4):
        """MaskFill.check_shapefile_geojson handles a raw GeoJSON string."""
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

    def test_check_shapefile_geojson_missing_file(self):
        """ If the BOUNDINGSHAPE argument is not specified, a
            `MissingParameterValue` error should be raised.

        """
        expected_message = 'A shapefile is required for the mask fill utility'

        with self.assertRaises(MissingParameterValue) as context:
            check_shapefile_geojson(None, self.output_dir)

        self.assertEqual(context.exception.message, expected_message)

    def test_validate_input_parameters_all_valid(self):
        """ No exception is raised when all parameters are valid. If this is
            the case, the return value is `None`.

        """
        self.assertIsNone(validate_input_parameters(self.input_h5_file,
                                                    self.shape_file,
                                                    self.output_dir,
                                                    None,
                                                    self.logger))

    def test_validate_input_parameters_valid_extensions(self):
        """Ensure the expected input file extensions are valid."""
        for extension in ['h5', 'H5', 'tif', 'TIF']:
            with self.subTest(extension=extension):
                input_file = f'{self.output_dir}/input.{extension}'
                with open(input_file, 'w'):
                    pass

                self.assertIsNone(validate_input_parameters(input_file,
                                                            self.shape_file,
                                                            self.output_dir,
                                                            None,
                                                            self.logger))

    def test_validate_input_parameters_invalid_extension(self):
        """Ensure invalid input file extensions are detected."""
        expected_message = ('The input data file must be a GeoTIFF, HDF5 '
                            'or NetCDF-4 file type')

        input_file = f'{self.output_dir}/input.json'
        with open(input_file, 'w'):
            pass

        with self.assertRaises(InvalidParameterValue) as context:
            validate_input_parameters(input_file, self.shape_file,
                                      self.output_dir, None, self.logger)

        self.assertEqual(context.exception.message, expected_message)

    def test_validate_input_parameters_no_input_file(self):
        """Validation should fail for either None or non-existant file."""
        expected_message = 'An input data file is required for the mask fill utility'

        input_file = None
        with self.assertRaises(MissingParameterValue) as context:
            validate_input_parameters(input_file, self.shape_file,
                                      self.output_dir, None, self.logger)

        self.assertEqual(context.exception.message, expected_message)

    def test_validate_input_parameters_bad_input_file(self):
        """Validation should fail for either None or non-existant file."""
        input_file = 'not_a_real_file.h5'
        expected_message = f'The path {input_file} does not exist'

        with self.assertRaises(MissingParameterValue) as context:
            validate_input_parameters(input_file, self.shape_file,
                                      self.output_dir, None, self.logger)

        self.assertEqual(context.exception.message, expected_message)

    def test_validate_input_parameters_bad_shape_file(self):
        """Validation should fail if the shape file doesn't exist."""
        expected_message = 'The path not_a_real_file does not exist'

        with self.assertRaises(MissingParameterValue) as context:
            validate_input_parameters(self.input_h5_file, 'not_a_real_file',
                                      self.output_dir, None, self.logger)

        self.assertEqual(context.exception.message, expected_message)

    def test_validate_input_parameters_bad_output_dir(self):
        """Validation should fail if the output directory doesn't exist."""
        bad_directory = '/this/does/not/exist'
        expected_message = f'The path {bad_directory} does not exist'

        with self.assertRaises(MissingParameterValue) as context:
            validate_input_parameters(self.input_h5_file, self.shape_file,
                                      bad_directory, None, self.logger)

        self.assertEqual(context.exception.message, expected_message)

    def test_validate_input_parameters_valid_fill_value_type(self):
        """Validation should only pass for floats or integers."""
        for fill_value in [1, 1.234]:
            self.assertIsNone(validate_input_parameters(self.input_h5_file,
                                                        self.shape_file,
                                                        self.output_dir,
                                                        fill_value,
                                                        self.logger))

    def test_get_xml_error_response(self):
        """Exceptions that are designed to be used in XML output should be
        directly placed in the output message.

        """
        test_exceptions = [InvalidParameterValue(), MissingParameterValue(),
                           NoMatchingData(),
                           MissingCoordinateDataset('file.hdf', 'dataset'),
                           InsufficientProjectionInformation('dataset'),
                           InvalidMetadata('science_variable', 'name', 'value')]

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

    def test_get_sdps_logger(self):
        """ Ensure a logger is correctly configured for running MaskFill in the
            SDPS environment. This should include two handlers, one for a log
            file and another for the terminal. Both should have identically
            formatted output. The level of the logging should be determined by
            the parsed input of "DEBUG", which defaults to false.

        """
        expected_log_name = f'{self.output_dir}/mask_fill.log'
        test_args = [['debug=True', True, DEBUG], ['debug=False', False, INFO]]

        for description, debug, expected_level in test_args:
            with self.subTest(description):
                logger = get_sdps_logger(self.output_dir, debug)

                self.assertEqual(logger.name, 'MaskFill')
                self.assertEqual(logger.getEffectiveLevel(), expected_level)
                self.assertEqual(len(logger.handlers), 2)
                self.assertIsInstance(logger.handlers[0], StreamHandler)
                self.assertIsInstance(logger.handlers[1], FileHandler)
                self.assertTrue(
                    logger.handlers[1].baseFilename.endswith(expected_log_name)
                )

    def test_debug_bool(self):
        """ Ensure the input value for DEBUG is correctly cast as a boolean. """
        test_args = [['Boolean value', True, True],
                     ['String', 'False', False],
                     ['String with quotes', '\'True\'', True],
                     ['Any other value', 1234, False]]

        for description, input_value, expected_value in test_args:
            with self.subTest(description):
                self.assertEqual(debug_bool(input_value), expected_value)

    @patch('maskfill.MaskFill.get_input_parameters')
    @patch('maskfill.MaskFill.mask_fill')
    def test_maskfill_sdps_converts_fill_value(self, mock_mask_fill,
                                               mock_get_input_parameters):
        """ Verifiy that the `maskfill_sdps` function converts an input fill
            value from a string to a float.

        """
        mock_get_input_parameters.return_value = self.create_parameters_namespace({
            'debug': False,
            'fill_value': '123.4',
            'identifier': 'test',
            'input_file': self.input_h5_file,
            'mask_grid_cache': DEFAULT_MASK_GRID_CACHE,
            'output_dir': 'tests/output',
            'shape_file': self.shape_file,
        })

        output_file_name = 'tests/output/SMAP_L4_SM_aup_input_mf.h5'
        mock_mask_fill.return_value = output_file_name

        expected_response = get_xml_success_response(self.input_h5_file,
                                                     self.shape_file,
                                                     output_file_name)

        self.assertEqual(maskfill_sdps(), expected_response)

        # Ensure the fill value sent to `mask_fill` is a float
        mock_mask_fill.assert_called_once_with(self.input_h5_file,
                                               self.shape_file,
                                               'tests/output/test',
                                               DEFAULT_MASK_GRID_CACHE,
                                               123.4, ANY)

    @patch('maskfill.MaskFill.get_input_parameters')
    @patch('maskfill.MaskFill.mask_fill')
    def test_maskfill_sdps_non_numeric_fill_value(self, mock_mask_fill,
                                                  mock_get_input_parameters):
        """ Verifiy that the `maskfill_sdps` function converts an input fill
            value from a string to a float.

        """
        mock_get_input_parameters.return_value = self.create_parameters_namespace({
            'debug': False,
            'fill_value': 'not a number',
            'identifier': 'test',
            'input_file': self.input_h5_file,
            'mask_grid_cache': DEFAULT_MASK_GRID_CACHE,
            'output_dir': 'tests/output',
            'shape_file': self.shape_file,
        })

        output_file_name = 'tests/output/SMAP_L4_SM_aup_input_mf.h5'
        mock_mask_fill.return_value = output_file_name

        expected_response = get_xml_error_response(
            'tests/output/test',
            InvalidParameterValue('Default fill value not a number')
        )

        self.assertEqual(maskfill_sdps(), expected_response)

        # Ensure the assertion was raised prior to calling `mask_fill`.
        mock_mask_fill.assert_not_called()

    @patch('maskfill.MaskFill.get_input_parameters')
    def test_maskfill_sdps_exception(self, mock_get_input_parameters):
        """ Verify an exception is caught, and that it returns an appropriate
            response. To simulate a failure, a `None` value is given for the
            BOUNDINGSHAPE argument.

        """
        mock_get_input_parameters.return_value = self.create_parameters_namespace({
            'debug': False,
            'fill_value': None,
            'identifier': 'test',
            'input_file': 'tests/data/SMAP_L4_SM_aup_input.h5',
            'mask_grid_cache': DEFAULT_MASK_GRID_CACHE,
            'output_dir': 'tests/output',
            'shape_file': None,
        })

        expected_response = get_xml_error_response(
            'tests/output/test',
            MissingParameterValue('A shapefile is required for the mask fill utility')
        )

        self.assertEqual(maskfill_sdps(), expected_response)
