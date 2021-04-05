from logging import (basicConfig as basic_log_config, getLogger,
                     Handler as LogHandler, INFO)
from os import sep
from os.path import basename
from shutil import copy
from unittest import TestCase
from unittest.mock import ANY, patch

from harmony.message import Message
from harmony.util import config, HarmonyException

from harmony_adapter import HarmonyAdapter


class StringEndsWith:
    """ A custom matcher that can be used in `unittest` assertions, ensuring
        a string ends with the expected arguments.

    """
    def __init__(self, expected_string_ending):
        self.expected_string_ending = expected_string_ending

    def __eq__(self, string_to_check):
        return string_to_check.endswith(self.expected_string_ending)


class TestLogHandler(LogHandler):
    """ A log handler to enable the capturing of Harmony logging messages
        during test runs.

    """
    messages = []

    def emit(self, record):
        self.messages.append(self.format(record))

    def reset(self):
        self.messages = []


def download_side_effect(file_path, working_dir, **kwargs):
    """ A side effect to be used when mocking the `harmony.util.download`
        function. This should copy the input file (assuming it is a local
        file path) to the working directory, and then return the new file
        path.

    """
    file_base_name = basename(file_path)
    output_file_path = sep.join([working_dir, file_base_name])

    copy(file_path, output_file_path)
    return output_file_path


@patch('harmony_adapter.stage', return_value='https://example.com/data')
@patch('harmony_adapter.download', side_effect=download_side_effect)
class TestHarmonyMaskFill(TestCase):
    """ A test class that will run the full MaskFill service using the
        `HarmonyAdapter` class.

    """
    @classmethod
    def setUpClass(cls):
        """ Define class properties that do not need to be re-instantiated
            between tests.

        """
        cls.access_token = 'fake_token'
        cls.bounding_box = [-180, -90, 180, 90]
        cls.callback = 'https://example.com/callback'
        cls.input_geotiff = 'tests/data/SMAP_L4_SM_aup_input.tif'
        cls.input_hdf5 = 'tests/data/SMAP_L4_SM_aup_input.h5'
        cls.log_handler = TestLogHandler()
        cls.logger = getLogger('test')
        cls.mimetype_geotiff = 'image/tiff'
        cls.mimetype_hdf5 = 'application/x-hdf5'
        cls.masked_geotiff = 'SMAP_L4_SM_aup_input_mf.tif'
        cls.masked_hdf5 = 'SMAP_L4_SM_aup_input_mf.h5'
        cls.shape_usa = 'tests/data/USA.geo.json'
        cls.staged_geotiff = 'SMAP_L4_SM_aup_input_subsetted.tif'
        cls.staged_hdf5 = 'SMAP_L4_SM_aup_input_subsetted.h5'
        cls.staging_location = 's3://example-bucket/example-path'
        cls.temporal = {'start': '2021-01-01T00:00:00.000Z',
                        'end': '2021-01-01T00:00:00.000Z'}
        cls.user = 'cyeager'

        basic_log_config(format='%(levelname)s: %(message)s',
                         handlers=[cls.log_handler], level=INFO)

    def setUp(self):
        self.log_handler.reset()

    def tearDown(self):
        self.log_handler.reset()

    def test_harmony_adapter_hdf5(self, mock_download, mock_stage):
        """ Successful MaskFill run using the HarmonyAdapter and an HDF-5
            granule.

        """
        test_data = Message({
            'accessToken': self.access_token,
            'callback': self.callback,
            'stagingLocation': self.staging_location,
            'sources': [{'granules': [{'bbox': self.bounding_box,
                                       'temporal': self.temporal,
                                       'url': self.input_hdf5}]}],
            'subset': {'shape': {'href': self.shape_usa,
                                 'type': 'application/geo+json'}},
            'user': self.user,
        })

        maskfill_config = config(False)
        maskfill_adapter = HarmonyAdapter(test_data, config=maskfill_config)
        maskfill_adapter.invoke()

        mock_download.asset_called_once_with(self.input_hdf5,
                                             ANY,
                                             logger=maskfill_adapter.logger,
                                             access_token=self.access_token,
                                             cfg=maskfill_config)
        mock_stage.assert_called_once_with(StringEndsWith(self.masked_hdf5),
                                           StringEndsWith(self.staged_hdf5),
                                           self.mimetype_hdf5,
                                           location=self.staging_location,
                                           logger=maskfill_adapter.logger)

    def test_harmony_adapter_geotiff(self, mock_download, mock_stage):
        """ Successful MaskFill run using the HarmonyAdapter and a GeoTIFF
            granule.

        """
        test_data = Message({
            'accessToken': self.access_token,
            'callback': self.callback,
            'stagingLocation': self.staging_location,
            'sources': [{'granules': [{'bbox': self.bounding_box,
                                       'temporal': self.temporal,
                                       'url': self.input_geotiff}]}],
            'subset': {'shape': {'href': self.shape_usa,
                                 'type': 'application/geo+json'}},
            'user': self.user,
        })

        maskfill_config = config(False)
        maskfill_adapter = HarmonyAdapter(test_data, config=maskfill_config)
        maskfill_adapter.invoke()

        mock_download.asset_called_once_with(self.input_geotiff,
                                             ANY,
                                             logger=maskfill_adapter.logger,
                                             access_token=self.access_token,
                                             cfg=maskfill_config)
        mock_stage.assert_called_once_with(StringEndsWith(self.masked_geotiff),
                                           StringEndsWith(self.staged_geotiff),
                                           self.mimetype_geotiff,
                                           location=self.staging_location,
                                           logger=maskfill_adapter.logger)

    def test_validate_message_no_message(self, mock_download, mock_stage):
        """ Ensure that a `NoneType` message will raise an exception, during
            validation.

        """
        maskfill_adapter = HarmonyAdapter(None, config=config(False))

        with self.assertRaises(HarmonyException) as context:
            maskfill_adapter.invoke()

        mock_download.assert_not_called()
        mock_stage.assert_not_called()
        self.assertEqual(context.exception.message, 'No message request')

    def test_validate_message_no_granules(self, mock_download, mock_stage):
        """ Ensure that a message with no listed granules raises a
            HarmonyException, and does not attempt to process any further.

        """
        test_data = Message({
            'accessToken': self.access_token,
            'callback': self.callback,
            'stagingLocation': self.staging_location,
            'subset': {'shape': {'href': self.shape_usa,
                                 'type': self.input_hdf5}},
            'user': self.user,
        })

        maskfill_adapter = HarmonyAdapter(test_data, config=config(False))

        with self.assertRaises(HarmonyException) as context:
            maskfill_adapter.invoke()

        mock_download.assert_not_called()
        mock_stage.assert_not_called()
        self.assertEqual(context.exception.message,
                         'No granules specified for reprojection')

    def test_validate_message_shape_file(self, mock_download, mock_stage):
        """ Ensure that if MaskFill is called without a shape file specified,
            or with a poorly specified shape file, it will raise a
            HarmonyException.

        """
        base_message_text = {
            'accessToken': self.access_token,
            'callback': self.callback,
            'sources': [{'granules': [{'bbox': self.bounding_box,
                                       'temporal': self.temporal,
                                       'url': self.input_geotiff}]}],
            'stagingLocation': self.staging_location,
            'user': self.user
        }

        with self.subTest('No shape specified in Message.subset'):
            message = Message(base_message_text)
            maskfill_adapter = HarmonyAdapter(message, config=config(False))

            with self.assertRaises(HarmonyException) as context:
                maskfill_adapter.invoke()

            mock_download.assert_not_called()
            mock_stage.assert_not_called()
            self.assertEqual(context.exception.message,
                             'Shape file must be specified for masking.')

        with self.subTest('NoneType Message.subset.shape.href'):
            message_text = base_message_text.copy()
            message_text['subset'] = {'shape': {'type': 'application/geo+json'}}
            message = Message(message_text)
            maskfill_adapter = HarmonyAdapter(message, config=config(False))

            with self.assertRaises(HarmonyException) as context:
                maskfill_adapter.invoke()

            mock_stage.assert_not_called()
            self.assertEqual(context.exception.message,
                             'Shape file must be specified for masking.')

        with self.subTest('Incorrect shapefile MIME type'):
            message_text = base_message_text.copy()
            message_text['subset'] = {'shape': {'href': self.shape_usa,
                                                'type': 'image/tiff'}}
            message = Message(message_text)
            maskfill_adapter = HarmonyAdapter(message, config=config(False))

            with self.assertRaises(HarmonyException) as context:
                maskfill_adapter.invoke()

            mock_stage.assert_not_called()
            self.assertEqual(context.exception.message,
                             'Shape file must be GeoJSON format.')

        with self.subTest('NoneType Message.subset.shape.type'):
            message_text = base_message_text.copy()
            message_text['subset'] = {'shape': {'href': self.shape_usa}}
            message = Message(message_text)
            maskfill_adapter = HarmonyAdapter(message, config=config(False))

            with self.assertRaises(HarmonyException) as context:
                maskfill_adapter.invoke()

            mock_stage.assert_not_called()
            self.assertEqual(context.exception.message,
                             'Shape file must be GeoJSON format.')
