from logging import (basicConfig as basic_log_config, getLogger,
                     Handler as LogHandler, INFO)
from os.path import exists as file_exists, join as path_join
from shutil import rmtree
from tempfile import mkdtemp
from unittest import TestCase
from unittest.mock import patch

from harmony.message import Message
from harmony.util import config, HarmonyException

from harmony_adapter import HarmonyAdapter
from tests.test_harmony_adapter import TestLogHandler


class TestHarmonyMaskFill(TestCase):
    """ A test class that will run the full MaskFill service using the
        `HarmonyAdapter` class.

    """
    @classmethod
    def setUpClass(cls):
        """ Define class properties that do not need to be re-instantiated
            between tests.

        """
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
        cls.temporal = {'start': '2021-01-01T00:00:00.000Z',
                        'end': '2021-01-01T00:00:00.000Z'}

        cls.message = Message({
            'accessToken': 'fake_token',
            'callback': 'https://example.com/callback',
            'sources': [{'granules': [{'bbox': [-180, -90, 180, 90],
                                       'temporal': cls.temporal,
                                       'url': cls.input_hdf5}]}],
            'stagingLocation': 's3://example-bucket/example-path',
            'subset': {'shape': {'href': cls.shape_usa,
                                 'type': 'application/geo+json'}},
            'user': 'cyeaget',
        })
        cls.harmony_adapter = HarmonyAdapter(cls.message, config=config(False))
        basic_log_config(format='%(levelname)s: %(message)s',
                         handlers=[cls.log_handler], level=INFO)

    def setUp(self):
        self.log_handler.reset()

    def tearDown(self):
        self.log_handler.reset()

    def test_validate_input_granule(self):
        """ Ensure that only a granule with the expected extension passes
            validation when that item is being processed.

        """
        with self.subTest('Valid HDF-5 granule name passes validation'):
            self.harmony_adapter.validate_input_granule(self.input_hdf5)

        with self.subTest('Valid GeoTIFF granule name passes validation'):
            self.harmony_adapter.validate_input_granule(self.input_geotiff)

        with self.subTest('Inferred MIME type of None fails validation'):
            with self.assertRaises(HarmonyException) as context:
                self.harmony_adapter.validate_input_granule('file.unknown')

            self.assertEqual(context.exception.message,
                             'Invalid granule format: .unknown')

        with self.subTest('Unexpected MIME type fails validation'):
            with self.assertRaises(HarmonyException) as context:
                self.harmony_adapter.validate_input_granule('file.json')

            self.assertEqual(context.exception.message,
                             'Invalid granule format: application/json')

    def test_get_file_mimetype(self):
        """ Ensure that if a MIME type cannot be gained from the native Python
            `mimetypes.guess_type` function, other formats can be retrieved.
            For example, HDF-5 or GeoTIFF.

        """
        test_arguments = [
            ['GeoTIFF, known to Python', self.input_geotiff, 'image/tiff'],
            ['HDF-5, with custom definition', self.input_hdf5, 'application/x-hdf5'],
            ['Unknown file extension', 'file.unknown', None]
        ]

        for description, file_name, expected_mimetype in test_arguments:
            with self.subTest(description):
                self.assertEqual(
                    self.harmony_adapter.get_file_mimetype(file_name),
                    expected_mimetype
                )

    @patch('harmony_adapter.download')
    def test_download_from_remote(self, mock_download):
        """ Ensure that a specified resource is downloaded using the
            `harmony-service-lib-py` via the `harmony.util.download` function.

            This function has an optional parameter to specify the name of the
            downloaded file. Harmony, by default will create a file with a UUID
            for the basename. This UUID prevents the identification of the
            collection to which the granule belongs, if the granule does not
            have a shortname global attribute.

        """
        remote_resource = 'www.example.com/amazing_file.nc4'
        local_basename = 'name_with_prefix.nc4'

        def download_side_effect(url, directory, logger, access_token,
                                 cfg) -> str:
            """ This side effect will create a mock downloaded file. """
            downloaded_file = path_join(directory, 'random.nc4')
            with open(downloaded_file, 'w') as file_handler:
                file_handler.write('content')

            return downloaded_file

        mock_download.side_effect = download_side_effect

        with self.subTest('No local basename specified'):
            test_dir = mkdtemp()
            downloaded_file = self.harmony_adapter.download_from_remote(
                remote_resource, test_dir
            )

            expected_file_name = path_join(test_dir, 'random.nc4')
            self.assertEqual(downloaded_file, expected_file_name)
            self.assertTrue(file_exists(expected_file_name))
            rmtree(test_dir)

        with self.subTest('Basename specified and used'):
            test_dir = mkdtemp()
            downloaded_file = self.harmony_adapter.download_from_remote(
                remote_resource, test_dir, local_basename
            )

            expected_file_name = path_join(test_dir, local_basename)
            self.assertEqual(downloaded_file, expected_file_name)
            self.assertFalse(file_exists(path_join(test_dir, 'random.nc4')))
            self.assertTrue(file_exists(expected_file_name))
            rmtree(test_dir)

    def test_message_has_valid_shape_file(self):
        """ Ensure that an input message is correctly parsed to determine it
            contains a fully defined shape file.

        """
        with self.subTest('Valid shape file definition'):
            message = Message({
                'subset': {'shape': {'href': 'www.example.com/shape.geo.json',
                                     'type': 'application/geo+json'}}
            })
            harmony_adapter = HarmonyAdapter(message, config=config(False))
            self.assertTrue(harmony_adapter.message_has_valid_shape_file())

        with self.subTest('Undefined Message.subset.shape.href'):
            message = Message({
                'subset': {'shape': {'type': 'application/geo+json'}}
            })
            harmony_adapter = HarmonyAdapter(message, config=config(False))

            with self.assertRaises(HarmonyException) as context_manager:
                harmony_adapter.message_has_valid_shape_file()

            self.assertEqual(str(context_manager.exception),
                             'Shape file must specify resource URL.')

        with self.subTest('Missing shape file MIME type'):
            message = Message({
                'subset': {'shape': {'href': 'www.example.com/shape.geo.json'}}
            })
            harmony_adapter = HarmonyAdapter(message, config=config(False))

            with self.assertRaises(HarmonyException) as context_manager:
                harmony_adapter.message_has_valid_shape_file()

            self.assertEqual(context_manager.exception.message,
                             'Shape file must be GeoJSON format.')

        with self.subTest('Incorrect MIME type'):
            message = Message({
                'subset': {'shape': {'href': 'www.example.com/shape.geo.json',
                                     'type': 'application/other'}}
            })
            harmony_adapter = HarmonyAdapter(message, config=config(False))

            with self.assertRaises(HarmonyException) as context_manager:
                harmony_adapter.message_has_valid_shape_file()

            self.assertEqual(context_manager.exception.message,
                             'Shape file must be GeoJSON format.')

        with self.subTest('Absent shape file'):
            message = Message({'subset': {'bbox': [10, 20, 30, 40]}})
            harmony_adapter = HarmonyAdapter(message, config=config(False))
            self.assertFalse(harmony_adapter.message_has_valid_shape_file())

    def test_message_has_valid_bounding_box(self):
        """ Ensure that an input message is correctly parsed to determine if it
            contains a valid bounding box.

        """
        with self.subTest('Valid bounding box'):
            message = Message({'subset': {'bbox': [10, 20, 30, 40]}})
            harmony_adapter = HarmonyAdapter(message, config=config(False))
            self.assertTrue(harmony_adapter.message_has_valid_bounding_box())

        with self.subTest('No bounding box'):
            message = Message({'subset': {'shape': {}}})
            harmony_adapter = HarmonyAdapter(message, config=config(False))
            self.assertFalse(harmony_adapter.message_has_valid_bounding_box())

        with self.subTest('Non-list bounding box'):
            message = Message({'subset': {'bbox': 10}})
            harmony_adapter = HarmonyAdapter(message, config=config(False))

            with self.assertRaises(HarmonyException) as context_manager:
                harmony_adapter.message_has_valid_bounding_box()

            self.assertEqual(context_manager.exception.message,
                             'Bounding box must be 4-element list.')

        with self.subTest('Bounding box has the wrong number of elements'):
            message = Message({'subset': {'bbox': [10, 20, 30]}})
            harmony_adapter = HarmonyAdapter(message, config=config(False))

            with self.assertRaises(HarmonyException) as context_manager:
                harmony_adapter.message_has_valid_bounding_box()

            self.assertEqual(context_manager.exception.message,
                             'Bounding box must be 4-element list.')
