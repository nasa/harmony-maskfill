from logging import (basicConfig as basic_log_config, getLogger,
                     Handler as LogHandler, INFO)
from unittest import TestCase

from harmony.message import Message
from harmony.util import config, HarmonyException

from harmony_adapter import HarmonyAdapter
from tests.python.test_harmony_adapter import TestLogHandler


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
