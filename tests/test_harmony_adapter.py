from logging import (basicConfig as basic_log_config, getLogger,
                     Handler as LogHandler, INFO)
from os import sep
from os.path import basename
from shutil import copy
from unittest.mock import ANY, patch

from harmony.message import Message
from harmony.util import config, HarmonyException

from harmony_adapter import HarmonyAdapter
from tests.utilities import create_input_stac, MaskFillTestCase


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
class TestHarmonyMaskFill(MaskFillTestCase):
    """ A test class that will run the full MaskFill service using the
        `HarmonyAdapter` class.

    """
    @classmethod
    def setUpClass(cls):
        """ Define class properties that do not need to be re-instantiated
            between tests.

        """
        super().setUpClass()
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
        super().setUp()
        self.log_handler.reset()

    def tearDown(self):
        super().tearDown()
        self.log_handler.reset()

    @patch('harmony_adapter.mkdtemp')
    @patch('harmony_adapter.rmtree')
    def test_harmony_adapter_hdf5(self, mock_rmtree, mock_mkdtemp,
                                  mock_download, mock_stage):
        """ Successful MaskFill run using the HarmonyAdapter and an HDF-5
            granule.

        """
        mock_mkdtemp.return_value = self.output_dir

        test_data = Message({
            'accessToken': self.access_token,
            'callback': self.callback,
            'stagingLocation': self.staging_location,
            'sources': [{
                'granules': [{
                    'bbox': self.bounding_box,
                    'temporal': self.temporal
                }]
            }],
            'subset': {
                'shape': {
                    'href': self.shape_usa,
                    'type': 'application/geo+json'
                }
            },
            'user': self.user,
        })
        input_stac = create_input_stac(self.input_hdf5, 'application/x-hdf5')

        maskfill_config = config(False)
        maskfill_adapter = HarmonyAdapter(
            test_data,
            config=maskfill_config,
            catalog=input_stac
        )
        maskfill_adapter.invoke()

        # Compare the output file to a template output file.
        expected_output_file = 'tests/data/SMAP_L4_SM_aup_output.h5'
        actual_output_file = self.create_output_file_name(self.input_hdf5,
                                                          use_identifier=False)

        self.compare_h5_files(actual_output_file, expected_output_file)

        # Check the functions to download the input data and stage the output
        # were called as expected.
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

        mock_rmtree.assert_called_once_with(self.output_dir, ignore_errors=True)

    @patch('harmony_adapter.mkdtemp')
    @patch('harmony_adapter.rmtree')
    def test_harmony_adapter_geotiff(self, mock_rmtree, mock_mkdtemp,
                                     mock_download, mock_stage):
        """ Successful MaskFill run using the HarmonyAdapter and a GeoTIFF
            granule.

        """
        mock_mkdtemp.return_value = self.output_dir

        test_data = Message({
            'accessToken': self.access_token,
            'callback': self.callback,
            'stagingLocation': self.staging_location,
            'sources': [{
                'granules': [{
                    'bbox': self.bounding_box,
                    'temporal': self.temporal
                }]
            }],
            'subset': {
                'shape': {
                    'href': self.shape_usa,
                    'type': 'application/geo+json'
                }
            },
            'user': self.user,
        })

        input_stac = create_input_stac(self.input_geotiff, 'image/tiff')

        maskfill_config = config(False)
        maskfill_adapter = HarmonyAdapter(test_data, config=maskfill_config,
                                          catalog=input_stac)
        maskfill_adapter.invoke()

        # Compare the output file to a template output file.
        expected_output_file = 'tests/data/SMAP_L4_SM_aup_output.tif'
        actual_output_file = self.create_output_file_name(self.input_geotiff,
                                                          use_identifier=False)

        self.compare_geotiff_files(actual_output_file, expected_output_file)

        # Check the functions to download the input data and stage the output
        # were called as expected.
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

        mock_rmtree.assert_called_once_with(self.output_dir, ignore_errors=True)

    @patch('harmony_adapter.mkdtemp')
    @patch('harmony_adapter.rmtree')
    def test_harmony_adapter_netcdf4_input(self, mock_rmtree, mock_mkdtemp,
                                           mock_download, mock_stage):
        """ Ensure MaskFill can run on a NetCDF-4 file (e.g., from HOSS). """
        mock_mkdtemp.return_value = self.output_dir

        input_file_name = 'tests/data/GPM_3IMERGHH_input.nc4'

        test_data = Message({
            'accessToken': self.access_token,
            'callback': self.callback,
            'stagingLocation': self.staging_location,
            'sources': [{
                'granules': [{
                    'bbox': self.bounding_box,
                    'temporal': self.temporal,
                }]
            }],
            'subset': {
                'shape': {
                    'href': self.shape_usa,
                    'type': 'application/geo+json'
                }
            },
            'user': self.user,
        })

        masked_name = 'GPM_3IMERGHH_input_mf.nc4'
        staged_name = 'GPM_3IMERGHH_input_subsetted.nc4'

        maskfill_config = config(False)
        input_stac = create_input_stac(input_file_name, 'application/netcdf-4')
        maskfill_adapter = HarmonyAdapter(test_data, config=maskfill_config,
                                          catalog=input_stac)
        maskfill_adapter.invoke()

        # Compare the output file to a template output file.
        expected_output_file = 'tests/data/GPM_3IMERGHH_output.nc4'
        actual_output_file = self.create_output_file_name(input_file_name,
                                                          use_identifier=False)

        self.compare_geotiff_files(actual_output_file, expected_output_file)

        mock_download.asset_called_once_with(self.input_geotiff,
                                             ANY,
                                             logger=maskfill_adapter.logger,
                                             access_token=self.access_token,
                                             cfg=maskfill_config)
        mock_stage.assert_called_once_with(StringEndsWith(masked_name),
                                           StringEndsWith(staged_name),
                                           'application/x-netcdf4',
                                           location=self.staging_location,
                                           logger=maskfill_adapter.logger)

        mock_rmtree.assert_called_once_with(self.output_dir, ignore_errors=True)

    @patch('harmony_adapter.mkdtemp')
    @patch('harmony_adapter.rmtree')
    def test_harmony_adapter_bbox_request(self, mock_rmtree, mock_mkdtemp,
                                          mock_download, mock_stage):
        """ Ensure MaskFill can handle a bounding box request for a
            non-geographic collection. The bounding box should encompass
            Norway, Sweden and Finland.

        """
        mock_mkdtemp.return_value = self.output_dir
        input_file_name = 'tests/data/SMAP_L3_FT_P_polar_3d_input.h5'
        masked_name = 'SMAP_L3_FT_P_polar_3d_input_mf.h5'
        staged_name = 'SMAP_L3_FT_P_polar_3d_input_subsetted.h5'

        test_data = Message({
            'accessToken': self.access_token,
            'callback': self.callback,
            'stagingLocation': self.staging_location,
            'sources': [{
                'granules': [{
                    'bbox': self.bounding_box,
                    'temporal': self.temporal,
                }]
            }],
            'subset': {'bbox': [0, 54, 44, 72]},
            'user': self.user,
        })

        maskfill_config = config(False)
        input_stac = create_input_stac(input_file_name, 'application/x-hdf5')
        maskfill_adapter = HarmonyAdapter(test_data, config=maskfill_config,
                                          catalog=input_stac)
        maskfill_adapter.invoke()

        # Compare the output file to a template output file.
        expected_output_file = 'tests/data/SMAP_L3_FT_P_polar_bbox_output.h5'
        actual_output_file = self.create_output_file_name(input_file_name,
                                                          use_identifier=False)

        self.compare_h5_files(actual_output_file, expected_output_file)

        # Check the functions to download the input data and stage the output
        # were called as expected.
        mock_download.asset_called_once_with(input_file_name,
                                             ANY,
                                             logger=maskfill_adapter.logger,
                                             access_token=self.access_token,
                                             cfg=maskfill_config)
        mock_stage.assert_called_once_with(StringEndsWith(masked_name),
                                           StringEndsWith(staged_name),
                                           'application/x-hdf5',
                                           location=self.staging_location,
                                           logger=maskfill_adapter.logger)

        mock_rmtree.assert_called_once_with(self.output_dir, ignore_errors=True)

    @patch('harmony_adapter.mkdtemp')
    @patch('harmony_adapter.rmtree')
    def test_harmony_adapter_h5_default_fill(self, mock_rmtree, mock_mkdtemp,
                                             mock_download, mock_stage):
        """ Ensure MaskFill can process a file that has no in-file fill value
            metadata, relying instead on default fill values that are selected
            based on the data type of each variable in the HDF-5 file.

        """
        mock_mkdtemp.return_value = self.output_dir
        input_file_name = 'tests/data/SMAP_L3_FT_P_fill_input.h5'
        masked_name = 'SMAP_L3_FT_P_fill_input_mf.h5'
        staged_name = 'SMAP_L3_FT_P_fill_input_subsetted.h5'

        test_data = Message({
            'accessToken': self.access_token,
            'callback': self.callback,
            'stagingLocation': self.staging_location,
            'sources': [{
                'granules': [{
                    'bbox': self.bounding_box,
                    'temporal': self.temporal,
                }]
            }],
            'subset': {
                'shape': {
                    'href': self.shape_usa,
                    'type': 'application/geo+json'
                }
            },
            'user': self.user,
        })

        maskfill_config = config(False)
        input_stac = create_input_stac(input_file_name, 'application/x-hdf5')
        maskfill_adapter = HarmonyAdapter(test_data, config=maskfill_config,
                                          catalog=input_stac)
        maskfill_adapter.invoke()

        # Compare the output file to a template output file.
        expected_output_file = 'tests/data/SMAP_L3_FT_P_fill_output.h5'
        actual_output_file = self.create_output_file_name(input_file_name,
                                                          use_identifier=False)

        self.compare_h5_files(actual_output_file, expected_output_file)

        # Check the functions to download the input data and stage the output
        # were called as expected.
        mock_download.asset_called_once_with(input_file_name,
                                             ANY,
                                             logger=maskfill_adapter.logger,
                                             access_token=self.access_token,
                                             cfg=maskfill_config)
        mock_stage.assert_called_once_with(StringEndsWith(masked_name),
                                           StringEndsWith(staged_name),
                                           'application/x-hdf5',
                                           location=self.staging_location,
                                           logger=maskfill_adapter.logger)

        mock_rmtree.assert_called_once_with(self.output_dir, ignore_errors=True)

    @patch('harmony_adapter.mkdtemp')
    @patch('harmony_adapter.rmtree')
    def test_harmony_adapter_geotiff_float_default_fill(
        self,
        mock_rmtree,
        mock_mkdtemp,
        mock_download,
        mock_stage
    ):
        """ Ensure MaskFill can process a file that has no in-file fill value
            metadata, relying instead on default fill values that are selected
            based on the data type of the band in the GeoTIFF file.

            This example tests floating point data, which has a default fill
            value of -9999.0.

        """
        mock_mkdtemp.return_value = self.output_dir

        test_data = Message({
            'accessToken': self.access_token,
            'callback': self.callback,
            'stagingLocation': self.staging_location,
            'sources': [{
                'granules': [{
                    'bbox': self.bounding_box,
                    'temporal': self.temporal,
                }]
            }],
            'subset': {
                'shape': {
                    'href': self.shape_usa,
                    'type': 'application/geo+json'
                }
            },
            'user': self.user,
        })

        maskfill_config = config(False)

        input_file_name = 'tests/data/SMAP_L3_FT_P_fill_float_input.tif'
        masked_name = 'SMAP_L3_FT_P_fill_float_input_mf.tif'
        staged_name = 'SMAP_L3_FT_P_fill_float_input_subsetted.tif'

        input_stac = create_input_stac(input_file_name, 'image/tiff')
        maskfill_adapter = HarmonyAdapter(
            test_data,
            config=maskfill_config,
            catalog=input_stac
        )
        maskfill_adapter.invoke()

        # Compare the output file to a template output file.
        expected_output_file = 'tests/data/SMAP_L3_FT_P_fill_float_output.tif'
        actual_output_file = self.create_output_file_name(
            input_file_name,
            use_identifier=False
        )

        self.compare_geotiff_files(actual_output_file, expected_output_file)

        # Check the functions to download the input data and stage the output
        # were called as expected.
        mock_download.asset_called_once_with(
            input_file_name,
            ANY,
            logger=maskfill_adapter.logger,
            access_token=self.access_token,
            cfg=maskfill_config
        )
        mock_stage.assert_called_once_with(
            StringEndsWith(masked_name),
            StringEndsWith(staged_name),
            'image/tiff',
            location=self.staging_location,
            logger=maskfill_adapter.logger
        )

        mock_rmtree.assert_called_once_with(self.output_dir, ignore_errors=True)

    @patch('harmony_adapter.mkdtemp')
    @patch('harmony_adapter.rmtree')
    def test_harmony_adapter_geotiff_uint_default_fill(
        self,
        mock_rmtree,
        mock_mkdtemp,
        mock_download,
        mock_stage
    ):
        """ Ensure MaskFill can process a file that has no in-file fill value
            metadata, relying instead on default fill values that are selected
            based on the data type of the band in the GeoTIFF file.

            This example tests unsigned integer data, which has a default fill
            value of 254.

        """
        mock_mkdtemp.return_value = self.output_dir

        test_data = Message({
            'accessToken': self.access_token,
            'callback': self.callback,
            'stagingLocation': self.staging_location,
            'sources': [{
                'granules': [{
                    'bbox': self.bounding_box,
                    'temporal': self.temporal,
                }]
            }],
            'subset': {
                'shape': {
                    'href': self.shape_usa,
                    'type': 'application/geo+json'
                }
            },
            'user': self.user,
        })

        maskfill_config = config(False)

        input_file_name = 'tests/data/SMAP_L3_FT_P_fill_uint_input.tif'
        masked_name = 'SMAP_L3_FT_P_fill_uint_input_mf.tif'
        staged_name = 'SMAP_L3_FT_P_fill_uint_input_subsetted.tif'

        input_stac = create_input_stac(input_file_name, 'image/tiff')
        maskfill_adapter = HarmonyAdapter(
            test_data,
            config=maskfill_config,
            catalog=input_stac
        )
        maskfill_adapter.invoke()

        # Compare the output file to a template output file.
        expected_output_file = 'tests/data/SMAP_L3_FT_P_fill_uint_output.tif'
        actual_output_file = self.create_output_file_name(
            input_file_name,
            use_identifier=False
        )

        self.compare_geotiff_files(actual_output_file, expected_output_file)

        # Check the functions to download the input data and stage the output
        # were called as expected.
        mock_download.asset_called_once_with(
            input_file_name,
            ANY,
            logger=maskfill_adapter.logger,
            access_token=self.access_token,
            cfg=maskfill_config
        )
        mock_stage.assert_called_once_with(
            StringEndsWith(masked_name),
            StringEndsWith(staged_name),
            'image/tiff',
            location=self.staging_location,
            logger=maskfill_adapter.logger
        )

        mock_rmtree.assert_called_once_with(self.output_dir, ignore_errors=True)

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
            'subset': {
                'shape': {
                    'href': self.shape_usa,
                    'type': self.input_hdf5
                }
            },
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
            'sources': [{
                'granules': [{
                    'bbox': self.bounding_box,
                    'temporal': self.temporal,
                    'url': self.input_geotiff
                }]
            }],
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
                             'MaskFill requires a shape file or bounding box '
                             'that describes a mask.')

        with self.subTest('NoneType Message.subset.shape.href'):
            message_text = base_message_text.copy()
            message_text['subset'] = {'shape': {'type': 'application/geo+json'}}
            message = Message(message_text)
            maskfill_adapter = HarmonyAdapter(message, config=config(False))

            with self.assertRaises(HarmonyException) as context:
                maskfill_adapter.invoke()

            mock_stage.assert_not_called()
            self.assertEqual(context.exception.message,
                             'Shape file must specify resource URL.')

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
