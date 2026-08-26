"""Unit tests for history.py."""

import json
import os
import shutil
import tempfile
from unittest import TestCase

import h5py
from freezegun import freeze_time

from maskfill.history import (
    PROGRAM,
    PROGRAM_REF,
    get_semantic_version,
    update_history_metadata,
)

# Constants for test data
SHAPE_HASH = 'e1e473eb0276a31c09d07509d0b75018f8950410735a5a4ecaa43bab'
FROZEN_TIME = '2000-01-02T03:04:05'


class TestHistory(TestCase):
    """Tests for the functions in maskfill/history.py."""

    def setUp(self):
        """Set up test-specific resources."""
        self.tmp_dir = tempfile.mkdtemp()
        self.shape_file = 'tests/data/USA.geo.json'
        self.bounding_box = []
        self.fillvalue = -9999.0
        self.version = get_semantic_version()

    def tearDown(self):
        """Clean up resources after tests."""
        shutil.rmtree(self.tmp_dir)

    def copy_source_file_to_temp_dir(self, source_path):
        """Helper to copy source file to temp directory."""
        dest_path = os.path.join(self.tmp_dir, os.path.basename(source_path))
        shutil.copy2(source_path, dest_path)
        return dest_path

    def get_history_json_record(self, input_file, bounding_box=None):
        """Returns the standard JSON structure for history metadata."""

        if bounding_box:
            shape, shape_value = ('bbox', bounding_box)
        else:
            shape, shape_value = ('shape_file_hash', SHAPE_HASH)

        return {
            '$schema': (
                'https://harmony.earthdata.nasa.gov/schemas/history/'
                '0.1.0/history-v0.1.0.json'
            ),
            'date_time': f'{FROZEN_TIME}+00:00',
            'program': PROGRAM,
            'version': self.version,
            'parameters': {
                shape: shape_value,
                'fill_value': self.fillvalue,
            },
            'derived_from': input_file,
            'program_ref': PROGRAM_REF,
        }

    def assert_history(self, input_filename, history, history_json, key='history'):
        """Centralized assertion logic for history attributes."""
        with h5py.File(input_filename, 'r') as h5_input_file:
            history_key = 'History' if key == 'history' else 'history'
            self.assertIn(key, h5_input_file.attrs)
            self.assertNotIn(history_key, h5_input_file.attrs)

            self.assertEqual(h5_input_file.attrs[key], history)
            actual_history_json = json.loads(h5_input_file.attrs['history_json'])
            self.assertEqual(actual_history_json, history_json)

    @freeze_time(FROZEN_TIME)
    def test_update_history_metadata_no_history(self):
        """Test creating a new history attribute when none exists."""
        source = 'tests/data/GPM_3IMERGHH_input.nc4'
        input_filename = self.copy_source_file_to_temp_dir(source)

        expected_history = (
            f'{FROZEN_TIME}+00:00 Harmony Maskfill {self.version} {{'
            f'"shape_file_hash": "{SHAPE_HASH}", '
            f'"fill_value": {self.fillvalue}}}'
        )

        expected_history_json = [self.get_history_json_record(input_filename)]

        update_history_metadata(
            input_filename, self.shape_file, self.fillvalue, self.bounding_box
        )
        self.assert_history(input_filename, expected_history, expected_history_json)

    @freeze_time(FROZEN_TIME)
    def test_update_history_metadata_append_history(self):
        """Test appending to existing history and history_json."""
        source = 'tests/data/SC_SPL3SMP_subsetted_with_maskfill_mf.nc4'
        input_filename = self.copy_source_file_to_temp_dir(source)
        url = (
            'https://opendap.uat.earthdata.nasa.gov/collections/'
            'C1268452365-EEDTEST/granules/SC:SPL3SMP.008:240468423.dap.nc4'
        )
        previous_history = (
            f'2025-03-03 20:49:33 GMT hyrax-1.17.1-63 {url}'
            '?A-api-request-uuid=bd99a1e3-f5ca-43c7-9d21-a3a95eea0cd9,'
            'dap4.ce=%2FSoil_Moisture_Retrieval_Data_AM%2Flatitude%5B0%3A26'
            '%5D%5B294%3A455%5D%3B%2FSoil_Moisture_Retrieval_Data_AM%2Flong'
            'itude%5B0%3A26%5D%5B294%3A455%5D%3B%2FSoil_Moisture_Retrieval_'
            'Data_AM%2Flandcover_class_fraction%5B0%3A26%5D%5B294%3A455%5D'
            '%5B%5D'
        )

        expected_history_json = [
            {
                '$schema': (
                    'https://harmony.earthdata.nasa.gov/schemas/history/'
                    '0.1.0/history-0.1.0.json'
                ),
                'date_time': '2025-03-03T20:49:33.135+0000',
                'program': 'hyrax',
                'version': '1.17.1-63',
                'parameters': [
                    {
                        'request_url': (
                            'https://opendap.uat.earthdata.nasa.gov/collections/'
                            'C1268452365-EEDTEST/granules/'
                            'SC:SPL3SMP.008:240468423.dap.nc4'
                            '?A-api-request-uuid=bd99a1e3-f5ca-43c7-9d21-'
                            'a3a95eea0cd9,dap4.ce=%2FSoil_Moisture_Retrieval_Data_AM'
                            '%2Flatitude%5B0%3A26%5D%5B294%3A455%5D%3B'
                            '%2FSoil_Moisture_Retrieval_Data_AM%2Flongitude%5B0%3A26'
                            '%5D%5B294%3A455%5D%3B%2FSoil_Moisture_Retrieval_Data_AM'
                            '%2Flandcover_class_fraction%5B0%3A26%5D%5B294%3A455%5D'
                            '%5B%5D'
                        )
                    },
                    {
                        'decoded_constraint': (
                            'A-api-request-uuid=bd99a1e3-f5ca-43c7-9d21-'
                            'a3a95eea0cd9,dap4.ce=/Soil_Moisture_Retrieval_Data_AM/'
                            'latitude[0:26][294:455];'
                            '/Soil_Moisture_Retrieval_Data_AM/longitude[0:26]'
                            '[294:455];'
                            '/Soil_Moisture_Retrieval_Data_AM/'
                            'landcover_class_fraction[0:26][294:455][]'
                        )
                    },
                ],
            }
        ]

        # Append previous histroy with new service history
        expected_history = (
            f'{previous_history}\n\n{FROZEN_TIME}+00:00 Harmony Maskfill '
            f'{self.version} {{"shape_file_hash": "{SHAPE_HASH}", '
            f'"fill_value": {self.fillvalue}}}'
        )

        new_history_json = self.get_history_json_record(url)
        expected_history_json.append(new_history_json)

        update_history_metadata(
            input_filename, self.shape_file, self.fillvalue, self.bounding_box
        )
        self.assert_history(input_filename, expected_history, expected_history_json)

    @freeze_time(FROZEN_TIME)
    def test_update_history_metadata_capital_history(self):
        """Test appending to a capitalized "History" attribute."""
        source = 'tests/data/SMAP_L4_SM_aup_UTM_output.h5'
        input_filename = self.copy_source_file_to_temp_dir(source)
        previous_history = 'File written by ldas2daac.x'

        expected_history = (
            f'{previous_history}\n{FROZEN_TIME}+00:00 Harmony Maskfill '
            f'{self.version} {{"shape_file_hash": "{SHAPE_HASH}", '
            f'"fill_value": {self.fillvalue}}}'
        )

        expected_history_json = [self.get_history_json_record(input_filename)]

        update_history_metadata(
            input_filename, self.shape_file, self.fillvalue, self.bounding_box
        )
        self.assert_history(
            input_filename, expected_history, expected_history_json, key='History'
        )

    @freeze_time(FROZEN_TIME)
    def test_update_history_metadata_annotator_service(self):
        """Test appending history from Metadata Annotator Service."""
        source = 'tests/data/SC_SPL2SMAP_S_subsetted_annotated.nc4'
        input_filename = self.copy_source_file_to_temp_dir(source)
        url = (
            'https://opendap.uat.earthdata.nasa.gov/collections/'
            'C1268429762-EEDTEST/granules/SC:SPL2SMAP_S.003:241398946.dap.nc4'
        )
        prev_parts = [
            f'2025-04-29 22:13:06 GMT hyrax-1.17.1-133 {url}'
            '?A-api-request-uuid=3db86352-615b-4b65-be7f-65b8c05ba695,'
            'dap4.ce=%2FSoil_Moisture_Retrieval_Data_3km%2Falbedo_3km%3B'
            '%2FSoil_Moisture_Retrieval_Data_1km%2FEASE_row_index_1km%3B'
            '%2FSoil_Moisture_Retrieval_Data_1km%2Falbedo_1km%3B'
            '%2FSoil_Moisture_Retrieval_Data_1km%2Flongitude_1km%3B'
            '%2FSoil_Moisture_Retrieval_Data_3km%2FEASE_row_index_3km%3B'
            '%2FSoil_Moisture_Retrieval_Data_1km%2Flatitude_1km%3B'
            '%2FSoil_Moisture_Retrieval_Data_3km%2Flatitude_3km%3B'
            '%2FSoil_Moisture_Retrieval_Data_3km%2FEASE_column_index_3km%3B'
            '%2FSoil_Moisture_Retrieval_Data_3km%2Flongitude_3km%3B'
            '%2FSoil_Moisture_Retrieval_Data_1km%2FEASE_column_index_1km',
            '2025-05-08T19:14:18.390538+00:00 Harmony Metadata Annotator 0.0.1',
        ]

        expected_history_json = [
            {
                '$schema': (
                    'https://harmony.earthdata.nasa.gov/schemas/history/0.1.0/'
                    'history-0.1.0.json'
                ),
                'date_time': '2025-04-29T22:13:06.875+0000',
                'program': 'hyrax',
                'version': '1.17.1-133',
                'parameters': [
                    {
                        'request_url': (
                            'https://opendap.uat.earthdata.nasa.gov/collections/'
                            'C1268429762-EEDTEST/granules/'
                            'SC:SPL2SMAP_S.003:241398946.dap.nc4'
                            '?A-api-request-uuid=3db86352-615b-4b65-be7f-65b8c05ba695,'
                            'dap4.ce=%2FSoil_Moisture_Retrieval_Data_3km%2Falbedo_3km'
                            '%3B%2FSoil_Moisture_Retrieval_Data_1km%2FEASE_row_index_'
                            '1km%3B%2FSoil_Moisture_Retrieval_Data_1km%2Falbedo_1km%3B'
                            '%2FSoil_Moisture_Retrieval_Data_1km%2Flongitude_1km%3B'
                            '%2FSoil_Moisture_Retrieval_Data_3km%2FEASE_row_index_3km'
                            '%3B%2FSoil_Moisture_Retrieval_Data_1km%2Flatitude_1km%3B'
                            '%2FSoil_Moisture_Retrieval_Data_3km%2Flatitude_3km%3B'
                            '%2FSoil_Moisture_Retrieval_Data_3km%2FEASE_column_index_'
                            '3km%3B%2FSoil_Moisture_Retrieval_Data_3km%2Flongitude_3km'
                            '%3B%2FSoil_Moisture_Retrieval_Data_1km%2FEASE_column_'
                            'index_1km'
                        )
                    },
                    {
                        'decoded_constraint': (
                            'A-api-request-uuid=3db86352-615b-4b65-be7f-65b8c05ba695,'
                            'dap4.ce=/Soil_Moisture_Retrieval_Data_3km/albedo_3km;'
                            '/Soil_Moisture_Retrieval_Data_1km/EASE_row_index_1km;'
                            '/Soil_Moisture_Retrieval_Data_1km/albedo_1km;'
                            '/Soil_Moisture_Retrieval_Data_1km/longitude_1km;'
                            '/Soil_Moisture_Retrieval_Data_3km/EASE_row_index_3km;'
                            '/Soil_Moisture_Retrieval_Data_1km/latitude_1km;'
                            '/Soil_Moisture_Retrieval_Data_3km/latitude_3km;'
                            '/Soil_Moisture_Retrieval_Data_3km/EASE_column_index_3km;'
                            '/Soil_Moisture_Retrieval_Data_3km/longitude_3km;'
                            '/Soil_Moisture_Retrieval_Data_1km/EASE_column_index_1km'
                        )
                    },
                ],
            }
        ]

        expected_history = (
            f'{prev_parts[0]}\n\n{prev_parts[1]}\n'
            f'{FROZEN_TIME}+00:00 Harmony Maskfill {self.version} '
            f'{{"shape_file_hash": "{SHAPE_HASH}", '
            f'"fill_value": {self.fillvalue}}}'
        )

        new_history_json = self.get_history_json_record(url)
        expected_history_json.append(new_history_json)

        update_history_metadata(
            input_filename, self.shape_file, self.fillvalue, self.bounding_box
        )
        self.assert_history(input_filename, expected_history, expected_history_json)

    @freeze_time(FROZEN_TIME)
    def test_update_history_metadata_append_history_bounding_box(self):
        """Test appending to existing history and history_json."""
        source = 'tests/data/SC_SPL3SMP_subsetted_with_maskfill_mf.nc4'
        input_filename = self.copy_source_file_to_temp_dir(source)
        url = (
            'https://opendap.uat.earthdata.nasa.gov/collections/'
            'C1268452365-EEDTEST/granules/SC:SPL3SMP.008:240468423.dap.nc4'
        )
        previous_history = (
            f'2025-03-03 20:49:33 GMT hyrax-1.17.1-63 {url}'
            '?A-api-request-uuid=bd99a1e3-f5ca-43c7-9d21-a3a95eea0cd9,'
            'dap4.ce=%2FSoil_Moisture_Retrieval_Data_AM%2Flatitude%5B0%3A26'
            '%5D%5B294%3A455%5D%3B%2FSoil_Moisture_Retrieval_Data_AM%2Flong'
            'itude%5B0%3A26%5D%5B294%3A455%5D%3B%2FSoil_Moisture_Retrieval_'
            'Data_AM%2Flandcover_class_fraction%5B0%3A26%5D%5B294%3A455%5D'
            '%5B%5D'
        )

        bounding_box = [0, 54, 44, 72]

        expected_history = (
            f'{previous_history}\n\n{FROZEN_TIME}+00:00 Harmony Maskfill '
            f'{self.version} {{"bbox": {bounding_box}, '
            f'"fill_value": {self.fillvalue}}}'
        )

        expected_history_json = [
            {
                '$schema': (
                    'https://harmony.earthdata.nasa.gov/schemas/history/0.1.0/'
                    'history-0.1.0.json'
                ),
                'date_time': '2025-03-03T20:49:33.135+0000',
                'program': 'hyrax',
                'version': '1.17.1-63',
                'parameters': [
                    {
                        'request_url': (
                            'https://opendap.uat.earthdata.nasa.gov/collections/'
                            'C1268452365-EEDTEST/granules/'
                            'SC:SPL3SMP.008:240468423.dap.nc4'
                            '?A-api-request-uuid=bd99a1e3-f5ca-43c7-9d21-'
                            'a3a95eea0cd9,dap4.ce=%2FSoil_Moisture_Retrieval_Data_AM'
                            '%2Flatitude%5B0%3A26%5D%5B294%3A455%5D%3B'
                            '%2FSoil_Moisture_Retrieval_Data_AM%2Flongitude%5B0%3A26'
                            '%5D%5B294%3A455%5D%3B%2FSoil_Moisture_Retrieval_Data_AM'
                            '%2Flandcover_class_fraction%5B0%3A26%5D%5B294%3A455%5D'
                            '%5B%5D'
                        )
                    },
                    {
                        'decoded_constraint': (
                            'A-api-request-uuid=bd99a1e3-f5ca-43c7-9d21-'
                            'a3a95eea0cd9,dap4.ce=/Soil_Moisture_Retrieval_Data_AM/'
                            'latitude[0:26][294:455];'
                            '/Soil_Moisture_Retrieval_Data_AM/longitude[0:26]'
                            '[294:455];'
                            '/Soil_Moisture_Retrieval_Data_AM/'
                            'landcover_class_fraction[0:26][294:455][]'
                        )
                    },
                ],
            }
        ]

        new_history_json = self.get_history_json_record(url, bounding_box)
        expected_history_json.append(new_history_json)

        update_history_metadata(
            input_filename, self.shape_file, self.fillvalue, bounding_box
        )
        self.assert_history(input_filename, expected_history, expected_history_json)
