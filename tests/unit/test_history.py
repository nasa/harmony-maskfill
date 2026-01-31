"""Unit tests for history.py."""

import os
import shutil
import tempfile
import json
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
SHAPE_HASH = "e1e473eb0276a31c09d07509d0b75018f8950410735a5a4ecaa43bab"
FROZEN_TIME = "2000-01-02T03:04:05"


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

    def prepare_file(self, source_path):
        """Helper to copy source file to temp directory."""
        dest_path = os.path.join(self.tmp_dir, os.path.basename(source_path))
        shutil.copy2(source_path, dest_path)
        return dest_path

    def get_base_json(self, input_file, bounding_box=None):
        """Returns the standard JSON structure for history metadata."""
        shape, shape_value = (
            ('bbox', bounding_box) if bounding_box
            else ('shape_file_hash', SHAPE_HASH)
        )

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

    def assert_history(self, file_path, exp_txt, exp_json, key="history"):
        """Centralized assertion logic for history attributes."""
        with h5py.File(file_path, 'r') as output_file:
            other_key = "History" if key == "history" else "history"
            self.assertIn(key, output_file.attrs)
            self.assertNotIn(other_key, output_file.attrs)

            self.assertEqual(output_file.attrs[key], exp_txt)
            actual_json = json.loads(output_file.attrs["history_json"])
            self.assertEqual(actual_json, exp_json)

    @freeze_time(FROZEN_TIME)
    def test_update_history_metadata_no_history(self):
        """Test creating a new history attribute when none exists."""
        source = 'tests/data/GPM_3IMERGHH_input.nc4'
        dest = self.prepare_file(source)

        expected_history = (
            f'{FROZEN_TIME}+00:00 Harmony Maskfill {self.version} {{'
            f'"shape_file_hash": "{SHAPE_HASH}", '
            f'"fill_value": {self.fillvalue}}}'
        )

        update_history_metadata(
            dest, self.shape_file, self.fillvalue, self.bounding_box
        )
        self.assert_history(dest, expected_history, self.get_base_json(dest))

    @freeze_time(FROZEN_TIME)
    def test_update_history_metadata_append_history(self):
        """Test appending to existing history and history_json."""
        source = 'tests/data/SC_SPL3SMP_subsetted_with_maskfill_mf.nc4'
        dest = self.prepare_file(source)
        url = (
            'https://opendap.uat.earthdata.nasa.gov/collections/'
            'C1268452365-EEDTEST/granules/SC:SPL3SMP.008:240468423.dap.nc4'
        )
        prev = (
            f'2025-03-03 20:49:33 GMT hyrax-1.17.1-63 {url}'
            '?A-api-request-uuid=bd99a1e3-f5ca-43c7-9d21-a3a95eea0cd9,'
            'dap4.ce=%2FSoil_Moisture_Retrieval_Data_AM%2Flatitude%5B0%3A26'
            '%5D%5B294%3A455%5D%3B%2FSoil_Moisture_Retrieval_Data_AM%2Flong'
            'itude%5B0%3A26%5D%5B294%3A455%5D%3B%2FSoil_Moisture_Retrieval_'
            'Data_AM%2Flandcover_class_fraction%5B0%3A26%5D%5B294%3A455%5D'
            '%5B%5D'
        )

        expected_history = (
            f'{prev}\n\n{FROZEN_TIME}+00:00 Harmony Maskfill {self.version} '
            f'{{"shape_file_hash": "{SHAPE_HASH}", '
            f'"fill_value": {self.fillvalue}}}'
        )

        expected_history_json = self.get_base_json(url)
        expected_history_json['cf_history'] = [prev]

        update_history_metadata(
            dest, self.shape_file, self.fillvalue, self.bounding_box
        )
        self.assert_history(dest, expected_history, expected_history_json)

    @freeze_time(FROZEN_TIME)
    def test_update_history_metadata_capital_history(self):
        """Test appending to a capitalized "History" attribute."""
        source = 'tests/data/SMAP_L4_SM_aup_UTM_output.h5'
        dest = self.prepare_file(source)
        prev = 'File written by ldas2daac.x'

        expected_history = (
            f'{prev}\n{FROZEN_TIME}+00:00 Harmony Maskfill {self.version} '
            f'{{"shape_file_hash": "{SHAPE_HASH}", '
            f'"fill_value": {self.fillvalue}}}'
        )

        expected_history_json = self.get_base_json(dest)
        expected_history_json['cf_history'] = [prev]

        update_history_metadata(
            dest, self.shape_file, self.fillvalue, self.bounding_box
        )
        self.assert_history(dest, expected_history, expected_history_json, key="History")

    @freeze_time(FROZEN_TIME)
    def test_update_history_metadata_annotator_service(self):
        """Test appending history from Metadata Annotator Service."""
        source = 'tests/data/SC_SPL2SMAP_S_subsetted_annotated.nc4'
        dest = self.prepare_file(source)
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
            '2025-05-08T19:14:18.390538+00:00 Harmony Metadata Annotator 0.0.1'
        ]

        expected_history = (
            f'{prev_parts[0]}\n\n{prev_parts[1]}\n'
            f'{FROZEN_TIME}+00:00 Harmony Maskfill {self.version} '
            f'{{"shape_file_hash": "{SHAPE_HASH}", '
            f'"fill_value": {self.fillvalue}}}'
        )

        expected_history_json = self.get_base_json(url)
        expected_history_json['cf_history'] = [prev_parts[0], '', prev_parts[1]]

        update_history_metadata(
            dest, self.shape_file, self.fillvalue, self.bounding_box
        )
        self.assert_history(dest, expected_history, expected_history_json)

    @freeze_time(FROZEN_TIME)
    def test_update_history_metadata_append_history_bounding_box(self):
        """Test appending to existing history and history_json."""
        source = 'tests/data/SC_SPL3SMP_subsetted_with_maskfill_mf.nc4'
        dest = self.prepare_file(source)
        url = (
            'https://opendap.uat.earthdata.nasa.gov/collections/'
            'C1268452365-EEDTEST/granules/SC:SPL3SMP.008:240468423.dap.nc4'
        )
        prev = (
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
            f'{prev}\n\n{FROZEN_TIME}+00:00 Harmony Maskfill {self.version} '
            f'{{"bbox": {bounding_box}, '
            f'"fill_value": {self.fillvalue}}}'
        )

        expected_history_json = self.get_base_json(url, bounding_box)
        expected_history_json['cf_history'] = [prev]

        update_history_metadata(
            dest, self.shape_file, self.fillvalue, bounding_box
        )
        self.assert_history(dest, expected_history, expected_history_json)
