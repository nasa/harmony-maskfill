from logging import getLogger
from os import mkdir
from os.path import isdir, join
from shutil import rmtree
from unittest.mock import patch

from pyproj import CRS
import h5py
import numpy as np

from maskfill.h5_maskfill import (
    get_mask_array,
    mask_fill,
    get_coordinates,
    get_exclusions,
    get_string_variables,
)
from maskfill.cf_config import CFConfigH5

from tests.utilities import MaskFillTestCase


class TestH5MaskFill(MaskFillTestCase):

    @classmethod
    def setUpClass(cls):
        cls.cf_config = CFConfigH5('tests/data/SMAP_L4_SM_aup_input.h5')
        cls.logger = getLogger('test')

    def setUp(self):
        self.cache_dir = 'cache'
        self.output_dir = 'test'
        self.saved_mask_arrays = {}
        self.shape_file = 'tests/data/USA.geo.json'
        self.shortname = 'test_output.h5'
        mkdir(self.output_dir)
        self.test_h5_name = join(self.output_dir, self.shortname)
        self.exclusions_set = {'cell_row', 'cell_column', 'EASE_column',
                               'EASE_row', 'EASE_column_index',
                               'EASE_row_index', '/GEO/latitude',
                               '/GEO/longitude'}

    def tearDown(self):
        if isdir(self.output_dir):
            rmtree(self.output_dir)

    @patch('maskfill.h5_maskfill.get_mask_array')
    def test_mask_fill_no_processing(self, mock_get_mask_array):
        """ Ensure that a dataset that fails to meet the required criteria is
            not processed in any way. Instead, the function should return prior
            to that point.

        """
        h5_file = h5py.File(self.test_h5_name, 'w')
        valid_min = -1
        valid_max = 2

        test_args = [['1-d data', np.ones((3)), True, 0],
                     ['No coordinates', np.ones((3, 2)), False, 0],
                     ['All fill values', np.ones((3, 2)), True, 1],
                     ['String data', 'A string', True, '']]

        for description, data, coordinates_present, fill_value in test_args:
            with self.subTest(description):
                dataset = h5_file.create_dataset(description, data=data, fillvalue=fill_value)
                dataset.attrs['valid_min'] = valid_min
                dataset.attrs['valid_max'] = valid_max
                if coordinates_present:
                    dataset.attrs['coordinates'] = (f'/{description}/latitude '
                                                    f'/{description}/longitude')

                mask_fill(dataset, self.shape_file, self.cache_dir, 'maskgrid_only',
                          fill_value, self.saved_mask_arrays, self.cf_config,
                          self.exclusions_set, self.logger)

                mock_get_mask_array.assert_not_called()

        with self.subTest('Variable name exactly matches an exclusion'):
            dataset = h5_file.create_dataset('/excluded/variable',
                                             data=np.ones((3, 2)), fillvalue=0)
            dataset.attrs['coordinates'] = '/latitude, /longitude'
            mask_fill(dataset, self.shape_file, self.cache_dir, 'maskgrid_only',
                      fill_value, self.saved_mask_arrays, self.cf_config,
                      {'/excluded/variable'}, self.logger)

            mock_get_mask_array.assert_not_called()

    @patch('maskfill.h5_maskfill.create_mask_array')
    def test_get_mask_array(self, mock_create_mask_array):
        """ Ensure that the following cases correctly occur:

            - `saved_mask_arrays` contains a matching mask, and so that is
              returned.
            - A matching cached file file is saved - the numpy array within the
              file is returned.
            - No pre-existing mask is saved in either the dictionary or a file,
              so a new one is calculated.

        """
        h5_file = h5py.File('tests/data/SMAP_L4_SM_aup_input.h5', 'r')
        dataset = h5_file['/Analysis_Data/sm_profile_analysis']
        # The following CRS uses the parameters for EASE-2 Grid Global, as
        # taken from the MaskFill configuration file.
        crs = CRS.from_cf({'false_easting': 0,
                           'false_northing': 0,
                           'grid_mapping_name': 'lambert_cylindrical_equal_area',
                           'longitude_of_central_meridian': 0,
                           'standard_parallel': 30,
                           'unit': 'm'})

        # Pre-calculated ID, to use for dictionary key and file name:
        mask_id = '45ec81bdef17350c3f1690a431203c0f0ee528e1e81bfec1525cef4e'

        saved_mask = np.ones((2, 3))
        cached_mask = np.ones((3, 4))
        new_mask = np.ones((4, 5))

        mock_create_mask_array.return_value = new_mask

        with self.subTest('Previously saved mask with matching ID'):
            saved_masks = {mask_id: saved_mask}
            mask_array = get_mask_array(dataset, self.shape_file,
                                        self.output_dir, 'use_cache',
                                        saved_masks, self.cf_config, self.logger)
            np.testing.assert_array_equal(mask_array, saved_mask)
            mock_create_mask_array.assert_not_called()

        with self.subTest('Previously cached mask with matching ID'):
            output_file_path = join(self.output_dir, f'{mask_id}.npy')
            np.save(output_file_path, cached_mask)
            mask_array = get_mask_array(dataset, self.shape_file,
                                        self.output_dir, 'use_cache', {},
                                        self.cf_config, self.logger)
            np.testing.assert_array_equal(mask_array, cached_mask)
            mock_create_mask_array.assert_not_called()
            rmtree(self.output_dir)

        with self.subTest('No prior mask (cached or saved)'):
            mask_array = get_mask_array(dataset, self.shape_file,
                                        self.output_dir, 'use_cache', {},
                                        self.cf_config, self.logger)
            np.testing.assert_array_equal(mask_array, new_mask)
            mock_create_mask_array.assert_called_once_with(dataset, crs,
                                                           self.shape_file,
                                                           self.cf_config,
                                                           self.logger)

        h5_file.close()

    def test_get_coordinates(self):
        """ Assert for maskfill.h5_maskfill.get_coordinates
             - set of strings is returned
             - strings are datasets contained in h5 file
             - all coordinates references exist in result
        """
        h5_file = h5py.File('tests/data/SMAP_L4_SM_aup_input.h5', 'r')
        coordinates = get_coordinates(h5_file)
        self.assertIsInstance(coordinates, set)
        for item in coordinates:
            self.assertIsInstance(item, str)

        self.assertTrue({'/cell_lat', '/cell_lon'}.issubset(coordinates))

    @patch('maskfill.h5_maskfill.get_string_variables')
    def test_get_exclusions(self, mock_get_string_variables):
        """ Assert for maskfill.h5_maskfill.get_exclusions:
             - set of strings is returned
             - coordinate exclusions are included
             - configuration exclusions are included
        """
        file_name = 'tests/data/SMAP_L4_SM_aup_input.h5'
        h5_file = h5py.File(file_name, 'r')
        exclusions = get_exclusions(file_name, self.cf_config)

        self.assertIsInstance(exclusions, set)

        for item in exclusions:
            self.assertIsInstance(item, str)

        coordinates = get_coordinates(h5_file)

        self.assertTrue(coordinates.issubset(exclusions))
        for item in coordinates:
            self.assertIn(item, exclusions)

        # check for exclusions (copied here from config file)
        config_file_exclusions = {'/cell_(column|row)', '/cell_l(at|on)'}
        self.assertTrue(config_file_exclusions.issubset(exclusions))

        mock_get_string_variables.assert_called_once()

    @patch('maskfill.h5_maskfill.get_exclusions')
    @patch('maskfill.h5_maskfill.get_mask_array')
    def test_no_exclusions(self, mock_get_mask_array, mock_get_exclusions):
        """ Assert for each given exclusions, maskfill processing is skipped
            (similar to test_mask_fill_no_processing)
        """
        exclusions = {'cell_row', 'cell_column', 'EASE_column',
                      'EASE_row', 'EASE_column_index', 'EASE_row_index'
                      '/cell_lat', '/cell_lon'}

        h5_file = h5py.File(self.test_h5_name + '2', 'w')

        for item in exclusions:
            dataset = h5_file.create_dataset(item, data=[0, 1, 2])
            mask_fill(dataset, self.shape_file, self.cache_dir, 'maskgrid_only',
                      0, self.saved_mask_arrays, self.cf_config,
                      self.exclusions_set, self.logger)

            mock_get_mask_array.assert_not_called()

    def test_get_string_variables(self):
        """Test that string variables are returned when they exist in the
        input file.

        This includes three types of string variables:

        - Fixed-length byte strings
        - Unicode strings
        - Object type / variable-length strings

        Note: Other variations of each type exist, but the ones I chose below
        are the most common.

        """
        # Data for all string types.
        ascii_data = ['hello', 'world', 'test']
        unicode_data = ['café', 'résumé', 'naïve', '中文']
        mixed_length_data = ['short', 'a much longer string', 'varying']

        with h5py.File(self.sample_nc4_file(), 'r+') as input_file:

            # 1. Fixed-length ASCII strings
            input_file.create_dataset('fixed_ascii_s10',
                                      dtype='S10')

            # 2. Fixed-length ASCII (alternative syntax)
            input_file.create_dataset('fixed_ascii_np',
                                      data=ascii_data,
                                      dtype=np.dtype('S15'))

            # 3. Variable-length Unicode strings
            vlen_str = h5py.special_dtype(vlen=str)
            input_file.create_dataset('vlen_unicode',
                                      data=unicode_data,
                                      dtype=vlen_str)

            # 4. Variable-length Unicode with UTF-8 encoding
            utf8_variable = h5py.string_dtype(encoding='utf-8')
            input_file.create_dataset('variable_utf8',
                                      data=mixed_length_data,
                                      dtype=utf8_variable)

            # Check that all the strings datasets are included in the output.
            expected_strings = ['fixed_ascii_s10',
                                'fixed_ascii_np',
                                'vlen_unicode',
                                'variable_utf8']

            actual_strings = get_string_variables(input_file)
            self.assertCountEqual(expected_strings, actual_strings)

    def test_get_string_variables_no_strings(self):
        """Test that an empty list is returned when the input file contains
        no strings.

        """
        with h5py.File(self.sample_nc4_file(), 'r+') as input_file:
            expected_strings = []
            actual_strings = get_string_variables(input_file)
            self.assertEqual(expected_strings, actual_strings)
