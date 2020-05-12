from os import mkdir
from os.path import isdir, join
from shutil import rmtree
from unittest import TestCase
from unittest.mock import patch

import h5py
import numpy as np

from H5MaskFill import \
    get_mask_array, mask_fill, \
    get_coordinates, get_exclusions


class TestH5MaskFill(TestCase):

    def setUp(self):
        self.cache_dir = 'cache'
        self.output_dir = 'test'
        self.saved_mask_arrays = {}
        self.shape_file = 'tests/data/USA.geo.json'
        self.shortname = 'test_output.h5'
        mkdir(self.output_dir)
        self.test_h5_name = join(self.output_dir, self.shortname)

    def tearDown(self):
        if isdir(self.output_dir):
            rmtree(self.output_dir)

    @patch('pymods.CFConfig.getShortName')
    @patch('H5MaskFill.get_mask_array')
    def test_mask_fill_no_processing(self, mock_get_mask_array, mock_getShortName):
        """ Ensure that a dataset that fails to meet the required criteria is
            not processed in any way. Instead, the function should return prior
            to that point.

        """
        mock_getShortName.return_value = 'SPL3FTP'
        h5_file = h5py.File(self.test_h5_name, 'w')

        test_args = [['1-d data', np.ones((3)), True, 0, -1, 2],
                     ['No coordinates', np.ones((3, 2)), False, 0, -1, 2],
                     ['All fill values', np.ones((3, 2)), True, 1, -1, 2]]

        for description, data, coordinates_present, fill_value, valid_min, valid_max in test_args:
            with self.subTest(description):
                dataset = h5_file.create_dataset(description, data=data, fillvalue=fill_value)
                dataset.attrs['valid_min'] = valid_min
                dataset.attrs['valid_max'] = valid_max
                if coordinates_present:
                    dataset.attrs['coordinates'] = (f'/{description}/latitude '
                                                    f'/{description}/longitude')

                mask_fill(dataset, self.shape_file, self.cache_dir, 'maskgrid_only',
                          fill_value, self.saved_mask_arrays, self.shortname)

                mock_get_mask_array.assert_not_called()

    @patch('H5MaskFill.create_mask_array')
    def test_get_mask_array(self, mock_create_mask_array):
        """ Ensure that the following cases correctly occur:

            - `saved_mask_arrays` contains a matching mask, and so that is
              returned.
            - A matching cached file file is saved - the numpy array within the
              file is returned.
            - No pre-existing mask is saved in either the dictionary or a file,
              so a new one is calculated.

        """
        h5_file = h5py.File('tests/data/SMAP_L4_SMAU_input.h5', 'r')
        dataset = h5_file['/Analysis_Data/sm_profile_analysis']

        # Pre-calculated ID, to use for dictionary key and file name:
        mask_id = 'a62e96c11d707f2153e4f6a7da7707fc681152a358b816af5c9bcd11'

        saved_mask = np.ones((2, 3))
        cached_mask = np.ones((3, 4))
        new_mask = np.ones((4, 5))

        mock_create_mask_array.return_value = new_mask

        with self.subTest('Previously saved mask with matching ID'):
            saved_masks = {mask_id: saved_mask}
            mask_array = get_mask_array(dataset, self.shape_file,
                                        self.output_dir, 'use_cache',
                                        saved_masks, 'SPL4SMAU')
            np.testing.assert_array_equal(mask_array, saved_mask)
            mock_create_mask_array.assert_not_called()

        with self.subTest('Previously cached mask with matching ID'):
            output_file_path = join(self.output_dir, f'{mask_id}.npy')
            np.save(output_file_path, cached_mask)
            mask_array = get_mask_array(dataset, self.shape_file,
                                        self.output_dir, 'use_cache', {},
                                        'SPL4SMAU')
            np.testing.assert_array_equal(mask_array, cached_mask)
            mock_create_mask_array.assert_not_called()
            rmtree(self.output_dir)

        with self.subTest('No prior mask (cached or saved)'):
            mask_array = get_mask_array(dataset, self.shape_file,
                                        self.output_dir, 'use_cache', {},
                                        'SPL4SMAU')
            np.testing.assert_array_equal(mask_array, new_mask)
            mock_create_mask_array.assert_called_once_with(dataset,
                                                           self.shape_file,
                                                           'SPL4SMAU')

        h5_file.close()

    def test_get_coordinates(self):
        ''' Assert for H5MaskFill.get_coordinates
             - set of strings is returned
             - strings are datasets contained in h5 file
             - all coordinates references exist in result
        '''
        h5_file = h5py.File('tests/data/SMAP_L4_SMAU_input.h5', 'r')
        coordinates = get_coordinates(h5_file)
        self.assertIsInstance(coordinates, set)
        for item in coordinates:
            self.assertIsInstance(item, str)
        for item in {'/cell_lat', '/cell_lon'}:
            self.assertIn(item, coordinates)

    def test_get_exclusions(self):
        ''' Assert for H5MaskFill.get_exclusions:
             - set of strings is returned
             - coordinates are included
             - configuration exclusions are included
        '''
        h5_file = h5py.File('tests/data/SMAP_L4_SMAU_input.h5', 'r')
        dataset = h5_file['/Analysis_Data/sm_profile_analysis']
        exclusions = get_exclusions(dataset)
        self.assertIsInstance(exclusions, set)
        for item in exclusions:
            self.assertIsInstance(item, str)
        coordinates = get_coordinates(h5_file)
        for item in coordinates:
            self.assertIn(item, exclusions)
        # check for exclusions (copied here from config file)
        for item in {'cell_row', 'cell_column', 'EASE_column',
                     'EASE_row', 'EASE_column_index', 'EASE_row_index'}:
            self.assertIn(item, exclusions)

    @patch('H5MaskFill.get_exclusions')
    @patch('H5MaskFill.get_mask_array')
    def test_no_exclusions(self, mock_get_mask_array, mock_get_exclusions):
        ''' Assert for each given exclusions, maskfill processing is skipped
            (similar to test_mask_fill_no_processing)
        '''
        exclusions = {'cell_row', 'cell_column', 'EASE_column',
                      'EASE_row', 'EASE_column_index', 'EASE_row_index'
                      '/cell_lat', '/cell_lon'}
        mock_get_exclusions.return_value = exclusions

        h5_file = h5py.File(self.test_h5_name + '2', 'w')

        for item in exclusions:
            dataset = h5_file.create_dataset(item, data=[0, 1, 2])
            mask_fill(dataset, self.shape_file, self.cache_dir, 'maskgrid_only',
                      0, self.saved_mask_arrays, self.shortname)

            mock_get_mask_array.assert_not_called()
