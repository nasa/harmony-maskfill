from logging import getLogger
from os import mkdir
from os.path import isdir, join
from shutil import rmtree
from unittest import TestCase
from unittest.mock import patch

import h5py
import numpy as np

from H5MaskFill import (get_mask_array, mask_fill, get_coordinates,
                        get_exclusions)
from pymods.cf_config import CFConfigH5


class TestH5MaskFill(TestCase):

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

    @patch('H5MaskFill.get_mask_array')
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
                     ['All fill values', np.ones((3, 2)), True, 1]]

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
        h5_file = h5py.File('tests/data/SMAP_L4_SM_aup_input.h5', 'r')
        dataset = h5_file['/Analysis_Data/sm_profile_analysis']

        # Pre-calculated ID, to use for dictionary key and file name:
        mask_id = '45a5aa3c6b02ae31b06ae524ee823474132b6f4a74a790bf37623f17'

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
            mock_create_mask_array.assert_called_once_with(dataset,
                                                           self.shape_file,
                                                           self.cf_config,
                                                           self.logger)

        h5_file.close()

    def test_get_coordinates(self):
        """ Assert for H5MaskFill.get_coordinates
             - set of strings is returned
             - strings are datasets contained in h5 file
             - all coordinates references exist in result
        """
        h5_file = h5py.File('tests/data/SMAP_L4_SM_aup_input.h5', 'r')
        coordinates = get_coordinates(h5_file)
        self.assertIsInstance(coordinates, set)
        for item in coordinates:
            self.assertIsInstance(item, str)
        for item in {'/cell_lat', '/cell_lon'}:
            self.assertIn(item, coordinates)

    def test_get_exclusions(self):
        """ Assert for H5MaskFill.get_exclusions:
             - set of strings is returned
             - coordinates are included
             - configuration exclusions are included
        """
        file_name = 'tests/data/SMAP_L4_SM_aup_input.h5'
        h5_file = h5py.File(file_name, 'r')
        exclusions = get_exclusions(file_name, self.cf_config)

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
