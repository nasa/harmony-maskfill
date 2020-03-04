from os import mkdir
from os.path import isdir, join
from shutil import rmtree
from unittest import TestCase
from unittest.mock import patch

import h5py
import numpy as np

from H5MaskFill import mask_fill


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


    @patch('H5MaskFill.get_mask_array')
    def test_mask_fill_no_processing(self, mock_get_mask_array):
        """Ensure that a dataset that fails to meet the required criteria is
        not processed in any way. Instead, the function should return prior to
        that point.

        """
        h5_file = h5py.File(self.test_h5_name, 'w')

        test_args = [['1-d data', np.ones((3)), True, 0, -1, 2],
                     ['No coordinates', np.ones((3, 2)), False, 0, -1, 2],
                     ['All fill values', np.ones((3, 2)), True, 1, -1, 2],
                     ['All outside range', np.ones((3, 2)), True, 0, 3, 4]]

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
