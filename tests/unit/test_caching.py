from logging import getLogger
from unittest import TestCase
from unittest.mock import Mock, patch

import numpy as np

from maskfill import utilities
from maskfill.caching import (
    cache_geotiff_mask_array,
    cache_h5_mask_arrays,
    get_geotiff_cached_mask_array,
    get_geotiff_mask_array_path,
    get_mask_array_path_from_id,
)


class TestCaching(TestCase):

    @classmethod
    def setUpClass(cls):
        cls.logger = getLogger('test')
        cls.mock_id = 'ba214cbe252'
        cls.tif_data_path = '/path/to/data.tif'
        cls.shape_path = '/path/to/shape.geo.json'
        cls.cache_dir = '/path/to/cache'
        cls.mask_path = f'{cls.cache_dir}/{cls.mock_id}.npy'

    @patch('maskfill.caching.utilities.get_geotiff_mask_array_id')
    @patch('maskfill.caching.np.save')
    def test_cache_geotiff_mask_array(self, mock_np_save,
                                      mock_get_geotiff_mask_array_id):
        """ Ensure that a single GeoTIFF mask is saved to a `.npy` file, if the
            correct grid caching option is selected. Otherwise, the mask should
            not be cached.

        """
        mock_get_geotiff_mask_array_id.return_value = self.mock_id

        geotiff_mask = np.array([[1, 2], [3, 4]])

        with self.subTest('"save" in the grid caching option'):
            cache_geotiff_mask_array(geotiff_mask, self.tif_data_path,
                                     self.shape_path, self.cache_dir,
                                     'use_and_save')
            mock_np_save.assert_called_once_with(self.mask_path, geotiff_mask)
            mock_np_save.reset_mock()

        with self.subTest('maskgrid_only grid caching option'):
            cache_geotiff_mask_array(geotiff_mask, self.tif_data_path,
                                     self.shape_path, self.cache_dir,
                                     'maskgrid_only')
            mock_np_save.assert_called_once_with(self.mask_path, geotiff_mask)
            mock_np_save.reset_mock()

        with self.subTest('Non saving grid caching option'):
            cache_geotiff_mask_array(geotiff_mask, self.tif_data_path,
                                     self.shape_path, self.cache_dir,
                                     'ignore_and_delete')
            mock_np_save.assert_not_called()

    @patch('maskfill.caching.np.save')
    def test_cache_h5_mask_arrays(self, mock_np_save):
        """ Ensure all masks are saved from a cache in local storage, to
            individual `.npy` files, if the correct grid caching option is
            selected. Otherwise, no masks should be cached.

        """
        h5_masks = {'mask_id_1': np.array([[1, 2], [3, 4]]),
                    'mask_id_2': np.array([[5, 6], [7, 8]]),
                    'mask_id_3': np.array([[9, 0], [1, 2]])}

        with self.subTest('Should save'):
            cache_h5_mask_arrays(h5_masks, self.cache_dir, 'use_and_save',
                                 self.logger)

            self.assertEqual(mock_np_save.call_count, 3)
            mock_np_save.assert_any_call(f'{self.cache_dir}/mask_id_1.npy',
                                         h5_masks['mask_id_1'])
            mock_np_save.assert_any_call(f'{self.cache_dir}/mask_id_2.npy',
                                         h5_masks['mask_id_2'])
            mock_np_save.assert_any_call(f'{self.cache_dir}/mask_id_3.npy',
                                         h5_masks['mask_id_3'])

        mock_np_save.reset_mock()

        with self.subTest('Does not save; "delete" in grid caching option'):
            cache_h5_mask_arrays(h5_masks, self.cache_dir, 'ignore_and_delete',
                                 self.logger)
            mock_np_save.assert_not_called()

    @patch('maskfill.utilities.get_geotiff_mask_array_id')
    @patch('maskfill.caching.os.path.exists')
    @patch('maskfill.caching.np.load')
    def test_get_geotiff_cached_mask_array(self, mock_np_load, mock_os_exists,
                                           mock_get_geotiff_mask_array_id):
        """ Ensure that cached information is retrieved using `numpy.load`, if
            the correct grid mapping option is specified and the mask file
            exists at the specified location.

        """
        mock_get_geotiff_mask_array_id.return_value = self.mock_id

        mock_data = np.array([[1, 2, 3], [4, 5, 6]])
        mock_np_load.return_value = mock_data

        with self.subTest('"use" in the grid caching option returns data'):
            mock_os_exists.return_value = True
            mask_array = get_geotiff_cached_mask_array(self.tif_data_path,
                                                       self.shape_path,
                                                       self.cache_dir,
                                                       'use_cache')
            np.testing.assert_array_equal(mask_array, mock_data)
            mock_get_geotiff_mask_array_id.assert_called_once_with(
                self.tif_data_path, self.shape_path
            )
            mock_os_exists.assert_called_once_with(self.mask_path)
            mock_np_load.assert_called_once_with(self.mask_path)
            mock_os_exists.reset_mock()
            mock_np_load.reset_mock()

        with self.subTest('The mask array path points to non-existant file'):
            mock_os_exists.return_value = False
            mask_array = get_geotiff_cached_mask_array(self.tif_data_path,
                                                       self.shape_path,
                                                       self.cache_dir,
                                                       'use_cache')
            self.assertEqual(mask_array, None)
            mock_os_exists.assert_called_once_with(self.mask_path)
            mock_np_load.assert_not_called()
            mock_os_exists.reset_mock()
            mock_np_load.reset_mock()

        with self.subTest('"use" not in grid caching option, returns `None`'):
            mock_os_exists.return_value = True
            mask_array = get_geotiff_cached_mask_array(self.tif_data_path,
                                                       self.shape_path,
                                                       self.cache_dir,
                                                       'ignore_and_delete')
            self.assertEqual(mask_array, None)
            mock_os_exists.assert_not_called()
            mock_np_load.assert_not_called()

    @patch('maskfill.caching.utilities.get_geotiff_mask_array_id')
    def test_get_geotiff_mask_array_path(self, mock_get_geotiff_mask_array_id):
        """ Ensure that the expected mask array path is constructed using the
            input file paths for the data, shape file and cache directory.

        """
        mock_utilties = Mock(spec=utilities)
        mock_get_geotiff_mask_array_id.return_value = self.mock_id

        mask_array_path = get_geotiff_mask_array_path(self.tif_data_path,
                                                      self.shape_path,
                                                      self.cache_dir)

        self.assertEqual(mask_array_path, self.mask_path)
        mock_get_geotiff_mask_array_id.assert_called_once_with(
            self.tif_data_path, self.shape_path
        )

    def test_get_mask_array_path_from_id(self):
        """ Ensure that a correctly formatted file name is returned. """
        self.assertEqual(
            get_mask_array_path_from_id(self.mock_id, '/directory/nested'),
            f'/directory/nested/{self.mock_id}.npy'
        )
