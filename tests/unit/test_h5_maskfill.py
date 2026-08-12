from logging import getLogger
from os import mkdir
from os.path import isdir, join
from shutil import rmtree
from unittest.mock import Mock, patch

from pyproj import CRS
import h5py
import numpy as np

from maskfill.h5_maskfill import (
    get_mask_array,
    mask_fill,
    get_coordinates,
    get_exclusions,
    get_string_variables,
    create_mask_array
)
from maskfill.cf_config import CFConfigH5
from maskfill.utilities import apply_2d, mask_fill_array

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

        with self.subTest('Dataset coordinates contain only fill values'):
            dataset_fill_value = -9999
            latitude_fill_value = 1
            longitude_fill_value = 0

            # The dataset itself does not contain only fill values.
            dataset = h5_file.create_dataset('variable',
                                             data=np.ones((3, 2)),
                                             fillvalue=dataset_fill_value)
            latitude_ds = h5_file.create_dataset('latitude',
                                                 data=np.ones((3, 2)),
                                                 fillvalue=latitude_fill_value)
            longitude_ds = h5_file.create_dataset('longitude',
                                                  data=np.zeros((3, 2)),
                                                  fillvalue=latitude_fill_value)
            dataset.attrs['coordinates'] = '/latitude, /longitude'
            latitude_ds.attrs['_FillValue'] = latitude_fill_value
            longitude_ds.attrs['_FillValue'] = longitude_fill_value

            mask_fill(dataset, self.shape_file, self.cache_dir, 'maskgrid_only',
                      dataset_fill_value, self.saved_mask_arrays, self.cf_config,
                      self.exclusions_set, self.logger)

            mock_get_mask_array.assert_not_called()

    @patch('maskfill.h5_maskfill.get_apply_2d_process')
    @patch('maskfill.h5_maskfill.get_mask_array')
    def test_mask_fill_write_paths(self, mock_get_mask_array,
                                   mock_get_apply_2d_process):
        """ Ensure a plain 2-D dataset is mask filled via the chunk-wise
            path, a dataset carrying observed statistics attributes retains
            the whole-array path and has its statistics updated, and that
            both paths write identical values.

        """
        whole_array_process = Mock(wraps=apply_2d)
        mock_get_apply_2d_process.return_value = whole_array_process

        fill_value = -9999.0
        data = np.arange(12, dtype=np.float64).reshape(4, 3)
        mask_array = np.array([[1, 0, 1], [1, 0, 1], [1, 0, 1], [1, 0, 1]],
                              dtype=np.uint8)
        mock_get_mask_array.return_value = mask_array
        expected_output = mask_fill_array(data, mask_array, fill_value)
        expected_unfilled = expected_output[expected_output != fill_value]

        h5_file = h5py.File(self.test_h5_name, 'w')

        with self.subTest('Plain 2-D dataset uses the chunk-wise path'):
            dataset = h5_file.create_dataset('plain', data=data, chunks=(3, 2))
            dataset.attrs['_FillValue'] = fill_value
            dataset.attrs['coordinates'] = '/latitude /longitude'

            mask_fill(dataset, self.shape_file, self.cache_dir,
                      'ignore_and_delete', fill_value,
                      self.saved_mask_arrays, self.cf_config,
                      self.exclusions_set, self.logger)

            np.testing.assert_array_equal(dataset[:], expected_output)
            whole_array_process.assert_not_called()

        with self.subTest('Observed statistics retain the whole-array path'):
            dataset = h5_file.create_dataset('with_statistics', data=data,
                                             chunks=(3, 2))
            dataset.attrs['_FillValue'] = fill_value
            dataset.attrs['coordinates'] = '/latitude /longitude'
            dataset.attrs['observed_max'] = 0.0
            dataset.attrs['observed_min'] = 0.0
            dataset.attrs['observed_mean'] = 0.0

            mask_fill(dataset, self.shape_file, self.cache_dir,
                      'ignore_and_delete', fill_value,
                      self.saved_mask_arrays, self.cf_config,
                      self.exclusions_set, self.logger)

            np.testing.assert_array_equal(dataset[:], expected_output)
            whole_array_process.assert_called_once()
            self.assertEqual(dataset.attrs['observed_max'],
                             np.max(expected_unfilled))
            self.assertEqual(dataset.attrs['observed_min'],
                             np.min(expected_unfilled))
            self.assertEqual(dataset.attrs['observed_mean'],
                             np.mean(expected_unfilled))

        h5_file.close()

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

    def test_create_mask_array(self):
        """Test that the right mask array is created when the spatial subset is
        multiple rows and columns, one row, one column and does not cause an
        exception even for a 1 pixel subset

        """
        crs = CRS.from_cf({'false_easting': 0,
                           'false_northing': 0,
                           'grid_mapping_name': 'lambert_cylindrical_equal_area',
                           'longitude_of_central_meridian': 0,
                           'standard_parallel': 30,
                           'unit': 'm'})

        with self.subTest('Multiple Row Col subset'):
            h5_file = h5py.File('tests/data/SPL2SMAP_S_subset.nc4', 'r')
            dataset = h5_file['/Soil_Moisture_Retrieval_Data_1km/surface_temperature_1km']
            expected_mask = np.zeros((48, 53))

            mask_array = create_mask_array(dataset, crs,
                                           'tests/data/EraNationalPark.geojson',
                                           self.cf_config,
                                           self.logger)
            np.testing.assert_array_equal(mask_array, expected_mask)
            h5_file.close()
        with self.subTest('Subset has one row, multiple columns'):
            h5_file = h5py.File('tests/data/SPL2SMAP_S_one_row.nc4', 'r')
            dataset = h5_file['/Soil_Moisture_Retrieval_Data_1km/surface_temperature_1km']
            expected_mask = np.zeros((1, 16))

            mask_array = create_mask_array(dataset, crs,
                                           'tests/data/one_row.geojson',
                                           self.cf_config,
                                           self.logger)
            np.testing.assert_array_equal(mask_array, expected_mask)
            h5_file.close()
        with self.subTest('Subset has one column, multiple rows'):
            h5_file = h5py.File('tests/data/SPL2SMAP_S_one_col.nc4', 'r')
            dataset = h5_file['/Soil_Moisture_Retrieval_Data_1km/surface_temperature_1km']
            expected_mask = np.zeros((25, 1))

            mask_array = create_mask_array(dataset, crs,
                                           'tests/data/one_col.geojson',
                                           self.cf_config,
                                           self.logger)
            np.testing.assert_array_equal(mask_array, expected_mask)
            h5_file.close()
        with self.subTest('Subset only has a single pixel'):
            h5_file = h5py.File('tests/data/SPL2SMAP_S_one_pixel.nc4', 'r')
            dataset = h5_file['/Soil_Moisture_Retrieval_Data_1km/surface_temperature_1km']

            mask_array = create_mask_array(dataset, crs,
                                           'tests/data/one_pixel.geojson',
                                           self.cf_config,
                                           self.logger)
            self.assertIsNone(mask_array)
            h5_file.close()
