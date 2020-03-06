from os import mkdir
from os.path import isdir, join
from shutil import rmtree
from unittest import TestCase
from unittest.mock import Mock, patch

import h5py
import numpy as np

from pymods.H5GridProjectionInfo import (BAD_FILL_VALUE_DATASETS,
                                         dataset_all_fill_value,
                                         dataset_all_outside_valid_range,
                                         euclidean_distance,
                                         extrapolate_coordinate,
                                         get_corner_points_from_lat_lon,
                                         get_fill_value,
                                         get_pixel_size_from_data_extent,
                                         get_valid_coordinates_extent)


class TestH5GridProjectionInfo(TestCase):

    def setUp(self):
        self.output_dir = 'tests/output'
        mkdir(self.output_dir)
        self.test_h5_name = join(self.output_dir, 'test_output.h5')


    def tearDown(self):
        """Clean up test artifacts after each test."""
        if isdir(self.output_dir):
            rmtree(self.output_dir)

    def test_dataset_all_fill_value_fill(self):
        """Ensure that a dataset is correctly identified as only containing
        fill values.
        
        """
        h5_file = h5py.File(self.test_h5_name, 'w')
        dimensions = (3, 2)
        data_ones = np.ones((3, 2))
        data = np.ones(dimensions)
        data[0][0] = 0

        test_args = [['All elements are fill value', 'filled', data_ones, 1.0, True],
                     ['Noelements are fill value', 'not_filled', data_ones, 0.0, False],
                     ['Only some elements are filled', 'some_filled', data, 1.0, False]]

        for description, dataset_name, data, fill_value, result in test_args:
            with self.subTest(description):
                dataset = h5_file.create_dataset(dataset_name, data=data, fillvalue = fill_value)
                self.assertEqual(dataset_all_fill_value(dataset, fill_value), result)

        h5_file.close()

    def test_dataset_all_outside_valid_range(self):
        """Ensure correct detection of when data in a Dataset all lies outside
        the valid range specified by the `valid_min` and `valid_max` attributes.
        
        Note, this function should also handle cases where only one of the
        attributes is present, or when neither are.
        
        """
        data_array = np.array([[3, 2, 3], [3, 3, 3], [6, 6, 6]])
        h5_file = h5py.File(self.test_h5_name, 'w')
        dataset = h5_file.create_dataset('valid', data=data_array)

        with self.subTest('neither valid_min or valid_max set'):
            self.assertFalse(dataset_all_outside_valid_range(dataset))

        both_attrs_tests = [['valid_min valid_max set, all within range', 2, 6, False],
                            ['valid_min, valid_max set, some less than min', 3, 6, False],
                            ['valid_min, valid_max set, some more than max', 2, 5, False],
                            ['valid_min, valid_max set, some more, some less, some in range', 3, 5, False],
                            ['valid_min, valid_max set, none in range', 4, 5, True]]

        for description, valid_min, valid_max, result in both_attrs_tests:
            with self.subTest(description):
                dataset.attrs['valid_min'] = valid_min
                dataset.attrs['valid_max'] = valid_max
                self.assertEqual(dataset_all_outside_valid_range(dataset), result)

        dataset.attrs.__delitem__('valid_min')
        dataset.attrs.__delitem__('valid_max')

        one_attr_tests = [['only valid_max, all below', 'valid_max', 7, False],
                          ['only valid_max, some below', 'valid_max', 5, False],
                          ['only valid_max, all above', 'valid_max', 1, True],
                          ['only valid_min, all above', 'valid_min', 1, False],
                          ['only valid_min, some above', 'valid_min', 4, False],
                          ['only valid_min, all below', 'valid_min', 7, True]]

        for description, attr_key, attr_value, result in one_attr_tests:
            with self.subTest(description):
                dataset.attrs[attr_key] = attr_value
                self.assertEqual(dataset_all_outside_valid_range(dataset), result)
                dataset.attrs.__delitem__(attr_key)

        h5_file.close()

    def test_get_fill_value(self):
        """The correct fill value should be returned. If it is entirely lacking,
        the default value supplied to the function should be returned instead.
        
        """
        data_array = np.ones((3,2))
        default_fill_value = 1.0
        fill_value_attr = 2.0
        fill_value_class = 3.0

        h5_file = h5py.File(self.test_h5_name, 'w')

        with self.subTest('Override default value for specific named datasets'):
            bad_dataset = list(BAD_FILL_VALUE_DATASETS.keys())[0]
            dataset = h5_file.create_dataset(bad_dataset,
                                             data=data_array,
                                             fillvalue=fill_value_class)
            dataset.attrs['_FillValue'] = fill_value_attr
            self.assertEqual(get_fill_value(dataset, default_fill_value),
                             BAD_FILL_VALUE_DATASETS[bad_dataset])

        with self.subTest('_FillValue attribute should take precedence.'):
            dataset = h5_file.create_dataset('attribute',
                                             data=data_array,
                                             fillvalue=fill_value_class)
            dataset.attrs['_FillValue'] = fill_value_attr
            self.assertEqual(get_fill_value(dataset, default_fill_value),
                             fill_value_attr)

        with self.subTest('Without _FillValue, fall back on the fillvalue class attribute'):
            dataset = h5_file.create_dataset('class_attribute',
                                             data=data_array,
                                             fillvalue=fill_value_class)
            self.assertEqual(get_fill_value(dataset, default_fill_value),
                             fill_value_class)

        with self.subTest('Return default when nothing is set.'):
            string_data = np.chararray((3, 3))
            string_data[:] = 'abc'
            dataset = h5_file.create_dataset('default', data=string_data)
            self.assertEqual(get_fill_value(dataset, default_fill_value),
                             default_fill_value)

        h5_file.close()

    def test_euclidean_distance(self):
        """The correct distance between two tuples of points should be calculated,
        or if the dimensions of the tuples do not match, there should be a None
        return.
        
        """
        tuple_one = (0, 0, 0)
        tuple_two = (4, 3, 12)

        test_args = [['3-D tuples, p1, p2', tuple_one, tuple_two, 13],
                     ['3-D tuples, p2, p1', tuple_two, tuple_one, 13],
                     ['2-D tuples', tuple_one[:2], tuple_two[:2], 5],
                     ['Non matching tuple lengths', tuple_one, tuple_two[:2], None]]

        for description, point_one, point_two, result in test_args:
            with self.subTest(description):
                self.assertEqual(euclidean_distance(point_one, point_two), result)

    def test_extrapolate_coordinate(self):
        """The coordinate should be correctly extrapolated, if the point being
        extrapolated to contains a fill value, then a coordinate should be
        calculated. If it doesn't contain the fill value, it should return
        the array value at the requested indices.
        
        """
        fill_value = -9999.0
        data_array = np.array([[0.0, 0.1, 0.2], [0.0, 0.1, 0.2], [0.0, 0.1, 0.2]])
        filled_array = np.array([[fill_value, 0.1, 0.2],
                                 [fill_value, 0.1, 0.2],
                                 [fill_value, 0.1, 0.2]])

        reference_indices = (2, 2)
        target_indices = (0, 0)
        pixel_scale_x = 0.1

        h5_file = h5py.File(self.test_h5_name, 'w')
        dataset = h5_file.create_dataset('test', data=data_array, fillvalue=fill_value)
        filled_dataset = h5_file.create_dataset('filled_test',
                                                data=filled_array,
                                                fillvalue=fill_value)
        
        with self.subTest('Corner point is extrapolated'):
            self.assertEqual(
                extrapolate_coordinate(filled_dataset, fill_value, 'x',
                                       filled_array[reference_indices],
                                       reference_indices, 1,
                                       target_indices, pixel_scale_x),
                0.0
            )
        with self.subTest('Corner point has data in it, and is returned.'):
            self.assertEqual(
                extrapolate_coordinate(dataset, fill_value, 'x',
                                       data_array[reference_indices],
                                       reference_indices, 1,
                                       target_indices, pixel_scale_x),
                0.0
            )

        h5_file.close()

    def test_get_pixel_size_from_data_extent(self):
        """The correct pixel extents should be calculated for both dimensions,
        and if one is invalid (largely due to date being in a single column or row),
        the assumption of square pixels should be applied. Finally, if both
        pixel scales are found to be zero, a ValueError should be raised.

        Note: Python indices for 2-D arrays are array[row][column]

        """
        x_two, y_two = (15.0, 5.0)
        x_one, y_one = (10.0, 10.0)
        lower_left_indices = (0, 0)
        upper_right_indices = (5, 5)
        scales = (1.0, -1.0)

        with self.subTest('Both extents non-zero'):
            self.assertEqual(
                get_pixel_size_from_data_extent(x_one, y_one, x_two, y_two,
                                                lower_left_indices,
                                                upper_right_indices),
                scales
            )

        with self.subTest('x extent zero'):
            same_column_indices = (upper_right_indices[0], lower_left_indices[1])
            self.assertEqual(
                get_pixel_size_from_data_extent(x_one, y_one, x_one, y_two,
                                                lower_left_indices,
                                                same_column_indices),
                scales
            )

        with self.subTest('y extent zero'):
            same_row_indices = (lower_left_indices[0], upper_right_indices[1])
            self.assertEqual(
                get_pixel_size_from_data_extent(x_one, y_one, x_two, y_one,
                                                lower_left_indices,
                                                same_row_indices),
                scales
            )

        with self.subTest('both extents zero'):
            with self.assertRaises(ValueError) as context_manager:
                get_pixel_size_from_data_extent(x_one, y_one, x_one, y_one,
                                                lower_left_indices,
                                                lower_left_indices)

    def test_get_valid_coordinates_extent(self):
        """The indices of the points containing valid data in both arrays,
        nearest to the upper right and lower left corners should be returned.

        """
        fill_value = -1
        data_array = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
        data_bad_ll = np.array([[fill_value, 2, 3], [4, 5, 6], [7, 8, 9]])
        data_bad_ur = np.array([[1, 2, 3], [4, 5, 6], [7, 8, fill_value]])
        data_bad_both = np.array([[fill_value, 2, 3], [4, 5, 6], [7, 8, fill_value]])
        lower_left_tuple = (0, 0)
        upper_right_tuple = (2, 2)

        test_args = [['Corners have valid data', data_array, data_array, lower_left_tuple, upper_right_tuple],
                     ['Lower left is invalid in one array', data_array, data_bad_ll, (0, 1), upper_right_tuple],
                     ['Lower left is invalid both arrays', data_bad_ll, data_bad_ll, (0, 1), upper_right_tuple],
                     ['Upper right is invalid in one array', data_array, data_bad_ur, lower_left_tuple, (1, 2)],
                     ['Upper right is invalid in both arrays', data_bad_ur, data_bad_ur, lower_left_tuple, (1, 2)],
                     ['Both corners invalid in one array', data_bad_both, data_array, (0, 1), (1, 2)],
                     ['Opposite corners invalid in each array', data_bad_ll, data_bad_ur, (0, 1), (1, 2)]]

        for description, data_one, data_two, result_ll, result_ur in test_args:
            with self.subTest(description):
                lower_left_valid, upper_right_valid = get_valid_coordinates_extent(
                    data_one, data_two, fill_value, fill_value, lower_left_tuple,
                    upper_right_tuple
                )
                self.assertEqual(lower_left_valid, result_ll)
                self.assertEqual(upper_right_valid, result_ur)

    @patch('pymods.H5GridProjectionInfo.get_proj4')
    @patch('pymods.H5GridProjectionInfo._get_short_name')
    @patch('pymods.H5GridProjectionInfo._get_grid_mapping_group')
    @patch('pymods.H5GridProjectionInfo._get_grid_mapping_data')
    def test_get_corner_points_from_lat_lon(self,
                                            mock_get_grid_data,
                                            mock_get_grid_group,
                                            mock_get_shortname,
                                            mock_get_proj4):
        """Ensure extrapolation occurs where expected, corners with valid points
        are used outright, and a ValueError is returned for entirely filled
        coordinate arrays.

        This test circumvents the retrieval of the Proj4 string at the start of
        the function, as the functionality being tested is that the extrapolation
        is consistent with expectations.

        """
        mock_get_proj4.return_value = {'proj': 'eqc'}

        fill_value = -1
        data_array = np.ones((3, 3))
        lat_array = np.array([[10.0, 10.0, 10.0, 10.0],
                              [15.0, 15.0, 15.0, 15.0],
                              [20.0, 20.0, 20.0, 20.0],
                              [25.0, 25.0, 25.0, 25.0]])
        lon_array = np.array([[5.0, 10.0, 15.0, 20.0],
                              [5.0, 10.0, 15.0, 20.0],
                              [5.0, 10.0, 15.0, 20.0],
                              [5.0, 10.0, 15.0, 20.0]])

        x_lower_left = 556597.453966368
        y_lower_left = 1113194.9079327357
        x_upper_right = 2226389.815865471
        y_upper_right = 2782987.269831839

        h5_file = h5py.File(self.test_h5_name, 'w')
        data = h5_file.create_dataset('data', data=data_array, fillvalue=fill_value)

        test_args = [['Neither_corner_filled', False, False],
                     ['Lower_leftcorner_filled', True, False],
                     ['Upper_right_corner_filled', False, True],
                     ['Both_corners_filled', True, True]]

        for description, fill_ll_corner, fill_ur_corner in test_args:
            with self.subTest(description):
                lat_copy = np.copy(lat_array)
                lon_copy = np.copy(lon_array)

                if fill_ll_corner:
                    lat_copy[0][0] = fill_value
                    lon_copy[0][0] = fill_value

                if fill_ur_corner:
                    lat_copy[-1][-1] = fill_value
                    lon_copy[-1][-1] = fill_value

                lat_name = f'/lat_{description}'
                lon_name = f'/lon_{description}'
                h5_file.create_dataset(lat_name, data=lat_copy, fillvalue=fill_value)
                h5_file.create_dataset(lon_name, data=lon_copy, fillvalue=fill_value)

                data.attrs['coordinates'] = f'{lat_name} {lon_name}'.encode('utf-8')

                x_ll, x_ur, y_ll, y_ur = get_corner_points_from_lat_lon(data)

                self.assertAlmostEqual(x_ll, x_lower_left)
                self.assertAlmostEqual(x_ur, x_upper_right)
                self.assertAlmostEqual(y_ll, y_lower_left)
                self.assertAlmostEqual(y_ur, y_upper_right)

        with self.subTest('Entirely filled latitude should raise a ValueError'):
            with self.assertRaises(ValueError) as context_manager:
                lat_copy = np.copy(lat_array)
                lon_copy = np.copy(lon_array)
                lat_copy.fill(fill_value)

                lat_name = f'/lat_filled'
                lon_name = f'/lon_filled'
                h5_file.create_dataset(lat_name, data=lat_copy, fillvalue=fill_value)
                h5_file.create_dataset(lon_name, data=lon_copy, fillvalue=fill_value)

                data.attrs['coordinates'] = f'{lat_name} {lon_name}'.encode('utf-8')

                x_ll, x_ur, y_ll, y_ur = get_corner_points_from_lat_lon(data)
