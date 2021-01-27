from logging import getLogger
from os import mkdir
from os.path import isdir, join
from shutil import rmtree
from unittest import TestCase
from unittest.mock import Mock, patch
import json

from pyproj.crs import CRS
import h5py
import numpy as np

from pymods.cf_config import CFConfigH5
from pymods.exceptions import (InsufficientDataError,
                               InsufficientProjectionInformation,
                               InvalidMetadata, MissingCoordinateDataset)
from pymods.H5GridProjectionInfo import (dataset_all_fill_value,
                                         dataset_all_outside_valid_range,
                                         euclidean_distance,
                                         extrapolate_coordinate,
                                         get_cell_size_from_dimensions,
                                         get_cell_size_from_lat_lon_extents,
                                         get_corner_points_from_dimensions,
                                         get_corner_points_from_lat_lon,
                                         get_crs, get_dataset_attributes,
                                         get_dimension_datasets,
                                         get_fill_value,
                                         get_grid_mapping_name,
                                         get_hdf_proj4,
                                         get_lon_lat_datasets,
                                         get_pixel_size_from_data_extent,
                                         get_transform,
                                         get_transform_information,
                                         get_valid_coordinates_extent,
                                         resolve_relative_dataset_path)


class TestH5GridProjectionInfo(TestCase):

    @classmethod
    def setUpClass(cls):
        cls.logger = getLogger('test')
        cls.cf_config = CFConfigH5('tests/data/SMAP_L3_FT_P_corners_input.h5')

    def setUp(self):
        self.output_dir = 'tests/output'
        mkdir(self.output_dir)
        self.test_h5_name = join(self.output_dir, 'test_output.h5')

    def tearDown(self):
        """Clean up test artifacts after each test."""
        if isdir(self.output_dir):
            rmtree(self.output_dir)

    def test_dataset_all_fill_value(self):
        """Ensure that a dataset is correctly identified as only containing
        fill values.

        """
        h5_file = h5py.File(self.test_h5_name, 'w')
        fill_value = 1.0
        dimensions = (3, 2)
        data_ones = np.ones(dimensions)
        data_zeros = np.zeros(dimensions)
        data = np.ones(dimensions)
        data[0][0] = 0

        test_args = [['All elements are fill value', 'filled', data_ones, True],
                     ['No elements are fill value', 'not_filled', data_zeros, False],
                     ['Only some elements are filled', 'some_filled', data, False]]

        for description, dataset_name, input_data, expected_result in test_args:
            with self.subTest(description):
                dataset = h5_file.create_dataset(dataset_name, data=input_data,
                                                 fillvalue=fill_value)
                result = dataset_all_fill_value(dataset, self.cf_config,
                                                self.logger, fill_value)
                self.assertEqual(result, expected_result)

        data_3d = np.stack([data_ones, data_zeros, data])
        dataset_3d = h5_file.create_dataset('data_3d', data=data_3d, fillvalue=fill_value)

        test_args_3d = [['3-D all band elements are fill value', 0, True],
                        ['3-D no band elements are fill value', 1, False],
                        ['3-D some band elements are fill value', 2, False]]

        for description, band, expected_result in test_args_3d:
            with self.subTest(description):
                result = dataset_all_fill_value(dataset_3d, self.cf_config,
                                                self.logger, fill_value, band)
                self.assertEqual(result, expected_result)

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
        bad_dataset_name = list(self.cf_config.fill_values.keys())[0]
        bad_dataset_value = self.cf_config.fill_values[bad_dataset_name]

        data_array = np.ones((3, 2))
        default_fill_value = 1.0
        fill_value_attr = 2.0
        fill_value_class = 3.0

        h5_file = h5py.File(self.test_h5_name, 'w')

        with self.subTest('Override default value for specific named datasets'):
            dataset = h5_file.create_dataset(bad_dataset_name,
                                             data=data_array,
                                             fillvalue=fill_value_class)
            dataset.attrs['_FillValue'] = fill_value_attr
            self.assertEqual(get_fill_value(dataset, self.cf_config,
                                            self.logger, default_fill_value),
                             bad_dataset_value)

        with self.subTest('_FillValue attribute should take precedence.'):
            dataset = h5_file.create_dataset('attribute',
                                             data=data_array,
                                             fillvalue=fill_value_class)
            dataset.attrs['_FillValue'] = fill_value_attr
            self.assertEqual(get_fill_value(dataset, self.cf_config,
                                            self.logger, default_fill_value),
                             fill_value_attr)

        with self.subTest('Without _FillValue, fall back on the fillvalue class attribute'):
            dataset = h5_file.create_dataset('class_attribute',
                                             data=data_array,
                                             fillvalue=fill_value_class)
            self.assertEqual(get_fill_value(dataset, self.cf_config,
                                            self.logger, default_fill_value),
                             fill_value_class)

        with self.subTest('Return default when nothing is set.'):
            string_data = np.chararray((3, 3))
            string_data[:] = 'abc'
            dataset = h5_file.create_dataset('default', data=string_data)
            self.assertEqual(get_fill_value(dataset, self.cf_config,
                                            self.logger, default_fill_value),
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
        data_3d = np.stack([data_array, filled_array])

        reference_indices = (2, 2)
        target_indices = (0, 0)
        pixel_scale_x = 0.1

        h5_file = h5py.File(self.test_h5_name, 'w')
        dataset = h5_file.create_dataset('test', data=data_array, fillvalue=fill_value)
        filled_dataset = h5_file.create_dataset('filled_test',
                                                data=filled_array,
                                                fillvalue=fill_value)
        dataset_3d = h5_file.create_dataset('three_dim', data=data_3d, fillvalue=fill_value)

        test_args_2d = [['2-D corner extrapolated.', filled_dataset],
                        ['2-D corner has data in it.', dataset]]

        for description, test_dataset in test_args_2d:
            with self.subTest(description):
                self.assertEqual(
                    extrapolate_coordinate(test_dataset, fill_value, 'x',
                                           data_array[reference_indices],
                                           reference_indices, 1,
                                           target_indices, pixel_scale_x,
                                           self.logger),
                    0.0
                )

        test_args_3d = [['3-D corner extrapolated.', 1],
                        ['3-D corner has data in it.', 0]]

        for description, band in test_args_3d:
            with self.subTest(description):
                self.assertEqual(
                    extrapolate_coordinate(dataset_3d, fill_value, 'x',
                                           data_3d[band][reference_indices],
                                           (band, 2, 2), 2, (band, 0, 0),
                                           pixel_scale_x, self.logger),
                    0.0
                )

        h5_file.close()

    def test_get_pixel_size_from_data_extent(self):
        """The correct pixel extents should be calculated for both dimensions,
        and if one is invalid (largely due to date being in a single column or row),
        the assumption of square pixels should be applied. Finally, if both
        pixel scales are found to be zero, a custom exception, InsufficientDataError,
        should be raised.

        Note: Python indices for 2-D arrays are array[row][column]

        """
        x_two, y_two = (15.0, 5.0)
        x_one, y_one = (10.0, 10.0)
        ll_indices = (0, 0)
        ur_indices = (5, 5)
        ll_indices_3d = (0, 0, 0)
        ur_indices_3d = (0, 5, 5)
        scales = (1.0, -1.0)

        valid_test_args = [['2-D both extents non-zero', ll_indices, ur_indices],
                           ['3-D both extents non-zero', ll_indices_3d, ur_indices_3d]]

        for description, ll_inds, ur_inds in valid_test_args:
            with self.subTest(description):
                self.assertEqual(
                    get_pixel_size_from_data_extent(x_one, y_one, x_two, y_two,
                                                    ll_inds, ur_inds),
                    scales
                )

        test_args = [['2-D x extent zero', x_one, y_one, x_one, y_two, ll_indices, ur_indices],
                     ['2-D y extent zero', x_one, y_one, x_two, y_one, ll_indices, ur_indices],
                     ['2-D both extents zero', x_one, y_one, x_one, y_one, ll_indices, ur_indices],
                     ['3-D x extent zero', x_one, y_one, x_one, y_two, ll_indices_3d, ur_indices_3d],
                     ['3-D y extent zero', x_one, y_one, x_two, y_one, ll_indices_3d, ur_indices_3d],
                     ['3-D both extents zero', x_one, y_one, x_one, y_one, ll_indices_3d, ur_indices_3d],
                     ['Divide by zero x', x_one, y_one, x_two, y_two, (0, 0), (5, 0)],
                     ['Divide by zero y', x_one, y_one, x_two, y_two, (0, 0), (0, 5)]]

        for description, x_0, y_0, x_N, y_M, ll_inds, ur_inds in test_args:
            with self.subTest(description):
                with self.assertRaises(InsufficientDataError):
                    get_pixel_size_from_data_extent(x_0, y_0, x_N, y_M,
                                                    ll_inds, ur_inds)

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

        data_3d = np.stack([data_array, data_bad_both])

        test_args = [['3-D corners valid', data_3d, data_3d, (0, 0, 0), (0, 2, 2), 0],
                     ['3-D corner invalid', data_3d, data_3d, (1, 0, 1), (1, 1, 2), 1]]

        for description, data_one, data_two, result_ll, result_ur, band in test_args:
            with self.subTest(description):
                lower_left_valid, upper_right_valid = get_valid_coordinates_extent(
                    data_one, data_two, fill_value, fill_value, lower_left_tuple,
                    upper_right_tuple, band
                )
                self.assertEqual(lower_left_valid, result_ll)
                self.assertEqual(upper_right_valid, result_ur)

    @patch('pymods.H5GridProjectionInfo.get_crs')
    def test_get_corner_points_from_lat_lon(self, mock_get_crs):
        """Ensure extrapolation occurs where expected, corners with valid points
        are used outright, and a InsufficientDataError is returned for entirely
        filled coordinate arrays.

        This test circumvents the retrieval of the Proj4 string at the start of
        the function, as the functionality being tested is that the extrapolation
        is consistent with expectations.

        """
        mock_get_crs.return_value = {'proj': 'eqc'}

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

        data_array_3d = np.stack([data_array, data_array, data_array, data_array])
        lat_array_3d = np.stack([lat_array, lat_array, lat_array, lat_array])
        lon_array_3d = np.stack([lon_array, lon_array, lon_array, lon_array])

        x_lower_left = 556597.453966368
        y_lower_left = 1113194.9079327357
        x_upper_right = 2226389.815865471
        y_upper_right = 2782987.269831839

        h5_file = h5py.File(self.test_h5_name, 'w')
        data = h5_file.create_dataset('data', data=data_array, fillvalue=fill_value)
        data_3d = h5_file.create_dataset('data_3d', data=data_array_3d, fillvalue=fill_value)

        test_args = [['Neither_corner_filled', False, False, False],
                     ['Lower_leftcorner_filled', True, False, False],
                     ['Upper_right_corner_filled', False, True, False],
                     ['Both_corners_filled', True, True, False],
                     ['3-D_Neither_corner_filled', False, False, True],
                     ['3-D_Lower_leftcorner_filled', True, False, True],
                     ['3-D_Upper_right_corner_filled', False, True, True],
                     ['3-D_Both_corners_filled', True, True, True]]

        for description, fill_ll_corner, fill_ur_corner, is_3d in test_args:
            with self.subTest(description):
                if is_3d:
                    dataset = data_3d
                    lat_copy = np.copy(lat_array_3d)
                    lon_copy = np.copy(lon_array_3d)
                    lat_slice = lat_copy[0]
                    lon_slice = lon_copy[0]
                else:
                    dataset = data
                    lat_copy = np.copy(lat_array)
                    lon_copy = np.copy(lon_array)
                    lat_slice = lat_copy
                    lon_slice = lon_copy

                if fill_ll_corner:
                    lat_slice[0][0] = fill_value
                    lon_slice[0][0] = fill_value

                if fill_ur_corner:
                    lat_slice[-1][-1] = fill_value
                    lon_slice[-1][-1] = fill_value

                lat_name = f'/lat_{description}'
                lon_name = f'/lon_{description}'
                h5_file.create_dataset(lat_name, data=lat_copy, fillvalue=fill_value)
                h5_file.create_dataset(lon_name, data=lon_copy, fillvalue=fill_value)

                dataset.attrs['coordinates'] = f'{lat_name} {lon_name}'.encode('utf-8')

                corners = get_corner_points_from_lat_lon(dataset,
                                                         self.cf_config,
                                                         self.logger)

                self.assertAlmostEqual(corners[0], x_lower_left)
                self.assertAlmostEqual(corners[1], x_upper_right)
                self.assertAlmostEqual(corners[2], y_lower_left)
                self.assertAlmostEqual(corners[3], y_upper_right)

        with self.subTest('Entirely filled latitude should raise an InsufficientDataError'):
            with self.assertRaises(InsufficientDataError) as context_manager:
                lat_copy = np.copy(lat_array)
                lon_copy = np.copy(lon_array)
                lat_copy.fill(fill_value)

                lat_name = '/lat_filled'
                lon_name = '/lon_filled'
                h5_file.create_dataset(lat_name, data=lat_copy, fillvalue=fill_value)
                h5_file.create_dataset(lon_name, data=lon_copy, fillvalue=fill_value)

                data.attrs['coordinates'] = f'{lat_name} {lon_name}'.encode('utf-8')

                corners = get_corner_points_from_lat_lon(data,
                                                         self.cf_config,
                                                         self.logger)

        h5_file.close()

    def test_get_cell_size_from_dimensions(self):
        """Given an input dataset, check the returned cell_width and cell_height."""
        data_array = np.ones((3, 4))
        x_array = np.array([1, 2, 3, 4])
        y_array = np.array([2, 4, 6])

        h5_file = h5py.File(self.test_h5_name, 'w')
        data = h5_file.create_dataset('data', data=data_array)
        x = h5_file.create_dataset('x', data=x_array)
        y = h5_file.create_dataset('y', data=y_array)
        data.attrs.create('DIMENSION_LIST', ((y.ref, ), (x.ref, )), dtype=h5py.ref_dtype)

        cell_width, cell_height = get_cell_size_from_dimensions(data)

        self.assertEqual(cell_width, 1)
        self.assertEqual(cell_height, 2)

        h5_file.close()

    @patch('pymods.H5GridProjectionInfo.get_crs')
    def test_get_cell_size_from_lat_lon_extents(self, mock_get_crs):
        """Given an input dataset, check the returned cell_width and cell_height."""
        mock_get_crs.return_value = {'proj': 'lonlat'}

        data_array = np.ones((3, 4))
        lon_array = np.array([[1, 2, 3, 4], [1, 2, 3, 4], [1, 2, 3, 4]])
        lat_array = np.array([[2, 2, 2, 2], [4, 4, 4, 4], [6, 6, 6, 6]])

        h5_file = h5py.File(self.test_h5_name, 'w')
        data = h5_file.create_dataset('data', data=data_array)
        h5_file.create_dataset('longitude', data=lon_array)
        h5_file.create_dataset('latitude', data=lat_array)

        data.attrs['coordinates'] = b'/longitude /latitude'

        cell_width, cell_height = get_cell_size_from_lat_lon_extents(data, 1,
                                                                     4, 2, 6)

        self.assertAlmostEqual(cell_width, 1)
        self.assertAlmostEqual(cell_height, 2)

        h5_file.close()

    def test_get_corner_points_from_dimensions(self):
        """Ensure the [0, 0] and [-1, -1] corner coordinates are returned in
        the case that they are specified via dimensions.

        Note - the corner point values are the cell corners.

        """
        data_array = np.ones((3, 4))
        x_array = np.array([1, 2, 3, 4])
        y_array = np.array([2, 4, 6])

        h5_file = h5py.File(self.test_h5_name, 'w')
        data = h5_file.create_dataset('data', data=data_array)
        x = h5_file.create_dataset('x', data=x_array)
        y = h5_file.create_dataset('y', data=y_array)
        data.attrs.create('DIMENSION_LIST', ((y.ref, ), (x.ref, )), dtype=h5py.ref_dtype)

        x_0, x_N, y_0, y_M = get_corner_points_from_dimensions(data)

        self.assertEqual(x_0, 0.5)
        self.assertEqual(x_N, 4.5)
        self.assertEqual(y_0, 1)
        self.assertEqual(y_M, 7)

        h5_file.close()

    @patch('pymods.H5GridProjectionInfo.get_cell_size_from_lat_lon_extents')
    @patch('pymods.H5GridProjectionInfo.get_corner_points_from_lat_lon')
    def test_get_transform_dimensions(self, mock_get_corner_points_from_lat_lon,
                                      mock_get_cell_size_from_lat_lon):
        """Ensure the correct Affine transformation matrix is formed for a
        dataset that has a DIMENSION_LIST attribute. This should not call
        `get_corner_points_from_lat_lon` or `get_cell_size_from_lat_lon_extents`.

        The expected transformation:

        [[a, b, c],      [[1, 0, 1.5],
         [d, e, f],   =   [0, 3, 2.5],
         [g, h, i]]       [0, 0, 1]]

        """
        data_array = np.ones((3, 4))
        x_array = np.array([2, 3, 4, 5])
        y_array = np.array([4, 7, 10])

        h5_file = h5py.File(self.test_h5_name, 'w')
        data = h5_file.create_dataset('data', data=data_array)
        x = h5_file.create_dataset('x', data=x_array)
        y = h5_file.create_dataset('y', data=y_array)
        data.attrs.create('DIMENSION_LIST', ((y.ref, ), (x.ref, )), dtype=h5py.ref_dtype)

        affine_transformation = get_transform(data, self.cf_config, self.logger)

        self.assertEqual(affine_transformation.a, 1)
        self.assertEqual(affine_transformation.b, 0)
        self.assertEqual(affine_transformation.c, 1.5)
        self.assertEqual(affine_transformation.d, 0)
        self.assertEqual(affine_transformation.e, 3)
        self.assertEqual(affine_transformation.f, 2.5)
        self.assertEqual(affine_transformation.g, 0)
        self.assertEqual(affine_transformation.h, 0)
        self.assertEqual(affine_transformation.i, 1)
        mock_get_corner_points_from_lat_lon.assert_not_called()
        mock_get_cell_size_from_lat_lon.assert_not_called()

        h5_file.close()

    @patch('pymods.H5GridProjectionInfo.get_cell_size_from_dimensions')
    @patch('pymods.H5GridProjectionInfo.get_corner_points_from_dimensions')
    @patch('pymods.H5GridProjectionInfo.get_crs')
    def test_get_transform_coordinates(self, mock_get_crs,
                                       mock_get_corner_points_from_dimensions,
                                       mock_get_cell_size_from_dimensions):
        """Ensure the correct Affine transformation matrix is formed for a
        dataset that uses the coordinate attribute. This should not call
        `get_corner_points_from_dimensions` or `get_cell_size_from_dimensions`.

        The expected transformation:

        [[a, b, c],      [[1, 0, 1.5],
         [d, e, f],   =   [0, 3, 2.5],
         [g, h, i]]       [0, 0, 1]]

        """
        mock_get_crs.return_value = {'proj': 'lonlat'}

        data_array = np.ones((3, 3))
        lon_array = np.array([[2, 3, 4], [2, 3, 4], [2, 3, 4]])
        lat_array = np.array([[4, 4, 4], [7, 7, 7], [10, 10, 10]])

        h5_file = h5py.File(self.test_h5_name, 'w')
        data = h5_file.create_dataset('data', data=data_array)
        h5_file.create_dataset('longitude', data=lon_array)
        h5_file.create_dataset('latitude', data=lat_array)

        data.attrs['coordinates'] = b'/longitude /latitude'
        data_array = np.ones((3, 4))

        affine_transformation = get_transform(data, self.cf_config, self.logger)

        self.assertEqual(affine_transformation.a, 1)
        self.assertEqual(affine_transformation.b, 0)
        self.assertEqual(affine_transformation.c, 1.5)
        self.assertEqual(affine_transformation.d, 0)
        self.assertEqual(affine_transformation.e, 3)
        self.assertEqual(affine_transformation.f, 2.5)
        self.assertEqual(affine_transformation.g, 0)
        self.assertEqual(affine_transformation.h, 0)
        self.assertEqual(affine_transformation.i, 1)
        mock_get_corner_points_from_dimensions.assert_not_called()
        mock_get_cell_size_from_dimensions.assert_not_called()

        h5_file.close()

    def test_get_transform_information(self):
        """ Ensure the correct string representation of the supporting dataset
            information is returned for a science dataset with either a
            DIMENSION_LIST attribute or a coordinate attribute.

        """
        with self.subTest('DIMENSION_LIST present'):
            h5_file = h5py.File('tests/data/SMAP_L4_SM_aup_input.h5', 'r')
            dataset = h5_file['/Analysis_Data/sm_profile_analysis']
            transform = get_transform_information(dataset)
            self.assertEqual(transform, 'DIMENSION_LIST: /y, /x')
            h5_file.close()

        with self.subTest('DIMENSION_LIST absent'):
            h5_file = h5py.File('tests/data/SMAP_L3_FT_P_corners_input.h5', 'r')
            group = '/Freeze_Thaw_Retrieval_Data_Global'
            dataset = h5_file[f'{group}/altitude_dem.Bands_01']
            expected_result = (f'coords: {group}/latitude.Bands_01 '
                               f'{group}/longitude.Bands_01')
            transform = get_transform_information(dataset)
            self.assertEqual(transform, expected_result)
            h5_file.close()

    def test_get_lon_lat_datasets(self):
        """Ensure the coordinate datasets specified in the 'coordinates'
        attribute are returned. If either of those datasets are absent, an
        exception should be raised.

        """
        lat_array = np.array([[1, 1], [2, 2]])
        lon_array = np.array([[1, 2], [1, 2]])
        lat_name = '/latitude'
        lon_name = '/longitude'
        bad_lat_name = '/absent_latitude'
        bad_lon_name = '/absent_longitude'

        h5_file = h5py.File(self.test_h5_name, 'w')
        dataset = h5_file.create_dataset('data', data=np.ones((2, 2)))
        lat = h5_file.create_dataset(lat_name, data=lat_array)
        lon = h5_file.create_dataset(lon_name, data=lon_array)
        dataset.attrs['coordinates'] = f'{lat_name} {lon_name}'.encode('utf-8')

        with self.subTest('latitude and longitude both present'):
            lon_out, lat_out = get_lon_lat_datasets(dataset)
            self.assertEqual(lon_out, lon)
            self.assertEqual(lat_out, lat)

        test_args = [[f'{lat_name} {bad_lon_name}', f'{bad_lon_name}'],
                     [f'{bad_lat_name} {lon_name}', f'{bad_lat_name}']]

        for coordinates_attr, missing_coords in test_args:
            with self.subTest(coordinates_attr):
                dataset.attrs['coordinates'] = coordinates_attr.encode('utf-8')
                with self.assertRaises(MissingCoordinateDataset) as context:
                    lon_out, lat_out = get_lon_lat_datasets(dataset)
                    self.assertTrue(missing_coords in context.exception.message)

    def test_get_hdf_proj4(self):
        """Ensure that a Proj4 string is returned, where possible. The order
        of projection information should be: DIMENSION_LIST, grid_mapping,
        configuration file or raise an exception.

        """
        global_proj4 = CRS('EPSG:6933').to_cf()
        dim_x_name = '/x'
        dim_y_name = '/y'
        dataset_name = '/Freeze_Thaw_Retrieval_polar/latitude'

        h5_file = h5py.File(self.test_h5_name, 'w')
        h5_file.create_dataset('data', data=np.ones((2, 2)))
        config_dataset = h5_file.create_dataset(dataset_name, data=np.ones((2, 2)))
        dim_x = h5_file.create_dataset(dim_x_name, data=np.ones((2, )))
        dim_y = h5_file.create_dataset(dim_y_name, data=np.ones((3, )))
        grid_mapping = h5_file.create_dataset('grid_mapping', (10, ))
        grid_mapping.attrs.update(global_proj4)

        with self.subTest('No information, uses configured defaults.'):
            proj4 = get_hdf_proj4(config_dataset, self.cf_config, self.logger)
            self.assertTrue(proj4.startswith('+proj=laea'))

        with self.subTest('grid_mapping attribute present.'):
            config_dataset.attrs['grid_mapping'] = grid_mapping.ref
            proj4 = get_hdf_proj4(config_dataset, self.cf_config, self.logger)
            self.assertTrue(proj4.startswith('+proj=cea'))

        with self.subTest('DIMENSION_LIST present, units degrees.'):
            dim_x.attrs['units'] = bytes('degrees', 'utf-8')
            config_dataset.attrs.create('DIMENSION_LIST',
                                        ((dim_x.ref, ), (dim_y.ref, )),
                                        dtype=h5py.ref_dtype)
            proj4 = get_hdf_proj4(config_dataset, self.cf_config, self.logger)
            self.assertTrue(proj4.startswith('+proj=longlat'))

        with self.subTest('DIMENSION_LIST, non degrees falls back to grid_mapping.'):
            dim_x.attrs['units'] = bytes('metres', 'utf-8')
            proj4 = get_hdf_proj4(config_dataset, self.cf_config, self.logger)
            self.assertTrue(proj4.startswith('+proj=cea'))

    def test_get_dimension_datasets(self):
        """If a dataset has a DIMENSION_LIST attribute, the listed references
        should be extracted and returned. Otherwise, the function should return
        None.

        """
        dim_x_name = '/x'
        dim_y_name = '/y'

        h5_file = h5py.File(self.test_h5_name, 'w')
        dataset = h5_file.create_dataset('data', data=np.ones((3, 2)))
        dim_x = h5_file.create_dataset(dim_x_name, data=np.ones((2, )))
        dim_y = h5_file.create_dataset(dim_y_name, data=np.ones((3, )))

        with self.subTest('No DIMENSION_LIST'):
            self.assertEqual(get_dimension_datasets(dataset), None)

        with self.subTest('Valid DIMENSION_LIST'):
            dataset.attrs.create('DIMENSION_LIST',
                                 ((dim_y.ref, ), (dim_x.ref, )),
                                 dtype=h5py.ref_dtype)

            dim_x_out, dim_y_out = get_dimension_datasets(dataset)
            self.assertEqual(dim_x_out, dim_x)
            self.assertEqual(dim_y_out, dim_y)

    def test_get_dataset_attributes(self):
        """ Ensure a dictionary is returned, with all string values decoded
            from bytes, which is the default return type from
            `h5py.Dataset.attrs`.

        """
        attribute_dictionary = {'float': 1.234, 'string': '123'}

        h5_file = h5py.File(self.test_h5_name, 'w')
        dataset = h5_file.create_dataset('data', data=np.ones((2, 2)))

        for attribute_name, attribute_value in attribute_dictionary.items():
            dataset.attrs.create(attribute_name, attribute_value)

        output_dictionary = get_dataset_attributes(dataset)
        self.assertDictEqual(output_dictionary, attribute_dictionary)

    def test_get_grid_mapping_name(self):
        """ Ensure the grid mapping name is returned, accounting for relative
            paths, extended naming format and whether the listed grid mapping
            is in the HDF-5 file.

        """
        h5_file = h5py.File(self.test_h5_name, 'w')
        h5_file.create_dataset('crs', data=np.ones((1,)))
        science_dataset = h5_file.create_dataset('/group/data',
                                                 data=np.ones((2, 2)))

        with self.subTest('No grid_mapping attribute'):
            self.assertEqual(get_grid_mapping_name(science_dataset), None)

        test_args = [['Full path grid_mapping', '/crs', '/crs'],
                     ['Extended grid_mapping', '/crs: grid_x grid_y', '/crs'],
                     ['Relative grid_mapping', '../crs', '/crs'],
                     ['Relative, extended grid_mapping',
                      '../crs: ../grid_x ../grid_y', '/crs']]

        for description, grid_mapping_attribute, expected_name in test_args:
            with self.subTest(description):
                science_dataset.attrs.create('grid_mapping',
                                             grid_mapping_attribute)
                grid_mapping_name = get_grid_mapping_name(science_dataset)
                self.assertEqual(grid_mapping_name, expected_name)

        with self.subTest('Reference not in file, raises exception'):
            with self.assertRaises(InvalidMetadata):
                science_dataset.attrs.create('grid_mapping', 'crs_2')
                get_grid_mapping_name(science_dataset)

    def test_resolve_relative_dataset_path(self):
        """ Ensure a relative path can be qualified to a full path using the
            location of the dataset making the reference. If the reference
            implies the referee is more nested than it is, a custom exception
            should be raised.

        """
        h5_file = h5py.File(self.test_h5_name, 'w')
        dataset = h5_file.create_dataset('/group/nested_group/science',
                                         data=np.ones((2, 3)))
        h5_file.create_dataset('/another_group/variable', data=np.ones((1,)))
        h5_file.create_dataset('/group/grid_mapping', data=np.ones((1,)))
        h5_file.create_dataset('/grid_mapping', data=np.ones((1,)))

        test_args = [['Already absolute path', '/another_group/variable',
                      '/another_group/variable'],
                     ['Nested relative path', '../grid_mapping',
                      '/group/grid_mapping'],
                     ['Double nested relative path', '../../grid_mapping',
                      '/grid_mapping']]

        for description, reference, expected_path in test_args:
            with self.subTest(description):
                resolved_path = resolve_relative_dataset_path(dataset,
                                                              reference)

                self.assertEqual(resolved_path, expected_path)

        test_args = [['Incorrect nesting', '../../../grid_mapping'],
                     ['Missing reference', 'non_existant_variable']]

        for description, relative_path in test_args:
            with self.subTest('Incorrect reference nesting'):
                with self.assertRaises(InvalidMetadata):
                    resolve_relative_dataset_path(dataset, relative_path)

    def test_get_crs(self):
        """ Ensure that this function can handle:

            * A dictionary of grid mapping attributes
            * A dataset with valid grid mapping metadata
            * A dataset where the grid mapping metadata failed, but has an SRID

        """
        proj4_string = ('+proj=cea +lat_ts=30 +lon_0=0 +x_0=0 +y_0=0 '
                        '+datum=WGS84 +units=m +no_defs +type=crs')

        h5_file = h5py.File(self.test_h5_name, 'w')
        dataset = h5_file.create_dataset('crs', data=np.ones((1,)))

        cf_attributes = {
            'grid_mapping_name': 'lambert_cylindrical_equal_area',
            'standard_parallel': 30.0,
            'longitude_of_central_meridian': 0.0,
            'false_easting': 0.0,
            'false_northing': 0.0
        }

        bad_wkt_attribute = {'crs_wkt': 'PROJCRS["THIS IS GARBLED"]',
                             'srid': 'urn:ogc:def:crs:EPSG::6933'}

        test_args = [['EPSG code', cf_attributes, None],
                     ['Good CF attributes', dataset, cf_attributes],
                     ['SRID backup', dataset, bad_wkt_attribute]]

        for description, input_object, dataset_attributes in test_args:
            with self.subTest(description):
                if dataset_attributes is not None:
                    dataset.attrs.clear()
                    dataset.attrs.update(dataset_attributes)

                self.assertEqual(get_crs(input_object).to_proj4(), proj4_string)
