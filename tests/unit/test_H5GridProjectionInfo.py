from logging import getLogger
from os import mkdir
from os.path import isdir, join
from shutil import rmtree
from unittest import TestCase
from unittest.mock import patch

from pyproj import CRS, Proj
import h5py
import numpy as np

from pymods.cf_config import CFConfigH5
from pymods.exceptions import (InsufficientDataError,
                               InsufficientProjectionInformation,
                               InvalidMetadata, MissingCoordinateDataset)
from pymods.H5GridProjectionInfo import (dataset_all_fill_value,
                                         dataset_all_outside_valid_range,
                                         get_cell_size_from_dimensions,
                                         get_cell_size_from_lat_lon_extents,
                                         get_corner_points_from_dimensions,
                                         get_corner_points_from_lat_lon,
                                         get_crs_from_grid_mapping,
                                         get_dataset_attributes,
                                         get_dimension_datasets,
                                         get_fill_value,
                                         get_grid_mapping_name,
                                         get_hdf_crs,
                                         get_lon_lat_datasets,
                                         get_projected_coordinate_extent,
                                         get_transform,
                                         is_projection_x_dimension,
                                         is_projection_y_dimension,
                                         is_x_y_flipped,
                                         has_geographic_dimensions,
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
        config_dataset_name = list(self.cf_config.fill_values.keys())[0]
        config_dataset_fill = self.cf_config.fill_values[config_dataset_name]

        data_array = np.ones((3, 2))
        default_fill_value = 1.0
        fill_value_attr = 2.0
        fill_value_class = 3.0

        # Create test file:
        h5_file = h5py.File(self.test_h5_name, 'w')

        # Add test dataset that matches a configuration file rule:
        dataset_with_config = h5_file.create_dataset(
            config_dataset_name,
            data=data_array,
            fillvalue=fill_value_class
        )
        dataset_with_config.attrs['_FillValue'] = fill_value_attr

        # Add test dataset with a fill value:
        dataset_with_fill = h5_file.create_dataset(
            'with_fill',
            data=data_array,
            fillvalue=fill_value_class
        )
        dataset_with_fill.attrs['_FillValue'] = fill_value_attr

        # Add test dataset without a fill value:
        dataset_without_fill = h5_file.create_dataset(
            'without_fill',
            data=data_array
        )

        with self.subTest('Override default value for specific named datasets'):
            self.assertEqual(
                get_fill_value(
                    dataset_with_config,
                    self.cf_config,
                    self.logger,
                    default_fill_value
                ),
                config_dataset_fill
            )

        with self.subTest('_FillValue attribute should take precedence.'):
            self.assertEqual(
                get_fill_value(
                    dataset_with_fill,
                    self.cf_config,
                    self.logger,
                    default_fill_value
                ),
                fill_value_attr
            )

        with self.subTest('No config or in-file fill value, user-specified default'):
            self.assertEqual(
                get_fill_value(
                    dataset_without_fill,
                    self.cf_config,
                    self.logger,
                    default_fill_value
                ),
                default_fill_value
            )

        with self.subTest('Last option: use variable-type default fill value.'):
            dataset = h5_file.create_dataset('type_default', data=data_array)
            self.assertEqual(
                get_fill_value(
                    dataset_without_fill,
                    self.cf_config,
                    self.logger,
                    None
                ),
                -9999.0
            )

        h5_file.close()

    def test_get_corner_points_from_lat_lon(self):
        """Ensure extrapolation occurs where expected, corners with valid points
        are used outright, and a InsufficientDataError is returned for entirely
        filled coordinate arrays.

        This test circumvents the retrieval of the Proj4 string at the start of
        the function, as the functionality being tested is that the extrapolation
        is consistent with expectations.

        """
        crs = CRS.from_proj4('+proj=eqc')

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
        data = h5_file.create_dataset('data', data=data_array)
        data.attrs.create('_FillValue', fill_value)

        data_3d = h5_file.create_dataset('data_3d', data=data_array_3d)
        data_3d.attrs.create('_FillValue', fill_value)

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
                lat_dataset = h5_file.create_dataset(lat_name, data=lat_copy)
                lat_dataset.attrs.create('_FillValue', fill_value)

                lon_dataset = h5_file.create_dataset(lon_name, data=lon_copy)
                lon_dataset.attrs.create('_FillValue', fill_value)

                dataset.attrs['coordinates'] = f'{lat_name} {lon_name}'.encode('utf-8')

                corners = get_corner_points_from_lat_lon(dataset, crs,
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
                lat_dataset = h5_file.create_dataset(lat_name, data=lat_copy)
                lat_dataset.attrs.create('_FillValue', fill_value)
                lon_dataset = h5_file.create_dataset(lon_name, data=lon_copy)
                lon_dataset.attrs.create('_FillValue', fill_value)

                data.attrs['coordinates'] = f'{lat_name} {lon_name}'.encode('utf-8')

                corners = get_corner_points_from_lat_lon(data, crs,
                                                         self.cf_config,
                                                         self.logger)

            self.assertEqual(context_manager.exception.message,
                             '/lon_filled or /lat_filled have no valid data.')

        h5_file.close()

    def test_get_projected_coordinate_extent(self):
        """ Ensure the x and y dimension can be correctly extrapolated when
            the coordinate arrays contain fill values in the corners.

            An InsufficientDataError should be returned if unfilled data only
            exist in a single row or column.

        """
        projection = Proj({'proj': 'eqc'})

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

        with self.subTest('x - No filled coordinates'):
            valid_data = np.where(data_array == 1)
            x_min, x_max = get_projected_coordinate_extent(projection,
                                                           lat_array,
                                                           lon_array,
                                                           1, valid_data)

            self.assertAlmostEqual(x_min, x_lower_left)
            self.assertAlmostEqual(x_max, x_upper_right)

        with self.subTest('x - Filled outer columns'):
            valid_data = np.where((lon_array > 5) & (lon_array < 20))
            x_min, x_max = get_projected_coordinate_extent(projection,
                                                           lat_array,
                                                           lon_array,
                                                           1, valid_data)

            self.assertAlmostEqual(x_min, x_lower_left)
            self.assertAlmostEqual(x_max, x_upper_right)

        with self.subTest('y - Filled outer rows'):
            valid_data = np.where((lat_array < 25) & (lat_array > 10))
            y_min, y_max = get_projected_coordinate_extent(projection,
                                                           lat_array,
                                                           lon_array,
                                                           0, valid_data)

            self.assertAlmostEqual(y_min, y_lower_left)
            self.assertAlmostEqual(y_max, y_upper_right)

        with self.subTest('x - only single column of data'):
            valid_data = np.where((lon_array > 10) & (lon_array < 20))

            with self.assertRaises(InsufficientDataError) as context_manager:
                x_min, x_max = get_projected_coordinate_extent(projection,
                                                               lat_array,
                                                               lon_array,
                                                               1, valid_data)

            self.assertEqual(context_manager.exception.message,
                             ('Only a single, unmasked column of data. Unable '
                              'to calculate x pixel size.'))

        with self.subTest('y - only single row of data'):
            valid_data = np.where((lat_array < 25) & (lat_array > 15))
            with self.assertRaises(InsufficientDataError) as context_manager:
                y_min, y_max = get_projected_coordinate_extent(projection,
                                                               lat_array,
                                                               lon_array,
                                                               0, valid_data)

            self.assertEqual(context_manager.exception.message,
                             ('Only a single, unmasked row of data. Unable '
                              'to calculate y pixel size.'))

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

    def test_get_cell_size_from_lat_lon_extents(self):
        """Given an input dataset, check the returned cell_width and cell_height."""
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
        """ Ensure the correct Affine transformation matrix is formed for a
            dataset that has a DIMENSION_LIST attribute. This should not call
            `get_corner_points_from_lat_lon` or
            `get_cell_size_from_lat_lon_extents`.

            The expected transformation (when array dimensions match the
            coordinate dimensions, e.g., rows = y and columns = x):

            [[a, b, c],      [[1, 0, 1.5],
             [d, e, f],   =   [0, 3, 2.5],
             [g, h, i]]       [0, 0, 1]]

            The expected transformation when the array dimensions are flipped
            with respect to the coordinate dimensions, e.g., rows = x and
            columns = y.

            [[a, b, c],      [[0, 1, 1.5],
             [d, e, f],   =   [3, 0, 2.5],
             [g, h, i]]       [0, 0, 1]]

        """
        crs = CRS(4326)
        data_array = np.ones((3, 4))
        x_array = np.array([2, 3, 4, 5])
        y_array = np.array([4, 7, 10])

        with h5py.File(self.test_h5_name, 'w') as h5_file:
            x = h5_file.create_dataset('x', data=x_array)
            x.attrs.create('standard_name', 'projection_x_coordinate')

            y = h5_file.create_dataset('y', data=y_array)
            y.attrs.create('standard_name', 'projection_y_coordinate')

            data = h5_file.create_dataset('data', data=data_array)
            data.attrs.create('DIMENSION_LIST', ((y.ref, ), (x.ref, )),
                              dtype=h5py.ref_dtype)

            flipped_data = h5_file.create_dataset('flipped_data',
                                                  data=np.ones((4, 3)))
            flipped_data.attrs.create('DIMENSION_LIST', ((x.ref, ), (y.ref, )),
                                      dtype=h5py.ref_dtype)

            with self.subTest('Unflipped dataset'):
                affine_transform = get_transform(data, crs, self.cf_config,
                                                 self.logger)

                self.assertEqual(affine_transform.a, 1)
                self.assertEqual(affine_transform.b, 0)
                self.assertEqual(affine_transform.c, 1.5)
                self.assertEqual(affine_transform.d, 0)
                self.assertEqual(affine_transform.e, 3)
                self.assertEqual(affine_transform.f, 2.5)
                self.assertEqual(affine_transform.g, 0)
                self.assertEqual(affine_transform.h, 0)
                self.assertEqual(affine_transform.i, 1)
                mock_get_corner_points_from_lat_lon.assert_not_called()
                mock_get_cell_size_from_lat_lon.assert_not_called()

            with self.subTest('Flipped dataset'):
                flipped_transform = get_transform(flipped_data, crs,
                                                  self.cf_config, self.logger)

                self.assertEqual(flipped_transform.a, 0)
                self.assertEqual(flipped_transform.b, 1)
                self.assertEqual(flipped_transform.c, 1.5)
                self.assertEqual(flipped_transform.d, 3)
                self.assertEqual(flipped_transform.e, 0)
                self.assertEqual(flipped_transform.f, 2.5)
                self.assertEqual(flipped_transform.g, 0)
                self.assertEqual(flipped_transform.h, 0)
                self.assertEqual(flipped_transform.i, 1)
                mock_get_corner_points_from_lat_lon.assert_not_called()
                mock_get_cell_size_from_lat_lon.assert_not_called()

    @patch('pymods.H5GridProjectionInfo.get_cell_size_from_dimensions')
    @patch('pymods.H5GridProjectionInfo.get_corner_points_from_dimensions')
    def test_get_transform_coordinates(self,
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
        crs = CRS.from_epsg(4326)

        data_array = np.ones((3, 3))
        lon_array = np.array([[2, 3, 4], [2, 3, 4], [2, 3, 4]])
        lat_array = np.array([[4, 4, 4], [7, 7, 7], [10, 10, 10]])

        h5_file = h5py.File(self.test_h5_name, 'w')
        data = h5_file.create_dataset('data', data=data_array)
        h5_file.create_dataset('longitude', data=lon_array)
        h5_file.create_dataset('latitude', data=lat_array)

        data.attrs['coordinates'] = b'/longitude /latitude'
        data_array = np.ones((3, 4))

        affine_transformation = get_transform(data, crs, self.cf_config,
                                              self.logger)

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

                self.assertEqual(context.exception.message,
                                 (f'Cannot find "{missing_coords}" in '
                                  f'"{self.test_h5_name}".'))

    def test_get_hdf_crs(self):
        """Ensure that a `pyproj.CRS` object is returned, where possible. The
            order of projection information should be: DIMENSION_LIST,
            grid_mapping, configuration file or raise an exception.

            - EPSG:4326: Geographic.
            - EPSG:6931: EASE-2 Grid North.
            - EPSG:6933: EASE-2 Grid Global.

        """
        global_grid_mapping_cf = CRS(6933).to_cf()
        dim_x_name = '/x'
        dim_y_name = '/y'
        dataset_name = '/Freeze_Thaw_Retrieval_polar/latitude'

        h5_file = h5py.File(self.test_h5_name, 'w')
        h5_file.create_dataset('data', data=np.ones((2, 2)))
        config_dataset = h5_file.create_dataset(dataset_name, data=np.ones((2, 2)))
        dim_x = h5_file.create_dataset(dim_x_name, data=np.ones((2, )))
        dim_y = h5_file.create_dataset(dim_y_name, data=np.ones((3, )))
        grid_mapping = h5_file.create_dataset('grid_mapping', (10, ))
        grid_mapping.attrs.update(global_grid_mapping_cf)

        with self.subTest('No projection information or configuration, raises exception'):
            with self.assertRaises(InsufficientProjectionInformation) as context:
                with patch.object(CFConfigH5,
                                  'get_dataset_grid_mapping_attributes',
                                  return_value=None):

                    cf_config = CFConfigH5('tests/data/SMAP_L3_FT_P_corners_input.h5')
                    crs = get_hdf_crs(config_dataset, cf_config,
                                      self.logger)

            self.assertEqual(context.exception.message,
                             ('Cannot find projection information for dataset:'
                              ' /Freeze_Thaw_Retrieval_polar/latitude.'))

        with self.subTest('No information, uses configured defaults.'):
            crs = get_hdf_crs(config_dataset, self.cf_config, self.logger)
            self.assertEqual(crs.to_epsg(), 6931)

        with self.subTest('grid_mapping attribute present (EASE-2 Global).'):
            config_dataset.attrs['grid_mapping'] = grid_mapping.ref
            crs = get_hdf_crs(config_dataset, self.cf_config, self.logger)
            self.assertTrue(crs.to_epsg(), 6933)

        with self.subTest('DIMENSION_LIST present, units degrees.'):
            dim_x.attrs['units'] = bytes('degrees', 'utf-8')
            config_dataset.attrs.create('DIMENSION_LIST',
                                        ((dim_x.ref, ), (dim_y.ref, )),
                                        dtype=h5py.ref_dtype)
            crs = get_hdf_crs(config_dataset, self.cf_config, self.logger)
            self.assertTrue(crs.to_epsg(), 4326)

        with self.subTest('DIMENSION_LIST, non degrees falls back to grid_mapping.'):
            dim_x.attrs['units'] = bytes('metres', 'utf-8')
            crs = get_hdf_crs(config_dataset, self.cf_config, self.logger)
            self.assertTrue(crs.to_epsg(), 6933)

        h5_file.close()

    def test_has_geographic_dimensions_true(self):
        """ Ensure if a variable can be correctly determined as geographically
            gridded based on its dimensions. This should be true for 2-D
            variables with only geographic dimensions as well as 3-D variables
            that have a non-geographic dimension first, e.g.: (time, lat, lon).

        """
        dimension_data = np.linspace(0, 10, num=11)
        science_data = np.ones((11, 11))
        h5_file = h5py.File(self.test_h5_name, 'w')
        lon_dimension = h5_file.create_dataset('lon', data=dimension_data)
        lon_dimension.attrs['units'] = 'degrees_east'
        lat_dimension = h5_file.create_dataset('lat', data=dimension_data)
        lat_dimension.attrs['units'] = 'degrees_north'

        temporal_dimension = h5_file.create_dataset('temporal_dim',
                                                    data=np.ones((1, )))
        temporal_dimension.attrs['unit'] = 'seconds since 2020-01-01T00:00:00'

        geo_dataset = h5_file.create_dataset('var_with_geo_dims',
                                             data=science_data)

        temporal_geo_dataset = h5_file.create_dataset('var_with_time_and_geo',
                                                      data=np.ones((1, 11, 11)))

        geo_dataset.attrs.create(
            'DIMENSION_LIST', ((lat_dimension.ref, ), (lon_dimension.ref, )),
            dtype=h5py.ref_dtype
        )
        temporal_geo_dataset.attrs.create(
            'DIMENSION_LIST', ((temporal_dimension.ref, ),
                               (lat_dimension.ref, ), (lon_dimension.ref, )),
            dtype=h5py.ref_dtype
        )

        test_args = [['2-D geographic variable', geo_dataset],
                     ['3-D geographic variable', temporal_geo_dataset]]

        for description, test_dataset in test_args:
            with self.subTest('Geographic dimensions'):
                self.assertTrue(has_geographic_dimensions(test_dataset))

        h5_file.close()

    def test_has_geographic_dimensions_false(self):
        """ Ensure a non-geographically gridded variable can be correctly
            determined as such based on its dimensions. Such variables could be
            projection gridded, not list dimensions at all, or have dimensions
            with no units.

        """
        dimension_data = np.linspace(0, 10, num=11)
        science_data = np.ones((11, 11))
        h5_file = h5py.File(self.test_h5_name, 'w')

        non_geo_dimension = h5_file.create_dataset('non_geo_dim',
                                                   data=dimension_data)
        non_geo_dimension.attrs['unit'] = 'm'
        no_unit_dimension = h5_file.create_dataset('no_unit_dim',
                                                   data=dimension_data)

        no_dim_dataset = h5_file.create_dataset('var_no_dims',
                                                data=science_data)

        non_geo_dataset = h5_file.create_dataset('var_with_non_geo_dims',
                                                 data=science_data)
        no_units_dataset = h5_file.create_dataset('var_with_unitless_dims',
                                                  data=science_data)

        non_geo_dataset.attrs.create('DIMENSION_LIST',
                                     ((non_geo_dimension.ref, ), ),
                                     dtype=h5py.ref_dtype)
        no_units_dataset.attrs.create('DIMENSION_LIST',
                                      ((no_unit_dimension.ref, ), ),
                                      dtype=h5py.ref_dtype)

        test_args = [['No dimensions present', no_dim_dataset],
                     ['Non-geographic dimensions', non_geo_dataset],
                     ['Dimension with no units', no_units_dataset]]

        for description, test_dataset in test_args:
            with self.subTest(description):
                self.assertFalse(has_geographic_dimensions(test_dataset))

        h5_file.close()

    def test_get_dimension_datasets(self):
        """ If a dataset has a DIMENSION_LIST attribute, the listed references
            should be extracted and returned. Otherwise, the function should
            return None.

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

        h5_file.close()

    def test_get_dimension_datasets_square_array(self):
        """ Ensure that if a square array is supplied, the correct dimensions
            are retrieved, rather than the same dimension twice. This function
            assumes standard Python index ordering of (row, column), meaning
            the x dimension will be assumed to be last.

        """
        dim_x_name = '/x'
        dim_y_name = '/y'

        h5_file = h5py.File(self.test_h5_name, 'w')
        dim_x = h5_file.create_dataset(dim_x_name, data=np.ones((3, )))
        dim_y = h5_file.create_dataset(dim_y_name, data=np.ones((3, )))
        dim_time = h5_file.create_dataset('/time', data=np.ones((1, )))

        flat_dataset = h5_file.create_dataset('flat_data',
                                              data=np.ones((3, 3)))
        flat_dataset.attrs.create('DIMENSION_LIST',
                                  ((dim_y.ref, ), (dim_x.ref, )),
                                  dtype=h5py.ref_dtype)

        banded_dataset = h5_file.create_dataset('banded_data',
                                                data=np.ones((1, 3, 3)))
        banded_dataset.attrs.create('DIMENSION_LIST',
                                    ((dim_time.ref, ), (dim_y.ref, ),
                                     (dim_x.ref, )), dtype=h5py.ref_dtype)

        with self.subTest('Flat, len(x) = len(y) variable'):
            dim_x_out, dim_y_out = get_dimension_datasets(flat_dataset)
            self.assertEqual(dim_x_out, dim_x)
            self.assertEqual(dim_y_out, dim_y)

        with self.subTest('Banded, len(x) = len(y) variable'):
            dim_x_out, dim_y_out = get_dimension_datasets(banded_dataset)
            self.assertEqual(dim_x_out, dim_x)
            self.assertEqual(dim_y_out, dim_y)

        h5_file.close()

    def test_is_x_y_flipped(self):
        """ Ensure that a collection is correctly identified as being either
            [..., y, x] or [..., x, y].

            Note, Python array dimensions are [..., row, column], so will
            appear flipped if the last two dimensions are ordered [..., x, y].

        """
        with h5py.File(self.test_h5_name, 'w') as h5_file:
            dim_x = h5_file.create_dataset('/x', data=np.ones((2, )))
            dim_x.attrs.create('standard_name', 'projection_x_coordinate')
            dim_x.attrs.create('units', 'm')

            dim_y = h5_file.create_dataset('/y', data=np.ones((3, )))
            dim_y.attrs.create('standard_name', 'projection_y_coordinate')
            dim_y.attrs.create('units', 'm')

            dim_lon = h5_file.create_dataset('/lon', data=np.ones((4, )))
            dim_lon.attrs.create('standard_name', 'longitude')
            dim_lon.attrs.create('units', 'degrees_east')

            dim_lat = h5_file.create_dataset('/lat', data=np.ones((5, )))
            dim_lat.attrs.create('standard_name', 'latitude')
            dim_lat.attrs.create('units', 'degrees_north')

            flipped_xy_dataset = h5_file.create_dataset('flipped_data',
                                                        data=np.ones((2, 3)))
            flipped_xy_dataset.attrs.create('DIMENSION_LIST',
                                            ((dim_x.ref, ), (dim_y.ref, )),
                                            dtype=h5py.ref_dtype)

            unflipped_xy_dataset = h5_file.create_dataset('unflipped_data',
                                                          data=np.ones((3, 2)))
            unflipped_xy_dataset.attrs.create('DIMENSION_LIST',
                                              ((dim_y.ref, ), (dim_x.ref, )),
                                              dtype=h5py.ref_dtype)

            flipped_geo_dataset = h5_file.create_dataset('flipped_geo',
                                                         data=np.ones((4, 5)))
            flipped_geo_dataset.attrs.create('DIMENSION_LIST',
                                             ((dim_lon.ref, ), (dim_lat.ref, )),
                                             dtype=h5py.ref_dtype)
            unflipped_geo_dataset = h5_file.create_dataset('unflipped_geo',
                                                           data=np.ones((5, 4)))
            unflipped_geo_dataset.attrs.create('DIMENSION_LIST',
                                               ((dim_lat.ref, ), (dim_lon.ref, )),
                                               dtype=h5py.ref_dtype)

            with self.subTest('Flipped projected data returns True'):
                self.assertTrue(is_x_y_flipped(flipped_xy_dataset))

            with self.subTest('Unflipped projected data returns False'):
                self.assertFalse(is_x_y_flipped(unflipped_xy_dataset))

            with self.subTest('Flipped geographic data returns True'):
                self.assertTrue(is_x_y_flipped(flipped_geo_dataset))

            with self.subTest('Unflipped geographic data returns False'):
                self.assertFalse(is_x_y_flipped(unflipped_geo_dataset))

    def test_is_projection_x_dimension(self):
        """ Ensure that an HDF-5 dataset is correctly determined as being an
            x spatial dimension (either longitude or projected x). Other
            dimension types should be identified as not being an x dimension.

        """
        with h5py.File(self.test_h5_name, 'w') as h5_file:
            dim_x = h5_file.create_dataset('/x', data=np.ones((2, )))
            dim_x.attrs.create('standard_name', 'projection_x_coordinate')
            dim_x.attrs.create('units', 'm')

            dim_y = h5_file.create_dataset('/y', data=np.ones((3, )))
            dim_y.attrs.create('standard_name', 'projection_y_coordinate')
            dim_y.attrs.create('units', 'm')

            dim_lon = h5_file.create_dataset('/lon', data=np.ones((4, )))
            dim_lon.attrs.create('standard_name', 'longitude')
            dim_lon.attrs.create('units', 'degrees_east')

            dim_lat = h5_file.create_dataset('/lat', data=np.ones((5, )))
            dim_lat.attrs.create('standard_name', 'latitude')
            dim_lat.attrs.create('units', 'degrees_north')

            dim_time = h5_file.create_dataset('/time', data=np.ones((6, )))
            dim_time.attrs.create('standard_name', 'time')
            dim_time.attrs.create('units', 'seconds since 1980-01-01T00:00:00')

            with self.subTest('Projected x dimension returns True'):
                self.assertTrue(is_projection_x_dimension(dim_x))

            with self.subTest('Longitude dimension returns True'):
                self.assertTrue(is_projection_x_dimension(dim_lon))

            with self.subTest('Projected y dimension returns False'):
                self.assertFalse(is_projection_x_dimension(dim_y))

            with self.subTest('Latitude dimension returns False'):
                self.assertFalse(is_projection_x_dimension(dim_lat))

            with self.subTest('Non-spatial dimension returns False'):
                self.assertFalse(is_projection_x_dimension(dim_time))

    def test_is_projection_y_dimension(self):
        """ Ensure that an HDF-5 dataset is correctly determined as being a
            y spatial dimension (either latitude or projected y). Other
            dimension types should be identified as not being a y dimension.

        """
        with h5py.File(self.test_h5_name, 'w') as h5_file:
            dim_x = h5_file.create_dataset('/x', data=np.ones((2, )))
            dim_x.attrs.create('standard_name', 'projection_x_coordinate')
            dim_x.attrs.create('units', 'm')

            dim_y = h5_file.create_dataset('/y', data=np.ones((3, )))
            dim_y.attrs.create('standard_name', 'projection_y_coordinate')
            dim_y.attrs.create('units', 'm')

            dim_lon = h5_file.create_dataset('/lon', data=np.ones((4, )))
            dim_lon.attrs.create('standard_name', 'longitude')
            dim_lon.attrs.create('units', 'degrees_east')

            dim_lat = h5_file.create_dataset('/lat', data=np.ones((5, )))
            dim_lat.attrs.create('standard_name', 'latitude')
            dim_lat.attrs.create('units', 'degrees_north')

            dim_time = h5_file.create_dataset('/time', data=np.ones((6, )))
            dim_time.attrs.create('standard_name', 'time')
            dim_time.attrs.create('units', 'seconds since 1980-01-01T00:00:00')

            with self.subTest('Projected x dimension returns False'):
                self.assertFalse(is_projection_y_dimension(dim_x))

            with self.subTest('Longitude dimension returns False'):
                self.assertFalse(is_projection_y_dimension(dim_lon))

            with self.subTest('Projected y dimension returns True'):
                self.assertTrue(is_projection_y_dimension(dim_y))

            with self.subTest('Latitude dimension returns True'):
                self.assertTrue(is_projection_y_dimension(dim_lat))

            with self.subTest('Non-spatial dimension returns False'):
                self.assertFalse(is_projection_y_dimension(dim_time))

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
            with self.assertRaises(InvalidMetadata) as context_manager:
                science_dataset.attrs.create('grid_mapping', 'crs_2')
                get_grid_mapping_name(science_dataset)

            self.assertEqual(context_manager.exception.message,
                             ('Invalid metadata in /group/data: '
                              'grid_mapping or coordinate="crs_2": '
                              'Variable reference not in file'))

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

        test_args = [['Relative path has incorrect nesting', '../../../grid_mapping'],
                     ['Variable reference not in file', 'non_existant_variable']]

        for description, relative_path in test_args:
            with self.subTest('Incorrect reference nesting'):
                with self.assertRaises(InvalidMetadata) as context_manager:
                    resolve_relative_dataset_path(dataset, relative_path)

                self.assertTrue(context_manager.exception.message.endswith(description))

    def test_get_crs_from_grid_mapping(self):
        """ Ensure that this function can handle:

            * A dictionary of grid mapping attributes
            * A dataset with valid grid mapping metadata
            * A dataset where the grid mapping metadata failed, but has an SRID

        """
        expected_epsg_code = 6933

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

        test_args = [['Dictionary of attributes', cf_attributes, None],
                     ['Good CF attributes in dataset', dataset, cf_attributes],
                     ['SRID backup in dataset', dataset, bad_wkt_attribute]]

        for description, input_object, dataset_attributes in test_args:
            with self.subTest(description):
                if dataset_attributes is not None:
                    dataset.attrs.clear()
                    dataset.attrs.update(dataset_attributes)

                self.assertEqual(get_crs_from_grid_mapping(input_object).to_epsg(),
                                 expected_epsg_code)
