from os.path import exists
from shutil import rmtree
from tempfile import mkdtemp
from unittest import TestCase
from unittest.mock import patch
import json

import h5py
import geopandas as gpd
import numpy as np
from affine import Affine
from geopandas import GeoDataFrame
from geopandas.testing import assert_geodataframe_equal
from numpy.testing import assert_allclose, assert_array_equal
from pyproj import CRS
from shapely.geometry import Polygon, shape

from pymods.MaskFillUtil import (create_bounding_box_shape_file,
                                 get_bounded_shape, get_decoded_attribute,
                                 get_default_fill_for_data_type,
                                 get_geographic_resolution, get_geotiff_crs,
                                 get_geotiff_info, get_grid_lat_lons,
                                 get_h5_mask_array_id, get_resolved_dataframe,
                                 get_resolved_line, get_resolved_polygon,
                                 get_resolved_ring, get_resolved_shape,
                                 get_transform_information,
                                 should_ignore_pyproj_bounds)


class TestMaskFillUtil(TestCase):
    """ Tests for the functions in pymods/MaskFillUtil.py. """

    @classmethod
    def setUpClass(cls):
        """ Create resources that can be reused by all tests. """
        cls.ease_two_north = CRS.from_epsg(6931)
        cls.exterior_ring = [(3, 5), (1, 4), (1, 2), (5, 1), (6, 2), (6, 4),
                             (3, 5)]
        cls.interior_ring = [(3, 4), (4, 3), (2, 3), (3, 4)]
        cls.resolution = 2
        cls.resolved_exterior_ring = [(3, 5), (2, 4.5), (1, 4), (1, 2),
                                      (2.333, 1.667), (3.667, 1.333), (5, 1),
                                      (6, 2), (6, 4), (4.5, 4.5)]
        cls.resolved_interior_ring = [(3, 4), (4, 3), (2, 3), (3, 4)]
        cls.solid_polygon = Polygon(cls.exterior_ring)
        cls.hollow_polygon = Polygon(cls.exterior_ring,
                                     holes=[cls.interior_ring])
        cls.resolved_solid_polygon = Polygon(cls.resolved_exterior_ring)
        cls.resolved_hollow_polygon = Polygon(cls.resolved_exterior_ring,
                                              holes=[cls.interior_ring])

    def setUp(self):
        """ Create test-specific resources that are reset for each test. """
        self.tmp_dir = mkdtemp()

    def tearDown(self):
        """ Clean-up operations to occur after every test. """
        rmtree(self.tmp_dir)

    def assert_points_equal(self, list_one, list_two):
        """ Check that all coordinates are equal to three decimal places. This
            separate utility function is needed for those values that are
            recurring decimals.

        """
        self.assertEqual(len(list_one), len(list_two),
                         'Input and output have different number of points')

        for point_index, point_one in enumerate(list_one):
            for coordinate_index, coordinate in enumerate(point_one):
                self.assertAlmostEqual(coordinate,
                                       list_two[point_index][coordinate_index],
                                       places=3)

    def test_get_h5_mask_array_id(self):
        """ Ensure that for datasets either with or without a DIMENSION_LIST
            attribute the expect hash is returned. Datasets that share
            projected coordinates should return the same hash. Datasets within
            the same file, with different projected coordinates should have
            different hashes.

            All variables being tested use the EASE-2 Grid Global.

        """
        shape_path = 'tests/data/USA.geo.json'
        crs = CRS.from_epsg(6933)

        with self.subTest('DIMENSION_LIST present, shared dimensions'):
            expected_id = '22336df3ff475d45e5b0a91c18161323c089f697fc4c70d702937abd'
            h5_file = h5py.File('tests/data/SMAP_L4_SM_aup_input.h5', 'r')
            dataset_one = h5_file['/Analysis_Data/sm_profile_analysis']
            dataset_two = h5_file['/Analysis_Data/sm_surface_analysis']

            mask_id_one = get_h5_mask_array_id(dataset_one, crs, shape_path)
            mask_id_two = get_h5_mask_array_id(dataset_two, crs, shape_path)

            self.assertEqual(mask_id_one, expected_id)
            self.assertEqual(mask_id_two, expected_id)
            h5_file.close()

        with self.subTest('DIMENSION_LIST absent, different coordinates'):
            expected_mask_id_one = '1f689a7016523c01a6436da39fba978f26d73ee4200fbd3a0ccd613e'
            expected_mask_id_two = '95aa4fb28d44ae780450d85e362385a762ed917d48dc8ccf6e4ae293'
            group = '/Freeze_Thaw_Retrieval_Data_Global'
            h5_file = h5py.File('tests/data/SMAP_L3_FT_P_corners_input.h5', 'r')
            dataset_one = h5_file[f'{group}/altitude_dem.Bands_01']
            dataset_two = h5_file[f'{group}/altitude_dem.Bands_02']

            mask_id_one = get_h5_mask_array_id(dataset_one, crs, shape_path)
            mask_id_two = get_h5_mask_array_id(dataset_two, crs, shape_path)

            self.assertEqual(mask_id_one, expected_mask_id_one)
            self.assertEqual(mask_id_two, expected_mask_id_two)
            h5_file.close()

    def test_get_geotiff_crs(self):
        """ Ensure that a `pyproj.crs.CRS` object can be extracted from a
            GeoTIFF file, and that it contains the correct projection
            information.

            EPSG codes:

            * EPSG:6931 - NSIDC EASE-Grid 2.0 North.
            * EPSG:6933 - NSIDC EASE-Grid 2.0 Global.

        """
        test_args = [
            ['Polar GeoTIFF example', 'SMAP_L3_FT_P_polar_3d_input.tif', 6931],
            ['Global GeoTIFF example', 'SMAP_L4_SM_aup_input.tif', 6933]
        ]

        for description, file_name, expected_epsg_code in test_args:
            with self.subTest(description):
                geotiff_path = f'tests/data/{file_name}'
                geotiff_crs = get_geotiff_crs(geotiff_path)
                self.assertEqual(geotiff_crs.to_epsg(), expected_epsg_code)

    def test_get_bounded_shape(self):
        """ Tests the method for getting a bounded GeoJSON shape geodataframe,
            which truncates the raw GeoJSON shape according to the valid
            extents of the projection. These are determined by:

            - Preferably the Area of Use associated with an EPSG code for the
              projection.
            - The geographic information of the grid and affine transformation
              matrix.

        """

        geotiff_path = 'tests/data/SMAP_L3_FT_P_polar_3d_input.tif'
        shape_path = 'tests/data/south_pole.geo.json'
        out_shape, transform = get_geotiff_info(geotiff_path)

        with self.subTest('Creating bounded shape from CRS with EPSG code'):
            # This will truncate the shape at 0 degrees north, as that's the
            # lower latitude limit.
            expected_gdf = gpd.read_file('tests/data/south_pole_bounded_epsg.geo.json')
            bounded_gdf = get_bounded_shape(shape_path, self.ease_two_north,
                                            out_shape, transform)

            assert_geodataframe_equal(expected_gdf, bounded_gdf)

        with self.subTest('Creating bounded shape from CRS without EPSG code'):
            # This will truncate the shape by the longitude and latitude values
            # associated with the grid.
            with patch.object(CRS, 'to_epsg', return_value=None):
                crs = CRS.from_epsg(6931)
                expected_gdf = gpd.read_file('tests/data/south_pole_bounded_grid.geo.json')
                bounded_gdf = get_bounded_shape(shape_path, crs, out_shape,
                                                transform)
                assert_geodataframe_equal(expected_gdf, bounded_gdf)

        with self.subTest('UTM projections do not use pyproj bounds'):
            utm_crs = CRS.from_epsg(32618)
            expected_gdf = gpd.read_file('tests/data/COL_UTM_bounded.geo.json')

            with patch('pymods.MaskFillUtil.CRS') as mock_crs:
                bounded_gdf = get_bounded_shape('tests/data/COL.geo.json',
                                                utm_crs, out_shape, transform)
                mock_crs.assert_not_called()

            assert_geodataframe_equal(expected_gdf, bounded_gdf)

    def test_should_ignore_pyproj_bounds(self):
        """ Ensure that if a UTM projection is used, this function indicates
            that the `pyproj.CRS` bounds should not be used (so the unmasked
            output extends outside of a 6 degree wide longitude stripe).
            Otherwise, indicate that the `pyproj.CRS` bounds can be used.

        """
        with self.subTest('UTM zone 18N'):
            self.assertTrue(should_ignore_pyproj_bounds(CRS.from_epsg(32618)))

        test_args = [['Geographic', 4326],
                     ['EASE-2 Grid Global', 6933],
                     ['EASE-2 Grid North', 6931],
                     ['EASE-2 Grid South', 6932],
                     ['NSIDC Sea Ice, North Polar Stereographic', 3411]]

        for description, epsg_code in test_args:
            with self.subTest(description):
                self.assertFalse(should_ignore_pyproj_bounds(CRS.from_epsg(epsg_code)))

    def test_create_bounding_box_shape_file(self):
        """ Ensure the function creates a file in the expected location, and
            that it contains the expected GeoJSON contents.

        """
        bounding_box = [10, 20, 30, 40]
        expected_coordinates = [[[10, 20], [30, 20], [30, 40], [10, 40],
                                 [10, 20]]]
        geojson_file_path = create_bounding_box_shape_file(bounding_box,
                                                           self.tmp_dir)

        # Ensure an output GeoJSON file is written with the expected structure
        self.assertTrue(exists(geojson_file_path), 'Output file not created')

        with open(geojson_file_path, 'r', encoding='utf-8') as file_handler:
            actual_geojson = json.load(file_handler)

        self.assertIn('features', actual_geojson)
        self.assertIn('geometry', actual_geojson['features'][0])
        self.assertIn('type', actual_geojson['features'][0]['geometry'])
        self.assertIn('coordinates', actual_geojson['features'][0]['geometry'])
        self.assertEqual(actual_geojson['features'][0]['geometry']['type'],
                         'Polygon')
        self.assertListEqual(
            actual_geojson['features'][0]['geometry']['coordinates'],
            expected_coordinates
        )

        # Ensure that geopandas can parse the output, and the contents are
        # correct
        parsed_dataframe = gpd.read_file(geojson_file_path)
        gpd_bbox = parsed_dataframe.iloc[0]
        self.assertEqual(parsed_dataframe['name'][0], 'Harmony bbox')
        self.assertTrue(gpd_bbox['geometry'].is_valid)
        self.assertTrue(gpd_bbox['geometry'].is_closed)
        self.assertEqual(gpd_bbox['geometry'].type, 'Polygon')
        self.assertTupleEqual(parsed_dataframe['geometry'][0].bounds,
                              tuple(bounding_box))

        # Expected longitude points: [W, E, E, W, W]
        expected_lon = np.array([bounding_box[0], bounding_box[2],
                                 bounding_box[2], bounding_box[0],
                                 bounding_box[0]])

        # Expected latitude points: [S, S, N, N, S]
        expected_lat = np.array([bounding_box[1], bounding_box[1],
                                 bounding_box[3], bounding_box[3],
                                 bounding_box[1]])
        actual_lon, actual_lat = parsed_dataframe['geometry'][0].exterior.xy
        assert_array_equal(actual_lon, expected_lon)
        assert_array_equal(actual_lat, expected_lat)

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

    def test_get_resolved_line(self):
        """ Ensure that a line, defined by its two end-points, will be
            converted so that there are evenly spaced points separated by,
            at most, the resolution supplied to the function.

            Note, in the first test, the distance between each point is 2.83,
            resulting from the smallest number of points possible being placed
            on the line at a distance of no greater than the requested
            resolution (3).

        """
        test_args = [
            ['Line needs additional points', (0, 0), (10, 10), 3,
             [(0, 0), (2, 2), (4, 4), (6, 6), (8, 8), (10, 10)]],
            ['Resolution bigger than line', (0, 0), (1, 1), 2,
             [(0, 0), (1, 1)]],
            ['Line flat in one dimension', (0, 0), (0, 10), 5,
             [(0, 0), (0, 5), (0, 10)]]
        ]

        for description, point_one, point_two, resolution, expected_output in test_args:
            with self.subTest(description):
                self.assertListEqual(get_resolved_line(point_one, point_two,
                                                       resolution),
                                     expected_output)

    def test_get_resolved_ring(self):
        """ Ensure that a full ring (either interior or exterior) for a
            polygon can be populated with points along each line segment, so
            the output ring has the required resolution along each edge.

            These rings will have some line segments that require additional
            points and others that do not.

        """
        test_args = [['Exterior ring', self.exterior_ring, self.resolved_exterior_ring],
                     ['Interior ring (no extra points)', self.interior_ring,
                         self.interior_ring[:-1]]]

        for description, input_ring, expected_output in test_args:
            with self.subTest(description):
                self.assert_points_equal(get_resolved_ring(input_ring,
                                                           self.resolution),
                                         expected_output)

    def test_get_resolved_polygon(self):
        """ Ensure a polygon, both with and without an interior hole, can be
            updated to include finer resolution points along each edge
            (external and internal).

            This test will construct a `geopandas.GeoDataFrame` to mimic the
            input to `get_resolved_polygon`.

        """
        test_args = [['Solid polygon', self.solid_polygon,
                      self.resolved_solid_polygon],
                     ['Polygon with hole', self.hollow_polygon,
                      self.resolved_hollow_polygon]]

        for description, input_polygon, expected_output in test_args:
            with self.subTest(description):
                gdf = GeoDataFrame(crs='epsg:4326', geometry=[input_polygon])
                resolved_output = get_resolved_polygon(next(gdf.iterfeatures()),
                                                       self.resolution)

                self.assert_points_equal(list(resolved_output.exterior.coords),
                                         list(expected_output.exterior.coords))

                for hole_index, hole in enumerate(resolved_output.interiors):
                    self.assert_points_equal(
                        list(hole.coords),
                        list(expected_output.interiors[hole_index].coords)
                    )

    def test_get_resolved_shape(self):
        """ Ensure a `geopandas.GeoDataFrame` with polygons in it can be
            converted into polygons at the requested resolution.

        """
        test_args = [['Solid polygon', self.solid_polygon,
                      self.resolved_solid_polygon],
                     ['Polygon with hole', self.hollow_polygon,
                      self.resolved_hollow_polygon]]

        for description, input_polygon, expected_output in test_args:
            with self.subTest(description):
                input_gdf = GeoDataFrame(crs='epsg:4326', geometry=[input_polygon])
                output_gdf = get_resolved_shape(input_gdf, self.resolution)
                output_polygon = shape(next(output_gdf.iterfeatures())['geometry'])

                self.assert_points_equal(list(output_polygon.exterior.coords),
                                         list(expected_output.exterior.coords))

                for hole_index, hole in enumerate(output_polygon.interiors):
                    self.assert_points_equal(
                        list(hole.coords),
                        list(expected_output.interiors[hole_index].coords)
                    )

    def test_get_resolved_shape_multipolygon(self):
        """ Ensure all polygons are resolved when a `geopandas.GeoDataFrame`
            contains multiple rows of polygons.

            The second polygon will use the class fixture of a solid polygon,
            but translate the points laterally.

        """
        second_polygon_points = [(point[0] + 10, point[1] + 10)
                                 for point in self.exterior_ring]
        second_polygon_input = Polygon(second_polygon_points)

        expected_second_output_points = [(point[0] + 10, point[1] + 10)
                                         for point
                                         in self.resolved_exterior_ring]

        expected_output_points = [self.resolved_solid_polygon,
                                  Polygon(expected_second_output_points)]

        input_gdf = GeoDataFrame(crs='epsg:4326',
                                 geometry=[self.solid_polygon,
                                           second_polygon_input])

        output_gdf = get_resolved_shape(input_gdf, self.resolution)

        for out_index, out_feature in enumerate(output_gdf.iterfeatures()):
            output_polygon = shape(out_feature['geometry'])
            self.assert_points_equal(
                list(output_polygon.exterior.coords),
                list(expected_output_points[out_index].exterior.coords)
            )

    def test_get_resolved_dataframe(self):
        """ Take a typical GeoJSON input file, in this case the USA, and
            ensure it can be resolved. Note, this file has a MultiPolygon
            geometry, so this test establishes that a MultiPolygon is
            correctly split into individual Polygon objects.

            The collection used is SPL3FTP, with a north polar band.

        """
        geotiff_path = 'tests/data/SMAP_L3_FT_P_polar_3d_input.tif'
        shape_path = 'tests/data/USA.geo.json'
        expected_geojson_path = 'tests/data/USA_resolved.geo.json'
        out_shape, transform = get_geotiff_info(geotiff_path)

        output_dataframe = get_resolved_dataframe(shape_path, transform,
                                                  self.ease_two_north,
                                                  out_shape)

        expected_dataframe = GeoDataFrame.from_file(expected_geojson_path)

        # check_less_precise is used below, as the written template output
        # truncates data values to fewer decimal places than the dataframe
        # returned by the function (template output is 6 d.p.).
        assert_geodataframe_equal(output_dataframe, expected_dataframe,
                                  check_less_precise=True)

    def test_get_grid_lat_lons(self):
        """ Ensure that an Affine transformation, CRS and array shape are
            correctly combined to return the expected arrays of latitude and
            longitude points.

            The Affine transformation is designed to have a starting point of
            approximately longitude = 10 degrees east and
            latitude = 10 degrees north.

        """
        transform = Affine(10000, 0, 1422897.377, 0, 10000, -8069652.027)
        out_shape = (3, 3)
        expected_latitudes = np.array([[10.0, 9.979, 9.959],
                                       [10.116, 10.096, 10.075],
                                       [10.232, 10.212, 10.191]])
        expected_longitudes = np.array([[10.0, 10.069, 10.138],
                                        [10.012, 10.081, 10.150],
                                        [10.024, 10.093, 10.162]])

        latitudes, longitudes = get_grid_lat_lons(transform,
                                                  self.ease_two_north,
                                                  out_shape)

        assert_allclose(expected_latitudes, latitudes, atol=1e-3)
        assert_allclose(expected_longitudes, longitudes, atol=1e-3)

    def test_get_geographic_resolution(self):
        """ Ensure the calculated resolution is the minimum Euclidean distance
            between diagonally adjacent pixels.

            The example coordinates below have the shortest diagonal difference
            between (10, 10) and (15, 15), resulting in a resolution of
            (5^2 + 5^2)^0.5 = 50^0.5 ~= 7.07.

        """
        latitudes = np.array([[10, 10, 10], [15, 15, 15], [25, 25, 25]])
        longitudes = np.array([[10, 15, 25], [10, 15, 25], [10, 15, 25]])
        expected_resolution = 7.071
        self.assertAlmostEqual(get_geographic_resolution(latitudes, longitudes),
                               expected_resolution, places=3)

    def test_get_decoded_attribute(self):
        """ Ensure attributes will be retrieved with the correct type. If the
            extracted type is a bytes object, it should be decoded to a string,
            otherwise the type of the retrieved metadata attribute should match
            the type as contained in the HDF-5 file.

            If the attribute is a single-element array, then the output should
            be only the element, not an array.

        """
        string_value = 'this is a string'
        decoded_bytes = 'bytes'
        bytes_value = bytes(decoded_bytes, 'utf-8')
        numerical_value = 123.456
        np_bytes_value = np.bytes_(decoded_bytes, 'utf-8')
        single_element_array = np.array([numerical_value])
        multi_element_array = np.array([numerical_value, numerical_value])

        test_args = [
            ['String attribute', 'string_value', string_value],
            ['Bytes attribute decoded', 'bytes_value', decoded_bytes],
            ['np.bytes_ attribute decoded', 'np_bytes_value', decoded_bytes],
            ['Numerical attribute', 'numerical_value', numerical_value],
            ['Absent attribute defaults to None', 'Missing', None],
            ['Single element array', 'single_element_array', numerical_value]
        ]

        default_test_args = [
            ['Default not used when value present', 'string_value', 'default', string_value],
            ['Default value used', 'missing', 'default', 'default'],
            ['Bytes default is decoded', 'missing', bytes_value, decoded_bytes],
        ]
        with h5py.File('test.h5', 'w', driver='core', backing_store=False) as h5_file:
            attributes = h5py.AttributeManager(parent=h5_file)
            attributes.create('string_value', string_value)
            attributes.create('bytes_value', bytes_value)
            attributes.create('numerical_value', numerical_value)
            attributes.create('np_bytes_value', np_bytes_value)
            attributes.create('single_element_array', single_element_array)
            attributes.create('multi_element_array', multi_element_array)

            for description, attribute_key, expected_value in test_args:
                with self.subTest(description):
                    self.assertEqual(
                        get_decoded_attribute(h5_file, attribute_key),
                        expected_value
                    )

            for description, key, default, expected_value in default_test_args:
                with self.subTest(description):
                    self.assertEqual(
                        get_decoded_attribute(h5_file, key, default),
                        expected_value
                    )

            np.testing.assert_array_equal(
                get_decoded_attribute(h5_file, 'multi_element_array'),
                multi_element_array
            )

    def test_get_default_fill_for_data_type(self):
        """ Ensure that the correct default fill value is retrieved based on
            the `numpy.dtype.name` supplied to the function. If there is an
            unrecognised input type, then -9999.0 should be returned.

            These tests are written out separately as parameterising them felt
            a bit too close to the actual code for independent testing.

        """
        with self.subTest('float64 returns -9999.0'):
            self.assertEqual(
                get_default_fill_for_data_type('float64'),
                -9999.0
            )

        with self.subTest('uint8 returns 254'):
            self.assertEqual(
                get_default_fill_for_data_type('uint8'),
                254
            )
        with self.subTest('None returns -9999.0'):
            self.assertEqual(
                get_default_fill_for_data_type(None),
                -9999.0
            )
        with self.subTest('Unrecognised type returns -9999.0'):
            self.assertEqual(
                get_default_fill_for_data_type('random_type_string'),
                -9999.0
            )
