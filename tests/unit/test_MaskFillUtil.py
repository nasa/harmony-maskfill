from logging import getLogger
from unittest import TestCase
from unittest.mock import patch

import h5py
import geopandas as gpd
from geopandas.testing import assert_geodataframe_equal
from pyproj import CRS

from pymods.MaskFillUtil import (get_h5_mask_array_id, get_bounded_shape,
                                 get_geotiff_crs, get_geotiff_info,
                                 should_ignore_pyproj_bounds)


class TestMaskFillUtil(TestCase):

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
            crs = CRS.from_epsg(6931)
            expected_gdf = gpd.read_file('tests/data/south_pole_bounded.geojson')
            bounded_gdf = get_bounded_shape(shape_path, crs, None, None)

            assert_geodataframe_equal(expected_gdf, bounded_gdf)

        with self.subTest('Creating bounded shape from CRS without EPSG code'):
            # This will truncate the shape by the longitude and latitude values
            # associated with the grid.
            with patch.object(CRS, 'to_epsg', return_value=None):
                crs = CRS.from_epsg(6931)
                expected_gdf = gpd.read_file('tests/data/south_pole_bounded2.geojson')
                bounded_gdf = get_bounded_shape(shape_path, crs, out_shape,
                                                transform)
                assert_geodataframe_equal(expected_gdf, bounded_gdf)

        with self.subTest('UTM projections do not use pyproj bounds'):
            utm_crs = CRS.from_epsg(32618)
            colombia_shape_path = 'tests/data/COL.geo.json'
            expected_gdf = gpd.read_file(colombia_shape_path)

            with patch('pymods.MaskFillUtil.CRS') as mock_crs:
                bounded_gdf = get_bounded_shape(colombia_shape_path, utm_crs,
                                                out_shape, transform)
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
