from unittest import TestCase
from unittest.mock import patch

import h5py
import geopandas as gpd
from geopandas.testing import geom_equals
from pyproj import CRS
import affine

from pymods.MaskFillUtil import get_h5_mask_array_id, get_bounded_shape
from GeotiffMaskFill import get_geotiff_proj4, get_geotiff_info


class TestMaskFillUtil(TestCase):

    def test_get_h5_mask_array_id(self):
        """ Ensure that for datasets either with or without a DIMENSION_LIST
            attribute the expect hash is returned. Datasets that share
            projected coordinates should return the same hash. Datasets within
            the same file, with different projected coordinates should have
            different hashes.

        """
        shape_path = 'tests/data/USA.geo.json'

        with self.subTest('DIMENSION_LIST present, shared dimensions'):
            expected_id = 'a62e96c11d707f2153e4f6a7da7707fc681152a358b816af5c9bcd11'
            short_name = 'SPL4SMAU'
            h5_file = h5py.File('tests/data/SMAP_L4_SMAU_input.h5', 'r')
            dataset_one = h5_file['/Analysis_Data/sm_profile_analysis']
            dataset_two = h5_file['/Analysis_Data/sm_surface_analysis']

            mask_id_one = get_h5_mask_array_id(dataset_one, shape_path,
                                               short_name)
            mask_id_two = get_h5_mask_array_id(dataset_two, shape_path,
                                               short_name)

            self.assertEqual(mask_id_one, expected_id)
            self.assertEqual(mask_id_two, expected_id)
            h5_file.close()

        with self.subTest('DIMENSION_LIST absent, different coordinates'):
            expected_mask_id_one = '5f0739e33f6e7c0a8b0692919d3d12f4bbf47fdf61fff07dacecbfcd'
            expected_mask_id_two = '60040a5542740b9f65063fb27f0ff90ace42f67234083b4802001899'
            short_name = 'SPL3FTP'
            group = '/Freeze_Thaw_Retrieval_Data_Global'
            h5_file = h5py.File('tests/data/SMAP_L3_corners_input.h5', 'r')
            dataset_one = h5_file[f'{group}/altitude_dem.Bands_01']
            dataset_two = h5_file[f'{group}/altitude_dem.Bands_02']

            mask_id_one = get_h5_mask_array_id(dataset_one, shape_path,
                                               short_name)
            mask_id_two = get_h5_mask_array_id(dataset_two, shape_path,
                                               short_name)

            self.assertEqual(mask_id_one, expected_mask_id_one)
            self.assertEqual(mask_id_two, expected_mask_id_two)
            h5_file.close()

    def test_get_bounded_shape(self):
        """ Tests the method for getting bounded shape geodataframes using EPSG codes/proj4
        strings to get the geographic extent of the data using pyproj, as well as calculating
        the geographic extent of the data using the shape of the data and the transform.
        """

        geotiff_path = 'tests/data/SMAP_L4_SMAU_input.tif'
        shape_path = 'tests/data/south_pole1.geo.json'

        proj4 = get_geotiff_proj4(geotiff_path)
        out_shape, transform = get_geotiff_info(geotiff_path)

        # Test creating bounding shape from EPSG code
        gdf1 = gpd.read_file('tests/data/south_pole1_bounded.geojson')
        gdf2 = get_bounded_shape(shape_path, 6933, None, None, None)

        assert geom_equals(gdf1, gdf2)

        # Test creating bounding shape from proj4 string
        epsg = CRS(proj4).to_epsg()
        gdf3 = get_bounded_shape(shape_path, epsg, None, None, None)
        assert geom_equals(gdf1, gdf3)

        # Test calculating bounding shape from proj4, out shape, and transform
        gdf4 = gpd.read_file('tests/data/south_pole1_bounded2.geojson')
        gdf5 = get_bounded_shape(shape_path, None, proj4, out_shape, transform)

        assert geom_equals(gdf4, gdf5)
