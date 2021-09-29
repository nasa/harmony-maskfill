from logging import getLogger
from unittest import TestCase

import h5py
import geopandas as gpd
from geopandas.testing import assert_geodataframe_equal
from pyproj import CRS

from pymods.cf_config import CFConfigH5
from pymods.MaskFillUtil import (get_h5_mask_array_id, get_bounded_shape,
                                 get_geotiff_info)
from GeotiffMaskFill import get_geotiff_proj4


class TestMaskFillUtil(TestCase):

    @classmethod
    def setUpClass(cls):
        cls.cf_config = CFConfigH5('tests/data/SMAP_L3_FT_P_corners_input.h5')
        cls.logger = getLogger('test')

    def test_get_h5_mask_array_id(self):
        """ Ensure that for datasets either with or without a DIMENSION_LIST
            attribute the expect hash is returned. Datasets that share
            projected coordinates should return the same hash. Datasets within
            the same file, with different projected coordinates should have
            different hashes.

        """
        shape_path = 'tests/data/USA.geo.json'
        spl3ftp_config = CFConfigH5('tests/data/SMAP_L3_FT_P_corners_input.h5')
        spl4smau_config = CFConfigH5('tests/data/SMAP_L4_SM_aup_input.h5')

        with self.subTest('DIMENSION_LIST present, shared dimensions'):
            expected_id = '45a5aa3c6b02ae31b06ae524ee823474132b6f4a74a790bf37623f17'
            h5_file = h5py.File('tests/data/SMAP_L4_SM_aup_input.h5', 'r')
            dataset_one = h5_file['/Analysis_Data/sm_profile_analysis']
            dataset_two = h5_file['/Analysis_Data/sm_surface_analysis']

            mask_id_one = get_h5_mask_array_id(dataset_one, shape_path,
                                               spl4smau_config, self.logger)
            mask_id_two = get_h5_mask_array_id(dataset_two, shape_path,
                                               spl4smau_config, self.logger)

            self.assertEqual(mask_id_one, expected_id)
            self.assertEqual(mask_id_two, expected_id)
            h5_file.close()

        with self.subTest('DIMENSION_LIST absent, different coordinates'):
            expected_mask_id_one = '5f0739e33f6e7c0a8b0692919d3d12f4bbf47fdf61fff07dacecbfcd'
            expected_mask_id_two = '60040a5542740b9f65063fb27f0ff90ace42f67234083b4802001899'
            group = '/Freeze_Thaw_Retrieval_Data_Global'
            h5_file = h5py.File('tests/data/SMAP_L3_FT_P_corners_input.h5', 'r')
            dataset_one = h5_file[f'{group}/altitude_dem.Bands_01']
            dataset_two = h5_file[f'{group}/altitude_dem.Bands_02']

            mask_id_one = get_h5_mask_array_id(dataset_one, shape_path,
                                               spl3ftp_config, self.logger)
            mask_id_two = get_h5_mask_array_id(dataset_two, shape_path,
                                               spl3ftp_config, self.logger)

            self.assertEqual(mask_id_one, expected_mask_id_one)
            self.assertEqual(mask_id_two, expected_mask_id_two)
            h5_file.close()

    def test_get_bounded_shape(self):
        """ Tests the method for getting bounded shape geodataframes using EPSG codes/proj4
        strings to get the geographic extent of the data using pyproj, as well as calculating
        the geographic extent of the data using the shape of the data and the transform.
        """

        geotiff_path = 'tests/data/SMAP_L3_FT_P_polar_3d_input.tif'
        shape_path = 'tests/data/south_pole.geo.json'
        epsg = 6931

        proj4 = get_geotiff_proj4(geotiff_path)
        out_shape, transform = get_geotiff_info(geotiff_path)

        # Test creating bounding shape from EPSG code
        gdf1 = gpd.read_file('tests/data/south_pole_bounded.geojson')
        gdf2 = get_bounded_shape(shape_path, epsg, None, None, None)

        assert_geodataframe_equal(gdf1, gdf2)

        # Test creating bounding shape from proj4 string
        epsg = CRS(proj4).to_epsg()
        gdf3 = get_bounded_shape(shape_path, epsg, None, None, None)
        assert_geodataframe_equal(gdf1, gdf3)

        # Test calculating bounding shape from proj4, out shape, and transform
        gdf4 = gpd.read_file('tests/data/south_pole_bounded2.geojson')
        gdf5 = get_bounded_shape(shape_path, None, proj4, out_shape, transform)

        assert_geodataframe_equal(gdf4, gdf5)
