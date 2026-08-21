import json
from unittest import TestCase

from maskfill.cf_config import CFConfig


class TestCFConfig(TestCase):
    """Test the common functionality of the CFConfig class."""

    @classmethod
    def setUpClass(cls):
        """Define objects to be reused between tests."""
        cls.cf_config = CFConfig('SPL3FTP')

        with open('maskfill/maskfill_config.json', 'r') as file_handler:
            cls.raw_config = json.load(file_handler)

    def test_cfconfig_instantiation(self):
        """Ensure that a CFConfig object was correctly created.

        Assertions test:

        - Collection short name.
        - Collection coordinate variables.
        - Collection fill values.
        - Collection grid-mapping information.

        """
        self.assertEqual(self.cf_config.shortname, 'SPL3FTP')
        self.assertListEqual(
            self.cf_config.coordinate_variables,
            self.raw_config['collection_coordinate_variables']['SPL3FTP'],
        )
        self.assertDictEqual(
            self.cf_config.fill_values,
            self.raw_config['corrected_fill_values']['SPL3FT(A|P|P_E)'],
        )
        self.assertDictEqual(
            self.cf_config.grid_mapping_groups,
            self.raw_config['grid_mapping_supplements']['SPL3FT(P|P_E)'],
        )

    def test_get_dataset_fill_value(self):
        """Ensure a fill value stored in the configuration file is returned
        for the requested dataset. If there is not match, `None` should be
        returned.

        """
        with self.subTest('Matching fill value in the configuration file'):
            self.assertEqual(
                self.cf_config.get_dataset_fill_value(
                    '/Freeze_Thaw_Retrieval_Data_Polar/latitude'
                ),
                -9999.0,
            )

        with self.subTest('No fill value in the configuration file'):
            self.assertEqual(
                self.cf_config.get_dataset_fill_value('/other_variable'), None
            )

    def test_get_coordinate_overrides(self):
        """Ensure the correct coordinate overrides are returned from the configuration file.
        If there is no match, `None` should be returned.

        """
        cf_config = CFConfig('SPL3SMP_E')
        self.assertEqual(cf_config.shortname, 'SPL3SMP_E')

        with self.subTest('Matching coordinate overrides in the configuration file'):
            self.assertEqual(
                cf_config.get_coordinate_overrides(
                    '/Soil_Moisture_Retrieval_Data_Polar_PM/surface_flag_pm'
                ),
                [
                    '/Soil_Moisture_Retrieval_Data_Polar_PM/latitude_pm',
                    '/Soil_Moisture_Retrieval_Data_Polar_PM/longitude_pm',
                ],
            )

        with self.subTest('No matching coordinate overrides in the configuration file'):
            self.assertEqual(
                cf_config.get_coordinate_overrides('/other_variable'), None
            )

    def test_get_dataset_grid_mapping_attributes(self):
        """Ensure that the grid mapping attributes are returned if a dataset
        name is specified that matches one of the keys in the
        `grid_mapping_supplements` section of the MaskFill configuration
        file. If there is no match, then `None` should be returned.

        """
        real_variable = '/Freeze_Thaw_Retrieval_Data_Global/latitude'
        fake_variable = '/group/variable'

        with self.subTest('Dataset name matches a key'):
            self.assertEqual(
                self.cf_config.get_dataset_grid_mapping_attributes(real_variable),
                self.cf_config.full_config['grid_mapping_definitions']['EASE2_global'],
            )

        with self.subTest('There is no matching key'):
            self.assertEqual(
                self.cf_config.get_dataset_grid_mapping_attributes(fake_variable), None
            )
