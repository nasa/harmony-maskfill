from os import remove
from unittest import TestCase
import json

from h5py import File

from pymods.cf_config import CFConfigGeotiff, CFConfigH5


class TestCFConfig(TestCase):
    """ Test the common functionality from the abstract base class. Use an
        instance of CFConfigH5 to do so, but the choice of child class is
        arbitrary.

    """
    @classmethod
    def setUpClass(cls):
        cls.cf_config = CFConfigH5('tests/data/SMAP_L3_FT_P_corners_input.h5')

    def test_get_shorname_from_config(self):
        """ Given a file path, match it to the config entries. """
        with self.subTest('Shortname is present in configuration file'):
            self.assertEqual(
                self.cf_config._get_shortname_from_config('ATL16_file.h5'),
                'ATL16'
            )

        with self.subTest('Shortname does not match shorter matching entry'):
            self.assertEqual(
                self.cf_config._get_shortname_from_config('SMAP_L3_FT_P_E_file.h5'),
                'SPL3FTP_E'
            )

        with self.subTest('Unknown shortname returns None'):
            self.assertEqual(
                self.cf_config._get_shortname_from_config('RANDOM.h5'),
                None
            )

    def test_get_dataset_fill_value(self):
        """ Ensure a fill value stored in the configuration file is returned
            for the requested dataset. If there is not match, `None` should be
            returned.

        """
        with self.subTest('Matching fill value in the configuration file'):
            self.assertEqual(
                self.cf_config.get_dataset_fill_value('/Freeze_Thaw_Retrieval_Data_Polar/latitude'),
                -9999.0
            )

        with self.subTest('No fill value in the configuration file'):
            self.assertEqual(
                self.cf_config.get_dataset_fill_value('/other_variable'), None
            )

    def test_get_dataset_grid_mapping_attributes(self):
        """ Ensure that the grid mapping attributes are returned if a dataset
            name is specified that matches one of the keys in the
            `grid_mapping_supplements` section of the MaskFill configuration
            file. If there is no match, then `None` should be returned.

        """
        real_variable = '/Freeze_Thaw_Retrieval_Data_Global/latitude'
        fake_variable = '/group/variable'

        with self.subTest('Dataset name matches a key'):
            self.assertEqual(
                self.cf_config.get_dataset_grid_mapping_attributes(real_variable),
                self.cf_config.full_config['grid_mapping_definitions']['EASE2_global']
            )

        with self.subTest('There is no matching key'):
            self.assertEqual(
                self.cf_config.get_dataset_grid_mapping_attributes(fake_variable),
                None
            )


class TestCFConfigH5(TestCase):
    """ Test that the CFConfigH5 child class successfully instantiates, and
        that class dependent methods return the expected values.

    """
    @classmethod
    def setUpClass(cls):
        with open('data/maskfill_config.json', 'r') as file_handler:
            cls.raw_config = json.load(file_handler)

    def test_instantiation(self):
        """ Ensure that a CFConfigH5 object can be created, using the shortname
            either from granule metadata, or the listed mapping of file name
            prefixes to shortnames. If the granule does not allow MaskFill to
            identify a collection, then ensure no collection specific metadata
            augmentations are retrieved from the configuration file.

        """
        with self.subTest('Shortname in metadata'):
            cf_config = CFConfigH5('tests/data/SMAP_L3_FT_P_corners_input.h5')
            self.assertEqual(cf_config.shortname, 'SPL3FTP')
            self.assertListEqual(
                cf_config.coordinate_variables,
                self.raw_config['collection_coordinate_variables']['SPL3FTP']
            )
            self.assertDictEqual(
                cf_config.fill_values,
                self.raw_config['corrected_fill_values']['SPL3FT(A|P|P_E)']
            )
            self.assertDictEqual(
                cf_config.grid_mapping_groups,
                self.raw_config['grid_mapping_supplements']['SPL3FT(P|P_E)']
            )

        with self.subTest('Shortname from the configuration file'):
            test_file_name = 'tests/data/SMAP_L3_SM_P_temp.h5'
            temp_file = File(test_file_name, 'w')
            temp_file.close()

            cf_config = CFConfigH5(test_file_name)
            self.assertEqual(cf_config.shortname, 'SPL3SMP')
            self.assertListEqual(
                cf_config.coordinate_variables,
                self.raw_config['collection_coordinate_variables']['SPL3SMP']
            )
            self.assertDictEqual(
                cf_config.fill_values,
                self.raw_config['corrected_fill_values']['SPL3SM(P|P_E)']
            )
            self.assertDictEqual(
                cf_config.grid_mapping_groups,
                self.raw_config['grid_mapping_supplements']['SPL3SM(P|P_E|A|AP)']
            )

            remove(test_file_name)

        with self.subTest('Unknown shortname'):
            test_file_name = 'tests/data/RANDOM_granule.h5'
            temp_file = File(test_file_name, 'w')
            temp_file.close()

            cf_config = CFConfigH5(test_file_name)
            self.assertEqual(cf_config.shortname, None)
            self.assertListEqual(cf_config.coordinate_variables, [])
            self.assertDictEqual(cf_config.fill_values, {})
            self.assertDictEqual(cf_config.grid_mapping_groups, {})

        remove('tests/data/RANDOM_granule.h5')

    def test_get_file_exclusions(self):
        """ Check the expected list of file exclusions are returned. For an
            HDF-5 file, these should be regular expressions for variable full
            paths.

        """
        cf_config = CFConfigH5('tests/data/SMAP_L3_FT_P_corners_input.h5')
        self.assertListEqual(
            cf_config.get_file_exclusions(),
            self.raw_config['collection_coordinate_variables']['SPL3FTP']
        )

    def test_shortname_attribute_present(self):
        """ Ensure an attribute is correctly identified as being present:

        * Attribute present
        * Attribute present in root group
        * Group present, but attribute absent
        * Attribute present in different group

        """
        test_args = [
            ['Attribute present', '/Metadata', 'iso_19139_series_xml', True],
            ['Attribute in root', '/', 'Conventions', True],
            ['Group missing attribute', '/Metadata', 'not_present', False],
            ['Group absent', '/random_group', 'shortName', False]
        ]

        with File('tests/data/SMAP_L4_SM_aup_input.h5', 'r') as h5_file:
            for description, group, attribute, result in test_args:
                with self.subTest(description):
                    present = CFConfigH5._shortname_attribute_present(
                        h5_file, {'attribute': attribute, 'group': group}
                    )
                    self.assertEqual(present, result)


class TestCFConfigGeotiff(TestCase):
    """ Test that the CFConfigGeotiff child class successfully instantiates,
        and that class dependent methods return the expected values.

    """

    @classmethod
    def setUpClass(cls):
        with open('data/maskfill_config.json', 'r') as file_handler:
            cls.raw_config = json.load(file_handler)

    def test_instantiation(self):
        """ Ensure that the CFConfigGeotiff class can successfully be
            instantiated. Or, if an unrecognised collection shortname is given,
            the class is instantiated, but the instance does not include any
            collection specific metadata augmentation.

        """
        with self.subTest('Valid collection'):
            cf_config = CFConfigGeotiff('tests/data/SMAP_L4_SM_aup_input.tif')
            self.assertEqual(cf_config.shortname, 'SPL4SMAU')
            self.assertListEqual(
                cf_config.coordinate_variables,
                self.raw_config['collection_coordinate_variables']['SPL4SMAU']
            )
            self.assertDictEqual(cf_config.fill_values, {})
            self.assertDictEqual(
                cf_config.grid_mapping_groups,
                self.raw_config['grid_mapping_supplements']['SPL4.*']
            )

        with self.subTest('Unknown shortname'):
            cf_config = CFConfigGeotiff('RANDOM_variable.tif')
            self.assertEqual(cf_config.shortname, None)
            self.assertListEqual(cf_config.coordinate_variables, [])
            self.assertDictEqual(cf_config.fill_values, {})
            self.assertDictEqual(cf_config.grid_mapping_groups, {})

    def test_get_file_exclusions(self):
        """ Check the expected list of file exclusions are returned. For a
            GeoTIFF file, these should be the variables that the configuration
            file lists for that collection under the
            `collection_coordinate_variables` key.

        """
        with self.subTest('SLP4SMAU'):
            cf_config = CFConfigGeotiff('tests/data/SMAP_L4_SM_aup_data.tif')
            self.assertListEqual(
                cf_config.get_file_exclusions(),
                self.raw_config['collection_coordinate_variables']['SPL4SMAU']
            )

        with self.subTest('ATL17'):
            cf_config = CFConfigGeotiff('tests/data/ATL17_variable.tif')
            self.assertListEqual(
                cf_config.get_file_exclusions(),
                self.raw_config['collection_coordinate_variables']['ATL17']
            )
