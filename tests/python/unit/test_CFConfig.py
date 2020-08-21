from unittest import TestCase
import json

from pymods.CFConfig import (get_dataset_config_fill_value,
                             get_grid_mapping_data, readConfigFile)


class TestCFConfig(TestCase):

    def setUp(self):
        with open('data/MaskFillConfig.json') as file_handler:
            config_data = json.load(file_handler)

        self.config_data = config_data
        self.short_name = 'SPL3FTP'

        self.dataset, self.fill_value = (
            list(config_data['Corrected_Fill_Value']['SPL3FT(A|P|P_E)'].items())[0]
        )

        readConfigFile()

    def test_get_dataset_config_fill_value(self):
        """Ensure that a dataset that fails to meet the required criteria is
        not processed in any way. Instead, the function should return prior to
        that point.

        """
        test_args = [['Dataset and FillValue present', self.short_name,
                      self.dataset, self.fill_value],
                     ['Dataset present, no FillValue', self.short_name,
                      'missing_ds', None],
                     ['Dataset and FillValue absent', 'missing_sn',
                      'missing_ds', None]]

        for description, short_name, dataset_name, expected_fill_value in test_args:
            with self.subTest(description):
                fill_value = get_dataset_config_fill_value(short_name, dataset_name)
                self.assertEqual(fill_value, expected_fill_value)

    def test_get_grid_mapping_data(self):
        """Ensure that datasets contained within the global configuration
        Grid_Mapping_Group and Grid_Mapping_Data return the projection
        information. Datasets that are not in the configuration should return
        None.

        """
        dataset = '/Freeze_Thaw_Retrieval_Global_Data/altitude.Bands_01'
        collection = self.short_name
        grid_mapping = self.config_data['Grid_Mapping_Data']['EASE2_Global']

        test_args = [['Valid collection and dataset', collection, dataset, grid_mapping],
                     ['Missing collection', 'Invalid', 'science', None],
                     ['Missing dataset', 'SPL3FTP', 'Invalid', None]]

        for description, short_name, dataset, result in test_args:
            with self.subTest(description):
                self.assertEqual(get_grid_mapping_data(short_name, dataset),
                                 result)
