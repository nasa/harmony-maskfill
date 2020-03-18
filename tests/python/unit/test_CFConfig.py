from unittest import TestCase
import json

from pymods.CFConfig import get_dataset_config_fill_value, readConfigFile


class TestH5MaskFill(TestCase):

    def setUp(self):
        with open('data/MaskFillConfig.json') as file_handler:
            config_data = json.load(file_handler)

        self.short_name = 'SPL3FTP'

        self.dataset, self.fill_value = (
            list(config_data['Corrected_Fill_Value']['SPL3FT(A|P|_E)'].items())[0]
        )

        readConfigFile('data/MaskFillConfig.json')

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
