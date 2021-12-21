from unittest import TestCase

from GeotiffMaskFill import convert_variable_path, variable_should_be_masked


class TestGeotiffMaskfill(TestCase):
    """ Unit tests for GeotiffMaskfill branch. These are not end-to-end. """

    def test_convert_variable_path(self):
        """ Ensure that the full variable path regular expression is correctly
            converted to remove slashes and full stops.

        """
        self.assertEqual(convert_variable_path('/group/variable.band'),
                         '_group_variable_band')

    def test_variable_should_be_masked(self):
        """ Ensure the correct determination is made as to whether a variable
            should be masked or not, as indicated by its GeoTIFF file path and
            the available list of exclusions.

        """
        exclusions = ['latitude', 'longitude']
        test_args = [['No matches', 'SMAP_L3_variable.tif', exclusions, True],
                     ['A match', 'SMAP_L3_latitude.tif', exclusions, False],
                     ['No exclusions', 'SMAP_L3_latitude.tif', [], True]]

        for description, geotiff_path, excluding, should_exclude in test_args:
            with self.subTest(description):
                self.assertEqual(
                    variable_should_be_masked(geotiff_path, excluding),
                    should_exclude
                )
