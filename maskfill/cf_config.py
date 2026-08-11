"""Utility class for reading and interpreting the CF Convention configuration file.

Allows processing of HDF-5 files that do not fully follow the CF
conventions. The configuration file provides the missing information.

"""
from typing import Any
import json
import os
import re


class CFConfig:
    """A class defining configuration information specific to a given collection.

    For both the HDF-5 and GeoTIFF branch of MaskFill, a configuration is
    instantiated by first reading in the JSON configuration file. With that in
    place, and collection short name supplied by the Harmony request, the
    relevant parts of the configuration file are extracted, allowing for
    simpler retrieval functions that are quicker to run once per associated
    variable.

    A data file often needs configuration information to fill in missing CF
    Convention attributes or sometimes even override values if they are
    incorrect, or not useful as stated in the file. This is especially true for
    GeoTIFF files which do not really have CF Convention attributes.

    """
    def __init__(self, collection_shortname: str):
        """Extract collection-specific configuration from supplied JSON file."""
        self.full_config = self._read_configuration_file()
        self.shortname = collection_shortname

        self.coordinate_variables = (
            self.full_config['collection_coordinate_variables'].get(self.shortname, [])
        )
        self.coordinate_overrides = (
            self.full_config['collection_coordinate_overrides'].get(self.shortname, {})
        )
        self.fill_values = self._get_configuration_item_by_shortname(
            'corrected_fill_values', {}
        )
        self.grid_mapping_groups = self._get_configuration_item_by_shortname(
            'grid_mapping_supplements', {}
        )

    def _read_configuration_file(self):
        """Locate and parse the MaskFill JSON configuration file.

        This method assumes the configuration file location relative to this
        module is consistent.

        """
        maskfill_directory = os.path.abspath(os.sep.join([
            os.path.dirname(os.path.abspath(__file__)),
            os.pardir
        ]))

        config_file_path = os.sep.join([maskfill_directory, 'maskfill',
                                        'maskfill_config.json'])

        with open(config_file_path, encoding='utf-8') as file_handler:
            config = json.load(file_handler)

        return config

    def _get_configuration_item_by_shortname(
        self, config_group: str,
        default_value: Any,
    ) -> Any:
        """Extract relevant configuration entries based on collection short name.

        Iterates through a dictionary in the MaskFill configuration file and
        attempts to match the key to the supplied collection shortname. The key
        will be a regular expression pattern.

        The value corresponding to the matching key will be returned. If there
        are no matching keys, a specified default value will be returned instead.

        """
        if self.shortname is not None:
            item = next((configuration_item
                         for shortname_pattern, configuration_item
                         in self.full_config[config_group].items()
                         if re.match(shortname_pattern, self.shortname)),
                        default_value)
        else:
            item = default_value

        return item

    def get_file_exclusions(self) -> list[str]:
        """Return regular expressions matching all variables that should not be masked.

        These will largely be coordinates or grid related.

        """
        return self.coordinate_variables

    def get_dataset_fill_value(self, dataset_name: str) -> Any:
        """Retrieve relevant fill value override for a variable.

        Search the collection specific dictionary containing corrected
        FillValue data. These are known data issues, where the FillValue
        attribute in a dataset either is missing or does not correspond to
        the used value. If the dataset name is stored as a key in
        this dictionary, the associated value is returned. If there are no
        matches, then a `None` value is returned.

        Note, in this method "dataset" refers to a variable in an HDF-5 or
        netCDF4 file.

        """
        return self.fill_values.get(dataset_name, None)

    def get_dataset_grid_mapping_attributes(
        self,
        dataset_name: str,
    ) -> dict | None:
        """Return relevant grid mapping supplements from the configuration file.

        Search the collection specific dictionary containing grid mapping
        supplements, trying to match the dataset name to the regular
        expression keys denoting the datasets to apply the grid mapping to.
        If a match is found, retrieve the definition for that grid mapping
        from the `grid_mapping_definitions` part of the configuration file.
        If no matches are found, return `None`.

        This method assumes that a dataset name can only match a single
        regular expression pattern in a `grid_mapping_supplements` item of
        the configuration file.

        Note, in this method "dataset" refers to a variable in an HDF-5 or
        netCDF4 file.

        """
        grid_mapping_name = next((config_grid_mapping_name
                                  for dataset_pattern, config_grid_mapping_name
                                  in self.grid_mapping_groups.items()
                                  if re.match(dataset_pattern, dataset_name)),
                                 None)

        if grid_mapping_name is not None:
            grid_mapping_attributes = (
                self.full_config['grid_mapping_definitions'].get(grid_mapping_name,
                                                                 None)
            )
        else:
            grid_mapping_attributes = None

        return grid_mapping_attributes

    def get_coordinate_overrides(self, dataset_name: str) -> list | None:
        """Retrieve coordinate overrides for a given dataset name.

        This method searches the `coordinate_overrides` dictionary for a
        pattern that matches the provided dataset name. If a match is found,
        it returns the associated list of coordinate names. Otherwise, it
        returns None.

        Note, in this method "dataset" refers to a variable in an HDF-5 or
        netCDF4 file.

        """
        for variable_pattern, coordinates in self.coordinate_overrides.items():
            if re.search(variable_pattern, dataset_name):
                return coordinates

        return None
