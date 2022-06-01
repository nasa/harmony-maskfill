""" Utilities for reading and interpreting the CF configuration file
    Allows processing of HDF-5 files that do not fully follow the CF
    conventions. The configuration file provides the missing information.

"""
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Set
import json
import os
import re

from h5py import File
from numpy import bytes_


class CFConfig(ABC):
    """ A base class defining functionality common to both the HDF-5 and
        GeoTIFF branch of MaskFill. In both instances, a configuration is
        instantiated by first reading in the JSON configuration file, then
        trying to identify the granule's collection shortname. With that in
        place, the relevant parts of the configuration file are extracted,
        allowing for simpler retrieval functions that are quicker to run once
        per associated variable.

        A data file often needs CF configuration backing support to fill in the
        gaps of CF convention attributes and sometimes even override values if
        they are incorrect, or not useful as stated in the file. This is
        especially true for GeoTIFF files which do not really have CF
        attributes.

        The retrieval of the collection shortname. and the identification of
        variables to be excluded from being processed by MaskFill, are both
        specific to each file type, and are defined in the two child classes:
        `CFConfigH5` and `CFConfigGeotiff`.

    """
    def __init__(self, file_path: str):
        """ Read in the configuration file. Use the path of the granule being
            processed to determine the shortname of the collection. Finally,
            extract the collection specific portions of the configuration file
            to make the retrieval of dataset-specific information simpler.

        """
        self.full_config = self._read_configuration_file()
        self.shortname = self._get_shortname(file_path)

        self.coordinate_variables = (
            self.full_config['collection_coordinate_variables'].get(self.shortname, [])
        )
        self.fill_values = self._get_configuration_item_by_shortname(
            'corrected_fill_values', {}
        )
        self.grid_mapping_groups = self._get_configuration_item_by_shortname(
            'grid_mapping_supplements', {}
        )

    def _read_configuration_file(self):
        """ Identify the location of the MaskFill configuration file by
            assuming it's location relative to this module is consistent. Then
            read the file using the standard `json` package.

        """
        maskfill_directory = os.path.abspath(os.sep.join([
            os.path.dirname(os.path.abspath(__file__)),
            os.pardir
        ]))

        config_file_path = os.sep.join([maskfill_directory, 'data',
                                        'maskfill_config.json'])

        with open(config_file_path) as file_handler:
            config = json.load(file_handler)

        return config

    @abstractmethod
    def _get_shortname(self, file_path: str):
        """ This method will be specific to whether the input file is HDF-5 or
            GeoTIFF, and will be defined in each of the child classes that
            inherit from CFConfig.

        """
        pass

    def _get_shortname_from_config(self, file_path: str) -> Optional[str]:
        """ Compare the start of the granule basename to a list of file
            prefixes in the configuration file. Return the shortname associated
            with the longest matching prefix, e.g. SMAP_L3_FT_P_E rather than
            SMAP_L3_FT_P. (If the collection is actually SPL3FTP, then
            "SMAP_L3_FT_P" will be the only match)

            If the file prefix is unknown, a value of `None` will be returned.

        """
        shortname = None
        file_basename = os.path.basename(file_path)
        prefix_mapping = self.full_config['collection_prefix_to_shortname_mapping']

        for file_prefix, config_shortname in prefix_mapping.items():
            if (
                    file_basename.startswith(file_prefix)
                    and (shortname is None or len(config_shortname) > len(shortname))
            ):
                shortname = config_shortname

        return shortname

    def _get_configuration_item_by_shortname(self, config_group: str,
                                             default_value: Any) -> Optional[Any]:
        """ A utility method to iterate through a dictionary in the MaskFill
            configuration file and attempt to match the key to the supplied
            collection shortname. The key will be a regular expression pattern.
            The value corresponding to the matching key will be returned. If
            there are no matching keys, a specified default value will be
            returned instead.

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

    def get_file_exclusions(self) -> List[str]:
        """ Return a list of regular expressions that will match to all
            variables within a collection that should not be masked. These will
            largely be coordinates or grid related.

        """
        return self.coordinate_variables

    def get_dataset_fill_value(self, dataset_name: str) -> Optional[Any]:
        """ Search the collection specific dictionary containing corrected
            FillValue data. These are known data issues, where the FillValue
            attribute in a dataset either is missing or does not correspond to
            the used value. If the dataset name is stored as a key in
            this dictionary, the associated value is returned. If there are no
            matches, then a `None` value is returned.

        """
        return self.fill_values.get(dataset_name, None)

    def get_dataset_grid_mapping_attributes(self,
                                            dataset_name: str) -> Optional[Dict]:
        """ Search the collection specific dictionary containing grid mapping
            supplements, trying to match the dataset name to the regular
            expression keys denoting the datasets to apply the grid mapping to.
            If a match is found, retrieve the definition for that grid mapping
            from the `grid_mapping_definitions` part of the configuration file.
            If no matches are found, return `None`.

            This method assumes that a dataset name can only match a single
            regular expression pattern in a `grid_mapping_supplements` item of
            the configuration file.

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


class CFConfigH5(CFConfig):

    def _get_shortname(self, file_path: str) -> str:
        """ Check the supplied granule in specified locations for metadata
            indication the granule collection. These locations are present in
            the MaskFill configuration file.

            If none of the expected attributes are present, potentially due to
            variable subsetting of the granule prior to the MaskFill step of
            processing, use a stored mapping between file name prefix and
            collection shortname.

            Finally, if there is no identified shortname, raise an exception.

        """
        with File(file_path, 'r') as h5_file:
            shortname = next(
                (h5_file[shortname_path['group']].attrs.get(shortname_path['attribute'])
                 for shortname_path
                 in self.full_config['collection_shortname_paths']
                 if self._shortname_attribute_present(h5_file, shortname_path)),
                None
            )

            if isinstance(shortname, (bytes, bytes_)):
                shortname = shortname.decode()

        if shortname is None:
            # Could not locate shortname in metadata, so try file name prefix:
            shortname = self._get_shortname_from_config(file_path)

        return shortname

    @staticmethod
    def _shortname_attribute_present(h5_file: File,
                                     collection_path: Dict[str, str]) -> bool:
        """ Check HDF-5 file for group that should have specified shortname
            attribute. If present, check whether that group has the expected
            attribute containing the collection shortname.

        """
        return (
            collection_path['group'] in h5_file
            and (collection_path['attribute']
                 in h5_file[collection_path['group']].attrs)
        )


class CFConfigGeotiff(CFConfig):

    def _get_shortname(self, file_path: str) -> str:
        """ Compare the base name of the GeoTIFF file to a list of prefixes
            in the MaskFill configuration file. Select the shortname associated
            with the longest matching prefix.

        """
        # If the shortname is not in the mapping, an exception will be raised
        return self._get_shortname_from_config(file_path)
