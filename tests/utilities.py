""" This module contains testing utilities to be used by multiple test
    classes.

"""
from datetime import datetime
from os.path import basename, isdir, join, splitext
from shutil import rmtree
from tempfile import mkdtemp
from unittest import TestCase

from harmony_service_lib.util import bbox_to_geometry
from numpy import array, array_equal, ndarray
from osgeo import gdal
from pystac import Asset as StacAsset, Catalog as StacCatalog, Item as StacItem
import h5py


def create_input_stac(granule_url: str, media_type: str) -> StacCatalog:
    """ A helper function to create a STAC catalog to be used as input when
        invoking the MaskFill HarmonyAdapter.

        The geometry and datetime are set to arbitrary values are these are
        not used in the tests.

    """
    catalog = StacCatalog(id='input catalog', description='test input')
    item = StacItem(
        id='input granule', bbox=[-180, -90, 180, 90],
        geometry=bbox_to_geometry([-180, -90, 180, 90]),
        datetime=datetime(2020, 1, 1), properties=None
    )
    item.add_asset(
        'input data',
        StacAsset(granule_url, media_type=media_type, roles=['data'])
    )
    catalog.add_item(item)
    return catalog


class MaskFillTestCase(TestCase):
    """ A base class for testing containing comparison methods for HDF-5 and
        GeoTIFF images.

    """
    @classmethod
    def setUpClass(cls):
        cls.identifier = 'test'

    def setUp(self):
        self.output_dir = mkdtemp()

    def tearDown(self):
        """Clean up test artifacts after each test."""
        if isdir(self.output_dir):
            rmtree(self.output_dir)

    def create_output_file_name(self, input_file_name, use_identifier=True):
        """ Determine the output name that MaskFill will give to an file,
            based on the target directory and input file name.

        """
        output_root, output_extension = splitext(basename(input_file_name))
        output_basename = f'{output_root}_mf{output_extension}'
        if use_identifier:
            file_name = join(self.output_dir, self.identifier, output_basename)
        else:
            file_name = join(self.output_dir, output_basename)

        return file_name

    def compare_geotiff_files(self, file_one_name, file_two_name):
        """Check both files have the same number of bands, and that the data
        within those bands match. Also, ensure any file-level metadata is
        identical.

        :type file_one_name: str
        :type file_two_name: str

        """

        dataset_one = gdal.Open(file_one_name)
        dataset_two = gdal.Open(file_two_name)
        band_one = array(dataset_one.ReadAsArray())
        band_two = array(dataset_two.ReadAsArray())
        self.assertEqual(band_one.shape, band_two.shape)
        self.assertTrue(array_equal(band_one, band_two))
        self.assertEqual(dataset_one.GetMetadata(), dataset_two.GetMetadata())

    def compare_h5_files(self, file_one_name, file_two_name):
        """Check all Attributes, Datasets and Groups within two HDF-5 files are
        equal.

        :type file_one_name: str
        :type file_two_name: str

        """
        file_one = h5py.File(file_one_name, 'r')
        file_two = h5py.File(file_two_name, 'r')

        self.compare_h5_file_datasets(file_one, file_two)
        self.compare_h5_file_attributes(file_one, file_two)

        file_one.close()
        file_two.close()

    def compare_h5_file_attributes(self, file_one, file_two):
        """For both files, extract dictionaries of attributes. Check both
        dictionaries contain the same keys (attribute names). Then compare the
        values of each attribute to ensure equality. Note, the attribute names
        are the full path from the root of the file, so also contain the
        hierarchy (groups) that the attributes belong to. This ensures the
        attributes location within the file is also being compared.

        :type file_one: h5py.File
        :type file_two: h5py.File

        """
        file_one_attributes = self.extract_all_h5_attributes(file_one, {})
        file_two_attributes = self.extract_all_h5_attributes(file_two, {})

        self.assertEqual(list(file_one_attributes.keys()),
                         list(file_two_attributes.keys()))

        for attribute_name, attribute_value in file_one_attributes.items():
            if isinstance(attribute_value, ndarray):
                if attribute_name.endswith('_LIST'):
                    # This catches 'DIMENSION_LIST' and 'REFERENCE_LIST' metadata
                    for ref_index, ref_one in enumerate(attribute_value):
                        dataset_ref_one = file_one[ref_one[0]]
                        ref_two = file_two_attributes[attribute_name][ref_index][0]
                        dataset_ref_two = file_two[ref_two]
                        self.assertEqual(dataset_ref_one.name,
                                         dataset_ref_two.name)
                else:
                    self.assertTrue(
                        array_equal(attribute_value,
                                    file_two_attributes[attribute_name]),
                        attribute_name
                    )

            else:
                self.assertEqual(attribute_value,
                                 file_two_attributes[attribute_name],
                                 attribute_name)

    def compare_h5_file_datasets(self, object_one, object_two):
        """For both files, traverse through all Groups and Datasets. Ensure
        that each Group or Dataset is present in both files, and each Group
        has the same child Groups and Datasets. For each Dataset compare the
        values between the two files.

        :type object_one: h5py.Dataset, h5py.File or h5py.Group
        :type object_two: h5py.Dataset, h5py.File or h5py.Group

        """

        for object_one_name, object_one_value in object_one.items():
            self.assertIn(object_one_name, list(object_two.keys()))
            object_two_value = object_two[object_one_name]

            if isinstance(object_one_value, h5py.Dataset):
                self.assertEqual(object_one_value.shape, object_two_value.shape)
                if isinstance(object_one_value[()], ndarray):
                    self.assertTrue(array_equal(object_one_value[()],
                                                object_two_value[()]))
                else:
                    self.assertEqual(object_one_value[()], object_two_value[()])

            else:
                self.assertEqual(list(object_one.keys()), list(object_two.keys()))
                self.compare_h5_file_datasets(object_one_value, object_two_value)

    def extract_all_h5_attributes(self, h5py_object, attribute_dictionary):
        """Starting at the root given of an HDF-5 file, recursively extract all
        attributes to a Python dictionary. The keys should be the full path of
        the attribute, for example: '/Metadata/Source/L1C_TB/version'

        :type h5py_object: h5py.File or h5py.Group
        :rtype: dict

        """
        if h5py_object.name.startswith('/'):
            key_prefix = h5py_object.name
        else:
            key_prefix = f'/{h5py_object.name}'

        for attr_key, attr_value in h5py_object.attrs.items():
            attribute_dictionary[f'{key_prefix}/{attr_key}'] = attr_value

        for iterable_object in h5py_object.values():
            if isinstance(iterable_object, h5py.Dataset):
                for attr_key, attr_value in iterable_object.attrs.items():
                    attribute_dictionary[f'{iterable_object.name}/{attr_key}'] = attr_value

            else:
                self.extract_all_h5_attributes(iterable_object, attribute_dictionary)

        return attribute_dictionary
