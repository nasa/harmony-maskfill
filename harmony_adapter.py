""" Data Services MaskFill service for Harmony. """
from argparse import ArgumentParser
from mimetypes import guess_type as guess_mimetype
from shutil import move as move_file, rmtree
from tempfile import mkdtemp
import os

from pystac import Asset, Item
from harmony import BaseHarmonyAdapter, run_cli, setup_cli
from harmony.message import Source as MessageSource
from harmony.util import (download, generate_output_filename, HarmonyException,
                          stage)

from MaskFill import DEFAULT_FILL_VALUE, DEFAULT_MASK_GRID_CACHE, mask_fill
from pymods.MaskFillUtil import create_bounding_box_shape_file


EXTENSION_MIMETYPES = {'.h5': 'application/x-hdf5',
                       '.hdf5': 'application/x-hdf5',
                       '.nc4': 'application/x-netcdf4',
                       '.tif': 'image/tiff',
                       '.tiff': 'image/tiff'}
VALID_MIMETYPES = {'application/x-hdf5', 'application/x-netcdf4', 'image/tiff'}


class HarmonyAdapter(BaseHarmonyAdapter):
    """ Data Services MaskFill service for Harmony

        This class uses the Harmony utility library for processing the
        service input options. First the `invoke` method is called, which adds
        validation to the raw Harmony message. The `BaseHarmonyAdapter.invoke`
        method is then used to call the `process_item` method of this class.

    """
    def invoke(self):
        """ Adds validation to default process_item-based invocation

            Returns
            -------
            pystac.Catalog
                the output catalog

        """
        self.logger.info('Starting Data Services MaskFill Service')
        os.environ['HDF5_DISABLE_VERSION_CHECK'] = '1'
        self.validate_message()
        return super().invoke()

    def process_item(self, item: Item, source: MessageSource):
        """ Processes a single input item. Services that are not aggregating
            multiple input files should prefer to implement this method rather
            than #invoke

            This example copies its input to the output, marking "dpi" and
            "variables" message attributes as having been processed

            Parameters
            ----------
            item : pystac.Item
                the item that should be processed
            source : harmony.message.Source
                the input source defining the variables, if any, to subset
                from the item

            Returns
            -------
            pystac.Item
                a STAC catalog whose metadata and assets describe the service
                output

        """
        result = item.clone()
        result.assets = {}

        # Create a temporary dir for processing we may do
        working_dir = mkdtemp()

        try:
            asset = next(item_asset for item_asset in item.assets.values()
                         if 'data' in (item_asset.roles or []))

            input_filename = self.download_from_remote(
                asset.href, working_dir, os.path.basename(asset.href)
            )
            self.logger.info('Granule data copied')
            self.validate_input_granule(input_filename)

            # Get the shape file data
            if self.message_has_valid_shape_file():
                shape_filename = self.download_from_remote(
                    self.message.subset.shape.process('href'), working_dir
                )
                self.logger.info('Shape file downloaded')
            else:
                shape_filename = create_bounding_box_shape_file(
                    self.message.subset.process('bbox'), working_dir
                )
                self.logger.info('Shape file constructed from bounding box.')

            # Call MaskFill utility
            working_filename = mask_fill(input_filename, shape_filename,
                                         working_dir, DEFAULT_MASK_GRID_CACHE,
                                         DEFAULT_FILL_VALUE, self.logger)

            # Stage the output file with a conventional filename
            output_filename = generate_output_filename(asset.href,
                                                       is_subsetted=True)

            output_mimetype = self.get_file_mimetype(output_filename)

            output_url = stage(working_filename, output_filename,
                               output_mimetype,
                               location=self.message.stagingLocation,
                               logger=self.logger)

            # Update the STAC record
            asset = Asset(output_url, title=output_filename,
                          media_type=output_mimetype, roles=['data'])

            result.assets['data'] = asset

            # Return the output file back to Harmony
            self.logger.info('MaskFill complete')

            return result

        except Exception as err:
            self.logger.error('MaskFill failed: ' + str(err), exc_info=1)
            raise HarmonyException('MaskFill failed with error: ' + str(err)) from err
        finally:
            # Clean up any intermediate resources
            rmtree(working_dir, ignore_errors=True)

    def download_from_remote(self, remote_resource_url: str,
                             output_directory: str,
                             local_basename: str = None) -> str:
        """ A class method to wrap the Harmony utility function to download a
            file from a remote source. This method automatically uses the
            Logger, access token and configuration object from the instance of
            the HarmonyAdapter class.

            If a file name is specified, then the downloaded file will be
            renamed. Otherwise, Harmony will use a UUID as the basename for
            any downloaded resource.

        """
        self.logger.info(f'Retrieving: {remote_resource_url}')

        local_file = download(remote_resource_url, output_directory,
                              logger=self.logger,
                              access_token=self.message.accessToken,
                              cfg=self.config)

        if local_basename is not None:
            full_local_name = os.path.join(output_directory, local_basename)
            move_file(local_file, full_local_name)
            local_file = full_local_name

        return local_file

    def validate_message(self):
        """ Check the service was triggered by a valid message containing
            STAC item assets and a valid shape file: e.g. GeoJSON format.

            This validation does not check the input file, as that has to be
            done per-item.

        """
        if not hasattr(self, 'message') or self.message is None:
            raise HarmonyException('No message request')

        has_granules = (hasattr(self.message, 'granules')
                        and self.message.granules)

        try:
            has_items = bool(self.catalog and next(self.catalog.get_all_items()))
        except StopIteration:
            has_items = False

        if not has_granules and not has_items:
            raise HarmonyException('No granules specified for reprojection')

        if not isinstance(self.message.granules, list):
            raise HarmonyException('Invalid granule list')

        # Ensure that either a GeoJSON shape file or a bounding box is
        # specified in the Harmony message.
        if (
            not self.message_has_valid_shape_file()
            and not self.message_has_valid_bounding_box()
        ):
            raise HarmonyException('MaskFill requires a shape file or bounding'
                                   ' box that describes a mask.')

    def message_has_valid_shape_file(self):
        """ A method that confirms if the Harmony message specifies a GeoJSON
            shape file. If either the URL is omitted or the MIME type of the
            shape file is not GeoJSON an exception will be raised.

        """
        if getattr(self.message.subset, 'shape', None) is not None:
            if self.message.subset.shape.href is None:
                raise HarmonyException('Shape file must specify resource URL.')
            elif self.message.subset.shape.type != 'application/geo+json':
                raise HarmonyException('Shape file must be GeoJSON format.')
            else:
                has_valid_shape = True
        else:
            has_valid_shape = False

        return has_valid_shape

    def message_has_valid_bounding_box(self):
        """ A method that confirms a message defines a bounding box for spatial
            subsetting. This will be used if a shape file is not defined in the
            input Harmony message. If a bounding box is defined with incorrect
            input (e.g., the wrong number of elements) then an exception will
            be raised.

        """
        if getattr(self.message.subset, 'bbox', None) is not None:
            if (
                isinstance(self.message.subset.bbox, list)
                and len(self.message.subset.bbox) == 4
            ):
                has_valid_bbox = True
            else:
                raise HarmonyException('Bounding box must be 4-element list.')
        else:
            has_valid_bbox = False

        return has_valid_bbox

    def validate_input_granule(self, input_filename):
        """ Check that the MIME type of the given file name is one of the
            expected types.

        """
        input_mimetype = self.get_file_mimetype(input_filename)

        if input_mimetype not in VALID_MIMETYPES:
            if input_mimetype is None:
                # Ensure message in the exception doesn't just contain "None"
                input_format = os.path.splitext(input_filename)[1]
            else:
                input_format = input_mimetype

            raise HarmonyException(f'Invalid granule format: {input_format}')

    @staticmethod
    def get_file_mimetype(file_name):
        """ Check the MIME type of the given file name, by checking the file
            extension against a selection of known options.

        """
        mimetype, _ = guess_mimetype(file_name, False)

        if mimetype is None:
            # Assumption: only one extension at the end of the file path
            file_extension = os.path.splitext(file_name)[1]
            mimetype = EXTENSION_MIMETYPES.get(file_extension, None)

        return mimetype


if __name__ == '__main__':
    PARSER = ArgumentParser(prog='MaskFill',
                            description=('Extract a polygon spatial subset of '
                                         'from HDF-5 or GeoTIFF file'))
    setup_cli(PARSER)
    ARGS, _ = PARSER.parse_known_args()
    run_cli(PARSER, ARGS, HarmonyAdapter)
