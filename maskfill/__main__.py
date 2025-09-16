"""Run the Harmony MaskFill services via the Harmony CLI/"""
from argparse import ArgumentParser
from sys import argv

from harmony_service_lib import is_harmony_cli, run_cli, setup_cli

from maskfill.adapter import MaskFillAdapter


def main(arguments: list[str]):
    """Parse command line arguments and invoke the appropriate method."""
    parser = ArgumentParser(
        prog='MaskFill',
        description=(
            'Extract a polygon spatial subset from an HDF-5 or GeoTIFF file'
        ),
    )

    setup_cli(parser)
    harmony_arguments, _ = parser.parse_known_args(arguments[1:])

    if is_harmony_cli(harmony_arguments):
        run_cli(parser, harmony_arguments, MaskFillAdapter)
    else:
        parser.error('Only --harmony CLIs are supported')


if __name__ == '__main__':
    main(argv)
