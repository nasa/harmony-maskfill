## Overview:

The `MaskFill` utility works with gridded data, applying a fill value in all pixels
outside of a provided shape.

The utility accepts HDF-5 files which follow CF conventions and GeoTIFFs.

## Installation:

MaskFill was developed using the Anaconda distribution of Python (https://www.anaconda.com/download) and conda virutal environment.
This simplifies dependency management. Run these commands to create a mask fill conda virtual environment and install all the needed packages:

```bash
conda create --name maskfill --file pymods/mask_fill_conda_requirements.txt
source activate maskfill
pip install -r pymods/mask_fill_pip_requirements.txt
```

## Running locally:

Within the `maskfill` Conda environment, the main `MaskFill` utility can be run via:

```bash
python MaskFill.py --FILE_URLS [data_file_path] --BOUNDINGSHAPE [shape_file_path] --OUTPUT_DIR [output_directory_path] --IDENTIFIER [output_subdirectory]
```

There are other parameters that can be supplied to the script, but these are optional:

```
--DEBUG True
--DEFAULT_FILL [value]
--MASK_GRID_CACHE [ignore_and_delete|ignore_and_save|use_cache|use_cache_delete|MaskGrid_Only]
```

## Testing:

### Unit tests:

This project has unit tests that utilize the standard `unittest` Python package. These
can be run from the root directory of this repository using the following command:

```bash
python -m unittest discover tests/python
```

The unit tests also contain basic tests for code style, ensuring that all Python
files conform to [PEP8](https://www.python.org/dev/peps/pep-0008/), excluding
checks on line-length.

Tests within `tests/python/test_MaskFill.py` are designed to test the full use
of the functionality, taking an input file, creating an output file and comparing
that output file to a template. Those within `tests/python/unit` are designed
as more granular unit tests of the logic and behaviour of individual functions.
