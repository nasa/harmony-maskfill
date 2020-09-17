## Overview:

The `MaskFill` utility works with gridded data, applying a fill value in all pixels
outside of a provided shape.

The utility accepts HDF-5 files which follow CF conventions and GeoTIFFs.

## Installation:

MaskFill was developed using the Anaconda distribution of Python (https://www.anaconda.com/download) and conda virutal environment.
This simplifies dependency management. Run these commands to create a mask fill conda virtual environment and install all the needed packages:

```bash
conda create --name maskfill --file ./data/mask_fill_conda_requirements.txt
source activate maskfill
pip install -r ./data/mask_fill_pip_requirements.txt
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

### Test coverage report:

To see how much of the code is covered by the unit and end-to-end tests, run
the following two commands.

```
coverage run -m unittest discover tests/python
coverage report --omit=tests/*
```

### Unit tests in Docker:

The unit tests can also be run within a Docker container:

```bash
mkdir test-reports
docker build -f tests/Dockerfile -t maskfill
docker run -v /full/path/to/test-reports:/home/tests/reports -v /full/path/to/maskfill-coverage:/home/tests/coverage maskfill:latest
```

The terminal should display output from the test results, with the failures
from `unittest`. Additionally, the XML test reports should be saved to the new
`test-reports` directory. Test coverage report should also be displayed in the 
terminal, and will also be saved to the 'coverage' directory in HTML format.
Coverage reports are being generate for each build in Bamboo, and saved as artifacts.
Following URL is an example coverage report in Bamboo:

https://ci.earthdata.nasa.gov/artifact/HITC-MAS18/JOB1/build-16/Coverage-Report/maskfill/test-coverage/index.html
