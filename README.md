## Overview:

The `MaskFill` utility works with gridded data, applying a fill value in all pixels
outside of a provided shape.

The utility accepts HDF-5 files which follow CF conventions and GeoTIFFs.

## Installation:

MaskFill was developed using the Anaconda distribution of Python
(https://www.anaconda.com/download) and conda virutal environment.
This simplifies dependency management. Run these commands to create a MaskFill
conda virtual environment and install all the needed packages:

```bash
conda create --name maskfill --file ./data/mask_fill_conda_requirements.txt
source activate maskfill
pip install -r ./data/mask_fill_pip_requirements.txt
```

To run the Harmony version of the service, replace the last command in the
snippet above with:

```bash
cd data
pip install -r mask_fill_harmony_pip_requirements.txt
```

Note, that installation most be done within the `data` directory so the
reference to `mask_fill_pip_requirements.txt` within the Harmony Pip
requirements file can be resolved.

## Development:

### General notes:

* Commit messages should use the ticket number as a prefix,
  e.g.: `DAS-123 - Awesome feature description.`
* Commit history should be squashed locally, to avoid minor commits (e.g.:
  `fix typo`, `update README`). This can be done via an interactive rebase,
  where `N` is the number of commits added during the feature development:
  ```
  git rebase -i HEAD~N
  ```

### Regular releases:

During regular development, developers should create feature branches from the
`dev` branch. When feature work is done, this branch should be merged back into
`dev` via a pull request (PR). When it is time for a full SDPS release, a
release branch will be created from the head of the `dev` branch. This release
branch should be named with the following format: `202_UPDATES`. When the
release branch is deemed ready, the branch should be merged into `master` and
back into `dev`. Merger into `master` will trigger the release of a new
`master` artefact in Maven.

### Hot shelves:

Hot shelves occur when bug fixes are required outside of the regular release
cycle. When one has been identified, and new hot shelf branch should be created
from the master branch. It should have a name with the format:

```
HOTSHELF-DAS-XYZ
```

The name of this branch will be used in naming the tarball saved in Maven, so
it is important to maintain a consistent naming convention for the hot shelf
branches.

When hot shelf work is complete, a pull request should be opened against the
master branch.

After the merge into the master branch, the developer who worked on the hot
shelf also needs to then merge the changes into the `dev` branch.

If there has been a lot of work since the last release, then this
step may be tricky. An additional branch to deal with merge conflicts may be
required.

## Running locally (SDPS method):

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

## Running locally (Harmony method):

You can also run the service within the `HarmonyAdapter`, starting within the root directory of the `maskfill` repository. 
When you do this you will need to set several environment variables, that Harmony expects. The first,
`ENV`, tells Harmony not to try and stage the results.

Additionally, if you want to inspect the output, you'll need to temporarily
comment out the `rmtree` call in the `finally` block of the
`HarmonyAdapter.process_item` method. It will probably be helpful to temporarily
log the `working_dir` in that method, to see the temporary directory produced,
which contains the output file.

```bash
export ENV=dev
export OAUTH_CLIENT_ID=''
export OAUTH_PASSWORD=''
export OAUTH_REDIRECT_URI=''
export OAUTH_UID=''
export STAGING_BUCKET=''
export STAGING_PATH=''
```

Then in a Python session:

```Python
from unittest.mock import patch
from harmony.message import Message
from harmony.util import config
from harmony_adapter import HarmonyAdapter
from tests.python.test_harmony_adapter import download_side_effect

message = Message({
    'accessToken': 'fake_token',
    'callback': 'https://www.example.com/callback',
    'stagingLocation': 's3://example-bucket/example-path',
    'sources': [{
        'granules': [{
            'bbox': [-180, -90, 180, 90],
            'temporal': {
                'start': '2020-01-01T00:00:00.000Z',
                'end': '2020-12-31T00:00:00.0000Z'
            },
            'url': 'tests/data/SMAP_L4_SM_aup_input.h5'
        }]
    }],
	'subset': {
        'shape': {
            'href': 'tests/data/USA.geo.json',
            'type': 'application/geo+json'
        }
    },
	'user': 'narmstrong'
})

with patch('harmony_adapter.download', side_effect=download_side_effect):
    maskfill_adapter = HarmonyAdapter(message, config=config(False))
    maskfill_adapter.invoke()
```

Note in the message above, the URL for a granule and shape file should be a
path to a local file.

## Testing:

### Unit tests:

This project has unit tests that utilize the standard `unittest` Python
package. These can be run from the root directory of this repository using the
following commands:

```bash
export ENV=test
python -m unittest discover tests/python
```

The environment variable `ENV` must be set to ensure that all unit tests that
invoke the `HarmonyAdapter` class do not try to stage their output files.

The unit tests also contain basic tests for code style, ensuring that all Python
files conform to [PEP8](https://www.python.org/dev/peps/pep-0008/), excluding
checks on line-length.

Tests within `tests/python/test_MaskFill.py` are designed to test the full use
of the functionality, taking an input file, creating an output file and comparing
that output file to a template. Those within `tests/python/unit` are designed
as more granular unit tests of the logic and behaviour of individual functions.

### Test coverage report:

To see how much of the code is covered by the unit and end-to-end tests, run
the following three commands.

```
export ENV=test
coverage run -m unittest discover tests/python
coverage report --omit=tests/*
```

A more detailed way to view the test coverage can be to run the coverage report
in HTML pages. This output will be automatically generated by the
`bin/run-test` script in the `maskfill/coverage` directory. Alternatively, one
can create a `coverage` directory and run the following commands:

```
export ENV=test
mkdir -p coverage
coverage run -m unittest discover tests/python
coverage html --omit=tests/* -d coverage
```

Then navigate in a web browser to:

```
file:///full/path/to/maskfill/coverage/index.html
```

This should display a page with a table of coverage percentages. Clicking on
each file should open a further page that renders the contents of the file,
indicating exactly the lines that have coverage, and those that don't.

### Unit tests in Docker:

The unit tests can also be run within a Docker container:

```bash
./bin/build-test
./bin/run-test
```

The terminal should display output from the test results, with the failures
from `unittest`. Additionally, the XML test reports should be saved to the new
`test-reports` directory. Test coverage report should also be displayed in the 
terminal, and will also be saved to the 'coverage' directory in HTML format.
Coverage reports are being generate for each build in Bamboo, and saved as artefacts.
Following URL is an example coverage report in Bamboo:

https://ci.earthdata.nasa.gov/artifact/HITC-MAS18/JOB1/build-16/Coverage-Report/maskfill/test-coverage/index.html

## Gotchas:

### New collection grid mappings:

MaskFill will try to determine the projection information for a variable by
using the following metadata (in the order specified):

* `DIMENSION_LIST` attribute. If present, and with units of 'degrees', the
  data are assumed to be geographic.
* `grid_mapping` attribute. If present, this will point to a `grid_mapping`
  variable in the granule. The metadata of that variable is used to define the
  projection of the variable being filled.
* Configuration file. If neither `DIMENSION_LIST` nor `grid_mapping` are
  included in the metadata attributes, the configuration file is checked for
  default values.
* If all of the above options do not return information from which a projection
  can be derived, MaskFill will raise an exception, and the service will fail.

When adding several SMAP collections, new entries were needed for the default
grid mapping when input data to MaskFill have not been reprojected. When adding
the MaskFill service to a new collection, care should be taken to ensure
whether the granule format can provide the necessary grid mapping information.
