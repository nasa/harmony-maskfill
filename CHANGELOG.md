# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [v1.3.5] - 2026-07-31

### Changed

- MaskFill computes the geographic bounds and resolution of projected grids from the grid perimeter and from streamed row blocks instead of computing latitude and longitude values for every grid cell.
- MaskFill mask fills two dimensional datasets without observed statistics attributes one chunk at a time instead of a full-array read and write, and no longer reads entire datasets into memory to obtain a shape or to check for fill values.

## [v1.3.4] - 2026-07-21

### Removed

- The unused SDPS command-line entry point and its helper functions have been removed from `maskfill/maskfill.py`.
- The SDPS-only unit tests in `tests/unit/test_maskfill.py`, which covered only the removed helper functions, have been removed.

### Changed

- The tests in `tests/test_maskfill.py` now call `mask_fill` directly rather than the removed `maskfill_sdps`.
- Bumped harmony-service-lib version to 3.0.0 in `pip_requirements.txt`.
- Bumped notebook version to 7.5.6 in `doc/requirements.txt`.

## [v1.3.3] - 2026-03-03

### Added

MaskFill raises a `MissingCoordinateDataset` when lat/lon datasets
cannot be found in `cf_config.coordinate_overrides` or the dataset's
'coordinates' attribute.

## [v1.3.2] - 2026-02-20

### Added

MaskFill raises a NoRetryException for errors that should not be retried.

## [v1.3.1] - 2026-02-06

### Changed

MaskFill handles small subsets where the spatial mask is a single row or column
without causing an exception.

## [v1.3.0] - 2026-01-29

### Added

MaskFill adds and updates history and history_json attributes to HDF5 and NetCDF4
 output file.

## [v1.2.1] - 2026-01-06

### Changed

This version of MaskFill removes changes python version from 3.12 to 3.13 and
updates the supporting packages.

## [v1.2.0] - 2025-11-06
This version of the Harmony MaskFill service now supports processing
 4D TEMPO O3PROF L3 datasets.

### Changed
- `get_dimension_datasets` method supports the identification of spatial
  dimensions within 4D datasets.
- `apply_2d_dataset_to_multidim` can now process both 3-dimensional
  and 4-dimensional data with less code.


## [v1.1.0] - 2025-11-03

## Changed

- Allow MaskFill to skip masking of valid datasets when the associated
  coordinate variables only contain fill values (this case previously
  threw an error).

## [v1.0.1] - 2025-10-14

### Removed

- bin/push-harmony-image. This script was used to push Harmony images to AWS
  ECR, which is no longer used. Images now live in ghcr.io.
- harmony_adapter.py. Previously this was left in the repository out of caution,
  but the entry point for the Harmony image has been moved to `maskfill/adapter.py`.
  This extra file is redundant and a potential cause for confusion.

## [v1.0.0] - 2025-10-14

This version of the Harmony MaskFill service contains all functionality
previously released internally to EOSDIS as `sds/maskfill:0.2.2`. Minor
reformatting of the repository structure has occurred to better comply with
recommended best practices for a Harmony backend service repository, but the
service itself is fundamentally unchanged.

For more information on internal releases prior to NASA open-source migration,
see legacy-CHANGELOG.md.

### Added

- LICENSE file as required by NASA Open Source Software guidelines.
- CODEOWNERS file to ensure default reviewers for pull requests.
- GitHub workflows for running tests and publishing Docker images to GHCR.

### Changed

- Dockerfiles and scripts in the `bin` directory have been updated to make use
  of new GHCR image names.
- The `pymods` directory has been renamed `maskfill`. All modules that can be
  non-disruptively moved or renamed have been placed in this directory and
  `snake_case` naming has been adopted for module names to adhere to standard
  Python best practices.
- The entry point for the Docker image has been moved to `maskfill.__main__.py`
  to conform with Harmony service repository best practices. `harmony_adapter.py`
  has been retained to enable continuous operations while Harmony CI/CD is
  updated to use the new entry point.

### Removed

- On-premises scripts and artefacts for the SDPS system have been removed from
  the repository.

[v1.3.5]: https://github.com/nasa/harmony-maskfill/releases/tag/1.3.5
[v1.3.4]: https://github.com/nasa/harmony-maskfill/releases/tag/1.3.4
[v1.3.3]: https://github.com/nasa/harmony-maskfill/releases/tag/1.3.3
[v1.3.2]: https://github.com/nasa/harmony-maskfill/releases/tag/1.3.2
[v1.3.1]: https://github.com/nasa/harmony-maskfill/releases/tag/1.3.1
[v1.3.0]: https://github.com/nasa/harmony-maskfill/releases/tag/1.3.0
[v1.2.1]: https://github.com/nasa/harmony-maskfill/releases/tag/1.2.1
[v1.2.0]: https://github.com/nasa/harmony-maskfill/releases/tag/1.2.0
[v1.1.0]: https://github.com/nasa/harmony-maskfill/releases/tag/1.1.0
[v1.0.1]: https://github.com/nasa/harmony-maskfill/releases/tag/1.0.1
[v1.0.0]: https://github.com/nasa/harmony-maskfill/releases/tag/1.0.0
