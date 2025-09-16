# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [v1.0.0] -  Unreleased

This version of the Harmony MaskFill service contains all functionality
previously released internally to EOSDIS as `sds/maskfill:0.2.1`. Minor
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

[v1.0.0]: https://github.com/nasa/harmony-maskfill/releases/tag/1.0.0
