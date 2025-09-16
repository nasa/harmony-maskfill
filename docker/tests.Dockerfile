#
# Test image for the Harmony MaskFill service. This image uses the main service
# image, ghcr.io/nasa/harmony-maskfill, as a base layer for the tests. This
# ensures that the contents of the service image are tested, preventing
# discrepancies between the service and test environments.
#
# The results of the test run will be saved to tests/reports, which should be
# mounted as a shared volume with the host.
#
# Commands to use this file locally:
#
# docker build -f tests/Dockerfile -t maskfill .
# docker run -v /full/path/to/host/directory/test-reports:/home/tests/reports maskfill:latest
#
# 2021-06-25: Updated
# 2025-09-15: Updated for migration to GitHub and GHCR images.
# 2025-09-16: Updated to install test dependencies.
#
FROM ghcr.io/nasa/harmony-maskfill

# Install additional Pip requirements (for testing)
COPY tests/pip_test_requirements.txt .
RUN conda run --name maskfill pip install --no-input -r pip_test_requirements.txt

# Copy test directory containing Python unittest suite, test data and utilities
COPY ./tests tests

# Set conda environment to maskfill, as `conda run` will not stream logging.
# Setting these environment variables is the equivalent of `conda activate`.
ENV _CE_CONDA='' \
    _CE_M='' \
    CONDA_DEFAULT_ENV=maskfill \
    CONDA_EXE=/opt/conda/bin/conda \
    CONDA_PREFIX=/opt/conda/envs/maskfill \
    CONDA_PREFIX_1=/opt/conda \
    CONDA_PROMPT_MODIFIER=(maskfill) \
    CONDA_PYTHON_EXE=/opt/conda/bin/python \
    CONDA_ROOT=/opt/conda \
    CONDA_SHLVL=2 \
    PATH="/opt/conda/envs/maskfill/bin:${PATH}" \
    SHLVL=1

# GDAL specific environment variables
ENV CPL_ZIP_ENCODING=UTF-8 \
    GDAL_DATA=/opt/conda/envs/maskfill/share/gdal \
    GSETTINGS_SCHEMA_DIR=/opt/conda/envs/maskfill/share/glib-2.0/schemas \
    GSETTINGS_SCHEMA_DIR_CONDA_BACKUP='' \
    PROJ_LIB=/opt/conda/envs/maskfill/share/proj

# An environment variable used by BaseHarmonyAdapter uses to not stage files
ENV ENV=test

# Configure a container to be executable via the `docker run` command.
ENTRYPOINT ["/home/tests/run_tests.sh"]
