#
# Service image for sds/maskfill-harmony, a Harmony backend service that masks
# gridded Earth Observation data according to a user-supplied GeoJSON shape
# file. This service can process either HDF-5 or GeoTIFF files, and will
# preserve the input file format and compression in the output product.
#
# This image instantiates a conda environment, with required packages, before
# installing additional dependencies via Pip. The service code is then copied
# into the Docker image, before environment variables are set to activate the
# created conda environment.
#
# Commands to use this file locally:
#
# docker build -f Harmony.Dockerfile -t sds/maskfill-harmony .
# docker run -v /full/path/to/host/directory:/home/results sds/maskfill-harmony:latest "<full list of arguments>"
#
# 2021-06-25: Updated
# 2025-09-15: Updated for migration to GitHub and GHCR Docker image names.
# 2025-09-16: Updated entry point to align with Harmony service repository best practices.
#
FROM continuumio/miniconda3:latest

WORKDIR "/home"

# Copy Conda requirements into the container
COPY data/mask_fill_conda_requirements.txt data/mask_fill_conda_requirements.txt

# Create Conda environment
RUN conda create -y --name maskfill --file data/mask_fill_conda_requirements.txt \
    python=3.12 --channel conda-forge --override-channels  -q && conda clean -a

# Copy additional Pip dependencies into the image
COPY data/mask_fill_pip_requirements.txt data/mask_fill_pip_requirements.txt
COPY data/mask_fill_harmony_pip_requirements.txt data/mask_fill_harmony_pip_requirements.txt

# Install additional Pip dependencies.
RUN conda run --name maskfill pip install --no-input -r data/mask_fill_harmony_pip_requirements.txt

# Place contents of the repository in the container.
COPY . /home/

# Create a directory to be the destination of a mounted volume:
RUN mkdir /home/results

# Set conda environment for MaskFill, as `conda run` will not stream logging.
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

# Set GDAL related environment variables.
ENV CPL_ZIP_ENCODING=UTF-8 \
    GDAL_DATA=/opt/conda/envs/maskfill/share/gdal \
    GSETTINGS_SCHEMA_DIR=/opt/conda/envs/maskfill/share/glib-2.0/schemas \
    GSETTINGS_SCHEMA_DIR_CONDA_BACKUP='' \
    PROJ_LIB=/opt/conda/envs/maskfill/share/proj

# Configure a container to be executable via the `docker run` command.
ENTRYPOINT ["python", "-m", "maskfill"]
