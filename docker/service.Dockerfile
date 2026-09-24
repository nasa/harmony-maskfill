#
# Service image for sds/maskfill-harmony, a Harmony backend service that masks
# gridded Earth Observation data according to a user-supplied GeoJSON shape
# file. This service can process either HDF-5 or GeoTIFF files, and will
# preserve the input file format and compression in the output product.
#
# This image installs all dependencies via Pip. The binary wheels for rasterio,
# pyproj, and h5py bundle their own GDAL, PROJ and HDF-5 libraries and data
# files, so the only system library required is expat, and no conda environment
# is needed. The service code is then copied into the Docker image.
#
# Commands to use this file locally:
#
# docker build -f docker/service.Dockerfile -t ghcr.io/nasa/harmony-maskfill .
# docker run -v /full/path/to/host/directory:/home/results ghcr.io/nasa/harmony-maskfill:latest "<full list of arguments>"
#
# 2021-06-25: Updated
# 2025-09-15: Updated for migration to GitHub and GHCR Docker image names.
# 2025-09-16: Updated entry point to align with Harmony service repository best practices.
# 2025-09-16: Updated paths to requirements files.
# 2026-09-24: Migrated from a conda environment to a Pip-only Python image.
#
FROM python:3.13-slim-trixie

WORKDIR "/home"

# The rasterio wheels link against, but do not bundle, the system expat library.
RUN apt-get update \
    && apt-get install -y --no-install-recommends libexpat1 \
    && rm -rf /var/lib/apt/lists/*

# Copy Pip requirements into the container
COPY ./pip_requirements.txt pip_requirements.txt

# Install Pip dependencies.
RUN pip install --no-input --no-cache-dir -r pip_requirements.txt

# Copy only the files the service needs at runtime.
COPY maskfill maskfill
COPY docker/service_version.txt docker/service_version.txt

# Create a directory to be the destination of a mounted volume:
RUN mkdir /home/results

# Configure a container to be executable via the `docker run` command.
ENTRYPOINT ["python", "-m", "maskfill"]
