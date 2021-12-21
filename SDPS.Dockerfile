#
# Commands to use this file locally:
#
# docker build -f SDPS.Dockerfile -t sds/maskfill .
# docker run -v /full/path/to/host/directory:/home/results sds/maskfill:latest "<full list of arguments>"
#
#
FROM continuumio/miniconda3

WORKDIR "/home"

# Create Conda environment.
COPY data/mask_fill_conda_requirements.txt data/mask_fill_conda_requirements.txt
RUN conda create -y --name maskfill --file data/mask_fill_conda_requirements.txt python=3.9 -q && conda clean -a

# Install additional Pip dependencies.
COPY data/mask_fill_pip_requirements.txt data/mask_fill_pip_requirements.txt
RUN conda run --name maskfill pip install --no-input -r data/mask_fill_pip_requirements.txt

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
ENTRYPOINT ["python", "MaskFill.py"]
