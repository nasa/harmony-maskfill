#
# The created container will have a Conda environment, in which it will run a
# Python unittest suite.
#
# The results of the test run will be saved to tests/reports, which should be
# mounted as a shared volume with the host.
#
# Commands to use this file locally:
#
# docker build -f tests/Dockerfile -t maskfill .
# docker run -v /full/path/to/host/directory/test-reports:/home/tests/reports maskfill:latest
#
# Using Conda within ENTRYPOINT was taken from:
# https://pythonspeed.com/articles/activate-conda-dockerfile/
#
FROM continuumio/miniconda3

WORKDIR "/home"

# Place contents of the repository in the container.
COPY . /home/

# Create Conda environment.
RUN conda create -y --name maskfill --file data/mask_fill_conda_requirements.txt python=3.6 -q

# Install additional Pip dependencies.
RUN conda run --name maskfill pip install --no-input -r data/mask_fill_pip_requirements.txt

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

ENV CPL_ZIP_ENCODING=UTF-8 \
    GDAL_DATA=/opt/conda/envs/maskfill/share/gdal \
    GSETTINGS_SCHEMA_DIR=/opt/conda/envs/maskfill/share/glib-2.0/schemas \
    GSETTINGS_SCHEMA_DIR_CONDA_BACKUP='' \
    PROJ_LIB=/opt/conda/envs/maskfill/share/proj

# Configure a container to be executable via the `docker run` command.
ENTRYPOINT ["/home/tests/run"]
