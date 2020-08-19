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
RUN conda create --name maskfill --file data/mask_fill_conda_requirements.txt python=3.6

# Install additional Pip dependencies.
RUN conda run --name maskfill pip install -r data/mask_fill_pip_requirements.txt

# Set conda environment to maskfill, as `conda run` will not stream logging.
# Setting these environment variables is the equivalent of `conda activate`.
ENV _CE_CONDA ''
ENV _CE_M ''
ENV CONDA_DEFAULT_ENV maskfill
ENV CONDA_EXE /opt/conda/bin/conda
ENV CONDA_PREFIX /opt/conda/envs/maskfill
ENV CONDA_PREFIX_1 /opt/conda
ENV CONDA_PROMPT_MODIFIER (maskfill)
ENV CONDA_PYTHON_EXE /opt/conda/bin/python
ENV CONDA_ROOT /opt/conda
ENV CONDA_SHLVL 2
ENV PATH "/opt/conda/envs/maskfill/bin:${PATH}"
ENV SHLVL 1

ENV CPL_ZIP_ENCODING UTF-8
ENV GDAL_DATA /opt/conda/envs/maskfill/share/gdal
ENV GSETTINGS_SCHEMA_DIR /opt/conda/envs/maskfill/share/glib-2.0/schemas
ENV GSETTINGS_SCHEMA_DIR_CONDA_BACKUP ''
ENV PROJ_LIB /opt/conda/envs/maskfill/share/proj

# Configure a container to be executable via the `docker run` command.
ENTRYPOINT ["/home/tests/run"]
