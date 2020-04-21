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

# Make RUN commands use the Conda environment.
SHELL ["conda", "run", "--name", "maskfill", "/bin/bash", "-c"]

# Install additional Pip dependencies.
RUN pip install -r data/mask_fill_pip_requirements.txt

# Set the tests to run when the container is created.
# The output is redirected from the terminal to ensure that Bamboo doesn't
# detect output and wrongly consider there to be a failure.
ENTRYPOINT ["conda", "run", "--name", "maskfill", "python", "-m", \
			"xmlrunner", "discover", "tests/python", "-o", "tests/reports", \
			">", "tests.log", "2>&1"]
