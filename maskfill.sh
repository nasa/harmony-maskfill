#!/bin/sh

# Commands to create an environment for the Mask Fill Utility:

# conda create --name maskfill --./data/file mask_fill_conda_requirements.txt
# source activate maskfill
# pip install -r ./data/mask_fill_pip_requirements.txt

# Activate environment
# Need path for conda in order to "source activate"
# e.g., path="/tools/miniconda/bin"
path="$(dirname `which conda`)"
path="$(dirname ${path})/bin"  # sometimes conda is in a special condabin directory
source ${path}/activate maskfill

# Call the Mask Fill Utility, passing on the input parameters
path="$(dirname ${0})"
exec ${path}/MaskFill.py "$@"
