#!/bin/sh

# Commands to create an environment for the Mask Fill Utility:
# Note: the environment name is stored in data/MASKFILL_CONDA_ENVIRONMENT

# conda create --name ${env_name} python=3.9 \
#     --file ./data/file mask_fill_conda_requirements.txt \
#     --channel conda-forge --channel default
# source activate ${env_name}
# pip install -r ./data/mask_fill_pip_requirements.txt

# Retrieve the path of the MaskFill directory
maskfill_path="$(dirname ${0})"

# Retrieve the name of the conda environment
maskfill_env="$(cat ${maskfill_path}/data/MASKFILL_CONDA_ENVIRONMENT.txt)"

# Activate environment
# Need path for conda in order to "source activate"
# conda_path="$(dirname `which conda`)"
# conda_path="$(dirname ${path})/bin"  # sometimes conda is in a special condabin directory
conda_path="/tools/miniconda/bin"
source ${conda_path}/activate ${maskfill_env}

# Call the Mask Fill Utility, passing on the input parameters
exec ${maskfill_path}/MaskFill.py "$@"
