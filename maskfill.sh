#!/bin/sh

# Activate environment
source activate maskfill

# Call the Mask Fill Utility, passing on the input parameters
./MaskFillUtility.py "$@"