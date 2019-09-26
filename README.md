OVERVIEW:

The mask fill utility works with gridded data, applying a fill value in all pixels outside of a provided shape.
Mask fill is the second step of Subset by Shape/Polygon, following the cropping of the grid to a minimally surrounding bounding box.
The utility accepts HDF5 files which follow CF conventions and GeoTIFFs. 

INSTALLATION:

MaskFill was developed using the Anaconda distribution of Python (https://www.anaconda.com/download) and conda virutal environment.
This simplifies dependency management. Run these commands to create a mask fill conda virtual environment and install all the needed packages:

    conda create --name maskfill --file pymods/mask_fill_conda_requirements.txt
    source activate maskfill
    pip install -r pymods/mask_fill_pip_requirements.txt

