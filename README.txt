INSTALLATION

MaskFill was developed using the Anaconda distribution of Python (https://www.anaconda.com/download) and conda virutal environment.
This simplifies dependency management. Run these commands to create a mask fill conda virtual environment and install all the needed packages:

conda create --name maskfill --file pymods/conda_requirements.txt
source activate maskfill
pip install -r pymods/pip_requirements.txt