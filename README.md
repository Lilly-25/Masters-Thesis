# Electric Motor Modelling with Graph Neural Networks

The aim of the Master Thesis is to train a neural network to learn the parameters of Electric Motors and thus be able to predict its KPIs(key performance indicators). 

We have developed and trained a MLP neural network on the tabular representation of data to predict 2 KPIs. 

The KPIs are 2D and 3D plots on Torque(Mgrenz) curve and Efficiency(ETA) grid.


## How to install dependecies with Linux OS?

When executing the program with miniconda,

Create conda environment with the dependencies from .yml file as below:

conda env create -f environment.yml

conda activate newenv


When executing the program with python virtual environment

python -m newenv venv

source ./newenv/bin/activate

pip install -r requirements.txt


## MLP Results

Run the jupyter notebook main_tabular.ipynb

