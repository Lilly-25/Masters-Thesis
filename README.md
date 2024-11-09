# Master Thesis

<center><h2>Electric Motor Modelling with Graph Neural Networks</h2></center>

<div>
The aim of the Master Thesis is to train a neural network to learn the parameters of Electric Motors and thus be able to predict its KPIs(key performance indicators). 

We have developed and trained a MLP neural network on the tabular representation of data to predict 2 KPIs. 

The KPIs are 2D and 3D plots on Torque(Mgrenz) curve and Efficiency(ETA) grid.

</div>

## How to install dependecies with Linux OS?

When executing the program with miniconda,

Create conda environment with the dependencies from .yml file as below:

```bash
conda env create -f environment.yml
conda activate newenv
```

When executing the program with python virtual environment

```bash
python -m newenv venv
source ./newenv/bin/activate
pip install -r requirements.txt
```

## Repo Structure

```python
.
├── src
│   ├── __init__.py
│   ├── data_preprocessing_tabular.py
│   ├── inference.py
│   ├── model.py
│   ├── scaling.py
│   ├── table_creation.py
│   ├── training.py
│   └── utils.py
├── GraphModelling
│   ├── src
│   │   ├── __init__.py
│   │   ├── data_preprocessing_graph.py
│   │   ├── dataset_creation.py
│   │   ├── model.py
│   │   ├── scaling.py
│   │   ├── graph_creation.py
│   │   ├── training.py
│   └── main_graph.ipynb  
│   └── main_graph.py
├── data
│   ├── raw
│   ├── Testing
│   │   ├── raw
├── Intermediate
│   ├── cross_val_splits.npy
│   ├── DoubleVGraph.json
│   ├── EMTabular.json
│   ├── max_mgrenz.pkl
│   ├── x_mean.pkl
│   ├── x_stddev.pkl
├── temp
│   ├── ReportPics
│   ├── wandb
│   │   ├── loss
│   │   ├── score
├── Manuscript
│   ├── ReportImages
│   ├── Report.pdf
│   ├── Report.tex
├── Presentations
├── .gitignore
├── environment.yaml
├── LICENSE
├── README.md
├── requirements.txt
├── main_tabular.py
├── main_tabular.ipynb
└── .env.local
```

## Secrets

In .env.local file store
 
WANDB_API_KEY=API KEY

## Data Preprocessing

Store the files for training within folder data -> raw

Run the python program main_tabular.py separately for generating preprocessed data

## MLP Results

Run the jupyter notebook main_tabular.ipynb

## Testing(New Files)

Within folder data -> Testing -> raw and store the new files for generating model predictions of it.