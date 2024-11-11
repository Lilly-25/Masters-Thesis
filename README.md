# Master Thesis

<center><h2>Electric Motor Modelling with Graph Neural Networks</h2></center>

## 📋 Overview

The aim of the Master Thesis is to train a neural network to learn the parameters of Electric Motors and thus be able to predict its KPIs(key performance indicators). 

We have developed and trained a MLP neural network on the tabular representation of data to predict 2 KPIs. 

The KPIs are 2D and 3D plots on Torque(Mgrenz) curve and Efficiency(ETA) grid.


## ⚙️ How to install dependecies with Linux OS?

[![Python Version](https://img.shields.io/badge/python-3.10.14-blue.svg)]()

When executing the program with miniconda,

Create conda environment with the dependencies from .yml file as below:

```bash
conda env create -f environment.yml
conda activate thesis
```

When executing the program with python virtual environment

```bash
python -m thesis venv
source ./thesis/bin/activate
pip install -r requirements.txt
```

## 📁 Repo Structure

```python
.
├── src
│   ├── __init__.py
│   ├── README.md   
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
│   │   ├── README.md 
│   │   ├── data_preprocessing_graph.py
│   │   ├── dataset_creation.py
│   │   ├── model.py
│   │   ├── scaling.py
│   │   ├── graph_creation.py
│   │   └── training.py
│   └── main_graph.ipynb  
│   └── main_graph.py
├── data
│   ├── README.md   
│   ├── raw
│   ├── DoubleVGraph.json
│   ├── EMTabular.json
│   ├── Testing
│       └── raw
├── Intermediate
│   ├── README.md
│   ├── cross_val_splits.npy
│   ├── max_mgrenz.pkl
│   ├── x_mean.pkl
│   └── x_stddev.pkl
├── Manuscript
│   ├── README.md
│   ├── ReportImages
│   ├── wandb
│   │   ├── loss
│   │   └── score
│   ├── Report.pdf
│   └── Report.tex
├── Presentations
│    └── README.md
├── .gitignore
├── environment.yaml
├── LICENSE
├── README.md
├── requirements.txt
├── data_preprocessing.py
├── main.ipynb
├── main_train.py
└── .env.local
```

## 📖 Usage

### Secrets

In .env.local file store
 
WANDB_API_KEY=API KEY

### Data Preprocessing

Store the files for training within folder data -> raw

Run the python program data_preprocessing.py separately for generating preprocessed data

### MLP Results

Run the jupyter notebook main.ipynb for data explorations, training and inference

### Testing(New Files)

Within folder data -> Testing -> raw and store the new files for generating model predictions of it.