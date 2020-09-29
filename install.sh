#!/bin/bash

rm -rf ./venv
python3 -m venv venv
source ./venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
jupyter labextension install @jupyter-widgets/jupyterlab-manager @jupyterlab/toc
