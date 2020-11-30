#!/bin/bash

jupyter labextension install @jupyter-widgets/jupyterlab-manager
jupyter labextension install @jupyterlab/toc
jupyter labextension install jupyter-matplotlib 
jupyter labextension install jupyterlab-datawidgets 
jupyter labextension install itkwidgets

jupyter lab build
