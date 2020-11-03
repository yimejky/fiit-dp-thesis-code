#!/bin/bash

jupyter labextension install @jupyter-widgets/jupyterlab-manager @jupyterlab/toc
jupyter labextension install jupyter-matplotlib jupyterlab-datawidgets itkwidgets
jupyter lab build
