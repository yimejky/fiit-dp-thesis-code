#!/bin/bash

# jupyter and basics
pip install jupyterlab numpy pandas matplotlib ipywidgets

# jupyter widgets
jupyter labextension install @jupyter-widgets/jupyterlab-manager @jupyterlab/toc

# opencv
# pip install opencv-python
pip install opencv-contrib-python

# scikit
pip install scikit-learn scikit-image scipy

# simple itk
pip install SimpleITK

# PyTorch
pip install torch torchvision torchsummary

# Tensorboard
pip install tensorboard
