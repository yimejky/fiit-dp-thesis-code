#!/bin/bash
cd "$(dirname "${BASH_SOURCE[0]}")" || exit
cd "../"

# jupyter and basics
pip install jupyterlab numpy pandas matplotlib ipywidgets

# saving jupyter state
pip install dill

# matplotlib 3d
pip install ipympl
pip install itkwidgets

# opencv
# pip install opencv-python
pip install opencv-contrib-python

# scikit
pip install scikit-learn scikit-image scipy

# simple itk
pip install SimpleITK

# PyTorch
pip install torch torchvision torchsummary torchio

# Tensorboard
pip install tensorboard
