#!/bin/sh

# DO NOT put the installation of pytorch in this file
# The reason is that users may have different cuda versions

# http related
pip install --upgrade requests~=2.31

# OpenCV
pip install --upgrade opencv-python~=4.9
pip install --upgrade opencv-contrib-python~=4.9

# For plotting images
pip install --upgrade matplotlib~=3.8

# For machine learning
pip install --upgrade scikit-learn~=1.4

# For TensorBoard
pip install --upgrade tb-nightly~=2.17
pip install --upgrade tensorflow~=2.16
pip install --upgrade future~=1.0
pip install --upgrade moviepy~=1.0

# For data analysis
pip install --upgrade pandas~=2.2

# For pytorch related
pip install --upgrade torchviz~=0.0
pip install --upgrade torchsummary~=1.5
