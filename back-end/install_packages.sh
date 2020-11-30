#!/bin/sh

# DO NOT put the installation of pytorch in this file
# As indicated in the README file, users need to use:
# $conda install pytorch torchvision cudatoolkit=10.0 -c pytorch
# The reason is that users may have different cuda versions

# http related
pip install --upgrade requests==2.22.0

# OpenCV
pip install --upgrade opencv-python==4.1.1.26
pip install --upgrade opencv-contrib-python==4.1.1.26

# For plotting images
pip install --upgrade matplotlib==3.1.1

# For machine learning
pip install --upgrade scikit-learn==0.21.3

# For TensorBoard
pip install --upgrade tb-nightly==2.1.0a20191103
pip install --upgrade tensorflow==2.0.0
pip install --upgrade future==0.18.2
pip install --upgrade moviepy==1.0.1

# For data analysis
pip install --upgrade pandas==0.25.3

# For pytorch related
pip install --upgrade torchviz==0.0.1
pip install --upgrade torchsummary==1.5.1
