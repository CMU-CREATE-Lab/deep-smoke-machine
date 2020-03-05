#!/bin/sh

# DO NOT put the installation of pytorch in this file
# As indicated in the README file, users need to use:
# $conda install pytorch torchvision cudatoolkit=10.0 -c pytorch
# The reason is that users may have different cuda versions

# http related
pip install requests==2.22.0

# OpenCV
pip install opencv-python==4.1.1.26
pip install opencv-contrib-python==4.1.1.26

# For plotting images
pip install matplotlib==3.1.1

# For machine learning
pip install scikit-learn==0.21.3

# For TensorBoard
pip install tb-nightly==2.1.0a20191103
pip install tensorflow==2.0.0
pip install future==0.18.2
pip install moviepy==1.0.1

# For data analysis
pip install pandas==0.25.3

# For pytorch related
pip install torchviz==0.0.1
pip install torchsummary==1.5.1
