#!/bin/sh

# DO NOT put the installation of pytorch in this file
# As indicated in the README file, users need to use:
# $conda install pytorch torchvision cudatoolkit=10.0 -c pytorch
# The reason is that users may have different cuda versions

# http related
pip install requests==2.21.0

# OpenCV
pip install opencv-python==4.1.0.25
pip install opencv-contrib-python==4.1.0.25

# For plotting images
pip install matplotlib==3.0.3

# For machine learning
pip install scikit-learn==0.21.2

# For TensorBoard
pip install tb-nightly==1.15.0a20190624
pip install tensorflow==1.14.0
pip install future==0.17.1
pip install moviepy==1.0.0
