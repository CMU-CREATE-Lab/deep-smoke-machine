#!/bin/sh

# http related
pip install requests==2.21.0

# OpenCV
pip install opencv-python==4.1.0.25

# pytorch
conda install pytorch torchvision cudatoolkit=9.0 -c pytorch

# For plotting images
pip install matplotlib==3.0.3

# For machine learning
pip install scikit-learn
