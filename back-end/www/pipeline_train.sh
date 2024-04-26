#!/bin/sh

# The "RGB-I3D" model in our AAAI paper
python train.py i3d-rgb-cv-1

# The "RGB-I3D-ND" model in our AAAI paper
python train.py i3d-rgb-cv-2

# The "RGB-I3D-FP" model in our AAAI paper 
python train.py i3d-rgb-cv-3

# The "Flow-I3D" model in our AAAI paper
python train.py i3d-flow-cv-1

# The "RGB-TC" model in our AAAI paper
python train.py i3d-ft-tc-rgb-cv-1

# The "RGB-TSM" model in our AAAI paper
python train.py i3d-tsm-rgb-cv-1

# The "RGB-NL" model in our AAAI paper
python train.py i3d-nl-rgb-cv-1

# The "RGB-LSTM" model in our AAAI paper
python train.py i3d-ft-lstm-rgb-cv-1

# The "RGB-SVM" model in our AAAI paper
python train.py svm-rgb-cv-1

# The "Flow-SVM" model in our AAAI paper
python train.py svm-flow-cv-1
