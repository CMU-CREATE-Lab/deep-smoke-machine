#!/bin/sh

# Download videos
python download_videos.py

# Process videos
# If you want to compute optical flow (for flow-based models, "Flow-SVM" and "Flow-I3D")
# Go to the process_videos.py file and change flow_type to 1
# If you do not need optical flow, you do not need to change anything
# Optionally you can set num_workers to a higher number for faster processing
python process_videos.py

# Extract features (optional)
# (only needed for training "RGB-SVM" and "Flow-SVM" models)
# Uncomment the following if you need them
#python extract_features.py i3d-rgb
#python extract_features.py i3d-flow

# Perturb frames (optional)
# This is only for the "RGB-I3D-FP" model that needs frame perturbation
# Uncomment the following if you need it
#python perturb_frames.py
