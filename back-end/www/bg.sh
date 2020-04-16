#!/bin/sh

# This script runs a python script using screen command
# For example:
# sh bg.sh python train.py i3d-rgb

# Get file path
if [ "$1" != "" ] && [ "$2" != "" ]
then
  echo "Run: $1 $2 $3 $4 $5 $6"
else
  echo "Usage examples:\n\
  sh bg.sh python download_videos.py\n\
  sh bg.sh python process_videos.py\n\
  sh bg.sh python perturb_frames.py\n\
  sh bg.sh python extract_features.py i3d-rgb\n\
  sh bg.sh python extract_features.py i3d-flow\n\
  sh bg.sh python train.py i3d-rgb\n\
  sh bg.sh python train.py i3d-rgb-cv-1\n\
  sh bg.sh python train.py i3d-rgb-mil-cv-1\n\
  sh bg.sh python train.py i3d-ft-tc-rgb-cv-1\n\
  sh bg.sh python train.py i3d-ft-tc-tsm-rgb-cv-1\n\
  sh bg.sh python train.py i3d-tc-rgb-cv-1\n\
  sh bg.sh python train.py i3d-tsm-rgb-cv-1\n\
  sh bg.sh python train.py i3d-nl-rgb-cv-1\n\
  sh bg.sh python train.py i3d-lstm-rgb-cv-1\n\
  sh bg.sh python train.py i3d-flow\n\
  sh bg.sh python train.py i3d-flow-cv-1\n\
  sh bg.sh python train.py svm-rgb\n\
  sh bg.sh python train.py svm-rgb-cv-1\n\
  sh bg.sh python train.py svm-flow\n\
  sh bg.sh python train.py svm-flow-cv-1\n\
  sh bg.sh python test.py i3d-rgb [model_path]\n\
  sh bg.sh python test.py i3d-rgb-cv-1 [model_path]\n\
  sh bg.sh python test.py i3d-rgb-mil-cv-1 [model_path]\n\
  sh bg.sh python test.py i3d-ft-tc-rgb-cv-1 [model_path]\n\
  sh bg.sh python test.py i3d-ft-tc-tsm-rgb-cv-1 [model_path]\n\
  sh bg.sh python test.py i3d-tc-rgb-cv-1 [model_path]\n\
  sh bg.sh python test.py i3d-tsm-rgb-cv-1 [model_path]\n\
  sh bg.sh python test.py i3d-nl-rgb-cv-1 [model_path]\n\
  sh bg.sh python test.py i3d-lstm-rgb-cv-1 [model_path]\n\
  sh bg.sh python test.py i3d-flow [model_path]\n\
  sh bg.sh python test.py i3d-flow-cv-1 [model_path]\n\
  sh bg.sh python test.py svm-rgb-cv-1 [model_path]\n\
  sh bg.sh python test.py svm-flow-cv-1 [model_path]\n\
  sh bg.sh python grad_cam_viz.py i3d-rgb [model_path]"
  exit 1
fi

# Delete existing screen
for session in $(sudo screen -ls | grep -o "[0-9]*.$1.$2.$3")
do
  sudo screen -S "${session}" -X quit
  sleep 2
done

# Delete the log
sudo rm screenlog.0

# For python in conda env in Ubuntu
sudo screen -dmSL "$1.$2.$3" bash -c "export PATH='/opt/miniconda3/bin:$PATH'; . '/opt/miniconda3/etc/profile.d/conda.sh'; conda activate deep-smoke-machine; $1 $2 $3 $4 $5 $6"

# List screens
sudo screen -ls
exit 0
