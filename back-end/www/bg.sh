#!/bin/sh

# This script runs a python script using screen command
# For example:
# sh bg.sh python download_videos.py
# sh bg.sh python process_videos.py
# sh bg.sh python extract_features.py i3d-rgb
# sh bg.sh python extract_features.py i3d-flow
# sh bg.sh python train.py i3d-flow
# sh bg.sh python train.py i3d-rgb
# sh bg.sh python train.py svm-flow
# sh bg.sh python train.py svm-rgb
# sh bg.sh python train.py ts-rgb
# sh bg.sh python train.py ts-flow
# sh bg.sh python train.py lstm

# Get file path
if [ "$1" != "" ] && [ "$2" != "" ]
then
  echo "Run: $1 $2 $3 $4"
else
  echo "Usage examples:\n  sh bg.sh python download_videos.py\n  sh bg.sh python process_videos.py\n  sh bg.sh python extract_features.py i3d-rgb\n  sh bg.sh python extract_features.py i3d-flow\n  sh bg.sh python train.py i3d-flow\n  sh bg.sh python train.py i3d-rgb\n  sh bg.sh python train.py svm-flow\n  sh bg.sh python train.py svm-rgb\n  sh bg.sh python train.py ts-rgb\n  sh bg.sh python train.py ts-flow\n  sh bg.sh python train.py lstm\n  sh bg.sh python test.py i3d-rgb [model_path]\n  sh bg.sh python test.py i3d-flow [model_path]"
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
sudo screen -dmSL "$1.$2.$3" bash -c "export PATH='/opt/miniconda3/bin:$PATH'; . '/opt/miniconda3/etc/profile.d/conda.sh'; conda activate deep-smoke-machine; $1 $2 $3 $4"

# List screens
sudo screen -ls
exit 0
